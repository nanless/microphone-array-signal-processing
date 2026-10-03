import importlib.util
import io
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import markdown


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_pdf = load("build_pdf", ROOT / "scripts" / "build_pdf.py")
build_site = load("build_site", ROOT / "scripts" / "build_site.py")
quality_check = load("quality_check", ROOT / "scripts" / "quality_check.py")


class PdfHeadingDestinationTest(unittest.TestCase):
    """Independent PDF objects distinguish heading identity from repeated text."""

    @staticmethod
    def fixture_bytes(named, bookmarks=(), page_texts=None):
        from pypdf import PdfWriter
        from pypdf.generic import (
            ArrayObject, BooleanObject, DecodedStreamObject, DictionaryObject, Fit, FloatObject,
            NameObject, NumberObject, TextStringObject,
        )

        writer = PdfWriter()
        pages = [writer.add_blank_page(width=595, height=842) for _ in range(3)]
        structure = DictionaryObject({NameObject("/Type"): NameObject("/StructTreeRoot")})
        structure_ref = writer._add_object(structure)
        structure_children, parent_entries = ArrayObject(), ArrayObject()
        font = writer._add_object(DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }))
        for index, (page, text) in enumerate(zip(
                pages, page_texts or ["Contents", "Body", "Body"])):
            # Literal ASCII heading text is intentionally repeated on distinct pages.
            stream = DecodedStreamObject()
            stream.set_data(("/P <</MCID 0>> BDC "
                             f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET EMC").encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            page[NameObject("/StructParents")] = NumberObject(index)
            element_ref = writer._add_object(DictionaryObject({
                NameObject("/Type"): NameObject("/StructElem"),
                NameObject("/S"): NameObject("/P"),
                NameObject("/P"): structure_ref,
                NameObject("/Pg"): page.indirect_reference,
                NameObject("/K"): NumberObject(0),
            }))
            structure_children.append(element_ref)
            parent_entries.extend([NumberObject(index), ArrayObject([element_ref])])
        structure[NameObject("/K")] = structure_children
        structure[NameObject("/ParentTree")] = writer._add_object(DictionaryObject({
            NameObject("/Nums"): parent_entries,
        }))
        writer._root_object[NameObject("/StructTreeRoot")] = structure_ref
        writer._root_object[NameObject("/MarkInfo")] = DictionaryObject({
            NameObject("/Marked"): BooleanObject(True),
        })
        for name, page_number, left, top, zoom in named:
            page_reference = (pages[page_number].indirect_reference
                              if 0 <= page_number < len(pages)
                              else NumberObject(page_number))
            destination = ArrayObject([
                page_reference, NameObject("/XYZ"), FloatObject(left),
                FloatObject(top), FloatObject(zoom),
            ])
            writer.add_named_destination_array(TextStringObject(name), destination)
        for title, page_number, left, top, zoom in bookmarks:
            writer.add_outline_item(title, page_number,
                                    fit=Fit.xyz(left=left, top=top, zoom=zoom))
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    def fixture_reader(self, named, bookmarks=(), page_texts=None):
        from pypdf import PdfReader
        return PdfReader(io.BytesIO(self.fixture_bytes(named, bookmarks, page_texts)))

    def test_heading_destination_accepts_bare_and_chrome_slash_names(self):
        for key in ("ch-0", "/ch-0"):
            with self.subTest(key=key):
                reader = self.fixture_reader([(key, 1, 37, 713, 1.25)])
                destination = build_pdf.heading_destination(reader, "ch-0")
                self.assertEqual(reader.get_destination_page_number(destination), 1)
                self.assertEqual(destination.typ, "/XYZ")
                self.assertEqual(float(destination.left), 37)
                self.assertEqual(float(destination.top), 713)
                self.assertEqual(float(destination.zoom), 1.25)

    def test_heading_destination_accepts_equivalent_dual_names(self):
        reader = self.fixture_reader([
            ("ch-0", 1, 37, 713, 1.25), ("/ch-0", 1, 37, 713, 1.25),
        ])
        destination = build_pdf.heading_destination(reader, "ch-0")
        self.assertEqual(reader.get_destination_page_number(destination), 1)
        self.assertEqual(float(destination.top), 713)

    def test_heading_destination_rejects_missing_and_unresolvable_pages(self):
        for named in ([], [("/ch-0", 999, 37, 713, 1.25)]):
            with self.subTest(named=named):
                reader = self.fixture_reader(named)
                with self.assertRaises(SystemExit):
                    build_pdf.heading_destination(reader, "ch-0")

    def test_heading_destination_rejects_ambiguous_page_or_view(self):
        for other in (("/ch-0", 2, 37, 713, 1.25),
                      ("/ch-0", 1, 37, 600, 1.25)):
            with self.subTest(other=other):
                reader = self.fixture_reader([("ch-0", 1, 37, 713, 1.25), other])
                with self.assertRaises(SystemExit):
                    build_pdf.heading_destination(reader, "ch-0")

    def test_bookmarks_follow_ids_despite_repeated_or_missing_heading_text(self):
        from pypdf import PdfReader
        named = [
            ("/ch-0", 1, 30, 760, 1.1),
            ("/ch-0-first", 1, 31, 640, 1.2),
            ("/ch-0-second", 2, 32, 720, 1.3),
            ("/ch-0-math", 2, 33, 510, 1.4),
        ]
        outline = [("Repeated title", "ch-0", [
            ("Repeated title", "ch-0-first", []),
            ("Repeated title", "ch-0-second", [(r"$\Delta$ math", "ch-0-math")]),
        ])]
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "heading-identities.pdf"
            pdf_path.write_bytes(self.fixture_bytes(
                named, page_texts=["Repeated title", "Repeated title", "Body only"]))
            original = PdfReader(pdf_path)
            self.assertIn("Repeated title", original.pages[0].extract_text())
            self.assertIn("Repeated title", original.pages[1].extract_text())
            self.assertNotIn("math", original.pages[2].extract_text())
            build_pdf.add_bookmarks(pdf_path, outline)
            reader = PdfReader(pdf_path)
            chapter, children = reader.outline
            first, second, grandchildren = children
            destinations = [chapter, first, second, grandchildren[0]]
            for bookmark, (page, left, top, zoom) in zip(destinations, [
                (1, 30, 760, 1.1), (1, 31, 640, 1.2),
                (2, 32, 720, 1.3), (2, 33, 510, 1.4),
            ]):
                self.assertEqual(reader.get_destination_page_number(bookmark), page)
                self.assertEqual(bookmark.typ, "/XYZ")
                self.assertEqual(float(bookmark.left), left)
                self.assertEqual(float(bookmark.top), top)
                self.assertAlmostEqual(float(bookmark.zoom), zoom)
            self.assertEqual(set(reader.named_destinations), {row[0] for row in named})
            self.assertTrue(reader.trailer["/Root"]["/MarkInfo"]["/Marked"])
            tree = reader.trailer["/Root"]["/StructTreeRoot"]
            self.assertEqual(len(tree["/K"]), 3)
            self.assertEqual(len(tree["/ParentTree"]["/Nums"]), 6)
            self.assertEqual([page["/StructParents"] for page in reader.pages], [0, 1, 2])

    def test_bookmark_gate_accepts_same_page_and_top_and_rejects_wrong_page(self):
        for bookmark_page, expected_ok in ((1, True), (0, False)):
            with self.subTest(bookmark_page=bookmark_page):
                reader = self.fixture_reader(
                    [("/ch-0", 1, 37, 713, 1.25)],
                    [("Repeated title", bookmark_page, 37, 713, 1.25)],
                    page_texts=["Repeated title", "Repeated title", "Body only"])
                bookmark = reader.outline[0]
                issues = quality_check.bookmark_destination_issues(reader, bookmark, "ch-0")
                self.assertEqual(issues == [], expected_ok)

    def test_bookmark_gate_rejects_wrong_top_even_when_page_and_title_match(self):
        reader = self.fixture_reader(
            [("/ch-0", 1, 37, 713, 1.25)], [("Repeated title", 1, 37, 600, 1.25)])
        self.assertTrue(quality_check.bookmark_destination_issues(
            reader, reader.outline[0], "ch-0"))

    def test_bookmark_gate_rejects_missing_and_unresolvable_named_target(self):
        for named in ([], [("/ch-0", 999, 37, 713, 1.25)]):
            with self.subTest(named=named):
                reader = self.fixture_reader(named, [("Repeated title", 1, 37, 713, 1.25)])
                self.assertTrue(quality_check.bookmark_destination_issues(
                    reader, reader.outline[0], "ch-0"))


class BuildHelpersTest(unittest.TestCase):
    @staticmethod
    def budget_fixture(headers, *, aligned=False):
        cells = ['`$x<y$`', '$z^H z$', '[证据](https://example.org/source)',
                 '完整说明', '第五列'][:len(headers)]
        separators = ['---'] * len(headers)
        if aligned:
            separators[0] = ':---:'
        return ('| ' + ' | '.join(headers) + ' |\n| ' +
                ' | '.join(separators) + ' |\n| ' + ' | '.join(cells) + ' |\n')

    def test_narrow_table_budget_matches_complete_headers_and_exact_source(self):
        # Published contracts are specified independently of the policy dictionary.
        cases = [
            (ROOT / 'chapters/00_overview.md',
             ('编号', '文件', '配图', '难度', '建议学习单元'), 60, {1: 3, 4: 6}),
            (ROOT / 'chapters/00_overview.md',
             ('图', '文件', '所在文档', '类型', '内容'), 60, {1: 5, 4: 6}),
            (ROOT / 'chapters/12_appendix-symbols-math.md',
             ('权重', '误差均方的逐项计算', '解析 MSE', '解析 NMSE（除以 $0.02$）'),
             52, {1: 6}),
            (build_site.RESEARCH_ROOT / '05_exercises_and_audio.md',
             ('题号', '输入和计算', '能支持的结论'), 36, {1: 6}),
        ]
        for path, headers, width, columns in cases:
            with self.subTest(path=path, headers=headers):
                html, _ = build_site.render(self.budget_fixture(headers), path)
                self.assertNotIn('tutorial-budget-table', build_site.render(
                    self.budget_fixture(headers), ROOT / 'chapters/fixture.md')[0])
                self.assertNotIn('tutorial-budget-table', build_site.render(
                    self.budget_fixture(tuple(reversed(headers))), path)[0])
                document = ET.fromstring('<fixture>' + html + '</fixture>')
                table = document.find('.//table')
                self.assertIsNotNone(table)
                self.assertIn('tutorial-budget-table', table.get('class', '').split())
                self.assertIn(f'--tutorial-table-width:{width}em', table.get('style', ''))
                self.assertIsNotNone(table.find('thead/tr'))
                self.assertIsNotNone(table.find('tbody/tr'))
                self.assertEqual([''.join(th.itertext()) for th in table.findall('thead/tr/th')],
                                 list(headers))
                for row in table.findall('.//tr'):
                    self.assertEqual(len(row), len(headers))
                    for index, cell in enumerate(row, 1):
                        if cell.tag == 'th':
                            self.assertEqual(cell.get('scope'), 'col')
                        if index in columns:
                            self.assertIn('tutorial-short-column', cell.get('class', '').split())
                            self.assertIn(f'--tutorial-column-width:{columns[index]}em',
                                          cell.get('style', ''))
                        else:
                            self.assertNotIn('tutorial-short-column', cell.get('class', '').split())
                region = document.find('.//div[@class="table-scroll"]')
                self.assertEqual(region.get('role'), 'region')
                self.assertEqual(region.get('tabindex'), '0')
                self.assertIs(region.find('table'), table)

    def test_narrow_table_budget_rejects_similar_headers_and_wrong_file(self):
        path = ROOT / 'chapters/12_appendix-symbols-math.md'
        headers = ('符号', '含义')
        source = self.budget_fixture(headers)
        # resolve() accepts the same file spelled through a parent directory.
        alias = path.parent / '..' / 'chapters' / path.name
        self.assertIn('tutorial-budget-table', build_site.render(source, alias)[0])
        for other in (ROOT / 'chapters/11_selection-guide.md',
                      build_site.RESEARCH_ROOT / path.name,
                      ROOT / 'copied' / path.name):
            with self.subTest(other=other):
                self.assertNotIn('tutorial-budget-table', build_site.render(source, other)[0])
        for near in (('含义', '符号'), ('符号', '含义说明'), ('符号',),
                     ('符号', '含义', '附加列'), ('符号', '含义', '含义')):
            with self.subTest(headers=near):
                self.assertNotIn('tutorial-budget-table',
                                 build_site.render(self.budget_fixture(near), path)[0])
        unrelated = self.budget_fixture(('观测', '条件'))
        combined, _ = build_site.render(source + '\n' + unrelated, path)
        plain, _ = build_site.render(unrelated, path)
        combined_tables = ET.fromstring('<fixture>' + combined + '</fixture>').findall('.//table')
        plain_table = ET.fromstring('<fixture>' + plain + '</fixture>').find('.//table')
        self.assertEqual(len(combined_tables), 2)
        self.assertEqual(ET.tostring(combined_tables[1]), ET.tostring(plain_table))

    def test_narrow_table_budget_preserves_alignment_math_code_and_links(self):
        path = build_site.RESEARCH_ROOT / '05_exercises_and_audio.md'
        source = self.budget_fixture(('题号', '输入和计算', '能支持的结论'), aligned=True)
        html, _ = build_site.render(source, path)
        baseline, _ = build_site.render(source, ROOT / 'chapters/fixture.md')
        table = ET.fromstring('<fixture>' + html + '</fixture>').find('.//table')
        ordinary = ET.fromstring('<fixture>' + baseline + '</fixture>').find('.//table')
        self.assertEqual(len(list(table.iter())), len(list(ordinary.iter())))
        for actual, original in zip(table.iter(), ordinary.iter()):
            self.assertEqual(actual.tag, original.tag)
            self.assertEqual(actual.text, original.text)
            self.assertEqual(actual.tail, original.tail)
            # Policy additions must retain the original attributes, especially alignment.
            for key, value in original.attrib.items():
                if key == 'style':
                    self.assertIn(value, actual.get(key, ''))
                elif key == 'class':
                    self.assertTrue(set(value.split()).issubset(actual.get(key, '').split()))
                else:
                    self.assertEqual(actual.get(key), value)
        self.assertIn('text-align: center;', table.find('thead/tr/th').get('style'))
        self.assertEqual(table.find('tbody/tr/td/code').text, '$x<y$')
        self.assertEqual(table.find('tbody/tr')[1].text, '$z^H z$')
        self.assertEqual(table.find('.//a').get('href'), 'https://example.org/source')
        self.assertEqual(table.find('.//a').text, '证据')
        for quote in ('"', "'"):
            with self.subTest(existing_attribute_quote=quote):
                raw = (f'<table class={quote}original-table{quote} '
                       f'style={quote}border-collapse:collapse{quote}>'
                       '<thead><tr>'
                       f'<th class={quote}original-header{quote} '
                       f'style={quote}text-align:center{quote}>题号</th>'
                       '<th>输入和计算</th><th>能支持的结论</th></tr></thead>'
                       '<tbody><tr><td>例</td><td>计算</td><td>边界</td></tr></tbody></table>')
                rendered, _ = build_site.render(raw, path)
                # XML parsing also rejects duplicate class/style attributes.
                preserved = ET.fromstring('<fixture>' + rendered + '</fixture>').find('.//table')
                self.assertIn('original-table', preserved.get('class').split())
                self.assertIn('tutorial-budget-table', preserved.get('class').split())
                self.assertIn('border-collapse:collapse', preserved.get('style'))
                first_header = preserved.find('thead/tr/th')
                self.assertIn('original-header', first_header.get('class').split())
                self.assertIn('tutorial-short-column', first_header.get('class').split())
                self.assertIn('text-align:center', first_header.get('style'))
                self.assertEqual(first_header.get('scope'), 'col')

    def test_narrow_table_budget_css_is_screen_only_and_keeps_native_display(self):
        mobile = ('@media screen and (max-width:900px){'
                  '.tutorial-budget-table{min-width:var(--tutorial-table-width)}'
                  '.tutorial-budget-table .tutorial-short-column{'
                  'min-width:var(--tutorial-column-width);overflow-wrap:normal}}')
        self.assertIn(mobile, build_site.CSS)
        self.assertNotIn('tutorial-budget-table', build_site.CSS.replace(mobile, ''))
        self.assertNotIn('tutorial-short-column', build_pdf.CSS)
        self.assertNotIn('tutorial-budget-table', build_pdf.CSS)
        self.assertNotIn('display:block', mobile)

    def test_shared_inline_layout_changes_both_publisher_and_gate_digests(self):
        # Substitute bytes in memory: no real asset or source file is rewritten.
        inline = (ROOT / 'scripts/inline_layout.js').resolve()
        contents = [b'window.fixtureLayout = 1;']
        visited = set()
        def fixture_bytes(path):
            path = path.resolve()
            visited.add(path)
            if path == inline:
                return contents[0]
            return ('unchanged fixture: ' + path.relative_to(ROOT).as_posix()).encode('utf-8')
        with mock.patch.object(Path, 'read_bytes', fixture_bytes):
            before = (build_site.source_digest(), quality_check.site_source_digest(),
                      build_pdf.source_digest(), quality_check.source_digest())
            self.assertIn(inline, visited)
            self.assertEqual(before[0], before[1])
            self.assertEqual(before[2], before[3])
            self.assertEqual(before, (build_site.source_digest(), quality_check.site_source_digest(),
                                      build_pdf.source_digest(), quality_check.source_digest()))
            contents[0] = b'window.fixtureLayout = 2;'
            after = (build_site.source_digest(), quality_check.site_source_digest(),
                     build_pdf.source_digest(), quality_check.source_digest())
        self.assertEqual(after[0], after[1])
        self.assertEqual(after[2], after[3])
        for old, new in zip(before, after):
            self.assertNotEqual(old, new)

    def test_terminal_rule_is_removed_without_touching_internal_section_rule(self):
        html = '<p>定义</p><hr />\n<p>章末正文</p>\n<hr />\n'
        self.assertEqual(build_pdf.strip_terminal_rule(html),
                         '<p>定义</p><hr />\n<p>章末正文</p>')
        self.assertEqual(build_pdf.strip_terminal_rule('<p>无结尾线</p>'),
                         '<p>无结尾线</p>')

    def test_empty_pdf_page_distinguishes_text_and_image_pages(self):
        self.assertTrue(quality_check.pdf_page_is_empty(' \n ', 0))
        self.assertFalse(quality_check.pdf_page_is_empty('一行正文', 0))
        self.assertFalse(quality_check.pdf_page_is_empty('', 1))

    def test_sparse_pdf_page_catches_orphan_without_flagging_figures(self):
        self.assertTrue(quality_check.pdf_page_is_sparse('章末一句转场。', 0, 233))
        self.assertFalse(quality_check.pdf_page_is_sparse('章末一句转场。', 1, 233))
        self.assertFalse(quality_check.pdf_page_is_sparse('章末一句转场。', 0, 3))
        self.assertFalse(quality_check.pdf_page_is_sparse('正文' * 60, 0, 233))

    def test_book_end_stays_with_final_paragraph(self):
        html = '<p>前段</p>\n<p>最后一段。</p>\n<hr />\n'
        result = build_pdf.append_book_end(html)
        self.assertTrue(result.startswith('<p>前段</p>\n<div class="book-ending"><p>最后一段。</p>'))
        self.assertEqual(result.count('全书完'), 1)
        self.assertTrue(result.endswith('<div class="book-end">全书完</div></div>'))
        with self.assertRaises(ValueError):
            build_pdf.append_book_end('<table><tr><td>没有结尾段落</td></tr></table>')

    def test_narrow_screen_math_keeps_local_scroll_and_accessible_copy(self):
        self.assertIn('max-width:100%;min-width:0!important', build_site.CSS)
        import re
        for label, css in (('site', build_site.CSS), ('pdf', build_pdf.CSS)):
            with self.subTest(publisher=label):
                # The body selector must outrank MathJax's later element-only rule.
                rule = re.search(r'body\s+mjx-assistive-mml\s*\{([^}]*)\}', css)
                self.assertIsNotNone(rule)
                properties = dict(item.strip().split(':', 1)
                                  for item in rule.group(1).split(';') if item.strip())
                for key, value in (('width', '1px!important'),
                                   ('max-width', '1px!important'),
                                   ('min-width', '0!important'),
                                   ('height', '1px!important'),
                                   ('overflow', 'hidden!important')):
                    self.assertEqual(properties.get(key), value)
                self.assertNotRegex(css, r'mjx-assistive-mml[^}]*display\s*:\s*none')
        self.assertIn('mjx-container[jax="CHTML"]:not([display="true"]){display:inline-block;vertical-align:middle}', build_site.CSS)
        self.assertNotIn('mjx-assistive-mml{display:none', build_site.CSS)

    def test_combined_links_become_internal(self):
        html = '<a href="./01_problem-definition.html">第一章</a>'
        self.assertEqual(
            build_pdf.rewrite_book_links(html),
            '<a href="#ch-1">第一章</a>',
        )

    def test_combined_links_keep_deep_section_anchor(self):
        html = '<a href="04_doa-estimation.html#sec-7">定位小节</a>'
        self.assertEqual(
            build_pdf.rewrite_book_links(html),
            '<a href="#ch-4-sec-7">定位小节</a>',
        )

    def test_combined_links_keep_semantic_section_anchor(self):
        html = '<a href="04_doa-estimation.html#sec-4-2">定位小节</a>'
        self.assertEqual(
            build_pdf.rewrite_book_links(html),
            '<a href="#ch-4-sec-4-2">定位小节</a>',
        )

    def test_combined_page_title_anchor_maps_to_chapter(self):
        self.assertEqual(
            build_pdf.rewrite_book_links(
                '<a href="04_doa-estimation.html#sec-1">定位章</a>'
            ),
            '<a href="#ch-4">定位章</a>',
        )

    def test_combined_links_reject_unknown_anchor_scheme(self):
        with self.assertRaisesRegex(ValueError, "无法映射章节锚点"):
            build_pdf.rewrite_book_links(
                '<a href="04_doa-estimation.html#custom">定位小节</a>'
            )

    def test_repository_links_become_portable_in_combined_pdf(self):
        html = '<a href="../codes/chapters/ch06/core/aec.py">AEC code</a>'
        rewritten = build_pdf.rewrite_repository_links(html)
        self.assertEqual(
            rewritten,
            '<a href="https://github.com/nanless/microphone-array-signal-processing/blob/main/codes/chapters/ch06/core/aec.py">AEC code</a>',
        )
        self.assertEqual(
            build_pdf.rewrite_repository_links('<a href="https://example.com/a">a</a>'),
            '<a href="https://example.com/a">a</a>',
        )

    def test_exercise_links_are_namespaced_and_unknown_ids_rejected(self):
        chapter = build_pdf.SRC / "06_aec.md"
        self.assertEqual(
            build_pdf.rewrite_repository_links('<a href="#e06-21">练习</a>', chapter),
            '<a href="#ch-6-e06-21">练习</a>')
        self.assertEqual(
            build_pdf.rewrite_book_links('<a href="06_aec.html#e06-21">练习</a>'),
            '<a href="#ch-6-e06-21">练习</a>')
        with self.assertRaisesRegex(ValueError, "无法映射章节锚点"):
            build_pdf.rewrite_repository_links('<a href="#e06-99">错链</a>', chapter)
        html, _ = build_pdf.build_html("2026-09-26")
        self.assertIn('id="ch-6-e06-21"', html)
        self.assertIn('href="#ch-6-e06-21"', html)
        self.assertNotIn('id="e06-21"', html)

    def test_page_info_block_removal_pattern_does_not_leave_empty_quote(self):
        html = '<hr><blockquote>\n<p>📄 <a href="#ch-0">回首页</a></p>\n</blockquote>'
        cleaned = build_pdf.remove_page_info(html)
        self.assertNotIn("blockquote", cleaned)

    def test_toc_exercise_id_protection_preserves_labels_links_and_outline(self):
        import re
        positives = (
            ('题15（E11-15）', ['E11-15']),
            ('E11-14 与 E11-15', ['E11-14', 'E11-15']),
            ('（E11-15）。', ['E11-15']),
            ('E11-15.', ['E11-15']),
        )
        negatives = (
            'codes/E11-15/chapter.py', 'E11-15.py', './E11-15',
            '题E11-15', 'E11-15题', 'αE11-15', 'E11-15α',
            'E111-15', 'E11-150', 'E111-150', 'E11-15-extra',
            r'C:\cases\E11-15.py', r'folder\E11-15', r'E11-15\result',
            '\u0301E11-15', 'E11-15\u0301',
            '\u0903E11-15', 'E11-15\u0903',
            '\u20ddE11-15', 'E11-15\u20dd',
            'E11-15.\u0301', 'E１１-１５',
        )
        for title, expected_ids in positives:
            with self.subTest(title=title):
                label = ET.fromstring('<label>' + build_pdf.toc_label(title) + '</label>')
                self.assertEqual(''.join(label.itertext()), title)
                self.assertEqual([span.text for span in label.findall('span')], expected_ids)
                for span in label.findall('span'):
                    self.assertEqual(span.attrib, {'class': 'tutorial-exercise-id'})
        for title in negatives:
            with self.subTest(unchanged_identifier=title):
                self.assertEqual(build_pdf.toc_label(title), title)

        # Exercise labels are wrapped only in the real generated TOC, not in
        # source headings, bookmark text, link destinations or code content.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            filename = '11_selection-guide.md'
            title = positives[0][0]
            subtitle = positives[1][0]
            (source / filename).write_text(
                '## 夹具\n\n### ' + title + '\n\n#### ' + subtitle +
                '\n\n`E11-15.py` 与 $x^H x$ 保留。\n\n### codes/E11-15/chapter.py\n\n末段保留。\n' +
                '\n'.join('\n### ' + negative + '\n\n反例正文保留。\n'
                          for negative in negatives[-11:]),
                encoding='utf-8')
            with mock.patch.object(build_pdf, 'SRC', source), \
                    mock.patch.object(build_pdf, 'CHAPTERS', [(filename, '选型夹具')]), \
                    mock.patch.object(build_pdf, 'PDF_THIRD_LEVEL_CHAPTER_IDS', {'ch-11'}), \
                    mock.patch.object(build_pdf, 'source_digest', return_value='fixture'), \
                    mock.patch.object(build_pdf, 'check_mathjax_assets'), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                actual, actual_outline = build_pdf.build_html('2026-10-02')
                with mock.patch.object(build_pdf, 'toc_label', side_effect=lambda text: text):
                    baseline, baseline_outline = build_pdf.build_html('2026-10-02')
                restored_outline = build_pdf.outline_from_html(actual)
        self.assertEqual(actual_outline, baseline_outline)
        self.assertEqual(actual_outline[0][2][0][0], title)
        self.assertEqual(actual_outline[0][2][0][2][0][0], subtitle)
        self.assertEqual(restored_outline, baseline_outline)
        def toc_links(page):
            fragment = re.search(r'<section class="toc-page">.*?</section>', page, re.S)
            self.assertIsNotNone(fragment)
            return ET.fromstring(fragment.group(0)).findall('.//a')
        actual_links, baseline_links = toc_links(actual), toc_links(baseline)
        self.assertEqual([(link.attrib, ''.join(link.itertext())) for link in actual_links],
                         [(link.attrib, ''.join(link.itertext())) for link in baseline_links])
        self.assertEqual([span.text for link in actual_links for span in link.findall('span')],
                         ['E11-15', 'E11-14', 'E11-15'])
        self.assertEqual(actual.split('</section>', 1)[1], baseline.split('</section>', 1)[1])

    def test_inserted_topic_keeps_appendix_heading_and_exercise_identity(self):
        """Reading order changes cannot retarget already-published ch-12 links."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            names = ('00_overview.md', '14_acoustic-imaging.md',
                     '12_appendix-symbols-math.md', '13_appendix-guide.md')
            (source / names[0]).write_text('# 导读\n\n## 路径\n\n末段。\n')
            (source / names[1]).write_text(
                '## 专题\n\n### 14.1 模型\n\n'
                '[旧附录](12_appendix-symbols-math.html#e12-01)\n\n末段。\n')
            (source / names[2]).write_text(
                '## 数学\n\n### 12.1 原主题\n\n<a id="e12-01"></a>\n\n末段。\n')
            (source / names[3]).write_text('## 路径\n\n### 13.1 原路径\n\n末段。\n')
            chapters = [(name, name) for name in names]
            with mock.patch.object(build_pdf, 'SRC', source), \
                    mock.patch.object(build_pdf, 'CHAPTERS', chapters), \
                    mock.patch.object(build_pdf, 'source_digest', return_value='fixture'), \
                    mock.patch.object(build_pdf, 'check_mathjax_assets'), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                page, outline = build_pdf.build_html('2026-10-04')
            self.assertEqual([item[1] for item in outline],
                             ['ch-0', 'ch-14', 'ch-12', 'ch-13'])
            self.assertIn('id="ch-12-sec-12-1"', page)
            self.assertIn('id="ch-12-e12-01"', page)
            self.assertIn('href="#ch-12-e12-01"', page)
            self.assertNotIn('id="ch-2-sec-12-1"', page)
            self.assertIn('全书完', page)
            documents = {name: (source / name).read_text() for name in names}
            with mock.patch.object(quality_check, 'EXPECTED_CHAPTERS', chapters):
                expected = quality_check.expected_outline_heading_ids(documents)
            self.assertEqual([ids[0] for ids in expected],
                             ['ch-0', 'ch-14', 'ch-12', 'ch-13'])

    def test_outline_restores_sections(self):
        html = (
            '<div class="chap" id="ch-0"><h1>导读</h1>'
            '<h2 id="ch-0-s0">开始</h2><p>正文</p></div></body>'
        )
        self.assertEqual(
            build_pdf.outline_from_html(html),
            [("导读", "ch-0", [("开始", "ch-0-s0", [])])],
        )

    def test_outline_restores_third_level_only_for_configured_chapters(self):
        html = (
            '<div class="chap" id="ch-6"><h1>AEC</h1>'
            '<h2 id="ch-6-sec-6-1">问题</h2>'
            '<h3 id="ch-6-sec-u-a">FDKF</h3></div>'
            '<div class="chap" id="ch-1"><h1>问题定义</h1>'
            '<h2 id="ch-1-sec-1-1">双耳</h2>'
            '<h3 id="ch-1-sec-u-b">局部说明</h3></div>'
            '<div class="chap" id="ch-7"><h1>去混响</h1>'
            '<h2 id="ch-7-sec-7-1">模型</h2>'
            '<h3 id="ch-7-sec-u-d">局部说明</h3></div>'
            '<div class="chap" id="ch-10"><h1>工程</h1>'
            '<h2 id="ch-10-sec-10-1">采样时钟</h2>'
            '<h3 id="ch-10-sec-u-c">漂移实验</h3></div></body>'
        )
        self.assertEqual(
            build_pdf.outline_from_html(html),
            [
                ("AEC", "ch-6", [("问题", "ch-6-sec-6-1", [("FDKF", "ch-6-sec-u-a")])]),
                ("问题定义", "ch-1", [("双耳", "ch-1-sec-1-1", [("局部说明", "ch-1-sec-u-b")])]),
                ("去混响", "ch-7", [("模型", "ch-7-sec-7-1", [("局部说明", "ch-7-sec-u-d")])]),
                ("工程", "ch-10", [("采样时钟", "ch-10-sec-10-1", [("漂移实验", "ch-10-sec-u-c")])]),
            ],
        )

    def test_pdf_outline_tree_keeps_three_level_parentage(self):
        chapter, section, subsection = object(), object(), object()
        self.assertEqual(
            quality_check.pdf_outline_tree(
                [chapter, [section, [subsection]]]
            ),
            [[chapter, [[section, [[subsection, []]]]]]],
        )

    def test_bookmark_title_match_allows_missing_mathjax_text_only(self):
        title = r"7.1.1 $\Delta$ 与 $K$ 的参数换算"
        self.assertTrue(quality_check.bookmark_title_matches_page(
            title, "7.1.1  与  的参数换算\n先把帧数换成物理时间。"))
        self.assertFalse(quality_check.bookmark_title_matches_page(
            title, "7.1.2 在线 WPE 的递推更新"))

    def test_pdf_math_probe_rejects_silent_missing_glyphs(self):
        beginning = "不提前舍入时，上例的精确分数"
        ending = "分区块频域卡尔曼滤波"
        build_pdf.validate_pdf_math_example(beginning + "  、 、 、 " + ending, [115, 74])
        with self.assertRaisesRegex(SystemExit, "数学字形探针"):
            build_pdf.validate_pdf_math_example(beginning + "  、 、 、 " + ending, [74])
        with self.assertRaisesRegex(SystemExit, "正文边界缺失"):
            build_pdf.validate_pdf_math_example("没有目标算例", [115, 74])

    def test_locate_falls_back_to_unique_title_prefix(self):
        pages = ["目录", "4.2 GCC-PHAT：\n对齐两段录音"]
        self.assertEqual(
            build_pdf.locate(
                pages,
                "4.2 GCC-PHAT：“对齐两段录音，找最吻合的错位”",
                0,
                {0},
            ),
            1,
        )

    def test_locate_handles_mathjax_text_missing_from_pdf_extraction(self):
        pages = [
            "目录\n7.1.1 Delta 与 K 的参数换算",
            "7.1.1  与  的参数换算\n先把帧数换成物理时间。",
        ]
        self.assertEqual(
            build_pdf.locate(
                pages,
                r"7.1.1 $\Delta$ 与 $K$ 的参数换算",
                0,
                {0},
            ),
            1,
        )

    def test_site_keeps_mathjax_parenthesis_escapes(self):
        self.assertIn("['\\\\(', '\\\\)']", build_site.PAGE)

    def test_content_headings_promote_to_one_h1(self):
        html = '<h2 id="sec-1">篇名</h2><h3 id="sec-2">小节</h3>'
        promoted, heads = build_site.promote_content_headings(
            html, [(2, "篇名"), (3, "小节")]
        )
        self.assertEqual(promoted.count("<h1"), 1)
        self.assertIn('<h2 id="sec-2">小节</h2>', promoted)
        self.assertEqual(heads, [(1, "篇名"), (2, "小节")])

    def test_content_navigation_includes_promoted_source_h2(self):
        heads = [(1, "篇名"), (2, "小节")]
        nav, _ = build_site.sub_list(
            "06_aec.html", heads, include_level1=True)
        self.assertIn("篇名", nav)
        self.assertIn("小节", nav)

    def test_source_digest_is_stable_length(self):
        self.assertRegex(build_pdf.source_digest(), r"^[0-9a-f]{12}$")

    def test_pdf_prefers_searchable_chinese_font(self):
        self.assertLess(build_pdf.CSS.index('"STHeiti"'),
                        build_pdf.CSS.index('"Hiragino Sans GB"'))

    def test_pdf_only_digest_parser(self):
        self.assertEqual(
            build_pdf.digest_from_html("源文件 sha256 0123456789ab"),
            "0123456789ab",
        )

    def test_unrendered_math_detector_catches_tex_but_not_plain_text(self):
        self.assertTrue(build_pdf.contains_unrendered_math(r"残留 \frac{a}{b}"))
        self.assertTrue(build_pdf.contains_unrendered_math(r"残留 \mathbf{x}"))
        self.assertTrue(build_pdf.contains_unrendered_math(r"残留 \sum_k x_k"))
        self.assertTrue(build_pdf.contains_unrendered_math("残留 $x+y$"))
        self.assertFalse(build_pdf.contains_unrendered_math("公式已经渲染为可搜索文字"))

    def test_site_source_digest_is_stable_length(self):
        self.assertRegex(build_site.source_digest(), r"^[0-9a-f]{12}$")

    def test_site_heading_parser_ignores_backtick_and_tilde_fences(self):
        markdown = (
            "## 正文标题\n"
            "```text\n### 代码里的井号\n```\n"
            "~~~~text\n### 另一段代码里的井号\n~~~~\n"
            "### 正文小节\n"
        )
        self.assertEqual(
            build_site.parse_headings(markdown),
            [(2, "正文标题"), (3, "正文小节")],
        )

    def test_site_render_uses_semantic_anchor_and_keeps_legacy_alias(self):
        # A synthetic, never-published document uses ordinal fallback aliases.
        # The real overview now has a frozen map and must not reuse sec-2 here.
        html, count = build_site.render("## 篇名\n### 10.1 延迟\n",
                                       ROOT / "chapters" / "fixture.md")
        self.assertEqual(count, 2)
        self.assertIn('id="sec-10-1"', html)
        self.assertIn('id="sec-2" class="anchor-alias"', html)
        self.assertEqual(html.count("<h2"), 1)
        self.assertEqual(html.count("<h3"), 1)

    def test_restructured_chapter_keeps_old_semantic_anchor(self):
        html, _ = build_site.render(
            "## AEC\n### 6.2 工程化自适应\n",
            ROOT / "chapters/06_aec.md",
        )
        self.assertIn('id="sec-6-2"', html)
        self.assertIn('id="sec-6-1-3" class="anchor-alias"', html)

    def test_math_shielding_preserves_fenced_and_inline_code(self):
        source = ('```python\na = "$x<y$"\n```\n'
                  '`$HOME < $PATH` 与数学 $x<y$')
        site_html, _ = build_site.render(source)
        self.assertIn('a = &quot;$x&lt;y$&quot;', site_html)
        self.assertIn('<code>$HOME &lt; $PATH</code>', site_html)
        self.assertIn('$x\\lt y$', site_html)
        shielded, repo = build_pdf.shield_math(source)
        pdf_html = build_pdf.unshield_math(
            markdown.markdown(shielded, extensions=["fenced_code"]), repo)
        self.assertIn('a = &quot;$x&lt;y$&quot;', pdf_html)
        self.assertIn('<code>$HOME &lt; $PATH</code>', pdf_html)
        self.assertIn('$x\\lt y$', pdf_html)

    def test_publishers_share_code_shielding_for_fence_edge_cases(self):
        self.assertIs(build_site.protect_code, build_pdf.protect_code)
        self.assertIs(build_site.restore_code, build_pdf.restore_code)
        for source in (
                '````python\n`$x$`\n```\nstill code\n````\n$y$\n',
                '~~~text\r\n$x$\r\n~~~\r\n`$y$`'):
            with self.subTest(source=source):
                shielded, repo = build_site.protect_code(source)
                self.assertIn('@@CODETOKEN0@@', shielded)
                self.assertEqual(build_pdf.restore_code(shielded, repo), source)
        # An unclosed fence is ordinary text in the pinned Markdown extension,
        # not a reason to hide the rest of a document from math or headings.
        source = '```python\nunterminated $x$\n'
        shielded, repo = build_site.protect_code(source)
        self.assertEqual(build_pdf.restore_code(shielded, repo), source)

    def test_publishers_share_link_policy_and_existing_errors(self):
        self.assertIs(build_site.validate_url_schemes, build_pdf.validate_url_schemes)
        for validator in (build_site.validate_url_schemes, build_pdf.validate_url_schemes):
            with self.subTest(validator=validator):
                validator('<a href="mailto:reader@example.org">邮件</a>'
                          '<img src="relative.png"><a href="#part">章内</a>')
                with self.assertRaisesRegex(ValueError,
                                            '不允许省略协议的外部 href：//example.org/x'):
                    validator('<a href="//example.org/x">外链</a>')
                with self.assertRaisesRegex(ValueError,
                                            '不安全或不支持的 href 协议：javascript'):
                    validator('<a href="javascript&#58;alert(1)">危险</a>')

    def test_link_protocol_policy_accepts_normal_links_and_rejects_active_content(self):
        build_site.validate_url_schemes(
            '<a href="https://example.org">外链</a><a href="#sec-1">节</a>'
            '<a href="javascript-not-a-scheme.html">近似名称</a>')
        for html in ('<a href="javascript:alert(1)">危险</a>',
                     "<a href='javascript&#58;alert(1)'>实体编码</a>",
                     '<img src="data:text/html,boom">',
                     '<a href="//example.org/path">省略协议</a>'):
            with self.assertRaises(ValueError):
                build_site.validate_url_schemes(html)
            with self.assertRaises(ValueError):
                build_pdf.validate_url_schemes(html)

    def test_site_page_has_keyboard_skip_link_and_visible_focus(self):
        self.assertIn('href="#main-content"', build_site.PAGE)
        self.assertIn('id="main-content"', build_site.PAGE)
        self.assertIn(":focus-visible", build_site.CSS)

    def test_web_and_pdf_define_paragraph_spacing_explicitly(self):
        self.assertIn(".main p{margin:0 0 1.05em}", build_site.CSS)
        self.assertIn("p{margin:0 0 .85em;break-inside:avoid;orphans:2;widows:2}", build_pdf.CSS)

    def test_aec_mobile_explanation_width_is_local_to_selected_headers(self):
        for label in ("训练参考", "仍需实测的资源", "它实际解决什么"):
            source = f"| 算法 | {label} |\n| --- | --- |\n| A | 完整说明 |\n"
            with self.subTest(label=label):
                html, _ = build_site.render(source, ROOT / "chapters/06_aec.md")
                self.assertIn('<table class="aec-readable-table">', html)
                self.assertIn('class="table-scroll" tabindex="0"', html)
                self.assertIn('scope="col"', html)
                unrelated, _ = build_site.render(source, ROOT / "chapters/05_beamforming.md")
                self.assertNotIn('aec-readable-table', unrelated)
        similar, _ = build_site.render(
            "| 算法 | 训练参考的另一种含义 |\n| --- | --- |\n| A | 说明 |\n",
            ROOT / "chapters/06_aec.md")
        self.assertNotIn('aec-readable-table', similar)

    def test_site_render_wraps_table_in_focusable_scroll_region(self):
        html, _ = build_site.render("| 列 |\n|---|\n| 值 |")
        self.assertIn('class="table-scroll" tabindex="0"', html)
        self.assertIn('role="region"', html)
        self.assertIn('<table>', html)

    def test_page_parser_records_accessibility_fields(self):
        parser = quality_check.PageParser()
        parser.feed('<html lang="zh-CN"><h1>标题</h1><h3>跳级</h3>'
                    '<img src="x.png" alt="阵列图"></html>')
        self.assertEqual(parser.html_lang, "zh-CN")
        self.assertEqual(parser.heading_levels, [1, 3])
        self.assertEqual(parser.images, [("x.png", "阵列图")])

    def test_paragraph_review_candidates_find_dense_plain_paragraph(self):
        text = "。".join(["同一自然段承担一个完整判断"] * 7) + "。"
        candidates = quality_check.paragraph_review_candidates(text)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["line"], 1)
        self.assertGreaterEqual(candidates[0]["sentences"], 6)

    def test_paragraph_review_candidates_do_not_flag_structured_list(self):
        item = "每项有独立结构和解释" * 30
        text = f"- {item}\n- {item}\n- {item}\n"
        self.assertEqual(quality_check.paragraph_review_candidates(text), [])

    def test_paragraph_review_candidates_do_not_force_short_paragraph_split(self):
        text = "条件与它限定的结论应当保留在一起。\n\n下一段只解释一个新任务。"
        self.assertEqual(quality_check.paragraph_review_candidates(text), [])

    def test_paragraph_review_candidates_do_not_treat_short_english_as_280_chinese_chars(self):
        text = (
            "This tutorial covers localization, beamforming, echo cancellation, "
            "dereverberation, separation, and tracking. It also includes worked examples."
        )
        self.assertEqual(quality_check.paragraph_review_candidates(text), [])

    def test_every_source_chapter_renders_in_site_pipeline(self):
        names = [build_site.HOME_FNAME] + [name for name, _label in build_site.CHAPTERS]
        for name in names:
            with self.subTest(name=name):
                html, heading_count = build_site.render(
                    (ROOT / "chapters" / name).read_text(encoding="utf-8")
                )
                self.assertTrue(html)
                self.assertGreater(heading_count, 0)

    def test_loose_ordered_list_keeps_answer_inside_original_item(self):
        html, _ = build_site.render(
            "1. 题干\n\n    答案段。\n\n2. 第二题\n"
        )
        self.assertEqual(html.count("<ol>"), 1)
        self.assertRegex(
            html,
            r"<li>\s*<p>题干</p>\s*<p>答案段。</p>\s*</li>",
        )

    def test_site_nav_gate_accepts_all_current_page_fragments(self):
        source = "## 章名\n### 6.1 主节\n#### 子节\n"
        links = [f"06_aec.html#{primary}" for level, _title, primary in
                 quality_check.semantic_heading_ids(source) if 2 <= level <= 4]
        self.assertEqual(
            quality_check.site_nav_fragment_issues(
                "06_aec.html", "06_aec.md", source, links),
            [],
        )

    def test_site_nav_gate_reports_missing_fragment(self):
        source = "## 章名\n### 6.1 主节\n#### 子节\n"
        title_id = next(primary for level, _title, primary in
                        quality_check.semantic_heading_ids(source) if level == 2)
        issues = quality_check.site_nav_fragment_issues(
            "06_aec.html", "06_aec.md", source,
            [f"06_aec.html#{title_id}", "06_aec.html#sec-6-1"],
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("导航缺少", issues[0])

    def test_body_link_does_not_count_as_navigation(self):
        source = "## 章名\n### 6.1 主节\n"
        parser = quality_check.PageParser()
        parser.feed('<main><a href="06_aec.html#sec-6-1">正文链接</a></main>'
                    '<nav><a href="index.html">首页</a></nav>')
        self.assertIn("06_aec.html#sec-6-1", parser.links)
        self.assertNotIn("06_aec.html#sec-6-1", parser.nav_links)
        self.assertTrue(quality_check.site_nav_fragment_issues(
            "06_aec.html", "06_aec.md", source, parser.nav_links))

    def test_structure_count_ignores_code_and_detects_deleted_section(self):
        valid = "## 章名\n### 1.1 一\n```md\n### 代码\n```\n### 1.2 二\n"
        self.assertEqual(quality_check.section_count_from_markdown(valid), 2)
        self.assertEqual(
            quality_check.section_count_from_markdown(valid.replace("### 1.2 二\n", "")),
            1,
        )

    def test_figure_semantics_accept_any_reuse_and_reject_mismatch_or_orphan(self):
        refs = [(f"图{i} 示意", f"fig{i:02d}_x.png", i) for i in range(1, 73)]
        refs.extend([("图1 复用", "fig01_x.png", 1),
                     ("图23 复用", "fig23_x.png", 23)])
        names = [f"fig{i:02d}_x.png" for i in range(1, 73)]
        self.assertEqual(quality_check.figure_inventory_issues(refs, names), [])
        bad_refs = list(refs)
        bad_refs[0] = ("图2 错配", "fig01_x.png", 1)
        issues = quality_check.figure_inventory_issues(bad_refs, names + ["fig39_orphan.png"])
        self.assertTrue(any("不匹配" in item for item in issues))
        self.assertTrue(any("孤立 PNG" in item for item in issues))

    def test_wpe_phone_implementation_table_uses_exact_headers_and_scope(self):
        source = '| 实现 | 主要源码 | 与教学基线的差别 | 最小核对实验 |\n| --- | --- | --- | --- |\n| A | code | 区别 | 核对 |\n'
        html, _ = build_site.render(source, ROOT/'chapters/07_wpe-dereverberation.md')
        self.assertIn('<table class="wpe-readable-table">', html)
        self.assertIn('class="table-scroll" tabindex="0" role="region"', html)
        self.assertEqual(html.count('scope="col"'), 4)
        for other in ('06_aec.md', '08_speech-separation.md'):
            html, _ = build_site.render(source, ROOT/'chapters'/other)
            self.assertNotIn('wpe-readable-table', html)
        for near in (source.replace('最小核对实验', '另一实验'),
                     source.replace('实现 | 主要源码', '主要源码 | 实现')):
            html, _ = build_site.render(near, ROOT/'chapters/07_wpe-dereverberation.md')
            self.assertNotIn('wpe-readable-table', html)

    def test_separation_phone_tables_keep_phrases_and_exact_chapter_scope(self):
        tables = [
            ("路线", "代表", "思想", "局限"),
            ("结构", "代表", "主要结构", "结果复现要求", "优点与限制"),
            ("固定时轴输出", "解析MSE", "PCM误差整数能量 $E$", "实际PCM MSE", "实际PCM相对平方误差"),
        ]
        for labels in tables:
            with self.subTest(headers=labels):
                source = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(['短语']*len(labels))+' |\n'
                html, _ = build_site.render(source, ROOT/'chapters/08_speech-separation.md')
                self.assertIn('<table class="separation-readable-table">', html)
                self.assertEqual(html.count('scope="col"'), len(labels))
                self.assertIn('class="table-scroll" tabindex="0" role="region"', html)
                for other in ('06_aec.md', '07_wpe-dereverberation.md'):
                    html, _ = build_site.render(source, ROOT/'chapters'/other)
                    self.assertNotIn('separation-readable-table', html)
                for near in (source.replace(labels[0], '其他表头', 1),
                             source.replace(' | '.join(labels), ' | '.join(reversed(labels)), 1)):
                    html, _ = build_site.render(near, ROOT/'chapters/08_speech-separation.md')
                    self.assertNotIn('separation-readable-table', html)
        screen, printed = build_site.CSS.split('@media print', 1)
        self.assertIn('.separation-readable-table td{min-width:8em}', screen)
        self.assertNotIn('separation-readable-table', printed)

    def test_tracking_phone_comparison_tables_have_local_readable_columns(self):
        for labels in (("方法", "原理", "优点", "缺点", "适用"),
                       ("方法族", "代表", "思想", "声学表现")):
            source = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(['短语']*len(labels))+' |\n'
            html, _ = build_site.render(source, ROOT/'chapters/09_source-tracking.md')
            self.assertIn('<table class="tracking-readable-table">', html)
            self.assertEqual(html.count('scope="col"'), len(labels))
            self.assertIn('class="table-scroll" tabindex="0" role="region"', html)
            for other in ('08_speech-separation.md', '10_engineering-practice.md'):
                html, _ = build_site.render(source, ROOT/'chapters'/other)
                self.assertNotIn('tracking-readable-table', html)
            for near in (source.replace(labels[0], '其他表头', 1),
                         source.replace(' | '.join(labels), ' | '.join(reversed(labels)), 1)):
                html, _ = build_site.render(near, ROOT/'chapters/09_source-tracking.md')
                self.assertNotIn('tracking-readable-table', html)
        screen, printed = build_site.CSS.split('@media print', 1)
        self.assertIn('.tracking-readable-table td{min-width:8em}', screen)
        self.assertNotIn('tracking-readable-table', printed)

    def test_selection_phone_tables_match_exact_headers_and_chapter(self):
        tables = [
            ("决策对象", "适合先尝试的方案", "何时换方案", "同时检查的风险"),
            ("任务条件", "一路波束/掩码波束", "连续语音分离（CSS）", "目标说话人提取（TSE）"),
("条件或证据", "一路掩码波束", "两路 CSS", "注册声纹 TSE", "决定"),
        ]
        for labels in tables:
            source = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(['短语']*len(labels))+' |\n'
            html, _ = build_site.render(source, ROOT/'chapters/11_selection-guide.md')
            self.assertIn('<table class="selection-readable-table">', html)
            self.assertEqual(html.count('scope="col"'), len(labels))
            self.assertIn('class="table-scroll" tabindex="0" role="region"', html)
            for other in ('10_engineering-practice.md', '12_appendix-symbols-math.md'):
                html, _ = build_site.render(source, ROOT/'chapters'/other)
                self.assertNotIn('selection-readable-table', html)
            for near in (source.replace(labels[0], '其他表头', 1),
                         source.replace(' | '.join(labels), ' | '.join(reversed(labels)), 1)):
                html, _ = build_site.render(near, ROOT/'chapters/11_selection-guide.md')
                self.assertNotIn('selection-readable-table', html)
        screen, printed = build_site.CSS.split('@media print', 1)
        self.assertIn('.selection-readable-table{min-width:760px}', screen)
        self.assertIn('.selection-readable-table td{min-width:10em}', screen)
        self.assertNotIn('selection-readable-table', printed)

    def test_industrial_phone_tables_match_exact_headers_and_research_source(self):
        tables = [
            ("对照", "输出总帧数", "六个脉冲相对参考时间的峰值采样点偏移（帧）", "1～9 s 偏移增量（帧）", "部分消费调用", "输入取尽后补出帧数"),
            ("接口", "需要核对的对象", "不能省略的条件"),
            ("阅读位置", "要核对的变量/步骤", "独立判据"),
            ("固定对象", "实际执行与输入", "独立核对结果", "代替项和未执行范围"),
        ]
        path = build_site.RESEARCH_ROOT/'03_industrial_deployment.md'
        for labels in tables:
            source = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(['短语']*len(labels))+' |\n'
            html, _ = build_site.render(source,path)
            self.assertIn('<table class="industrial-readable-table">',html)
            self.assertIn('class="table-scroll" tabindex="0" role="region"',html)
            self.assertEqual(html.count('scope="col"'),len(labels))
            for other in (ROOT/'chapters/10_engineering-practice.md',
                          build_site.RESEARCH_ROOT/'04_source_reproduction.md',
                          ROOT/'chapters/03_industrial_deployment.md'):
                html, _ = build_site.render(source,other)
                self.assertNotIn('industrial-readable-table',html)
            for near in (source.replace(labels[0],'另一个表头',1),
                         source.replace(' | '.join(labels),' | '.join(reversed(labels)),1)):
                html, _ = build_site.render(near,path)
                self.assertNotIn('industrial-readable-table',html)
        screen, printed = build_site.CSS.split('@media print',1)
        self.assertIn('.industrial-readable-table td{min-width:8em}',screen)
        self.assertNotIn('industrial-readable-table',printed)

    def test_noise_tables_preserve_readable_first_column_in_own_chapter(self):
        for labels in (("支路", "噪声功率来源", "能检查的问题"),
                       ("输出", "阶跃前浮点 MSE", "阶跃前 PCM MSE", "阶跃后浮点 MSE", "阶跃后 PCM MSE")):
            source = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(['短语']*len(labels))+' |\n'
            html, _ = build_site.render(source,ROOT/'chapters/10_engineering-practice.md')
            self.assertIn('<table class="noise-readable-table">',html)
            for other in (ROOT/'chapters/09_source-tracking.md',build_site.RESEARCH_ROOT/'03_industrial_deployment.md'):
                html, _ = build_site.render(source,other)
                self.assertNotIn('noise-readable-table',html)
            html, _ = build_site.render(source.replace(labels[0],'其它',1),ROOT/'chapters/10_engineering-practice.md')
            self.assertNotIn('noise-readable-table',html)
        screen, printed = build_site.CSS.split('@media print',1)
        self.assertIn('.noise-readable-table td:first-child{min-width:6em}',screen)
        self.assertNotIn('noise-readable-table',printed)

    def test_source_control_characters_catch_damaged_tex(self):
        self.assertEqual(quality_check.source_control_character_issues("正文\n\t数学\r\n"), [])
        issues = quality_check.source_control_character_issues("正文\n$"+chr(7)+"pprox$\n")
        self.assertEqual(len(issues),1)
        self.assertIn("U+0007",issues[0])
        self.assertIn("第2行",issues[0])

    def test_formula_semantics_valid_duplicate_missing_and_arithmetic_near_miss(self):
        valid = {"02_x.md": "$$x=1\\tag{2-1}$$\n见式(2-1)。\n算术 (10-1) 不是引用。"}
        self.assertEqual(quality_check.formula_semantic_issues(valid), [])
        duplicate = {
            "02_x.md": "$$x=1\\tag{2-1}$$",
            "03_x.md": "$$y=1\\tag{2-1}$$\n见式(9-9)",
        }
        issues = quality_check.formula_semantic_issues(duplicate)
        self.assertTrue(any("重复" in item for item in issues))
        self.assertTrue(any("章号不匹配" in item for item in issues))
        self.assertTrue(any("无定义" in item for item in issues))

    def test_formula_semantics_rejects_malformed_outside_and_legacy_tags(self):
        documents = {
            "02_x.md": (
                "正文 \\tag{2-1}\n"
                "$$x=1\\tag{2_1}$$\n"
                "$$y=2$$\n(2-2)\n"
            )
        }
        issues = quality_check.formula_semantic_issues(documents)
        self.assertTrue(any("不在" in item for item in issues))
        self.assertTrue(any("语法错误" in item for item in issues))
        self.assertTrue(any("旧式" in item for item in issues))

    def test_section_reference_requires_existing_section_and_semantic_fragment(self):
        valid = {
            "04_x.md": "## 章\n### 4.2 GCC\n",
            "11_x.md": "[§4.2](04_x.md#sec-4-2)\n",
        }
        self.assertEqual(quality_check.section_reference_issues(valid), [])
        bad = {
            "04_x.md": "## 章\n### 4.2 GCC\n",
            "11_x.md": "[§4.2](04_x.md) 与 §9.9\n",
        }
        issues = quality_check.section_reference_issues(bad)
        self.assertTrue(any("未指向语义片段" in item for item in issues))
        self.assertTrue(any("引用不存在" in item for item in issues))

    def test_section_references_distinguish_external_citations_from_book(self):
        documents = {'04_x.md': '## 章\n### 4.2 GCC\n'
                     '[NIST §7.2.4.1](https://example.org/manual.htm "citation")\n'
                     '[原论文 §2.2](https://example.org/paper.pdf)\n'
                     '[§9.9](04_x.md#sec-9-9) 与 §8.8\n'}
        issues = quality_check.section_reference_issues(documents)
        self.assertFalse(any('7.2.4.1' in item or '§2.2' in item for item in issues))
        self.assertTrue(any('§9.9' in item for item in issues))
        self.assertTrue(any('§8.8' in item for item in issues))

    def test_section_semantics_rejects_wrong_chapter_and_duplicate_number(self):
        documents = {
            "03_x.md": "## 章\n### 4.2 错章\n",
            "04_x.md": "## 章\n### 4.2 重复\n",
        }
        issues = quality_check.section_reference_issues(documents)
        self.assertTrue(any("章号不匹配" in item for item in issues))
        self.assertTrue(any("编号重复" in item for item in issues))

    def test_expected_outline_comes_from_markdown_without_importing_builder(self):
        documents = {
            "00_overview.md": "# 导读\n## 1. 开始\n",
            "01_problem-definition.md": "## 第一章\n### 1.1 问题\n",
        }
        chapters = [("00_overview.md", "导读"),
                    ("01_problem-definition.md", "第一章")]
        with mock.patch.object(quality_check, "EXPECTED_CHAPTERS", chapters):
            self.assertEqual(
                quality_check.expected_outline(documents),
                [("导读", [["1. 开始", []]]),
                 ("第一章", [["1.1 问题", []]])],
            )

    def test_source_content_expectation_tracks_heading_and_repeated_images(self):
        source = ("## 章名\n### 10.1 小节\n"
                  "![图1 一](../figures/fig01_x.png)\n"
                  "![图1 再用](../figures/fig01_x.png)\n")
        ids, images = quality_check.expected_site_content("10_x.md", source)
        self.assertIn("sec-10-1", ids)
        self.assertEqual(images["../figures/fig01_x.png"], 2)

    def test_strip_fenced_code_preserves_following_line_numbers(self):
        source = "第一行\n```text\n乱飞\n```\n第五行乱飞\n"
        prose = quality_check.strip_fenced_code(source)
        self.assertEqual(prose.count("\n", 0, prose.rfind("乱飞")) + 1, 5)

    def test_url_scheme_checker_has_valid_invalid_and_near_miss_cases(self):
        valid = ["https://example.org", "mailto:a@example.org", "chapter.html#sec-1",
                 "javascript-not-a-scheme.html"]
        self.assertEqual(quality_check.url_scheme_issues(valid), [])
        issues = quality_check.url_scheme_issues(
            ["javascript:alert(1)", "data:text/html,boom", "//example.org/path"])
        self.assertEqual(len(issues), 3)

    def test_build_date_accepts_explicit_or_source_date_epoch(self):
        self.assertEqual(build_pdf.resolve_build_date("2026-09-21"), "2026-09-21")
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "0"}):
            self.assertEqual(build_pdf.resolve_build_date(), "1970-01-01")
        with self.assertRaises(ValueError):
            build_pdf.resolve_build_date("2026-99-99")

    def test_png_provenance_accepts_current_digest_and_rejects_stale_digest(self):
        from PIL import Image, PngImagePlugin
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            base = Path(temp_dir)
            script = base / "make_figures.py"
            script.write_text("print('stable')\n", encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(script.read_bytes()).hexdigest()
            image_path = base / "fig01_x.png"
            info = PngImagePlugin.PngInfo()
            info.add_text("SourceScript", "make_figures.py")
            info.add_text("SourceScriptDigest", digest)
            Image.new("RGB", (8, 8)).save(image_path, pnginfo=info)
            self.assertEqual(
                quality_check.png_provenance_issues(image_path, script), [])
            script.write_text("print('changed')\n", encoding="utf-8")
            self.assertTrue(quality_check.png_provenance_issues(image_path, script))

    def test_ascii_range_rule_ignores_version_near_miss(self):
        self.assertIsNotNone(quality_check.NUMERIC_ASCII_RANGE.search("2~3 周"))
        self.assertIsNone(quality_check.NUMERIC_ASCII_RANGE.search("v1.2~1.3"))


if __name__ == "__main__":
    unittest.main()
