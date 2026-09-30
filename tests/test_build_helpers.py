import importlib.util
import io
import os
import tempfile
import unittest
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
        self.assertIn('mjx-assistive-mml{width:1px!important;height:1px!important}', build_site.CSS)
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
                '~~~text\r\n$x$\r\n~~~\r\n`$y$`',
                '```python\nunterminated $x$\n'):
            with self.subTest(source=source):
                shielded, repo = build_site.protect_code(source)
                self.assertIn('@@CODETOKEN0@@', shielded)
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
        refs = [(f"图{i} 示意", f"fig{i:02d}_x.png", i) for i in range(1, 50)]
        refs.extend([("图1 复用", "fig01_x.png", 1),
                     ("图23 复用", "fig23_x.png", 23)])
        names = [f"fig{i:02d}_x.png" for i in range(1, 50)]
        self.assertEqual(quality_check.figure_inventory_issues(refs, names), [])
        bad_refs = list(refs)
        bad_refs[0] = ("图2 错配", "fig01_x.png", 1)
        issues = quality_check.figure_inventory_issues(bad_refs, names + ["fig39_orphan.png"])
        self.assertTrue(any("不匹配" in item for item in issues))
        self.assertTrue(any("孤立 PNG" in item for item in issues))

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
