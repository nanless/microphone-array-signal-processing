"""Research publishing regressions; no network, browser or final output writes."""
import contextlib
import io
import json
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlsplit

from scripts import build_pdf, build_site


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "codes" / "research"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.ids = set()
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.hrefs.append(attrs["href"])
        if tag == "img" and "src" in attrs:
            self.images.append((attrs["src"], attrs.get("alt")))


class ResearchBuildTest(unittest.TestCase):
    def test_mathjax_chinese_fallback_uses_body_font_and_radicals_are_rejected(self):
        self.assertIn('mjx-mtext>mjx-utext{font-family:MJXZERO,"STHeiti",', build_pdf.CSS)
        self.assertIn('"Microsoft YaHei",sans-serif!important}', build_pdf.CSS)
        build_pdf.validate_pdf_text_codepoints("一次预测：两次预测：")
        with self.assertRaisesRegex(SystemExit, "U\\+2F00"):
            build_pdf.validate_pdf_text_codepoints("\u2f00次预测：")
        with self.assertRaisesRegex(SystemExit, "U\\+2E85"):
            build_pdf.validate_pdf_text_codepoints("\u2e85")

    def test_publish_failure_restores_existing_files_and_removes_new_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "existing.html"
            old.write_text("old", encoding="utf-8")
            new = root / "new.html"
            first, second = root / "first.tmp", root / "second.tmp"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            replace = build_site.os.replace

            def fail_second(source, target):
                if target == old:
                    raise OSError("injected replacement failure")
                return replace(source, target)

            with mock.patch.object(build_site.os, "replace", fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    build_site.publish_files([(first, new), (second, old)])
            self.assertFalse(new.exists())
            self.assertEqual(old.read_text(encoding="utf-8"), "old")
            self.assertFalse(list(root.glob(".publish-backup-*")))

    def test_stale_cleanup_failure_restores_whole_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, staged = root / "page.html", root / "page.tmp"
            stale_a, stale_b = root / "stale-a.html", root / "stale-b.html"
            for path in (target, stale_a, stale_b):
                path.write_text("old " + path.name, encoding="utf-8")
            staged.write_text("new", encoding="utf-8")
            unlink = Path.unlink

            def fail_last(path, *args, **kwargs):
                if path == stale_b:
                    raise OSError("injected stale cleanup failure")
                return unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", fail_last):
                with self.assertRaisesRegex(OSError, "injected"):
                    build_site.publish_files([(staged, target)], [stale_a, stale_b])
            for path in (target, stale_a, stale_b):
                self.assertEqual(path.read_text(encoding="utf-8"), "old " + path.name)

    def test_pdf_failures_before_or_during_publication_preserve_old_pair(self):
        for failure in ("print", "bookmarks", "links", "publish"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                combined = output / "combined.html"
                pdf = output / "microphone-array-tutorial.pdf"
                combined.write_text("old HTML", encoding="utf-8")
                pdf.write_bytes(b"old PDF")
                replace = build_site.os.replace

                def print_stub(source, target, _minimum):
                    self.assertNotEqual(source, combined)
                    self.assertEqual(source.parent, combined.parent)
                    self.assertEqual(combined.read_text(encoding="utf-8"), "old HTML")
                    target.write_bytes(b"new PDF")
                    if failure == "print":
                        raise RuntimeError("injected print failure")

                def replace_stub(source, target):
                    if failure == "publish" and target == pdf:
                        raise OSError("injected publish failure")
                    return replace(source, target)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(build_pdf, "OUT", output))
                    stack.enter_context(mock.patch.object(build_pdf, "check_figures"))
                    stack.enter_context(mock.patch.object(build_pdf, "build_html", return_value=("new HTML", [])))
                    stack.enter_context(mock.patch.object(build_pdf, "print_pdf", side_effect=print_stub))
                    stack.enter_context(mock.patch.object(build_pdf, "add_bookmarks",
                        side_effect=RuntimeError("injected bookmarks failure") if failure == "bookmarks" else None))
                    stack.enter_context(mock.patch.object(build_pdf, "validate_pdf_links",
                        side_effect=RuntimeError("injected links failure") if failure == "links" else None))
                    stack.enter_context(mock.patch.object(build_site.os, "replace", side_effect=replace_stub))
                    with self.assertRaisesRegex((RuntimeError, OSError), "injected"):
                        build_pdf.main([])
                self.assertEqual(combined.read_text(encoding="utf-8"), "old HTML")
                self.assertEqual(pdf.read_bytes(), b"old PDF")
                self.assertEqual({path.name for path in output.iterdir()}, {combined.name, pdf.name})

    def test_print_figures_reserve_space_for_chapter_and_section_headings(self):
        self.assertIn("h1,h2,h3,h4{break-after:avoid}", build_pdf.CSS)
        self.assertIn("img{max-height:225mm;object-fit:contain}", build_pdf.CSS)
        self.assertIn(".chap>h1{margin:0 0 3mm;line-height:1.3}", build_pdf.CSS)
        self.assertIn(".chap>h2:first-of-type{margin-top:3mm;margin-bottom:2mm;line-height:1.3}", build_pdf.CSS)
        self.assertNotIn("max-height:92vh", build_pdf.CSS)

    def test_pdf_body_scale_accepts_normal_and_rejects_global_shrink(self):
        def reader(scale, font_size=16, text="这是用于检查全书打印尺度的中文正文"):
            page = mock.Mock()
            page.extract_text.side_effect = lambda visitor_text: visitor_text(
                text, [scale, 0, 0, -scale, 0, 0], [1, 0, 0, -1, 0, 0], {}, font_size)
            return mock.Mock(pages=[page])
        build_pdf.validate_pdf_body_scale(reader(0.75))
        with self.assertRaisesRegex(SystemExit, "额外缩小"):
            build_pdf.validate_pdf_body_scale(reader(0.544))
        with self.assertRaisesRegex(SystemExit, "缺少"):
            build_pdf.validate_pdf_body_scale(reader(0.5, font_size=12))
        with self.assertRaisesRegex(SystemExit, "缺少"):
            build_pdf.validate_pdf_body_scale(reader(0.5, text="x+y"))

    def test_inline_identifiers_wrap_without_disabling_block_scroll(self):
        self.assertIn("background:#fff;overflow-wrap:anywhere}", build_site.CSS)
        self.assertIn('pre,.table-scroll,mjx-container[jax="CHTML"]{overflow-wrap:normal}', build_site.CSS)
        self.assertIn(".table-scroll{max-width:100%;overflow-x:auto}", build_site.CSS)
        self.assertIn('mjx-container[jax="CHTML"]{font-size:110%!important;overflow-x:auto;', build_site.CSS)
        self.assertIn("border-radius:8px;overflow-x:auto}", build_site.CSS)

    def test_heading_whitespace_has_one_canonical_anchor(self):
        plain = "A01 NLMS 与输入"
        self.assertEqual(build_site.clean_label(" A01\u3000NLMS   与输入 "), plain)
        self.assertEqual(build_site.heading_anchor("A01\u3000NLMS   与输入", 1),
                         build_site.heading_anchor(plain, 1))

    def test_table_header_scope_keeps_thead_semantics(self):
        html = self.render("| 名称 | 数值 |\n|---|---|\n| 示例 | 1 |",
                           ROOT / "chapters" / "03_array-geometry.md")
        self.assertIn("<thead>", html)
        self.assertIn("</thead>", html)
        self.assertEqual(html.count('<th scope="col">'), 2)
        self.assertNotIn('scope="col"ead', html)

    def render(self, markdown, source):
        return build_site.render(markdown, source)[0]

    def test_explicit_page_map_has_14_tutorial_and_6_research_pages(self):
        paths = list(build_site.source_outputs().values())
        self.assertEqual(len(paths), 20)
        self.assertEqual(sum(path.startswith("research/") for path in paths), 6)
        self.assertEqual(build_site.source_outputs()[RESEARCH / "README.md"], "research/index.html")

    def test_chapter_links_to_research_and_source_documents(self):
        html = self.render(
            "[研究](../codes/research/02_aec_wpe_separation.md#wpe)\n\n"
            "[覆盖](../codes/COVERAGE.md#说明)", ROOT / "chapters" / "07_wpe-dereverberation.md")
        self.assertIn('href="research/02_aec_wpe_separation.html#wpe"', html)
        self.assertIn('href="' + build_site.REPOSITORY_BLOB_BASE + 'codes/COVERAGE.md#说明"', html)

    def test_research_links_to_home_peer_chapter_and_code(self):
        html = self.render(
            "[首页](../../chapters/00_overview.md)\n\n"
            "[研究](./README.md)\n\n"
            "[AEC](../../chapters/06_aec.md#sec-6-1)\n\n"
            "[测试](../../tests/test_codes_aec_wpe_sep_track.py)",
            RESEARCH / "02_aec_wpe_separation.md")
        for href in ("../index.html", "index.html", "../06_aec.html#sec-6-1",
                     build_site.REPOSITORY_BLOB_BASE + "tests/test_codes_aec_wpe_sep_track.py"):
            self.assertIn(f'href="{href}"', html)

    def test_external_markdown_urls_and_text_are_not_rewritten(self):
        url = "https://github.com/example/project/blob/main/README.md?plain=1#license"
        html = self.render(f'[上游]({url})\n\n`note.md`', RESEARCH / "README.md")
        self.assertIn(f'href="{url}"', html)
        self.assertIn("<code>note.md</code>", html)
        self.assertNotIn("README.html", html)

    def test_query_fragment_and_uri_encoded_path_preserved(self):
        html = self.render('[文档](../codes/%43OVERAGE.md?plain=1&view=source#说明)',
                           ROOT / "chapters" / "06_aec.md")
        self.assertIn('codes/COVERAGE.md?plain=1&amp;view=source#说明', html)

    def test_same_basename_is_resolved_using_source_context(self):
        html = self.render('[上游说明](../upstream/README.md)', RESEARCH / "README.md")
        self.assertIn(build_site.REPOSITORY_BLOB_BASE + "codes/upstream/README.md", html)
        self.assertNotIn('href="index.html"', html)

    def test_research_heading_alias_accepts_published_markdown_fragment(self):
        html = self.render("## 2. 噪声、语音活动与增益\n", RESEARCH / "03_industrial_deployment.md")
        self.assertIn('id="2-噪声语音活动与增益"', html)

    def test_combined_links_keep_real_source_suffix_and_external_url(self):
        html = ('<a href="../codes/COVERAGE.md#说明">覆盖</a>'
                '<a href="../codes/research/README.md">研究</a>'
                '<a href="https://example.org/README.md">外部</a>'
                '<a href="./07_wpe-dereverberation.md#sec-7-1">WPE</a>')
        html = build_pdf.rewrite_repository_links(html, ROOT / "chapters" / "06_aec.md")
        self.assertIn(build_site.REPOSITORY_BLOB_BASE + "codes/COVERAGE.md#说明", html)
        self.assertIn(build_site.REPOSITORY_BLOB_BASE + "codes/research/README.md", html)
        self.assertIn('href="https://example.org/README.md"', html)
        self.assertIn('href="#ch-7-sec-7-1"', html)
        self.assertNotIn(".html", html)

    def test_both_digests_include_every_research_source(self):
        original_read = Path.read_bytes
        for builder in (build_site, build_pdf):
            before = builder.source_digest()
            for name, _ in build_site.RESEARCH:
                changed = RESEARCH / name

                def read(path):
                    return original_read(path) + (b"\nchanged" if path == changed else b"")

                with self.subTest(builder=builder.__name__, source=name):
                    with mock.patch.object(Path, "read_bytes", read):
                        self.assertNotEqual(before, builder.source_digest())

    def test_pdf_same_chapter_fragment_gets_chapter_prefix(self):
        result = build_pdf.rewrite_repository_links(
            '<a href="#sec-4-9">MDL</a><a href="#sec-1">章首</a>',
            ROOT / "chapters/04_doa-estimation.md")
        self.assertEqual(result, '<a href="#ch-4-sec-4-9">MDL</a><a href="#ch-4">章首</a>')

    def test_chapter_four_mdl_precedes_audio_and_summary_exercises(self):
        source = (ROOT / "chapters/04_doa-estimation.md").read_text(encoding="utf-8")
        headings = re.findall(r"^### (4\.(?:9|10|11)) (.+)$", source, re.MULTILINE)
        self.assertEqual(headings, [
            ("4.9", "MDL 源数估计：逐候选评分与数值例"),
            ("4.10", "四麦亚采样时差：从几何到可听信号"),
            ("4.11", "本章练习"),
        ])
        self.assertLess(source.index("**E04-05"), source.index("**E04-06"))
        self.assertLess(source.index("**E04-06"), source.index("**E04-07"))
        self.assertLess(source.index("**E04-07"), source.index("**E04-01"))

    def test_pdf_same_chapter_unknown_fragment_is_rejected(self):
        with self.assertRaises(ValueError):
            build_pdf.rewrite_repository_links('<a href="#missing">缺失</a>',
                                               ROOT / "chapters/04_doa-estimation.md")

    def test_pdf_external_fragment_is_not_a_chapter_link(self):
        source = '<a href="https://example.org/#sec-4-10">外部</a>'
        self.assertEqual(build_pdf.rewrite_repository_links(source,
                         ROOT / "chapters/04_doa-estimation.md"), source)

    def test_temporary_build_navigation_and_all_local_deep_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "site"
            with mock.patch.object(build_site, "OUT", output), contextlib.redirect_stdout(io.StringIO()):
                build_site.main()
            room_source = ROOT / "codes" / "room_audio"
            room_manifest = json.loads((room_source / "MANIFEST.json").read_text(encoding="utf-8"))
            room_names = {record["file"] for record in room_manifest["files"]}
            self.assertEqual(len(room_names), 18)
            room_names.update({"MANIFEST.json", "ROOM_RESULTS.png"})
            self.assertEqual({path.name for path in (output / "room_audio").iterdir()}, room_names)
            for name in room_names:
                with self.subTest(room_asset=name):
                    self.assertEqual((output / "room_audio" / name).read_bytes(),
                                     (room_source / name).read_bytes())
            pages = {}
            for path in output.rglob("*.html"):
                parser = Links()
                parser.feed(path.read_text(encoding="utf-8"))
                pages[path.resolve()] = parser
            self.assertEqual(len(pages), 20)
            room_links = set()
            room_images = set()
            for path, parsed in pages.items():
                if path.parent.name == "research":
                    self.assertIn("../index.html", parsed.hrefs)
                else:
                    self.assertIn("research/index.html", parsed.hrefs)
                for href in parsed.hrefs:
                    uri = urlsplit(href)
                    if uri.scheme or uri.netloc:
                        continue
                    target = (path.parent / unquote(uri.path)).resolve() if uri.path else path
                    with self.subTest(page=path.name, href=href):
                        if target.parent == (output / "room_audio").resolve():
                            self.assertIn(target.name, room_names)
                            self.assertTrue(target.is_file())
                            self.assertEqual(target.read_bytes(), (room_source / target.name).read_bytes())
                            self.assertFalse(uri.fragment)
                            room_links.add(target.name)
                            continue
                        if target.parent == (output / "real_audio").resolve():
                            self.assertIn(target.name, {
                                "demand_nriver_16ch_10s.wav", "demand_nriver_ch01_10s.wav",
                                "demand_nriver_mean02_10s.wav", "demand_nriver_mean16_10s.wav",
                                "MANIFEST.json", "README.md", "ATTRIBUTION.txt", "LICENSE.txt",
                            })
                            self.assertTrue(target.is_file())
                            self.assertEqual(target.read_bytes(), (ROOT / "codes/real_audio" / target.name).read_bytes())
                            self.assertFalse(uri.fragment)
                            continue
                        if target.parent in {(output / "gss_audio").resolve(),
                                             (output / "moving_audio").resolve()}:
                            source = ROOT / "codes" / target.parent.name / target.name
                            self.assertTrue(target.is_file())
                            self.assertEqual(target.read_bytes(), source.read_bytes())
                            self.assertFalse(uri.fragment)
                            continue
                        if target.suffix == ".wav":
                            self.assertEqual(target.parent, (output / "audio").resolve())
                            self.assertTrue(target.is_file())
                            self.assertFalse(uri.fragment)
                            continue
                        self.assertIn(target, pages)
                        if uri.fragment:
                            self.assertIn(unquote(uri.fragment), pages[target].ids)
                for src, alt in parsed.images:
                    uri = urlsplit(src)
                    if uri.scheme or uri.netloc:
                        continue
                    target = (path.parent / unquote(uri.path)).resolve()
                    if target.parent == (output / "room_audio").resolve():
                        with self.subTest(page=path.name, image=src):
                            self.assertEqual(target.name, "ROOM_RESULTS.png")
                            self.assertTrue(alt and alt.strip())
                            self.assertEqual(target.read_bytes(),
                                             (room_source / "ROOM_RESULTS.png").read_bytes())
                            room_images.add(target.name)
            self.assertEqual(room_links, room_names)
            self.assertEqual(room_images, {"ROOM_RESULTS.png"})


if __name__ == "__main__":
    unittest.main()
