"""Independent research-site quality gates; fixtures never write final outputs."""

import tempfile
import unittest
import io
from contextlib import ExitStack, redirect_stdout
from html import escape
from pathlib import Path
from unittest.mock import patch

from scripts import quality_check as quality


class ResearchQualityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.site = self.root / "site"
        self.sources = self.root / "codes" / "research"
        self.sources.mkdir(parents=True)
        (self.site / "research").mkdir(parents=True)
        self.stack.enter_context(patch.object(quality, "ROOT", self.root))
        self.stack.enter_context(patch.object(quality, "SITE", self.site))
        self.stack.enter_context(patch.object(quality, "site_source_digest", return_value="0123456789ab"))
        (self.site / "index.html").write_text('<h1 id="home">Tutorial</h1>', encoding="utf-8")
        self.external = "https://example.org/project/README.md?plain=1&view=source#license"
        self.source = ("# Guide\n\n## Part\n\n### Algorithm\n\n#### Detail\n\n"
                       f"[Official source]({self.external})\n")
        for source_name, output_name in quality.EXPECTED_RESEARCH_PAGES:
            (self.sources / source_name).write_text(self.source, encoding="utf-8")
            (self.site / "research" / output_name).write_text(self.valid_page(), encoding="utf-8")

    def valid_page(self):
        headings = quality.semantic_heading_ids(self.source)
        nav = ''.join(f'<a href="#{anchor}">{escape(title)}</a>'
                      for level, title, anchor in headings if 2 <= level <= 4)
        body = ''.join(f'<h{level} id="{anchor}">{escape(title)}</h{level}>'
                       for level, title, anchor in headings)
        return (
            '<html lang="zh-CN"><head><meta name="source-digest" content="0123456789ab">'
            '<style>a:focus-visible{outline:2px solid}</style></head><body>'
            '<header><a href="../index.html#home">Tutorial home</a></header>'
            '<a href="#main-content">Skip to content</a>'
            '<nav><a aria-current="page" href="index.html">Research</a>' + nav + '</nav>'
            '<main id="main-content">' + body +
            f'<a href="{escape(self.external, quote=True)}">Official source</a>'
            '</main><footer>Research</footer></body></html>'
        )

    def issues(self):
        errors = []
        quality.check_research_site(errors)
        quality.check_site_links(errors)
        return errors

    def mutate_page(self, old, new, name="index.html"):
        path = self.site / "research" / name
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new), encoding="utf-8")

    def test_valid_six_research_pages_pass(self):
        self.assertEqual(self.issues(), [])

    def test_explicit_baselines_preserve_tutorial_pdf_and_figure_counts(self):
        self.assertEqual(quality.EXPECTED_CHAPTER_COUNT, 14)
        self.assertEqual(quality.EXPECTED_SECTION_COUNT, 119)
        self.assertEqual(quality.EXPECTED_SUBSECTION_COUNT, 121)
        self.assertEqual(quality.EXPECTED_OUTLINE_ITEM_COUNT, 254)
        self.assertEqual(quality.EXPECTED_FIGURE_NUMBERS, set(range(1, 40)))
        self.assertEqual(quality.EXPECTED_RESEARCH_PAGE_COUNT, 6)
        self.assertEqual(quality.EXPECTED_RESEARCH_PAGES, (
            ("README.md", "index.html"),
            ("01_spatial_and_tracking.md", "01_spatial_and_tracking.html"),
            ("02_aec_wpe_separation.md", "02_aec_wpe_separation.html"),
            ("03_industrial_deployment.md", "03_industrial_deployment.html"),
            ("04_source_reproduction.md", "04_source_reproduction.html"),
            ("05_exercises_and_audio.md", "05_exercises_and_audio.html"),
        ))

    def test_missing_page_fails_without_file_read_crash(self):
        (self.site / "research" / "04_source_reproduction.html").unlink()
        self.assertTrue(any("研究页面集" in issue and "04_source_reproduction.html" in issue
                            for issue in self.issues()))

    def test_same_count_wrong_filename_does_not_satisfy_manifest(self):
        (self.site / "research" / "04_source_reproduction.html").rename(
            self.site / "research" / "wrong.html")
        self.assertTrue(any("研究页面集" in issue for issue in self.issues()))

    def test_stale_source_digest_fails(self):
        self.mutate_page("0123456789ab", "stale")
        self.assertTrue(any("不是当前源文件生成" in issue for issue in self.issues()))

    def test_each_h2_h3_h4_must_be_inside_navigation(self):
        for level, title, anchor in quality.semantic_heading_ids(self.source):
            if level not in (2, 3, 4):
                continue
            with self.subTest(level=level):
                path = self.site / "research" / "index.html"
                path.write_text(self.valid_page().replace(
                    f'<a href="#{anchor}">{title}</a>', ''), encoding="utf-8")
                self.assertTrue(any("导航不完整" in issue and anchor in issue
                                    for issue in self.issues()))

    def test_missing_html_heading_is_not_hidden_by_nav_link(self):
        anchor = quality.semantic_heading_ids(self.source)[2][2]
        self.mutate_page(f'id="{anchor}"', 'id="wrong-anchor"')
        self.assertTrue(any("缺少源标题锚点" in issue for issue in self.issues()))

    def test_missing_local_file_fails(self):
        self.mutate_page('../index.html#home', '../absent.html#home')
        self.assertTrue(any("站内链接目标不存在" in issue for issue in self.issues()))

    def test_fragment_lookup_distinguishes_same_basename_in_different_directories(self):
        # "home" exists on the tutorial index, not the research index.
        self.mutate_page('../index.html#home', 'index.html#home')
        self.assertTrue(any("站内锚点不存在" in issue and "index.html#home" in issue
                            for issue in self.issues()))

    def test_encoded_fragment_and_parent_relative_link_pass(self):
        self.mutate_page('../index.html#home', '../index.html#%68ome')
        self.assertEqual(self.issues(), [])

    def test_external_markdown_suffix_query_and_fragment_are_preserved(self):
        self.assertEqual(quality.external_markdown_link_issues(self.source, [self.external]), [])
        self.mutate_page('README.md?plain=1', 'README.html?plain=1')
        self.assertTrue(any("外部 Markdown 链接被改写" in issue for issue in self.issues()))

    def test_fenced_and_inline_example_links_are_not_required_external_links(self):
        source = ('```md\n[not a link](https://example.org/a.md)\n```\n\n'
                  '`[not a link](https://example.org/b.md)`')
        self.assertEqual(quality.external_markdown_link_issues(source, []), [])


class PublishedResearchQualityTests(unittest.TestCase):
    def test_independent_source_digests_match_publishers(self):
        from scripts import build_pdf, build_site
        self.assertEqual(quality.site_source_digest(), build_site.source_digest())
        self.assertEqual(quality.source_digest(), build_pdf.source_digest())

    def test_every_research_source_invalidates_both_quality_digests(self):
        original_read = Path.read_bytes
        for digest in (quality.site_source_digest, quality.source_digest):
            before = digest()
            for name, _output in quality.EXPECTED_RESEARCH_PAGES:
                changed = quality.ROOT / "codes" / "research" / name

                def read(path):
                    return original_read(path) + (b"\nchanged" if path == changed else b"")

                with self.subTest(digest=digest.__name__, source=name):
                    with patch.object(Path, "read_bytes", read):
                        self.assertNotEqual(before, digest())

    def test_temporary_published_site_passes_tutorial_and_research_gates(self):
        from scripts import build_site
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            site = temporary / "site"
            (temporary / "figures").symlink_to(quality.ROOT / "figures", target_is_directory=True)
            with patch.object(build_site, "OUT", site), redirect_stdout(io.StringIO()):
                build_site.main()
            errors = []
            with patch.object(quality, "SITE", site):
                quality.check_site(errors)
                quality.check_research_site(errors)
                quality.check_room_audio(errors)
            self.assertEqual(errors, [])

    def test_room_asset_check_rejects_changed_published_wav(self):
        from scripts import build_site
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site"
            with patch.object(build_site, "OUT", site), redirect_stdout(io.StringIO()):
                build_site.main()
            published = site / "room_audio" / "fixed_near_left_source.wav"
            data = bytearray(published.read_bytes())
            data[-1] ^= 1
            published.write_bytes(data)
            errors = []
            with patch.object(quality, "SITE", site):
                quality.check_room_audio(errors)
            self.assertTrue(any("房间合成样本检查失败" in issue and
                                "fixed_near_left_source.wav" in issue for issue in errors))


if __name__ == "__main__":
    unittest.main()
