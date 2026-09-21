import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_pdf = load("build_pdf", ROOT / "scripts" / "build_pdf.py")
build_site = load("build_site", ROOT / "scripts" / "build_site.py")


class BuildHelpersTest(unittest.TestCase):
    def test_combined_links_become_internal(self):
        html = '<a href="./01_problem-definition.html">第一章</a>'
        self.assertEqual(
            build_pdf.rewrite_book_links(html),
            '<a href="#ch-1">第一章</a>',
        )

    def test_outline_restores_sections(self):
        html = (
            '<div class="chap" id="ch-0"><h1>导读</h1>'
            '<h2 id="ch-0-s0">开始</h2><p>正文</p></div></body>'
        )
        self.assertEqual(
            build_pdf.outline_from_html(html),
            [("导读", "ch-0", [("开始", "ch-0-s0")])],
        )

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

    def test_site_page_has_keyboard_skip_link_and_visible_focus(self):
        self.assertIn('href="#main-content"', build_site.PAGE)
        self.assertIn('id="main-content"', build_site.PAGE)
        self.assertIn(":focus-visible", build_site.CSS)


if __name__ == "__main__":
    unittest.main()
