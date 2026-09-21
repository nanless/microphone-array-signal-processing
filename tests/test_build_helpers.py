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


if __name__ == "__main__":
    unittest.main()
