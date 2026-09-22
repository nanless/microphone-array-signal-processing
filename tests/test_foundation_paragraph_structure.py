"""Inspect rendered list ownership, not just Markdown line prefixes."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

from scripts.build_site import render


class ListItems(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self.current = []

    def handle_data(self, text):
        if self.current is not None:
            self.current.append(text)

    def handle_endtag(self, tag):
        if tag == "li" and self.current is not None:
            self.items.append("".join(self.current))
            self.current = None


class FoundationParagraphTest(unittest.TestCase):
    def test_wng_di_example_has_two_items_and_owned_explanation(self):
        path = Path(__file__).resolve().parents[1]/"chapters/02_basics-signal-model.md"
        source = path.read_text(encoding="utf-8")
        excerpt = source.split("这个例子用同一组权重比较 WNG 与 DI：",1)[1].split("量级结论：",1)[0]
        parser = ListItems()
        parser.feed(render(excerpt, path)[0])
        self.assertEqual(len(parser.items),2)
        self.assertIn("WNG",parser.items[0])
        self.assertNotIn("口径注",parser.items[0])
        self.assertIn("DI",parser.items[1])
        self.assertIn("口径注",parser.items[1])


if __name__ == "__main__":
    unittest.main()
