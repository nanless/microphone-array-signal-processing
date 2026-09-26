"""Published fragments must keep their topic, not merely remain present.

The affected sequential maps were captured from the pre-2026-09-26 source and
checked against its committed site HTML before adding new research sections.
These tests need neither Git history nor an on-disk rebuild.
"""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import unittest

from scripts import build_pdf, build_site
from scripts.heading_aliases import HISTORICAL_SEQUENTIAL_IDS


ROOT = Path(__file__).resolve().parents[1]


class AliasDestinations(HTMLParser):
    """Associate each alias with the immediately following heading's identity."""
    def __init__(self, source):
        super().__init__()
        self.pending = []
        self.targets = {}
        self.ids = Counter()
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids[element_id] += 1
        if tag == "span" and "anchor-alias" in attributes.get("class", "").split():
            self.pending.append(element_id)
        if tag in {"h1", "h2", "h3", "h4"}:
            for alias in self.pending:
                self.targets[alias] = element_id
            self.pending.clear()


class HistoricalHeadingCompatibilityTest(unittest.TestCase):
    SOURCES = (
        "chapters/00_overview.md",
        "codes/research/01_spatial_and_tracking.md",
        "codes/research/02_aec_wpe_separation.md",
    )
    OVERVIEW_RENAMES = {
        "sec-u-b7a71b077a": "sec-u-3937b1b94e",
        "sec-u-efc5552983": "sec-u-f958564d39",
        "sec-u-6ef8e18126": "sec-u-d538d6d0a5",
    }

    def test_every_frozen_sequence_stays_with_its_original_topic(self):
        for relative in self.SOURCES:
            source = ROOT / relative
            html, _ = build_site.render(source.read_text(), source)
            parsed = AliasDestinations(html)
            for alias, target in HISTORICAL_SEQUENTIAL_IDS[source.name].items():
                with self.subTest(source=relative, alias=alias):
                    self.assertEqual(parsed.targets[alias], target)
                    self.assertEqual(parsed.ids[alias], 1)
                    self.assertEqual(parsed.ids[target], 1)

    def test_inserted_research_sections_cannot_steal_published_numbers(self):
        cases = (
            ("01_spatial_and_tracking.md", "sec-66", "收录边界",
             "从源代码接口到完整数学模型：四个确定性核对"),
            ("02_aec_wpe_separation.md", "sec-52", "6. 复现实验的共同记录表",
             "N15 TF-Locoformer：局部卷积、全局注意力与复谱接口"),
        )
        for name, alias, original_topic, inserted_topic in cases:
            with self.subTest(source=name):
                source = ROOT / "codes" / "research" / name
                html, _ = build_site.render(source.read_text(), source)
                parsed = AliasDestinations(html)
                expected = build_site.heading_anchor(original_topic, 1)
                inserted = build_site.heading_anchor(inserted_topic, 1)
                self.assertEqual(parsed.targets[alias], expected)
                self.assertNotEqual(parsed.targets[alias], inserted)
                self.assertNotIn(alias, [key for key, value in parsed.targets.items()
                                         if value == inserted])

    def test_overview_hashes_and_sequences_retain_reading_paths_and_map(self):
        source = ROOT / "chapters" / "00_overview.md"
        html, _ = build_site.render(source.read_text(), source)
        parsed = AliasDestinations(html)
        for old, current in self.OVERVIEW_RENAMES.items():
            self.assertEqual(parsed.targets[old], current)
            self.assertEqual(parsed.ids[old], 1)
        for sequence, current in zip(("sec-20", "sec-21", "sec-23"),
                                     self.OVERVIEW_RENAMES.values()):
            self.assertEqual(parsed.targets[sequence], current)

    def test_combined_book_keeps_the_same_overview_destinations(self):
        # build_html returns a string and performs no filesystem writes.
        html, _ = build_pdf.build_html(build_date="2026-09-26")
        parsed = AliasDestinations(html)
        for old, current in self.OVERVIEW_RENAMES.items():
            self.assertEqual(parsed.targets["ch-0-" + old], "ch-0-" + current)
            self.assertEqual(parsed.ids["ch-0-" + old], 1)
        for sequence, current in zip(("sec-20", "sec-21", "sec-23"),
                                     self.OVERVIEW_RENAMES.values()):
            self.assertEqual(parsed.targets["ch-0-" + sequence], "ch-0-" + current)


if __name__ == "__main__":
    unittest.main()
