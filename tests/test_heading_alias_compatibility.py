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
        self.heading_titles = {}
        self.heading_id = None
        self.heading_text = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids[element_id] += 1
        if tag == "span" and "anchor-alias" in attributes.get("class", "").split():
            self.pending.append(element_id)
        if tag in {"h1", "h2", "h3", "h4"}:
            self.heading_id = element_id
            self.heading_text = []
            for alias in self.pending:
                self.targets[alias] = element_id
            self.pending.clear()

    def handle_data(self, data):
        if self.heading_id is not None:
            self.heading_text.append(data)

    def handle_endtag(self, tag):
        if tag in {"h1", "h2", "h3", "h4"} and self.heading_id is not None:
            self.heading_titles[self.heading_id] = "".join(self.heading_text).strip()
            self.heading_id = None


class HistoricalHeadingCompatibilityTest(unittest.TestCase):
    SOURCES = (
        "chapters/00_overview.md",
        "codes/chapters/ch00/research/01_spatial_and_tracking.md",
        "codes/chapters/ch00/research/02_aec_wpe_separation.md",
    )
    OVERVIEW_RENAMES = {
        "sec-u-d12b48e159": "sec-u-bd5fecaa8e",
        "sec-u-10c3704f3f": "sec-u-bd5fecaa8e",
        "sec-u-f96924fd43": "sec-u-bd5fecaa8e",
        "sec-u-8838edd879": "sec-u-bd5fecaa8e",
        "sec-u-55e537d650": "sec-u-bd5fecaa8e",
        "sec-u-dccee38a3d": "sec-u-bd5fecaa8e",
        "sec-u-e445ebf98d": "sec-u-bd5fecaa8e",
        "sec-u-726cf32300": "sec-u-bd5fecaa8e",
        "sec-u-dacd520d2c": "sec-u-bd5fecaa8e",
        "sec-u-9c77d218fe": "sec-u-bd5fecaa8e",
        "sec-u-2afd7c6509": "sec-u-461c9549d0",
        "sec-u-ec3f837100": "sec-u-bd5fecaa8e",
        "sec-u-06e6cf7b78": "sec-u-461c9549d0",
        "sec-u-8657cda209": "sec-u-bd5fecaa8e",
        "sec-u-0d7a1562fa": "sec-u-bd5fecaa8e",
        "sec-u-8a5dd76e40": "sec-u-bd5fecaa8e",
        "sec-u-f19f47c7c1": "sec-u-bd5fecaa8e",
        "sec-u-2b17484f3b": "sec-u-bd5fecaa8e",
        "sec-u-cb1ede6c60": "sec-u-bd5fecaa8e",
        "sec-u-5edae53c5f": "sec-u-bd5fecaa8e",
        "sec-u-5c63f4d83e": "sec-u-bd5fecaa8e",
        "sec-u-36fa20efde": "sec-u-bd5fecaa8e",
        "sec-u-b7a71b077a": "sec-u-3937b1b94e",
        "sec-u-efc5552983": "sec-u-f958564d39",
        "sec-u-1e72416790": "sec-u-bd5fecaa8e",
        "sec-u-6ef8e18126": "sec-u-bd5fecaa8e",
        "sec-u-1e5d8a2bad": "sec-u-bd5fecaa8e",
        "sec-u-627ed3907e": "sec-u-bd5fecaa8e",
        "sec-u-3a0278b879": "sec-u-bd5fecaa8e",
        "sec-u-3ebc13ca5a": "sec-u-bd5fecaa8e",
        "sec-u-d538d6d0a5": "sec-u-bd5fecaa8e",
        "sec-u-0943a9ed3c": "sec-u-bd5fecaa8e",
        "sec-u-d8fd1002de": "sec-u-bd5fecaa8e",
    }

    def test_previous_41_figure_map_hash_still_reaches_same_map(self):
        source = ROOT / "chapters/00_overview.md"
        html, _ = build_site.render(source.read_text(), source)
        parsed = AliasDestinations(html)
        self.assertEqual(parsed.targets["sec-u-d538d6d0a5"], "sec-u-bd5fecaa8e")
        self.assertEqual(parsed.targets["sec-u-d8fd1002de"], "sec-u-bd5fecaa8e")
        html, _ = build_pdf.build_html(build_date="2026-09-28")
        parsed = AliasDestinations(html)
        self.assertEqual(parsed.targets["ch-0-sec-u-d538d6d0a5"], "ch-0-sec-u-bd5fecaa8e")
        self.assertEqual(parsed.targets["ch-0-sec-u-d8fd1002de"], "ch-0-sec-u-bd5fecaa8e")

    def test_previous_79_figure_and_imaging_budget_topics_keep_their_identity(self):
        cases = (
            ('00_overview.md', 'sec-u-d12b48e159', 'sec-u-bd5fecaa8e',
             '6. 插图地图：80 张编号图在哪篇', 'ch-0-'),
            ('14_acoustic-imaging.md', 'sec-u-c93fb52739', 'sec-u-8378eb57b1',
             '噪声与源量预算怎样改变拟合目标', 'ch-14-'),
        )
        combined, _ = build_pdf.build_html(build_date='2026-10-05')
        for name, old, current, title, chapter_prefix in cases:
            source = ROOT/'chapters'/name
            site_html, _ = build_site.render(source.read_text(), source)
            for html, prefix in ((site_html, ''), (combined, chapter_prefix)):
                with self.subTest(source=name, prefix=prefix):
                    parsed = AliasDestinations(html)
                    self.assertEqual(parsed.targets[prefix+old], prefix+current)
                    self.assertEqual(parsed.ids[prefix+old], 1)
                    self.assertEqual(parsed.ids[prefix+current], 1)
                    self.assertEqual(parsed.heading_titles[prefix+current], title)

    def test_previous_52_figure_primary_anchor_is_unique_and_keeps_map_topic(self):
        source = ROOT / "chapters/00_overview.md"
        site_html, _ = build_site.render(source.read_text(), source)
        combined_html, _ = build_pdf.build_html(build_date="2026-10-01")
        for html, prefix in ((site_html, ""), (combined_html, "ch-0-")):
            with self.subTest(prefix=prefix):
                parsed = AliasDestinations(html)
                old = prefix + "sec-u-3ebc13ca5a"
                current = prefix + "sec-u-bd5fecaa8e"
                self.assertEqual(parsed.targets[old], current)
                self.assertEqual(parsed.ids[old], 1)
                self.assertEqual(parsed.ids[current], 1)
                self.assertEqual(parsed.heading_titles[current], "6. 插图地图：80 张编号图在哪篇")

    def test_previous_60_and_original_sequential_map_keep_same_figure_topic(self):
        source = ROOT / "chapters/00_overview.md"
        site_html, _ = build_site.render(source.read_text(), source)
        combined_html, _ = build_pdf.build_html(build_date="2026-10-01")
        for html, prefix in ((site_html, ""), (combined_html, "ch-0-")):
            parsed = AliasDestinations(html)
            current = prefix + "sec-u-bd5fecaa8e"
            for old in (prefix + "sec-u-f19f47c7c1", prefix + "sec-u-2b17484f3b",
                        prefix + "sec-u-0d7a1562fa", prefix + "sec-23"):
                self.assertEqual(parsed.targets[old], current)
                self.assertEqual(parsed.ids[old], 1)
            self.assertEqual(parsed.heading_titles[current], "6. 插图地图：80 张编号图在哪篇")

    def test_previous_69_figure_and_15_file_primary_anchors_keep_map_topics(self):
        source = ROOT / "chapters/00_overview.md"
        site_html, _ = build_site.render(source.read_text(), source)
        combined_html, _ = build_pdf.build_html(build_date="2026-10-04")
        cases = (
            ("sec-u-9c77d218fe", "sec-u-bd5fecaa8e",
             "6. 插图地图：80 张编号图在哪篇"),
            ("sec-u-2afd7c6509", "sec-u-461c9549d0",
             "2. 16 个文件一览"),
        )
        for html, prefix in ((site_html, ""), (combined_html, "ch-0-")):
            parsed = AliasDestinations(html)
            for old, current, title in cases:
                with self.subTest(prefix=prefix, old=old):
                    self.assertEqual(parsed.targets[prefix + old], prefix + current)
                    self.assertEqual(parsed.ids[prefix + old], 1)
                    self.assertEqual(parsed.ids[prefix + current], 1)
                    self.assertEqual(parsed.heading_titles[prefix + current], title)

    def test_chapter_nine_relocated_boundary_keeps_its_semantic_anchor(self):
        source = ROOT / 'chapters/09_source-tracking.md'
        site_html, _ = build_site.render(source.read_text(), source)
        combined_html, _ = build_pdf.build_html(build_date='2026-10-05')
        for html, prefix in ((site_html, ''), (combined_html, 'ch-9-')):
            parsed = AliasDestinations(html)
            boundary = prefix + 'sec-u-b51bd23596'
            self.assertEqual(parsed.ids[boundary], 1)
            self.assertEqual(parsed.heading_titles[boundary], '新练习的实现边界')
            self.assertLess(html.index('id="' + boundary + '"'),
                            html.index('E09-01：把“每帧”改为“每秒”'))
            # Neither was a published sequence: preserve the existing six aliases.
            self.assertNotIn(prefix + 'sec-43', parsed.ids)
            self.assertNotIn(prefix + 'sec-63', parsed.ids)

    def test_chapter_one_task_heading_keeps_published_topic(self):
        source = ROOT / "chapters/01_problem-definition.md"
        site_html, _ = build_site.render(source.read_text(), source)
        combined_html, _ = build_pdf.build_html(build_date="2026-10-04")
        for html, prefix in ((site_html, ""), (combined_html, "ch-1-")):
            parsed = AliasDestinations(html)
            old = prefix + "sec-u-81896116fa"
            current = prefix + "sec-u-5e074a7e92"
            self.assertEqual(parsed.targets[old], current)
            self.assertEqual(parsed.ids[old], 1)
            self.assertEqual(parsed.ids[current], 1)
            self.assertEqual(parsed.heading_titles[current], "任务定义与理想阵列收益")

    def test_chapter_one_distance_heading_keeps_original_topic_link(self):
        source = ROOT / "chapters/01_problem-definition.md"
        html, _ = build_site.render(source.read_text(), source)
        parsed = AliasDestinations(html)
        self.assertEqual(parsed.targets["sec-u-05ec932a34"], "sec-u-c36b7ac16e")
        self.assertEqual(parsed.ids["sec-u-05ec932a34"], 1)
        html, _ = build_pdf.build_html(build_date="2026-09-29")
        parsed = AliasDestinations(html)
        self.assertEqual(parsed.targets["ch-1-sec-u-05ec932a34"],
                         "ch-1-sec-u-c36b7ac16e")
        self.assertEqual(parsed.ids["ch-1-sec-u-05ec932a34"], 1)

    def test_every_frozen_sequence_stays_with_its_original_topic(self):
        for relative in self.SOURCES:
            source = ROOT / relative
            html, _ = build_site.render(source.read_text(), source)
            parsed = AliasDestinations(html)
            for alias, target in HISTORICAL_SEQUENTIAL_IDS[source.name].items():
                if source.name == "00_overview.md":
                    target = self.OVERVIEW_RENAMES.get(target, target)
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
                source = ROOT / "codes" / "chapters" / "ch00" / "research" / name
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
        for sequence, current in {"sec-20": "sec-u-3937b1b94e",
                                  "sec-21": "sec-u-f958564d39",
                                  "sec-23": "sec-u-bd5fecaa8e"}.items():
            self.assertEqual(parsed.targets[sequence], current)

    def test_combined_book_keeps_the_same_overview_destinations(self):
        # build_html returns a string and performs no filesystem writes.
        html, _ = build_pdf.build_html(build_date="2026-09-26")
        parsed = AliasDestinations(html)
        for old, current in self.OVERVIEW_RENAMES.items():
            self.assertEqual(parsed.targets["ch-0-" + old], "ch-0-" + current)
            self.assertEqual(parsed.ids["ch-0-" + old], 1)
        for sequence, current in {"sec-20": "sec-u-3937b1b94e",
                                  "sec-21": "sec-u-f958564d39",
                                  "sec-23": "sec-u-bd5fecaa8e"}.items():
            self.assertEqual(parsed.targets["ch-0-" + sequence], "ch-0-" + current)


if __name__ == "__main__":
    unittest.main()
