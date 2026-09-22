"""Render real chapter passages in memory to protect list membership."""

from html.parser import HTMLParser
from pathlib import Path
import unittest

from scripts import build_site


ROOT = Path(__file__).resolve().parents[1]


class ListParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lists = []
        self.open_lists = []
        self.open_items = []

    def handle_starttag(self, tag, attrs):
        if tag in ("ol", "ul"):
            record = {"tag": tag, "items": []}
            self.lists.append(record)
            self.open_lists.append(record)
        elif tag == "li":
            item = {"text": "", "paragraphs": 0}
            self.open_lists[-1]["items"].append(item)
            self.open_items.append(item)
        elif tag == "p" and self.open_items:
            self.open_items[-1]["paragraphs"] += 1

    def handle_endtag(self, tag):
        if tag in ("ol", "ul"):
            self.open_lists.pop()
        elif tag == "li":
            self.open_items.pop()

    def handle_data(self, data):
        for item in self.open_items:
            item["text"] += data


class EnhancementParagraphTest(unittest.TestCase):
    def passage(self, filename, start, end):
        path = ROOT / "chapters" / filename
        source = path.read_text(encoding="utf-8")
        begin = source.index(start)
        return source[begin:source.index(end, begin)], path

    def render_items(self, source, path):
        html, _ = build_site.render(source, path)
        parser = ListParser()
        parser.feed(html)
        self.assertEqual(len(parser.lists), 1, "Expected one contiguous top-level list")
        self.assertEqual(parser.lists[0]["tag"], "ol")
        return parser.lists[0]["items"]

    def assert_labels(self, items, labels):
        self.assertEqual(len(items), len(labels))
        for item, label in zip(items, labels):
            self.assertTrue(item["text"].strip().startswith(label), item["text"])

    def test_dtd_five_methods_and_their_continuations(self):
        source, path = self.passage("06_aec.md", "**常见 DTD 方法**", "> **Geigel 判决")
        items = self.render_items(source, path)
        self.assert_labels(items, ["Geigel 能量比较", "互相关 /", "频域 NCC", "相干函数法", "双滤波器法"])
        self.assertEqual(items[0]["paragraphs"], 3)
        self.assertIn("经典示例阈值", items[0]["text"])
        self.assertEqual(items[4]["paragraphs"], 3)
        self.assertIn("WebRTC AEC3", items[4]["text"])

    def test_three_aec_beamforming_orders(self):
        source, path = self.passage("06_aec.md", "#### 6.1.7", "**联合分析**")
        items = self.render_items(source, path)
        self.assert_labels(items, ["AEC→BF", "BF→AEC", "联合优化"])
        self.assertEqual(items[0]["paragraphs"], 2)
        self.assertEqual(items[1]["paragraphs"], 3)
        self.assertIn("等效回声传递函数", items[1]["text"])
        self.assertNotIn("联合优化", items[1]["text"])

    def test_gss_three_parts_and_equation_stays_in_first_item(self):
        source, path = self.passage("08_speech-separation.md", "**GSS（导引源分离", "**紧凑实现顺序**")
        items = self.render_items(source, path)
        self.assert_labels(items, ["cACGMM 空间聚类", "导引（Guided）", "级联"])
        self.assertEqual(items[0]["paragraphs"], 3)
        self.assertIn(r"\tag{8-4}", items[0]["text"])
        self.assertIn("其中", items[0]["text"])
        self.assertIn("正定的形状矩阵", items[0]["text"])
        for item in items[1:]:
            self.assertNotIn(r"\tag{8-4}", item["text"])

    def test_eight_research_directions_preserve_arraydps(self):
        source, path = self.passage("13_appendix-guide.md", "### 13.3", "#### 专栏：")
        items = self.render_items(source, path)
        self.assert_labels(items, ["预训练模型", "阵列无关", "目标说话人提取", "生成式语音增强", "TF-GridNet", "几何无关前端", "CHiME-9", "扩散先验"])
        self.assertIn("FlowSep", items[3]["text"])
        self.assertIn("DiCoW", items[6]["text"])
        self.assertIn("ArrayDPS", items[7]["text"])
        self.assertIn("最小复现", items[7]["text"])

    def test_eight_debugging_cases_and_continuations(self):
        source, path = self.passage("13_appendix-guide.md", "### 13.4", "### 13.5")
        items = self.render_items(source, path)
        self.assert_labels(items, ["MUSIC", "GCC-PHAT", "波束形成后", "超指向", "AEC", "SRP-PHAT", "多麦录音", "仿真很好"])
        for index, expected in [(0, "再检查源数"), (2, "噪声协方差既可"), (4, "若线性路径已收敛"), (7, "位置容差")]:
            self.assertEqual(items[index]["paragraphs"], 2)
            self.assertIn(expected, items[index]["text"])

    def test_missing_indent_cannot_pass_gss_membership_check(self):
        source, path = self.passage("08_speech-separation.md", "**GSS（导引源分离", "**紧凑实现顺序**")
        broken = source.replace("\n    ", "\n   ")
        html, _ = build_site.render(broken, path)
        parser = ListParser()
        parser.feed(html)
        self.assertNotIn(r"\tag{8-4}", parser.lists[0]["items"][0]["text"])


if __name__ == "__main__":
    unittest.main()
