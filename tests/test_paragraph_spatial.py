"""Render actual chapter passages in memory; never rewrite publication outputs."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

from scripts.build_site import render


ROOT = Path(__file__).resolve().parents[1]


class TopLevelLists(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lists = []
        self.depth = 0
        self.in_item = False

    def handle_starttag(self, tag, attrs):
        if tag in ("ol", "ul"):
            if self.depth == 0:
                self.lists.append([])
            self.depth += 1
        elif tag == "li" and self.depth == 1:
            self.lists[-1].append("")
            self.in_item = True

    def handle_endtag(self, tag):
        if tag == "li" and self.depth == 1:
            self.in_item = False
        elif tag in ("ol", "ul"):
            self.depth -= 1

    def handle_data(self, data):
        if self.in_item:
            self.lists[-1][-1] += data


def chapter_fragment(filename, start, end):
    path = ROOT / "chapters" / filename
    source = path.read_text(encoding="utf-8")
    return source.split(start, 1)[1].split(end, 1)[0], path


def rendered_lists(source, path):
    html, _ = render(source, path)
    parser = TopLevelLists()
    parser.feed(html)
    return html, parser.lists


class SpatialParagraphTest(unittest.TestCase):
    def test_spatial_aliasing_has_three_separate_steps(self):
        source, path = chapter_fragment(
            "02_basics-signal-model.md", "**为什么 ", "等号边界可直接代入核对"
        )
        _, lists = rendered_lists("**为什么 " + source, path)
        self.assertEqual([len(items) for items in lists], [3])
        for item, phrase in zip(lists[0], ("相邻两麦", "相位以", "取值范围")):
            self.assertIn(phrase, item)
        self.assertIn(r"\psi = 2\pi d\sin\theta/\lambda", lists[0][0])
        self.assertIn(r"\sin\theta-\sin\theta_0=\dfrac{\lambda}{d}", lists[0][1])
        self.assertIn(r"\lambda/d>2", lists[0][2])

    def test_mvdr_implementation_has_three_separate_questions(self):
        source, path = chapter_fragment(
            "05_beamforming.md", "**实现中的三个问题**", "**对角加载、WNG"
        )
        _, lists = rendered_lists("**实现中的三个问题**" + source, path)
        self.assertEqual([len(items) for items in lists], [3])
        for item, phrase in zip(lists[0], ("从哪来", "病态与对角加载", "混响中的约束对象")):
            self.assertIn(phrase, item)
        self.assertIn(r"\hat{\mathbf R}+\varepsilon\mathbf I", lists[0][1])

    def test_kalman_five_steps_keep_equations_and_explanations_in_their_items(self):
        source, path = chapter_fragment(
            "09_source-tracking.md", "**卡尔曼滤波（KF）的五步递推**", "方位角跨越表示区间"
        )
        html, lists = rendered_lists("**卡尔曼滤波（KF）的五步递推**" + source, path)
        self.assertEqual([len(items) for items in lists], [5])
        labels = ("预测状态", "预测协方差", "计算卡尔曼增益", "更新状态", "更新协方差")
        for number, (item, label) in enumerate(zip(lists[0], labels), start=2):
            self.assertIn(label, item)
            equation = rf"\tag{{9-{number}}}"
            self.assertEqual(item.count(equation), 1)
            self.assertEqual(html.count(equation), 1)
            self.assertEqual(item.count(r"\tag{"), 1)
        self.assertIn("未建模运动的过程噪声", lists[0][1])
        self.assertIn("更新更依赖观测", lists[0][2])
        self.assertIn("括号里的差称为新息", lists[0][3])
        self.assertNotIn("<pre>", html)

    def test_missing_intro_separator_reproduces_the_original_failure(self):
        source, path = chapter_fragment(
            "02_basics-signal-model.md", "**为什么 ", "等号边界可直接代入核对"
        )
        broken = ("**为什么 " + source).replace("：\n\n1.", "：\n1.", 1)
        _, lists = rendered_lists(broken, path)
        self.assertEqual(lists, [])

    def test_decimal_prose_is_not_a_list(self):
        _, lists = rendered_lists("频率为 3.4 kHz。距离为 0.05 m。", ROOT / "chapters/example.md")
        self.assertEqual(lists, [])


if __name__ == "__main__":
    unittest.main()
