"""Render actual chapter passages in memory; never rewrite publication outputs."""
from html.parser import HTMLParser
import math
from pathlib import Path
import re
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

    def test_mask_beamforming_keeps_math_and_explanation_inside_three_steps(self):
        source, path = chapter_fragment(
            "05_beamforming.md", "一个可复现的处理顺序如下。", "**两麦小例子**"
        )
        html, lists = rendered_lists(source, path)
        self.assertEqual([len(items) for items in lists], [3])
        self.assertIn(r"\hat{\mathbf R}_{ss}(k)", lists[0][0])
        self.assertIn("零掩码拥有统计依据", lists[0][0])
        self.assertIn(r"\lambda_{\max}", lists[0][1])
        self.assertIn("尺度规则", lists[0][1])
        self.assertIn("滤波与检查", lists[0][2])
        self.assertGreaterEqual(html.count("<p>"), 6)
        self.assertNotIn("<pre>", html)

    def test_reviewed_topic_changes_are_separate_rendered_paragraphs(self):
        checks = [
            ("02_basics-signal-model.md", "周期 Hann 窗", "会在两端补半窗"),
            ("02_basics-signal-model.md", "需要加载时采用", "实际运行时还应记录"),
            ("04_doa-estimation.md", "信噪比高、快拍充足时", "相干源场景还存在"),
            ("04_doa-estimation.md", "这些方法的源码也应分别阅读", "DCASE2022 基线则通过"),
            ("04_doa-estimation.md", "最小输出检查可以先做手算", "换到 DCASE2025 基线时"),
            ("04_doa-estimation.md", "本书的原创 NumPy 基线位于", "对应式(4-1)"),
            ("04_doa-estimation.md", "接入实际录音时还要固定", "Capon/MVDR 使用"),
            ("04_doa-estimation.md", "Capon/MVDR 使用", "MUSIC 使用厄米特征分解"),
            ("05_beamforming.md", "§5.7 的噪声估计也需要", "Cohen 的"),
            ("05_beamforming.md", "以参考麦选择向量", "分子、分母要用同一频带"),
        ]
        for filename, first, second in checks:
            path = ROOT / "chapters" / filename
            html, _ = render(path.read_text(encoding="utf-8"), path)
            paragraphs = re.findall(r"<p>(.*?)</p>", html, flags=re.S)
            with self.subTest(filename=filename, first=first):
                containing_first = [p for p in paragraphs if first in p]
                containing_second = [p for p in paragraphs if second in p]
                self.assertTrue(containing_first)
                self.assertTrue(containing_second)
                self.assertFalse(any(second in p for p in containing_first))

    def test_noise_reduction_formula_has_its_own_block(self):
        source, path = chapter_fragment(
            "05_beamforming.md", "上表“目标保持条件”", "#### 延伸阅读"
        )
        html, _ = render("上表“目标保持条件”" + source, path)
        paragraphs = re.findall(r"<p>(.*?)</p>", html, flags=re.S)
        def paragraph_index(needle):
            matches = [index for index, paragraph in enumerate(paragraphs) if needle in paragraph]
            self.assertEqual(len(matches), 1, f"Expected one paragraph containing {needle}")
            return matches[0]

        intro = paragraph_index("目标保持条件")
        nr_formula = paragraph_index(r"\tag{5-9}")
        nr_explanation = paragraph_index("分子、分母要用同一频带")
        example = paragraph_index("同输入算例：零陷")
        example_conditions = paragraph_index("设计使用精确的干扰加噪声协方差")
        sinr_definition = paragraph_index("真实目标输出信干噪比按同一功率口径计算")
        sinr_formula = paragraph_index(r"\tag{5-10}")
        hand_check = paragraph_index("可先独立核对 DSB 行")
        boundary = paragraph_index("表中数字不代表语音可懂度")

        self.assertEqual(
            [intro, nr_formula, nr_explanation, example, example_conditions,
             sinr_definition, sinr_formula, hand_check, boundary],
            sorted({intro, nr_formula, nr_explanation, example, example_conditions,
                    sinr_definition, sinr_formula, hand_check, boundary}),
            "Baseline, formulas, conditions, hand check and boundary must stay in distinct ordered paragraphs",
        )
        self.assertTrue(paragraphs[nr_formula].strip().startswith("$$"))
        self.assertTrue(paragraphs[nr_formula].strip().endswith("$$"))
        self.assertTrue(paragraphs[sinr_formula].strip().startswith("$$"))
        self.assertTrue(paragraphs[sinr_formula].strip().endswith("$$"))
        self.assertLess(html.index(r"\tag{5-10}"), html.index("<table>"))
        self.assertLess(html.index("</table>"), html.index("可先独立核对 DSB 行"))

    def test_music_and_nearfield_formula_belongs_to_third_step(self):
        for start, end, formula, explanation in (
                ("**思想三步**", "**第 2 步为什么成立**", r"\tag{4-3}", "正交检验的含义"),
                ("**锚点算法：角度 × 距离球面扫描。**", "实际 SRP 或宽带最大似然",
                 r"J(r,\theta)", "分母为 3")):
            source, path = chapter_fragment("04_doa-estimation.md", start, end)
            html, lists = rendered_lists(start+source, path)
            self.assertEqual([len(items) for items in lists], [3])
            self.assertIn(formula, lists[0][2])
            self.assertIn(explanation, lists[0][2])
            self.assertNotIn("<pre>", html)

    def test_rank_one_mwf_relation_stays_in_its_parameter_item(self):
        source, path = chapter_fragment("05_beamforming.md", "- $\\mathbf{R}_{ss},", "前文给出的是 LCMV")
        html, lists = rendered_lists("- $\\mathbf{R}_{ss},"+source, path)
        self.assertEqual([len(items) for items in lists], [2])
        self.assertIn(r"\frac{\phi_s q}{\mu+\phi_s q}", lists[0][1])
        self.assertIn("目标协方差满秩", lists[0][1])
        self.assertNotIn("<pre>", html)

    def test_accdoa_external_and_book_azimuths_are_complementary(self):
        source = (ROOT / "chapters/04_doa-estimation.md").read_text(encoding="utf-8")
        self.assertIn("同一方向记为 60°", source)
        self.assertIn("用 $90°$ 减去外部方位角，得到约 60°", source)
        # The published three-decimal vector is rounded, not an exact 30-degree ray.
        x, y = .433, .25
        external = math.degrees(math.atan2(y, x))
        book = math.degrees(math.atan2(x, y))
        self.assertAlmostEqual(external + book, 90)
        self.assertLess(abs(external - 30), .001)
        self.assertLess(abs(book - 60), .001)
        self.assertLess(abs(math.hypot(x, y) - .5), .0001)


if __name__ == "__main__":
    unittest.main()
