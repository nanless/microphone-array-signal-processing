"""Protect the two final engineering paragraph fixes with in-memory rendering."""

import re
import unittest

from tests.test_paragraph_spatial import chapter_fragment, rendered_lists


class EngineeringRoundTwoParagraphTest(unittest.TestCase):
    def test_project_boundaries_are_four_items_not_one_prose_paragraph(self):
        start = "开源实现也要按任务筛选。"
        source, path = chapter_fragment("11_selection-guide.md", start, "本书按")
        html, lists = rendered_lists(start + source, path)
        self.assertEqual([len(items) for items in lists], [4])
        for item, name, boundary in zip(
                lists[0],
                ("PortAudio", "ONNX Runtime", "pyroomacoustics", "WebRTC AEC3"),
                ("不解决跨设备同步", "不规定模型训练目标", "不证明目标设备实时", "不是其他产品的默认值")):
            self.assertTrue(item.startswith(name))
            self.assertIn(boundary, item)
        self.assertIn("<p>选择开源项目时", html.split("</ul>", 1)[1])
        self.assertNotIn("选择开源项目时", lists[0][-1])

    def test_symbol_reuse_and_stft_grid_are_distinct_paragraphs(self):
        start = "**字母复用**"
        source, path = chapter_fragment("12_appendix-symbols-math.md", start, "| 符号")
        html, lists = rendered_lists(start + source, path)
        paragraphs = re.findall(r"<p>(.*?)</p>", html, re.S)
        self.assertEqual(lists, [])
        self.assertEqual(len(paragraphs), 2)
        self.assertIn("具体含义以所在公式的定义为准", paragraphs[0])
        self.assertNotIn("需要区分 STFT 网格时", paragraphs[0])
        self.assertTrue(paragraphs[1].startswith("需要区分 STFT 网格时"))
        self.assertIn("f_k=kf_s/N", paragraphs[1])
        self.assertIn("离散采样索引用", paragraphs[1])


if __name__ == "__main__":
    unittest.main()
