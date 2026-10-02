"""Independent syntax/semantic expectations for shared publication Markdown."""

import unittest
from unittest.mock import patch

import markdown

from scripts import build_markdown_helpers as helpers
from scripts import build_pdf, build_site, quality_check


class MarkdownBoundaryTests(unittest.TestCase):
    def both(self, source):
        site, _ = build_site.render(source)
        protected, repo = build_pdf.shield_math(source)
        pdf = build_pdf.unshield_math(markdown.markdown(
            protected, extensions=["tables", "fenced_code", "sane_lists"]), repo)
        return site, pdf

    def test_shared_math_implementation(self):
        self.assertIs(build_pdf.shield_math, helpers.shield_math)
        self.assertIs(build_pdf.unshield_math, helpers.unshield_math)
        with patch.object(build_site, "render_markdown", wraps=helpers.render_markdown) as call:
            build_site.render("$x_y$")
        call.assert_called_once_with("$x_y$")

    def test_indented_and_nested_code_are_literal(self):
        for source in ('    price = "$x<y$"\n\n$x<y$',
                       '1. 示例\n\n        price = "$x<y$"\n\n    $x<y$'):
            for html in self.both(source):
                self.assertIn('price = "$x&lt;y$"', html)
                self.assertIn('$x\\lt y$', html)
                self.assertNotIn('price = "$x\\lt y$"', html)

    def test_unpaired_code_dollar_does_not_eat_later_math(self):
        for html in self.both('    $unpaired\n\n$x_y$'):
            self.assertIn('<pre><code>$unpaired\n</code></pre>', html)
            self.assertIn('<p>$x_y$</p>', html)

    def test_raw_code_and_attributes_keep_original_values(self):
        source = ('<pre><code>price = "$x<y$"</code></pre>\n\n'
                  '<div title="$a<b$">literal</div>\n\n$x<y$')
        for html in self.both(source):
            self.assertIn('<pre><code>price = "$x<y$"</code></pre>', html)
            self.assertIn('<div title="$a<b$">literal</div>', html)
            self.assertIn('$x\\lt y$', html)

    def test_list_math_and_table_pipes_are_not_code_or_columns(self):
        source = ('1. 公式\n\n    $$x<y$$\n\n'
                  '| 量 | 含义 |\n|---|---|\n| $|X|^2$ | 功率 |')
        for html in self.both(source):
            self.assertIn('<p>$$x\\lt y$$</p>', html)
            self.assertIn('<td>$|X|^2$</td>', html)
            self.assertEqual(html.count('<td>'), 2)

    def test_backslash_parity_and_escaped_inner_dollar(self):
        source = r'\$cash\$ 与 $x+\$5$，\\$x_y$，\\\$literal\$'
        for html in self.both(source):
            self.assertIn(r'\$cash\$', html)
            self.assertIn(r'$x+\$5$', html)
            self.assertIn(r'\\$x_y$', html)
            self.assertIn(r'\\\$literal\$', html)
            self.assertNotIn('<em>', html)

    def test_literal_old_and_new_stash_names_do_not_collide(self):
        source = '原文 @@MATH0@@ @@MATHX0@@ @@CODETOKEN0@@；`$a<b$` 与 $x_y$'
        for html in self.both(source):
            for value in ('@@MATH0@@', '@@MATHX0@@', '@@CODETOKEN0@@'):
                self.assertEqual(html.count(value), 1)
            self.assertIn('<code>$a&lt;b$</code>', html)
            self.assertIn('$x_y$', html)

    def test_actual_fence_closing_grammar_controls_headings(self):
        for invalid_close in ('``` trailing', '````'):
            source = ('## 前\n\n```text\nx\n' + invalid_close +
                      '\n## 仍在代码\n```\n\n## 后')
            self.assertEqual(build_site.parse_headings(source), [(2, '前'), (2, '后')])
            self.assertEqual(quality_check.markdown_headings(source), [(2, '前'), (2, '后')])
            html, count = build_site.render(source)
            self.assertEqual(count, 2)
            expected = build_site.heading_anchor('后', 2)
            self.assertIn(f'<h2 id="{expected}">后</h2>', html)

    def test_raw_pre_and_indentation_do_not_create_phantom_headings(self):
        source = '<pre>\n## 原生代码\n</pre>\n\n    ## 缩进代码\n\n## 真标题'
        self.assertEqual(build_site.parse_headings(source), [(2, '真标题')])
        # The validator may share grammar, but must not ask the publisher to
        # supply its expected headings or anchor counts.
        with patch.object(build_site, 'parse_headings', side_effect=AssertionError):
            self.assertEqual(quality_check.markdown_headings(source), [(2, '真标题')])

    def test_fence_crlf_roundtrip_and_quality_line_positions(self):
        source = '第一行\r\n~~~text\r\n## 假标题\r\n~~~\r\n## 第五行'
        protected, repo = helpers.protect_code(source)
        self.assertEqual(helpers.restore_code(protected, repo), source)
        self.assertEqual(build_site.parse_headings(source), [(2, '第五行')])
        stripped = quality_check.strip_fenced_code(source)
        self.assertEqual(stripped.count('\n', 0, stripped.index('第五行')) + 1, 5)

    def test_backticks_with_different_run_lengths_are_literal(self):
        for html in self.both('`` `$x<y$` `` 与 $x_y$'):
            self.assertIn('<code>`$x&lt;y$`</code>', html)
            self.assertIn('$x_y$', html)

    def test_only_table_cell_file_labels_become_links(self):
        for source in ('`00_overview.md`', '```text\n00_overview.md\n```',
                       '    00_overview.md', '<pre>00_overview.md</pre>'):
            html, _ = build_site.render(source)
            self.assertNotIn('<a ', html)
            self.assertIn('00_overview.md', html)
        html, _ = build_site.render('| 文件 |\n|---|\n| `00_overview.md` |')
        self.assertIn('<td><code><a href="index.html">00_overview.md</a></code></td>', html)

    def test_table_link_mapping_excludes_attributes_pre_and_existing_links(self):
        source = ('<table><tr><td title="00_overview.md">00_overview.md '
                  '<pre>00_overview.md</pre>'
                  '<a href="https://example.org/other">00_overview.md</a></td></tr></table>')
        html, _ = build_site.render(source)
        self.assertIn('title="00_overview.md"', html)
        self.assertIn('<pre>00_overview.md</pre>', html)
        self.assertIn('<a href="https://example.org/other">00_overview.md</a>', html)
        self.assertEqual(html.count('href="index.html"'), 1)


if __name__ == '__main__':
    unittest.main()
