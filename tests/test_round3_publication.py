"""Independent checks for the nonlinear PCM figure and real appendix lists."""
from pathlib import Path
import unittest

import numpy as np

from scripts import build_site, make_figures
from tests.test_paragraph_enhancement import ListParser, ParagraphParser

ROOT = Path(__file__).resolve().parents[1]


class Round3PublicationTest(unittest.TestCase):
    def test_appendix_following_exercises_are_not_swallowed_by_hints(self):
        path = ROOT / 'chapters/13_appendix-guide.md'
        source = path.read_text()
        cases = [('**第 4 章（定位）**', '**第 5 章（波束形成）**', '打开', '在 fig_doa_spectrum()'),
                 ('**第 5 章（波束形成）**', '**第 7 章（去混响）**', '从拉格朗日', '在 fig_wng_di()')]
        for start, end, first, second in cases:
            with self.subTest(start=start):
                passage = source.split(start, 1)[1].split(end, 1)[0]
                html, _ = build_site.render(passage, path)
                parser = ListParser()
                parser.feed(html)
                lists = [x for x in parser.lists if x['depth'] == 0]
                self.assertEqual(len(lists), 1)
                self.assertEqual(len(lists[0]['items']), 2)
                self.assertTrue(lists[0]['items'][0]['text'].strip().startswith(first))
                self.assertTrue(lists[0]['items'][1]['text'].strip().startswith(second))
                self.assertNotIn(second, lists[0]['items'][0]['text'])
                self.assertGreaterEqual(lists[0]['items'][0]['paragraphs'], 2)

    def test_nonlinear_figure_pcm_measurements_have_analytic_bounds(self):
        measured, digest = make_figures.nonlinear_echo_measurements()
        self.assertEqual(len(digest), 64)
        for name, amplitudes, power in [('reference', [.4, 0], .08),
                                         ('echo', [.496, .032], .12352),
                                         ('estimate', [.496, 0], .123008),
                                         ('residual', [0, .032], .000512)]:
            with self.subTest(name=name):
                np.testing.assert_allclose(measured[name]['amplitudes'], amplitudes, atol=1/32768, rtol=0)
                # |mean(q^2)-mean(x^2)| <= 2*peak(x)*half_LSB+half_LSB^2.
                half_lsb = .5/32768
                self.assertAlmostEqual(measured[name]['power'], power,
                                       delta=2*sum(amplitudes)*half_lsb+half_lsb**2)


if __name__ == '__main__':
    unittest.main()
