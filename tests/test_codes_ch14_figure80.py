"""Figure80 outputs stay isolated and match independent scalar objectives."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import make_figures as drawing


class ImagingObjectiveFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        cls.image = cls.root/'fig80_imaging_objectives.png'
        cls.fonts = []
        cls.axes_ranges = []
        save = drawing.save
        def capture(fig, name, extra_metadata=None):
            drawing.finalize_figure(fig)
            cls.canvas = tuple(fig.get_size_inches())
            cls.fonts = [text.get_fontsize() for ax in fig.axes for text in
                         [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                          *ax.get_xticklabels(), *ax.get_yticklabels()]]
            cls.axes_ranges = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
            save(fig, name, extra_metadata=extra_metadata)
        # Every actual drawing/report write is confined to this directory.
        with patch.object(drawing, 'OUT', cls.root), \
                patch.object(drawing, 'CODE_CHAPTERS', cls.root/'codes/chapters'), \
                patch.object(drawing, 'save', side_effect=capture):
            drawing.fig_imaging_objectives()
        cls.report_path = cls.root/'codes/chapters/ch12/reports/figure80_imaging_objectives.json'
        cls.report = json.loads(cls.report_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_every_plotted_grid_value_matches_independent_closed_forms(self):
        plotted = self.report['results']['plotted']
        q1, q2 = np.meshgrid(plotted['q1'], plotted['q2'])
        scan = (q1+.25*q2-1)**2+(.25*q1+q2)**2
        csm = 64/9+4*(q1*q1+q2*q2)+2*q1*q2-8*q1
        np.testing.assert_allclose(plotted['scan_squared'], scan, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(plotted['csm_squared'], csm, rtol=2e-13, atol=2e-14)
        self.assertEqual(plotted['axis_range'], [[.8, 1.1], [0., .2]])
        self.assertEqual(plotted['view'], 'local nonnegative domain; not full domain')

    def test_two_minima_cross_objective_values_and_valid_input_are_bound(self):
        result = self.report['results']['experiment']
        np.testing.assert_allclose(result['q_scan'], [16/17, 0], atol=1e-14)
        np.testing.assert_allclose(result['q_CSM'], [1, 0], atol=1e-14)
        np.testing.assert_allclose(result['q_GS'], [1, 0], atol=1e-14)
        for name, expected in [('scan', (1/17, 8128/2601)), ('CSM', (1/16, 28/9)), ('GS', (1/16, 28/9))]:
            with self.subTest(solution=name):
                values = result['losses'][name]
                self.assertAlmostEqual(values['scan_squared'], expected[0], places=13)
                self.assertAlmostEqual(values['csm_frobenius_squared'], expected[1], places=13)
        r = np.array(result['R']['real'])+1j*np.array(result['R']['imag'])
        expected = np.array([[4/3, 2/3-2j*np.sqrt(3)/3], [2/3+2j*np.sqrt(3)/3, 4/3]])
        np.testing.assert_allclose(r, expected, atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(r)[0], -1e-14)

    def test_actual_png_and_report_bind_current_sources(self):
        expected_paths = {'scripts/make_figures.py', 'codes/chapters/ch12/core/imaging.py',
                          'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/io_contracts.py'}
        self.assertEqual(set(self.report['source_sha256']), expected_paths)
        for name, digest in self.report['source_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((drawing.REPOSITORY_ROOT/name).read_bytes()).hexdigest())
        with Image.open(self.image) as image:
            self.assertEqual(image.info['NumericalReportDigest'], hashlib.sha256(self.report_path.read_bytes()).hexdigest())
            self.assertEqual(image.info['SourceScriptDigest'], drawing.source_script_digest())
            # Auxiliary ticks/contour values and primary labels are readable
            # at the declared print and web widths, not merely on a large PNG.
            effective_pt = min(self.fonts)*(165/25.4)/(image.width/150)
            effective_css_px = min(self.fonts)*150/72*860/image.width
            self.assertGreaterEqual(effective_pt, 8.)
            self.assertGreaterEqual(effective_css_px, 11.)
        self.assertEqual(self.canvas, (9., 4.8))
        self.assertGreaterEqual(min(self.fonts), 11.5)
        self.assertEqual(self.axes_ranges, [((.8, 1.1), (0., .2))]*2)


if __name__ == '__main__':
    unittest.main()
