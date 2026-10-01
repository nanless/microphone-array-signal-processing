"""Independent ratio identities and domain checks for Figure 21 metrics."""
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from scripts import make_figures as figures


class WPEMetricContractTests(unittest.TestCase):
    def test_direct_script_entry_resolves_chapter_kernel(self):
        # Reproduce the direct-entry search path, rather than importing scripts
        # from the test runner's already-populated repository path.
        script = Path(__file__).resolve().parents[1] / 'scripts/make_figures.py'
        code = ('import runpy,sys; sys.path[0]=sys.argv[1]; '
                'ns=runpy.run_path(sys.argv[1]+"/make_figures.py"); '
                'import numpy as np; '
                'result=ns["wpe_dereverb"](np.array([[1.,2.,3.,4.]],complex),K=1,delay=1,iters=1); '
                'assert result.shape==(1,4) and np.isfinite(result).all()')
        result = subprocess.run([sys.executable, '-c', code, str(script.parent)],
                                cwd=script.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nmse_orthogonal_projection_ratio_across_scales(self):
        # r=(1,3), x=(1,2): c=7/5, ||r-cx||²=1/5, ||r||²=10.
        expected = 10 * math.log10(1 / 50)
        for scale in (1.0, 1e-150, 1e-10, 1e150, 1e300):
            for estimate_scale in (1.0, 1e-150, 1e150):
                # Independent scales are allowed because the fitted gain is free.
                result = figures.scale_aligned_spectral_nmse_db(
                    np.array([[1, 3]], complex) * scale,
                    np.array([[1, 2]], complex) * estimate_scale,
                    np.array([True, True]))
                self.assertAlmostEqual(result, expected, places=12)

    def test_exact_zero_and_undefined_reference_are_distinct(self):
        mask = np.array([True, True])
        reference = np.array([[1, 2]], complex)
        for scale in (1.0, 1e-150, 1e150):
            self.assertEqual(figures.scale_aligned_spectral_nmse_db(
                reference * scale, reference * (2 * scale), mask), -math.inf)
            self.assertEqual(figures.scale_aligned_spectral_nmse_db(
                reference * scale, np.zeros_like(reference), mask), 0.0)
        with self.assertRaises(ValueError):
            figures.scale_aligned_spectral_nmse_db(np.zeros_like(reference), reference, mask)

    def test_unscored_frames_do_not_change_fit(self):
        self.assertAlmostEqual(figures.scale_aligned_spectral_nmse_db(
            [[1, 3, 1e300]], [[1, 2, -1e300]], [True, True, False]),
            10 * math.log10(1 / 50), places=12)

    def test_ratio_uses_all_frequency_bins_and_true_zero(self):
        spectrum = np.array([[1, 2], [2, 4]], complex)
        mask = np.array([True, False])
        for scale in (1.0, 1e-150, 1e150):
            self.assertAlmostEqual(figures.quiet_energy_ratio_db(
                spectrum * scale, mask), 10 * math.log10(1 / 5), places=11)
        self.assertEqual(figures.quiet_energy_ratio_db([[0, 2]], mask), -math.inf)
        with self.assertRaises(ValueError):
            figures.quiet_energy_ratio_db([[0, 0]], mask)

    def test_tiny_positive_ratio_is_not_epsilon_or_zero(self):
        self.assertAlmostEqual(figures.quiet_energy_ratio_db(
            [[1e-200, 1]], [True, False]), -4000.0, places=10)

    def test_invalid_dimensions_mask_and_finiteness(self):
        valid = np.array([[1, 2]], complex)
        for reference, estimate, mask in (
                ([1, 2], [1, 2], [True, True]),
                (valid, valid, [False, False]),
                (valid, valid, [1, 1]),
                (valid, valid, [True]),
                (valid, [[1]], [True, True]),
                ([[math.inf, 2]], valid, [True, True]),
                (valid, [[math.nan, 2]], [True, True]),
                (np.empty((0, 2)), np.empty((0, 2)), [True, True])):
            with self.subTest(reference=reference, estimate=estimate, mask=mask):
                with self.assertRaises(ValueError):
                    figures.scale_aligned_spectral_nmse_db(reference, estimate, mask)

    def test_current_float_support_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '支持范围'):
            figures.scale_aligned_spectral_nmse_db(
                [[1e308, 1e-308]], [[1, 2]], [True, True])


if __name__ == '__main__':
    unittest.main()
