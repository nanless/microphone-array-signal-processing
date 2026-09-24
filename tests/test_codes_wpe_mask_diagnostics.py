"""Independent small-input checks for mask scaling and WPE solve diagnostics."""

import unittest

import numpy as np

from codes.array_tutorial.dereverberation import offline_wpe
from codes.array_tutorial.separation import mask_mvdr_2x2


class TestMaskScalingBoundary(unittest.TestCase):
    def test_positive_tiny_mask_and_zero_mask_have_different_outcomes(self):
        # Target steering [1,1] and interference steering [1,-1] are
        # orthogonal. The constrained two-mic minimum-interference weight is
        # analytically [1/2,1/2], giving [1,1,0,0] on these four frames.
        spectrum = np.array([[[1, 1, 1, 1], [1, 1, -1, -1]]], dtype=complex)
        target = np.array([[1, 1, 0, 0]], dtype=float)
        interference = np.array([[0, 0, 1, 1]], dtype=float)

        for scale in (1.0, 1e-300):
            with self.subTest(scale=scale):
                output, weights = mask_mvdr_2x2(spectrum, target * scale, interference)
                np.testing.assert_allclose(weights, [[0.5, 0.5]], rtol=0, atol=1e-12)
                np.testing.assert_allclose(output, [[1, 1, 0, 0]], rtol=0, atol=1e-12)

        output, weights = mask_mvdr_2x2(spectrum, np.zeros_like(target), interference)
        np.testing.assert_array_equal(weights, [[1.0 + 0j, 0j]])
        np.testing.assert_array_equal(output, spectrum[:, 0, :])


class TestWpeSolveDiagnostics(unittest.TestCase):
    def test_singular_constant_history_reports_least_squares_and_target_loss(self):
        # Four effective regressors are all [1,1]. Their unweighted normal
        # matrix is 4*[[1,1],[1,1]], with rank 1 and r=[4,4]. The minimum
        # norm predictor [1/2,1/2] removes every effective target frame.
        spectrum = np.ones((1, 6), dtype=complex)
        ordinary = offline_wpe(spectrum, taps=2, delay=1, iterations=1,
                               diagonal_loading=0.0)
        output, diagnostics = offline_wpe(spectrum, taps=2, delay=1, iterations=1,
                                          diagonal_loading=0.0, return_diagnostics=True)

        np.testing.assert_array_equal(output, ordinary)
        np.testing.assert_array_equal(output[:, :2], spectrum[:, :2])
        np.testing.assert_allclose(output[:, 2:], 0.0, rtol=0, atol=1e-14)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].frequency, 0)
        self.assertEqual(diagnostics[0].min_rank, 1)
        self.assertEqual(diagnostics[0].least_squares_count, 1)
        self.assertIsNone(diagnostics[0].bypass_reason)
        self.assertGreater(diagnostics[0].max_condition_number, 1e15)

    def test_loading_restores_rank_but_does_not_establish_reverb(self):
        # Loading = 1 adds (trace(R)/2) I = 4 I. Solving
        # [[8,4],[4,8]] g = [4,4] gives g=[1/3,1/3] and residual 1/3.
        output, diagnostics = offline_wpe(np.ones((1, 6), dtype=complex),
                                          taps=2, delay=1, iterations=1,
                                          diagonal_loading=1.0,
                                          return_diagnostics=True)
        np.testing.assert_allclose(output[:, 2:], 1 / 3, rtol=0, atol=1e-14)
        self.assertEqual(diagnostics[0].min_rank, 2)
        self.assertEqual(diagnostics[0].least_squares_count, 0)

    def test_bypass_reasons_and_default_shape(self):
        spectrum = np.ones((2, 4), dtype=complex)
        self.assertIsInstance(offline_wpe(spectrum, taps=0, delay=1), np.ndarray)
        for options, reason in (({"taps": 0, "delay": 1}, "zero_taps"),
                                ({"taps": 2, "delay": 3}, "short_record")):
            with self.subTest(reason=reason):
                result, diagnostics = offline_wpe(spectrum, return_diagnostics=True, **options)
                np.testing.assert_array_equal(result, spectrum)
                self.assertEqual([item.bypass_reason for item in diagnostics], [reason] * 2)

        result, diagnostics = offline_wpe(np.zeros((1, 6), dtype=complex),
                                          taps=2, delay=1, return_diagnostics=True)
        np.testing.assert_array_equal(result, np.zeros((1, 6), dtype=complex))
        self.assertEqual(diagnostics[0].bypass_reason, "zero_input")


if __name__ == "__main__":
    unittest.main()
