"""MDL expectations from high-precision scalar arithmetic, not the tested code."""

from decimal import Decimal, localcontext
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.array_tutorial.doa import mdl_source_count


def decimal_scores(values, snapshots):
    """Independent literal AM/GM formula with a wide Decimal exponent range."""
    with localcontext() as context:
        context.prec = 80
        spectrum = sorted((Decimal.from_float(float(x)) for x in values), reverse=True)
        n = Decimal(int(snapshots))
        scores = []
        for k in range(len(spectrum)):
            tail = spectrum[k:]
            q = Decimal(len(tail))
            product = Decimal(1)
            for value in tail:
                product *= value
            arithmetic = sum(tail) / q
            fit = n * (q * arithmetic.ln() - product.ln())
            penalty = Decimal(k * (2 * len(spectrum) - k)) * n.ln() / 2
            scores.append(float(fit + penalty))
        return np.asarray(scores)


class MDLTest(unittest.TestCase):
    def test_four_candidate_hand_calculation(self):
        selected, scores = mdl_source_count([9, 4, 1.1, .9], 100)
        self.assertEqual(selected, 2)
        np.testing.assert_allclose(scores, decimal_scores([9, 4, 1.1, .9], 100), rtol=1e-13)

    def test_all_orders_are_sorted_without_mutation(self):
        expected = decimal_scores([9, 4, 1.1, .9], 100)
        for order in itertools.permutations([9, 4, 1.1, .9]):
            values = np.array(order)
            before = values.copy()
            selected, scores = mdl_source_count(values, np.int64(100))
            self.assertEqual(selected, 2)
            np.testing.assert_array_equal(values, before)
            np.testing.assert_allclose(scores, expected, rtol=1e-13)

    def test_scale_invariance_and_extreme_positive_values(self):
        reference = decimal_scores([9, 4, 1.1, .9], 100)
        for scale in [1e-300, 1e-200, 1., 1e200, 1e300]:
            values = np.array([9, 4, 1.1, .9]) * scale
            selected, scores = mdl_source_count(values, 100)
            self.assertEqual(selected, 2)
            np.testing.assert_allclose(scores, reference, rtol=2e-12, atol=1e-11)
        for values in ([1e308, 1e100, 1e-100, np.nextafter(0., 1.)],
                       [4e-320, 3e-320, 2e-320, 1e-320],
                       [np.finfo(float).max] * 4):
            with np.errstate(all="raise"):
                _, scores = mdl_source_count(values, 100)
            np.testing.assert_allclose(scores, decimal_scores(values, 100), rtol=2e-12, atol=1e-10)

    def test_noise_only_single_channel_and_last_candidate(self):
        for common in [1., 1e-300, np.nextafter(0., 1.)]:
            selected, scores = mdl_source_count(np.full(4, common), 100)
            self.assertEqual(selected, 0)
            np.testing.assert_allclose(scores, [0, 3.5*math.log(100), 6*math.log(100), 7.5*math.log(100)])
        selected, scores = mdl_source_count([1.], 2)
        self.assertEqual(selected, 0)
        np.testing.assert_array_equal(scores, [0.])
        selected, scores = mdl_source_count([10000., 100., 1.], 100)
        self.assertEqual(selected, 2)
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[-1], 4*math.log(100))

    def test_n_equals_m_is_accepted_but_not_a_reliability_guarantee(self):
        selected, scores = mdl_source_count([9, 4, 1.1, .9], 4)
        self.assertEqual(selected, 0)
        np.testing.assert_allclose(scores, decimal_scores([9, 4, 1.1, .9], 4))

    def test_invalid_spectra_and_snapshots_rejected(self):
        for invalid in ([], [[1., 2.]], [0., 1.], [-1., 1.], [np.nan], [np.inf],
                        [1+0j], [True, False], ["1", "2"], np.array([1], dtype=object)):
            with self.subTest(eigenvalues=invalid), self.assertRaises(ValueError):
                mdl_source_count(invalid, 100)
        for invalid in (True, np.bool_(True), 100., "100", 0, -1, 3, float("nan"), 10**1000):
            with self.subTest(snapshots=invalid), self.assertRaises(ValueError):
                mdl_source_count([9., 4., 1.1, .9], invalid)
        with self.assertRaises(ValueError):
            mdl_source_count([1.], 1)

    def test_overflow_reports_error_without_mutation(self):
        values = np.array([1e308, 1e-300])
        before = values.copy()
        with self.assertRaises(ValueError):
            mdl_source_count(values, 10**308)
        np.testing.assert_array_equal(values, before)


if __name__ == "__main__":
    unittest.main()
