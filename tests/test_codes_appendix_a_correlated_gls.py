"""Independent Fraction/Cholesky oracles for one known correlated GLS model."""

from fractions import Fraction
import json
import math
import unittest

import numpy as np

from codes.chapters.appendix_a.appendix_a_experiments import correlated_gls_demo


class CorrelatedGLSTests(unittest.TestCase):
    def setUp(self):
        self.row = correlated_gls_demo()

    def test_full_covariance_adjugate_and_actual_variance(self):
        # A scalar adjugate with exact rationals is independent of whitening
        # and the lstsq implementation used by the teaching example.
        one, off, four = Fraction(1), Fraction(3, 2), Fraction(4)
        determinant = one * four - off * off
        inverse = [[four / determinant, -off / determinant],
                   [-off / determinant, one / determinant]]
        raw = [sum(row) for row in inverse]
        expected = [value / sum(raw) for value in raw]
        self.assertEqual(expected, [Fraction(5, 4), Fraction(-1, 4)])
        self.assertAlmostEqual(self.row['determinant'], float(determinant))
        np.testing.assert_allclose(self.row['noise_precision'],
                                   np.array(inverse, dtype=float), atol=7e-16, rtol=0)
        case = self.row['cases']['full_covariance']
        np.testing.assert_allclose(case['effective_original_weight'],
                                   list(map(float, expected)), atol=3e-16, rtol=0)
        self.assertAlmostEqual(case['target_response'], 1, places=14)
        self.assertAlmostEqual(case['variance_under_given_covariance'], 7/8, places=14)

    def test_all_candidates_use_same_correlated_covariance(self):
        covariance = [[Fraction(1), Fraction(3, 2)],
                      [Fraction(3, 2), Fraction(4)]]
        definitions = (
            ('ols', [Fraction(1, 2)] * 2, Fraction(2)),
            ('diagonal_only', [Fraction(4, 5), Fraction(1, 5)], Fraction(32, 25)),
            ('full_covariance', [Fraction(5, 4), Fraction(-1, 4)], Fraction(7, 8)),
        )
        for name, weights, variance in definitions:
            actual_variance = sum(weights[i] * covariance[i][j] * weights[j]
                                  for i in range(2) for j in range(2))
            self.assertEqual(actual_variance, variance)
            case = self.row['cases'][name]
            np.testing.assert_allclose(case['effective_original_weight'],
                                       list(map(float, weights)), atol=3e-16, rtol=0)
            self.assertAlmostEqual(case['target_response'], 1, places=14)
            self.assertAlmostEqual(case['variance_under_given_covariance'],
                                   float(variance), places=14)
        # The diagonal-only model predicts .8, but actual correlation adds .48.
        self.assertGreater(self.row['cases']['diagonal_only']['variance_under_given_covariance'], .8)

    def test_cholesky_joint_transform_and_weight_coordinate(self):
        root = math.sqrt(7)
        expected_factor = [[1, 0], [1.5, root/2]]
        expected_whitener = [[1, 0], [-3/root, 2/root]]
        np.testing.assert_allclose(self.row['cholesky_factor'], expected_factor,
                                   atol=0, rtol=0)
        np.testing.assert_allclose(self.row['whitening_matrix'], expected_whitener,
                                   atol=3e-16, rtol=0)
        np.testing.assert_allclose(self.row['whitened_noise_covariance'], np.eye(2),
                                   atol=3e-16, rtol=0)
        np.testing.assert_allclose(self.row['whitened_design'], [[1], [-1/root]],
                                   atol=2e-16, rtol=0)
        np.testing.assert_allclose(self.row['whitened_observation'], [0, 4/root],
                                   atol=3e-16, rtol=0)
        np.testing.assert_allclose(self.row['whitened_coordinate_weight'],
                                   [7/8, -root/8], atol=2e-16, rtol=0)
        self.assertEqual(self.row['whitened_design_rank'], 1)
        self.assertAlmostEqual(self.row['whitened_design_singular_values'][0], math.sqrt(8/7))

    def test_wrong_only_rhs_changes_actual_target_response(self):
        root = math.sqrt(7)
        wrong = self.row['wrong_only_observation_whitened']
        np.testing.assert_allclose(wrong['effective_original_weight'],
                                   [(1-3/root)/2, 1/root], atol=2e-16, rtol=0)
        self.assertAlmostEqual(wrong['target_response'], (1-1/root)/2, places=14)
        self.assertLess(wrong['target_response'], .32)
        self.assertAlmostEqual(wrong['fixed_observation_estimate'], 2/root, places=14)

    def test_completed_square_complex_weights_and_one_observation(self):
        # Include imaginary t to check the complex-conjugate cross term;
        # this scalar expansion never calls lstsq or the exercise function.
        for t in (0, .5, .8, 1.25, 1.25+2j, -1+1j):
            first, second = t, 1-t
            variance = (abs(first)**2 + 4*abs(second)**2
                        + 3*(complex(first).conjugate()*second).real)
            self.assertAlmostEqual(variance, 2*abs(t-1.25)**2 + 7/8, places=13)
            self.assertGreaterEqual(variance, 7/8)
        expected = {'ols': 1, 'diagonal_only': .4, 'full_covariance': -.5}
        for name, value in expected.items():
            self.assertAlmostEqual(self.row['cases'][name]['fixed_observation_estimate'],
                                   value, places=14)
        self.assertAlmostEqual(self.row['whitened_lstsq_solution'][0], -.5, places=14)
        self.assertEqual(self.row['exercise_id'], 'E14-20')
        json.dumps(self.row, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
