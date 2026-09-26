"""Independent fractions, complex geometry and boundaries for five exercises."""
from fractions import Fraction as F
import unittest
import numpy as np
from codes.examples.enhancement_step_exercises import ip_row, run_exercises


class EnhancementSteps(unittest.TestCase):
    def test_stable_ids(self):
        self.assertEqual(set(run_exercises()), {'E06-21', 'E07-06', 'E08-08', 'E08-09', 'E08-10'})

    def test_aec_batch_and_online_fraction_oracle(self):
        result = run_exercises()['E06-21']
        expected, errors, w = [], [], F(4, 5)
        for x, s in zip([1, 1, -1, -1], [F(1, 2), -F(1, 2), F(1, 2), -F(1, 2)]):
            e = F(4, 5)*x+s-w*x
            w += F(1, 2)*x*e
            expected.append(float(w)); errors.append(float(e))
        np.testing.assert_allclose(result['online_paths'], expected, atol=1e-15)
        np.testing.assert_allclose(result['prior_residuals'], errors, atol=1e-15)
        self.assertEqual(result['cross_sum'], 0)
        self.assertAlmostEqual(result['batch_path'], .8)
        self.assertAlmostEqual(result['correlated_batch_path'], 1.05)

    def test_shared_power_derivative_and_gain(self):
        result = run_exercises()['E07-06']
        for field, energy in [('shared_power', 6), ('one_channel_scaled', 18), ('all_channels_scaled', 24)]:
            power = result[field]
            self.assertAlmostEqual(2/power-energy/power**2, 0)
            self.assertGreater(-2/power**2+2*energy/power**3, 0)
        self.assertEqual(result['zero_raw_power'], 0)

    def test_ip_real_analytic(self):
        result = run_exercises()['E08-08']
        np.testing.assert_allclose(result['column_real'], [np.sqrt(2/3), -1/np.sqrt(6)])
        self.assertAlmostEqual(result['weighted_norm'], 1)
        self.assertAlmostEqual(result['euclidean_norm_squared'], 5/6)

    def test_ip_complex_conjugation_and_covariance_scale(self):
        v = np.array([[2, 1j], [-1j, 2]])
        w = ip_row(np.eye(2), v, 0)
        # Hand inverse: V^-1 e0 = [2/3, i/3]; quadratic = 2/3.
        np.testing.assert_allclose(w, [np.sqrt(2/3), 1j/np.sqrt(6)])
        np.testing.assert_allclose(ip_row(np.eye(2), v/2, 0), np.sqrt(2)*w)
        for scale in [1e-100, 1e100]:
            np.testing.assert_allclose(ip_row(np.eye(2), v*scale, 0)*np.sqrt(scale), w)
        self.assertAlmostEqual(np.vdot(w, v@w).real, 1)
        self.assertAlmostEqual((w.conj()@np.array([1, 1j])).real, 3/np.sqrt(6))

    def test_ip_invalid_and_single_channel(self):
        np.testing.assert_allclose(ip_row([[1]], [[4]], 0), [.5])
        for w, v, n in [(np.eye(2), np.ones((2, 2)), 0),
                        (np.zeros((2, 2)), np.eye(2), 0),
                        (np.eye(2), [[1, 1], [0, 1]], 0),
                        (np.eye(2), np.eye(2), True),
                        (np.eye(2), np.eye(2), 2),
                        (np.eye(2), [[np.nan, 0], [0, 1]], 0)]:
            with self.assertRaises(ValueError):
                ip_row(w, v, n)

    def test_nmf_multiplication_and_scale_ambiguity(self):
        result = run_exercises()['E08-09']
        expected = [[float(F(101, 50)), float(F(11, 10))], [float(F(11, 10)), float(F(13, 4))]]
        np.testing.assert_allclose(result['power'], expected)
        np.testing.assert_allclose(result['rescaled_power'], expected)
        rank_one = result['one_basis_power']
        self.assertEqual(rank_one, [[2, .5], [1, .25]])
        self.assertAlmostEqual(np.linalg.det(rank_one), 0)

    def test_mvdr_constraint_outputs_and_gev(self):
        result = run_exercises()['E08-10']
        np.testing.assert_allclose(result['weights'], [.5, .5])
        np.testing.assert_allclose(result['snapshot_outputs'], [1, 0], atol=1e-15)
        np.testing.assert_allclose(result['gev_eigenvalues'], [1/9, 9])
        self.assertAlmostEqual(result['target_output_power'], .9)
        self.assertAlmostEqual(result['interference_output_power'], .1)


if __name__ == '__main__':
    unittest.main()
