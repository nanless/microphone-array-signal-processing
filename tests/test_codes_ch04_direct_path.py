"""Independent lag-boundary and direct-path counterexamples for chapter 4."""
import math
import unittest

import numpy as np

from codes.chapters.ch04.core.doa import gcc_phat
from codes.chapters.ch04 import chapter04_experiments as ex


def complex_values(pairs):
    values = np.asarray(pairs)
    return values[..., 0]+1j*values[..., 1]


class GCCPhysicalBoundaryTests(unittest.TestCase):
    def test_one_sample_nextafter_before_equal_and_after(self):
        # A delayed impulse has one exact positive sample of delay. The
        # represented seconds, not a dimensionless absolute epsilon, set the
        # admissible grid. The reverse call must use the same negative bound.
        for sample_rate in (16000., 1e20):
            seconds = 1/sample_rate
            bounds = (np.nextafter(seconds, 0.), seconds,
                      np.nextafter(seconds, math.inf))
            for bound, allowed in zip(bounds, ([0.], [-seconds, 0., seconds],
                                              [-seconds, 0., seconds])):
                for first, second, sign in (([0, 1], [1, 0], 1),
                                            ([1, 0], [0, 1], -1)):
                    with self.subTest(fs=sample_rate, bound=bound, sign=sign):
                        result = gcc_phat(first, second, sample_rate, max_tau=bound)
                        np.testing.assert_array_equal(result[2], allowed)
                        self.assertEqual(result[0], sign*seconds if len(allowed) == 3 else 0.)
                        self.assertLessEqual(abs(result[0]), bound)

    def test_zero_bound_and_interpolation_cannot_reintroduce_excluded_lags(self):
        for sample_rate in (16000., 1e20):
            for interpolation in (False, True):
                with self.subTest(fs=sample_rate, interpolation=interpolation):
                    result = gcc_phat([0, 1], [1, 0], sample_rate, max_tau=0.,
                                      parabolic_interpolation=interpolation)
                    self.assertEqual(result[0], 0.)
                    np.testing.assert_array_equal(result[2], [0.])
                    bound = 1/sample_rate
                    endpoint = gcc_phat([0, 1], [1, 0], sample_rate, max_tau=bound,
                                        parabolic_interpolation=interpolation)
                    self.assertEqual(endpoint[0], bound)


class DirectPathTeachingTests(unittest.TestCase):
    def test_coherent_paths_have_rank_one_but_a_finite_false_peak(self):
        report = ex.coherent_reflection_false_peak()
        expected_signal = np.array([[4, 2-2j], [2+2j, 2]])
        np.testing.assert_array_equal(complex_values(report['signal_covariance_real_imag']),
                                      expected_signal)
        self.assertEqual(report['physical_emitter_count'], 1)
        self.assertEqual(report['propagation_path_count'], 2)
        self.assertEqual(report['signal_rank'], 1)
        self.assertEqual(report['signal_coherence_magnitude'], 1.)
        np.testing.assert_allclose(report['signal_eigenvalues'], [0, 6], atol=1e-14)
        np.testing.assert_allclose(report['virtual_total_eigenvalues'], [.1, 6.1], atol=1e-14)
        self.assertAlmostEqual(report['virtual_eigenvalue_ratio'], 61., places=11)
        self.assertAlmostEqual(report['spacing_m'], .042875)
        self.assertEqual(report['frequency_hz'], 4000.)
        self.assertEqual(report['sample_rate_hz'], 16000.)
        self.assertAlmostEqual(report['relative_response_magnitude'], 1/math.sqrt(2))
        self.assertAlmostEqual(report['false_peak_angle_deg'], math.degrees(math.asin(.25)))
        # Scalar trigonometry supplies the independent denominator, without
        # using the MUSIC eigensolver or its projector as the expected value.
        for angle, denominator, score in zip(report['comparison_angles_deg'],
                report['comparison_projection_energies'], report['comparison_music_scores']):
            phase = math.pi*math.sin(math.radians(angle))
            expected = 1-2*math.sqrt(2)*math.cos(phase-math.pi/4)/3
            self.assertAlmostEqual(denominator, expected, places=13)
            self.assertAlmostEqual(score, 1/expected, places=11)
        self.assertAlmostEqual(report['minimum_projection_energy'], 1-2*math.sqrt(2)/3)
        self.assertAlmostEqual(report['maximum_music_score'], 9+6*math.sqrt(2), places=11)
        self.assertGreater(report['minimum_projection_energy'], 0.)

    def test_ctf_plain_cross_relation_complete_tail_and_manual_solution(self):
        report = ex.exact_ctf_cross_relation()
        np.testing.assert_array_equal(complex_values(report['first_output_real_imag']),
                                      [1, .5, 0, 1, .5])
        np.testing.assert_array_equal(complex_values(report['second_output_real_imag']),
                                      [1+1j, .25-.5j, 0, 1+1j, .25-.5j])
        expected_design = np.array([[.5, 1, 1+1j], [0, .5, .25-.5j], [1, 0, 0]])
        expected_coefficients = np.array([1+1j, .25-.5j, -.5])
        actual_design = complex_values(report['design_real_imag'])
        np.testing.assert_array_equal(actual_design, expected_design)
        np.testing.assert_allclose(complex_values(report['estimated_coefficients_real_imag']),
                                   expected_coefficients, atol=2e-15)
        target = np.array([.25-.5j, 0, 1+1j])
        np.testing.assert_array_equal(expected_design@expected_coefficients, target)
        # Conjugating the cross-relation coefficients changes this equation.
        self.assertGreater(np.linalg.norm(expected_design.conj()@expected_coefficients-target), 1.)
        self.assertEqual(report['design_rank'], 3)
        self.assertAlmostEqual(abs(complex_values(report['design_determinant_real_imag'])
                                   -(-.25-1j)), 0., places=14)
        self.assertLess(report['regression_residual_norm'], 3e-15)
        self.assertEqual(report['full_cross_convolution_residual_max'], 0.)

    def test_ctf_condition_number_from_scalar_characteristic_polynomial(self):
        report = ex.exact_ctf_cross_relation()
        # Z^H Z has trace 77/16, principal-minor sum 305/64 and determinant
        # 17/16. Bisection of its three scalar roots is independent of SVD.
        def polynomial(value):
            return value**3-(77/16)*value**2+(305/64)*value-17/16
        roots = []
        for lower, upper in ((.3, .4), (.9, 1.), (3.5, 3.6)):
            for _ in range(100):
                midpoint = (lower+upper)/2
                if polynomial(lower)*polynomial(midpoint) <= 0:
                    upper = midpoint
                else:
                    lower = midpoint
            roots.append((lower+upper)/2)
        np.testing.assert_allclose(np.array(report['design_singular_values'])**2,
                                   roots[::-1], atol=2e-14)
        self.assertAlmostEqual(report['design_condition_2'], math.sqrt(roots[-1]/roots[0]), places=13)

    def test_first_ctf_ratio_differs_from_whole_response_and_tone_is_rank_one(self):
        report = ex.exact_ctf_cross_relation()
        self.assertAlmostEqual(abs(complex_values(report['first_coefficient_ratio_real_imag'])
                                   -(1+1j)), 0., places=14)
        # At frame modulation pi/2, exp(-j omega)=-j, hence B/A=(.5+.75j)/(1-.5j).
        self.assertAlmostEqual(abs(complex_values(report['whole_ctf_frequency_ratio_real_imag'])
                                   -(.1+.8j)), 0., places=14)
        self.assertAlmostEqual(report['whole_ctf_frequency_ratio_phase_deg'],
                               math.degrees(math.atan2(.8, .1)))
        tone_design = complex_values(report['steady_sinusoid_design_real_imag'])
        self.assertEqual(report['steady_sinusoid_design_rank'], 1)
        np.testing.assert_allclose(tone_design[1], 1j*tone_design[0], atol=1e-15)
        np.testing.assert_allclose(tone_design[2], -tone_design[0], atol=1e-15)


if __name__ == '__main__':
    unittest.main()
