"""Analytic edge cases for the Chapter 10 numeric contracts."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from codes.chapters.ch10.core.engineering import resample_sro_to_reference, q15_dot, validate_telemetry
from codes.chapters.ch10.core.noise_suppression import power_spectral_subtraction
from codes.chapters.ch10.sro_closed_loop_demo import StatefulLinearClockCorrector


class SpectralRangeTests(unittest.TestCase):
    def test_tiny_complex_gain_and_phase(self):
        y = 1e-200 * np.array([[2j, 3j, 1j]])
        out, power = power_spectral_subtraction(y, np.array([0]))
        np.testing.assert_allclose(out / 1e-200, [[.4j, np.sqrt(5)*1j, .2j]], rtol=3e-13)
        self.assertEqual(power[0], 0.)  # physical 4e-400 is unrepresentable

    def test_subnormal_powers_keep_floor_amplitude(self):
        for scale in (1., 1e-160, 2e-162):
            out, _ = power_spectral_subtraction(np.array([[scale+0j]]), np.array([0]))
            self.assertAlmostEqual(out[0, 0].real / scale, .2, places=14)

    def test_zero_subtraction_identity_even_tiny(self):
        y = np.array([[1e-310 + 2e-310j, 0j, -1e-200j]])
        out, _ = power_spectral_subtraction(y, np.array([0]), oversubtraction=0)
        np.testing.assert_array_equal(out, y)

    def test_finite_large_mean_does_not_overflow_sum(self):
        out, power = power_spectral_subtraction(np.array([[1e154, 1e154]]), np.array([0, 1]))
        self.assertAlmostEqual(power[0] / 1e308, 1., places=12)
        np.testing.assert_allclose(out / 1e153, [[2., 2.]], rtol=1e-12)

    def test_dynamic_range_preserves_tiny_floor(self):
        out, _ = power_spectral_subtraction(np.array([[1e150, 1e-200j]]), np.array([0]))
        self.assertAlmostEqual(out[0, 1].imag / 1e-200, .2)

    def test_truly_unrepresentable_mean_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mean noise power'):
            power_spectral_subtraction(np.array([[1e308]]), np.array([0]))


class EngineeringRangeTests(unittest.TestCase):
    def test_convex_interpolation_opposite_extremes(self):
        out = resample_sro_to_reference(np.array([-1e308, 1e308]), -500000)
        np.testing.assert_array_equal(out, [-1e308, 0, 1e308])

    def test_budget_checked_before_allocation(self):
        with patch('codes.chapters.ch10.core.engineering.np.arange', side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError, 'max_output_samples'):
                resample_sro_to_reference(np.array([0., 1.]), np.nextafter(-1e6, 0))
        with self.assertRaises(ValueError):
            resample_sro_to_reference([0., 1.], 0, max_output_samples=True)
        np.testing.assert_array_equal(resample_sro_to_reference([0., 1.], 0, max_output_samples=2), [0, 1])

    def test_q15_huge_broadcast_rejected_without_traversal(self):
        value = np.broadcast_to(np.int16(-32768), (2**33,))
        with self.assertRaisesRegex(ValueError, 'accumulator bound'):
            q15_dot(value, value)

    def test_telemetry_huge_numeric_and_nonmapping_report_errors(self):
        for field in ('rtf', 'sro_ppm', 'clipping_fraction', 'agc_gain'):
            self.assertIn(f'{field} must be finite', validate_telemetry({field: 10**400}))
        self.assertEqual(validate_telemetry(None), ['record must be a mapping'])
        self.assertIn('wrong type for rtf', validate_telemetry({'rtf': np.bool_(True)}))

    def test_clock_integer_configuration(self):
        for length in (True, 2.5, 0):
            with self.assertRaises(ValueError):
                StatefulLinearClockCorrector(2, 3, 0, length)
        for rate in (True, 1+0j, 10**400):
            with self.assertRaises(ValueError):
                StatefulLinearClockCorrector(rate, 3, 0, 4)

    def test_clock_failed_push_atomic(self):
        state = StatefulLinearClockCorrector(2, 3, 0, 4)
        state.push([0, 1], [0, 1/3])
        saved = copy.deepcopy(vars(state))
        for indices, samples in [([2, 3], [1., np.nan]), ([1, 2], [1., 2.]), ([2, 3.5], [1., 2.])]:
            with self.assertRaises(ValueError):
                state.push(indices, samples)
            self.assertEqual(vars(state), saved)

    def test_clock_arithmetic_failure_atomic(self):
        state = StatefulLinearClockCorrector(1e-308, 1e308, 0, 4)
        saved = copy.deepcopy(vars(state))
        with self.assertRaises(ValueError):
            state.push([0, 1], [0., 1.])
        self.assertEqual(vars(state), saved)

    def test_clock_waits_for_actual_right_endpoint(self):
        state = StatefulLinearClockCorrector(1., 1. + 5e-11, 0., 2)
        self.assertEqual(state.push([0, 1], [0., 1.]), [(0, 0.)])
        self.assertEqual(state.push([2], [2.]), [(1, 1. + 5e-11)])

    def test_clock_convex_large_values(self):
        state = StatefulLinearClockCorrector(2, 1, 0, 3)
        self.assertEqual(state.push([0, 1], [-1e308, 1e308]), [(0, -1e308), (1, 0.), (2, 1e308)])
