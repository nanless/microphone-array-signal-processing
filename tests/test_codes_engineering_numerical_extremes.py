"""Independent exact arithmetic for Chapter 10's exceptional float64 paths."""
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
import numpy as np

from codes.chapters.ch10.core.engineering import (
    PeakProtectAGC, estimate_sro_ppm, safe_convex_combination,
    resample_sro_to_reference, validate_telemetry,
)
from codes.chapters.ch10.core.noise_suppression import power_spectral_subtraction
from codes.chapters.ch10.sro_closed_loop_demo import StatefulLinearClockCorrector, fit_delay_ppm


class NumericalExtremesTests(unittest.TestCase):
    def test_complete_convex_sum_not_separately_rounded_products(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        expected = float(Fraction(smallest)/2 + Fraction(smallest)/2)
        self.assertEqual(float(safe_convex_combination(smallest, smallest, .5)), expected)
        np.testing.assert_array_equal(resample_sro_to_reference([smallest, smallest], -500000),
                                      [smallest]*3)
        state = StatefulLinearClockCorrector(2, 1, 0, 3)
        self.assertEqual(state.push([0, 1], [smallest, smallest]), list(enumerate([expected]*3)))
        self.assertEqual(float(safe_convex_combination(1e308, -1e308, .5)), 0.)
        with self.assertRaisesRegex(ValueError, 'support'):
            safe_convex_combination(smallest, 0, .5)

    def test_agc_positive_gain_and_final_output_support_are_atomic(self):
        state = PeakProtectAGC(target_peak=1e-308, attack=1)
        with self.assertRaisesRegex(ValueError, 'positive AGC gain'):
            state.process([1e308])
        self.assertEqual(state.gain, 1.)
        state = PeakProtectAGC(target_peak=.1, attack=1)
        with self.assertRaisesRegex(ValueError, 'state unchanged'):
            state.process([1., float.fromhex('0x0.0000000000001p-1022')])
        self.assertEqual(state.gain, 1.)
        np.testing.assert_array_equal(state.process([0.])[0], [0.])

    def test_adjacent_tiny_powers_keep_a_representable_residual(self):
        for scale in (1e-200, 1e-300):
            later = float(np.nextafter(scale, np.inf))
            difference = Fraction(later)**2-Fraction(scale)**2
            with localcontext() as ctx:
                ctx.prec = 100
                expected = float((Decimal(difference.numerator)/Decimal(difference.denominator)).sqrt())
            actual, power = power_spectral_subtraction(np.array([[scale*1j, later*1j]]),
                                                       np.array([0]), floor_ratio=0)
            self.assertEqual(actual[0, 0], 0j)
            self.assertEqual(actual[0, 1], expected*1j)
            self.assertEqual(power[0], 0.)  # physical power itself is below support
        exact_zero, _ = power_spectral_subtraction(np.array([[1e-300, 1e-300]]),
                                                  np.array([0]), floor_ratio=0)
        np.testing.assert_array_equal(exact_zero, [[0, 0]])

    def test_unrepresentable_final_spectral_component_is_not_physical_zero(self):
        with self.assertRaisesRegex(ValueError, 'component'):
            power_spectral_subtraction(np.array([[1., complex(0, np.nextafter(0., 1.))]]),
                                       np.array([0]), floor_ratio=.04)
        with self.assertRaisesRegex(ValueError, 'component'):
            power_spectral_subtraction(np.array([[1., complex(1., np.nextafter(0., 1.))]]),
                                       np.array([0]), floor_ratio=.04)

    def test_complex_component_hidden_by_rounded_total_power_is_recovered(self):
        tiny = 1e-10
        power = Fraction(1)+Fraction(tiny)**2
        ratio = (power-1)/power
        with localcontext() as ctx:
            ctx.prec = 120
            gain = (Decimal(ratio.numerator)/Decimal(ratio.denominator)).sqrt()
            expected = complex(float(gain), float(Decimal.from_float(tiny)*gain))
        actual, noise = power_spectral_subtraction(np.array([[1, 1+tiny*1j]]),
                                                  np.array([0]), floor_ratio=0)
        self.assertEqual(actual[0, 0], 0j)
        self.assertEqual(actual[0, 1], expected)
        self.assertEqual(noise[0], 1.)
        equal, _ = power_spectral_subtraction(np.array([[1+tiny*1j, 1+tiny*1j]]),
                                             np.array([0]), floor_ratio=0)
        np.testing.assert_array_equal(equal, [[0, 0]])

    def test_unique_line_fit_and_nonzero_ppm_underflow(self):
        for scale in (1e200, 1e-200):
            self.assertEqual(fit_delay_ppm([0, scale], [0, scale]), (1e6, 0.))
        for fit in (estimate_sro_ppm, fit_delay_ppm):
            with self.assertRaisesRegex(ValueError, 'nonzero SRO'):
                fit([0, 1e308], [0, 1e-308])
            self.assertEqual(fit([0, 1], [0, 0]), (0., 0.))
            for bad in ([False, True], [0, '1'], [0, 1j]):
                with self.assertRaises(ValueError):
                    fit(bad, [0., 1.])

    def test_clock_positions_strict_start_and_atomic_rejection(self):
        state = StatefulLinearClockCorrector(1, 1, 1e-13, 2)
        output = state.push([0, 1], [0, 1])
        self.assertEqual([i for i, _ in output], [1])
        state = StatefulLinearClockCorrector(1e308, 1e-308, 0, 3)
        snapshot = state.next_output_index, state.previous
        with self.assertRaisesRegex(ValueError, 'clock position'):
            state.push([0, 1], [0., 1e308])
        self.assertEqual((state.next_output_index, state.previous), snapshot)
        for arguments in [(1, 1, 0, 10**400, 10**400-2),
                          (1e-200, 1, 1e-200, 3, 0)]:
            with self.assertRaises(ValueError):
                StatefulLinearClockCorrector(*arguments)
        state = StatefulLinearClockCorrector(1, 1, 0, 2)
        with self.assertRaises(ValueError):
            state.push([0, 10**400], [0, 1])
        self.assertEqual(state.previous, None)

    def test_unknown_drop_count_keeps_known_subset_type_and_bounds(self):
        record = dict(timestamp_ns=0, frame_index=0, sample_rate_hz=16000, queue_depth=0,
                      xrun_count=1, dropped_samples=None, clipping_fraction=0., sro_ppm=0.,
                      rtf=.1, vad_active=False, deadline_miss=False, agc_gain=1., model_version='demo')
        self.assertEqual(validate_telemetry(record), [])
        self.assertEqual(validate_telemetry(record | {'known_dropped_samples': 160}), [])
        for value in (True, -1, 1.2, '160', None):
            self.assertTrue(validate_telemetry(record | {'known_dropped_samples': value}))
        self.assertTrue(validate_telemetry(record | {'dropped_samples': 100, 'known_dropped_samples': 160}))
        self.assertFalse(validate_telemetry(record | {'dropped_samples': 160, 'known_dropped_samples': 160}))


if __name__ == '__main__':
    unittest.main()
