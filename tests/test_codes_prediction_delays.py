"""Independent source-index, rational regression and dB controls for E07-22/23."""
import json
import math
import unittest
import numpy as np
from codes.chapters.ch07.core.prediction_delays import (
    source_gap, prediction_delays, complete_history_interval, shift_observation,
    fit_single_real_history, source_clock_example, late_power_decay,
)


class PredictionDelayTeaching(unittest.TestCase):
    def test_source_indices_by_enumerated_clock_not_matching_helper(self):
        t, ref, arrivals, delays = 12, 3, [3, 0, 5], [3, 6, 1]
        target_source = t-ref
        historical_sources = [t-lag-d for lag, d in zip(delays, arrivals)]
        self.assertEqual(historical_sources, [6, 6, 6])
        self.assertEqual(prediction_delays(arrivals, 0, 3), (3, 6, 1))
        for d, lag, index in zip(arrivals, delays, historical_sources):
            self.assertEqual(source_gap(lag, d, ref), target_source-index)
        self.assertEqual(source_gap(3, 0, 3), 0)

    def test_reference_change_and_origin_shift_preserve_equations(self):
        self.assertEqual(prediction_delays([3, 0, 5], 1, 3), (0, 3, -2))
        self.assertEqual(prediction_delays([103, 100, 105], 0, 3), (3, 6, 1))
        # The negative delay is an honest algebraic result, not accepted history.
        with self.assertRaises(ValueError):
            complete_history_interval(20, [0, 3, -2], 2)

    def test_complete_history_by_explicit_index_sets(self):
        valid = [t for t in range(10)
                 if all(t-delay-k >= 0 for delay in [3, 6, 1] for k in range(2))]
        self.assertEqual(valid, [7, 8, 9])
        self.assertEqual(complete_history_interval(10, [3, 6, 1], 2), (7, 10))
        self.assertEqual(complete_history_interval(4, [6], 1), (4, 4))
        self.assertEqual(complete_history_interval(3, [0], 1), (0, 3))

    def test_known_source_leakage_against_exact_expected_vectors(self):
        r = source_clock_example()
        self.assertEqual(r['target'], [0, 1, 0, 0])
        self.assertEqual(r['cases']['common']['history'], [0, 1, 0, 0])
        self.assertEqual(r['cases']['common']['coefficient'], 1)
        self.assertEqual(r['cases']['common']['residual'], [0, 0, 0, 0])
        self.assertEqual(r['cases']['aligned']['history'], [1, 0, 0, 0])
        self.assertEqual(r['cases']['aligned']['coefficient'], 0)
        self.assertEqual(r['cases']['aligned']['residual'], [0, 1, 0, 0])
        json.dumps(r, allow_nan=False)

    def test_delay_preserves_input_and_declares_truncation(self):
        source = np.array([1., 2., 3., 0., 0.])
        np.testing.assert_array_equal(shift_observation(source, 2), [0, 0, 1, 2, 3])
        np.testing.assert_array_equal(shift_observation(source, 5), np.zeros(5))
        np.testing.assert_array_equal(shift_observation(source, 0), source)
        np.testing.assert_array_equal(source, [1, 2, 3, 0, 0])

    def test_fit_uses_declared_window_and_applies_same_fixed_coefficient(self):
        # Only [1,3) is fitted: R=1+4=5, r=2+6=8, g=8/5.
        q = np.array([10., 1., 2., 20.])
        y = np.array([-9., 2., 3., -8.])
        fit = fit_single_real_history(y, q, (1, 3))
        self.assertEqual(fit['history_energy'], 5)
        self.assertEqual(fit['cross'], 8)
        self.assertEqual(fit['coefficient'], 8/5)
        np.testing.assert_allclose(fit['residual'], [-25., .4, -.2, -40.], atol=5e-16)
        np.testing.assert_array_equal(q, [10, 1, 2, 20])

    def test_same_target_predictability_is_not_fixed_by_clock_alignment(self):
        # A constant source gives identical old and current observations even
        # with a correctly declared positive source gap; target is still erased.
        fit = fit_single_real_history(np.ones(4), np.ones(4), (0, 4))
        self.assertEqual(fit['coefficient'], 1)
        np.testing.assert_array_equal(fit['residual'], np.zeros(4))

    def test_power_and_amplitude_have_distinct_sixty_db_ratios(self):
        r = late_power_decay()
        self.assertAlmostEqual(r['level_db_per_hop'], -.8)
        rows = {row['frames']: row for row in r['rows']}
        for frame, power, amplitude, db in [(10, 10**(-.8), 10**(-.4), -8),
                                            (25, .01, .1, -20),
                                            (75, 1e-6, .001, -60)]:
            self.assertAlmostEqual(rows[frame]['power_ratio'], power, places=14)
            self.assertAlmostEqual(rows[frame]['amplitude_ratio'], amplitude, places=14)
            self.assertAlmostEqual(rows[frame]['level_db'], db, places=13)
        self.assertAlmostEqual(20*math.log10(r['amplitude_ratio_per_hop']), -.8)
        self.assertAlmostEqual(10*math.log10(r['amplitude_ratio_per_hop']**75), -30)
        json.dumps(r, allow_nan=False)

    def test_same_elapsed_time_and_zero_offset(self):
        a = late_power_decay(.6, .008, [0, 75])
        b = late_power_decay(.6, .004, [150])
        self.assertEqual(a['rows'][0]['power_ratio'], 1)
        self.assertAlmostEqual(a['rows'][1]['power_ratio'], b['rows'][0]['power_ratio'])

    def test_integer_and_finite_input_boundaries(self):
        for bad in [True, 1.5, 1j, np.nan]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): shift_observation([1.], bad)
        for target, history, window in [([1j], [1], (0, 1)), ([1], [1j], (0, 1)),
                                         ([1], [0], (0, 1)), ([1], [1], (1, 1)),
                                         ([1], [1], (0, 2)), ([1], [1], (False, 1)),
                                         ([1, 2], [1], (0, 1)), ([np.inf], [1], (0, 1)),
                                         ([1], [1e308], (0, 1))]:
            with self.subTest(target=target, history=history, window=window):
                with self.assertRaises(ValueError): fit_single_real_history(target, history, window)
        for bad in [True, 0, -1, np.inf, 1j]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): late_power_decay(bad)
        with self.assertRaises(ValueError): late_power_decay(frame_offsets=[True])
        with self.assertRaises(ValueError): late_power_decay(1., 1e308, [])
        with self.assertRaises(ValueError): prediction_delays([], 0, 3)


if __name__ == '__main__':
    unittest.main()
