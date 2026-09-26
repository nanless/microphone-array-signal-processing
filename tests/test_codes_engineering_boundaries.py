"""Independent event, amplitude, timestamp and sign-test expectations."""

from itertools import product
import math
import unittest

import numpy as np

from codes.array_tutorial.engineering import PeakProtectAGC, simulate_deadline_queue
from codes.examples.engineering_boundary_exercises import (
    callback_timing_ns, paired_sign_test_lower_is_better, run_exercises,
)


class EngineeringBoundaryTests(unittest.TestCase):
    def test_equal_service_period_has_no_phantom_drops(self):
        for period in (0.1, 0.3, 1/3, 10.0, 1e-200, np.nextafter(0., 1.), 1e308):
            with self.subTest(period=period):
                result = simulate_deadline_queue(np.full(30, period), period, 1)
                self.assertEqual(result, {"frames": 30, "processed": 30, "dropped": 0,
                                          "deadline_misses": 0, "queue_high_water": 1,
                                          "max_wait_ms": 0.0})

    def test_true_one_ulp_overrun_is_not_hidden(self):
        for period in (0.1, 0.3, 10., 1e-200):
            duration = np.nextafter(period, np.inf)
            result = simulate_deadline_queue(np.full(6, duration), period, 1)
            self.assertEqual(result["processed"], 3)
            self.assertEqual(result["dropped"], 3)
            self.assertEqual(result["deadline_misses"], 3)

    def test_queue_matches_independent_integer_tick_reference(self):
        # Tick = 1/8 ms is exactly representable. Enumerate all 4-frame schedules
        # and implement a separate waiting/server recurrence in integer time.
        for costs in product((0, 1, 3), repeat=4):
            completion = 0
            pending = []
            misses = drops = high = wait = 0
            for arrival, cost in zip(range(0, 8, 2), costs):
                pending = [end for end in pending if end > arrival]
                if len(pending) == 2:
                    drops += 1
                    continue
                start = max(arrival, completion)
                completion = start + cost
                pending.append(completion)
                misses += completion > arrival + 2
                high = max(high, len(pending))
                wait = max(wait, start - arrival)
            actual = simulate_deadline_queue(np.array(costs) / 8, .25, 2)
            self.assertEqual((actual["dropped"], actual["deadline_misses"],
                              actual["queue_high_water"], actual["max_wait_ms"]),
                             (drops, misses, high, wait / 8))

    def test_unrepresentable_wait_raises_and_empty_schedule_is_valid(self):
        with self.assertRaisesRegex(ValueError, "wait"):
            simulate_deadline_queue(np.full(4, 1.7e308), 1e307, 4)
        result = simulate_deadline_queue([], .1, 1)
        self.assertEqual(result["processed"], 0)
        self.assertEqual(result["queue_high_water"], 0)

    def test_full_attack_preserves_small_representable_gain(self):
        for amplitude in (1., 1e16, 1e17, 1e200, np.finfo(float).max):
            with self.subTest(amplitude=amplitude), np.errstate(over="raise", invalid="raise"):
                output, gain = PeakProtectAGC().process(np.array([-amplitude, amplitude]))
                np.testing.assert_allclose(output, [-.8, .8], rtol=3e-15, atol=0.)
                self.assertGreater(gain, 0.)

    def test_agc_smallest_input_silence_and_partial_attack(self):
        smallest = np.nextafter(0., 1.)
        output, gain = PeakProtectAGC(release=1).process(np.array([smallest]))
        self.assertEqual(gain, 4.)
        self.assertEqual(output[0], smallest * 4)
        output, gain = PeakProtectAGC(release=1).process(np.zeros(2))
        np.testing.assert_array_equal(output, [0., 0.])
        self.assertEqual(gain, 4.)
        output, gain = PeakProtectAGC(attack=.5, gain=2).process(np.array([1.]))
        self.assertEqual(gain, 1.)  # .5*2 + .5*.8 = 1.4, safety limit is 1.

    def test_timestamp_differences_ignore_integer_origin(self):
        for origin in (0, 10**18, -10**18):
            actual = callback_timing_ns(origin, origin+15_000_000, origin+45_000_000)
            self.assertEqual(actual, {"input_age_ms": 15., "output_lead_ms": 30.,
                                      "matched_first_sample_latency_ms": 45.})
        for args in ((1, 0, 2), (0, 2, 1), (0., 1, 2), (False, 1, 2), (0j, 1, 2)):
            with self.assertRaises(ValueError):
                callback_timing_ns(*args)
        with self.assertRaises(ValueError):
            callback_timing_ns(0, 10**400, 10**401)

    def test_sign_test_tail_by_independent_enumeration(self):
        result = paired_sign_test_lower_is_better([12, 8, 20, 5, 15, 10],
                                                  [10, 9, 14, 5, 12, 9])
        outcomes = list(product((0, 1), repeat=5))
        probability = sum(sum(outcome) >= 4 for outcome in outcomes) / len(outcomes)
        self.assertEqual(result["one_sided_pvalue"], probability)
        self.assertEqual((result["pairs"], result["wins"], result["losses"], result["ties"]),
                         (6, 4, 1, 1))

    def test_sign_test_all_ties_extremes_and_invalid_inputs(self):
        self.assertIsNone(paired_sign_test_lower_is_better([1, 2], [1, 2])["one_sided_pvalue"])
        self.assertEqual(paired_sign_test_lower_is_better([1, 1], [0, 0])["one_sided_pvalue"], .25)
        self.assertEqual(paired_sign_test_lower_is_better([0, 0], [1, 1])["one_sided_pvalue"], 1.)
        with np.errstate(over="raise", invalid="raise"):
            self.assertEqual(paired_sign_test_lower_is_better([1e308], [-1e308])["wins"], 1)
        for a, b in (([], []), ([1], [1, 2]), ([1j], [1]), ([math.nan], [1]),
                     ([True], [1]), ([[1]], [[1]])):
            with self.assertRaises(ValueError):
                paired_sign_test_lower_is_better(a, b)

    def test_exercise_ids_and_hand_arithmetic(self):
        results = run_exercises()
        self.assertEqual(set(results), {"E10-16", "E10-17", "E11-09"})
        self.assertEqual(results["E10-16"]["longer_duration"]["dropped"], 15)
        self.assertEqual(results["E10-16"]["longer_duration"]["deadline_misses"], 15)
        self.assertEqual(results["E10-17"]["latency_after_one_extra_output_block_ms"], 55.)
        self.assertAlmostEqual(results["E11-09"]["baseline_pooled_wer"], 7/60)
        self.assertAlmostEqual(results["E11-09"]["candidate_pooled_wer"], 59/600)
        self.assertAlmostEqual(results["E11-09"]["reduction_percentage_points"], 11/6)


if __name__ == "__main__":
    unittest.main()
