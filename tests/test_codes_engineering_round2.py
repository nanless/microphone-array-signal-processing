"""Independent arithmetic and failure fixtures for the second engineering audit."""
import math
import unittest
import warnings

import numpy as np

from codes.array_tutorial.engineering import (
    HysteresisVAD, PeakProtectAGC, RingBuffer, estimate_sro_ppm,
    q15_quantize, resample_sro_to_reference, simulate_deadline_queue,
)
from codes.examples.exercises_engineering import block_consumption, run_exercises


class EngineeringInputTests(unittest.TestCase):
    def test_real_interfaces_reject_complex_before_casting(self):
        values = np.array([.25 + 4j, .5 - 1j])
        calls = (
            lambda: q15_quantize(values),
            lambda: HysteresisVAD(.2, .1, 0).update(values),
            lambda: PeakProtectAGC().process(values),
            lambda: RingBuffer(4).write(values),
            lambda: estimate_sro_ppm(values, np.array([0., .1])),
            lambda: estimate_sro_ppm(np.array([0., 1.]), values),
            lambda: resample_sro_to_reference(values, 100.),
            lambda: simulate_deadline_queue(values, 10., 2),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for call in calls:
                with self.subTest(call=call), self.assertRaises(ValueError):
                    call()

    def test_real_arrays_still_work_and_non_numeric_arrays_do_not(self):
        np.testing.assert_array_equal(q15_quantize([0, 1]), [0, 32767])
        for values in ([".25", ".5"], [True, False], np.array([.25], dtype=object)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                q15_quantize(values)

    def test_sro_scaled_fit_large_finite_time_axis(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ppm, intercept = estimate_sro_ppm(
                np.array([0., 1e200, 2e200]), np.array([.002, .003, .004]))
        self.assertTrue(math.isclose(ppm, 1e-197, rel_tol=1e-12, abs_tol=0.))
        self.assertAlmostEqual(intercept, .002)

    def test_sro_mean_overflow_and_zero_delays(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ppm, offset = estimate_sro_ppm(np.array([1e308, 1.2e308, 1.4e308]),
                                           np.array([1e307, 1.2e307, 1.4e307]))
            self.assertTrue(math.isclose(ppm, 1e5, rel_tol=1e-12))
            self.assertLess(abs(offset), 1e294)
            self.assertEqual(estimate_sro_ppm(np.array([0., 1.]), np.zeros(2)), (0., 0.))

    def test_sro_large_time_origin_preserves_representable_increments(self):
        for origin, step in ((1e16, 2.), (1.7e9, .001), (1e308, 2e292)):
            times = origin + np.arange(4) * step
            # The target is linear in the actual stored timestamps, not in
            # ideal increments that float64 might be unable to represent.
            delays = (times - times[0]) * 1e-4
            for order in (np.arange(4), np.array([3, 1, 0, 2])):
                with self.subTest(origin=origin, order=order), np.errstate(
                        over="raise", invalid="raise", divide="raise"):
                    ppm, offset = estimate_sro_ppm(times[order], delays[order])
                self.assertTrue(math.isclose(ppm, 100., rel_tol=1e-12))
                self.assertTrue(math.isclose(offset, -origin * 1e-4, rel_tol=1e-12))

    def test_sro_large_delay_origin_preserves_representable_increments(self):
        times = np.arange(4) * 2
        delays = 1e16 + np.arange(4) * 2
        ppm, offset = estimate_sro_ppm(times, delays)
        self.assertAlmostEqual(ppm, 1e6)
        self.assertEqual(offset, 1e16)

    def test_sro_opposite_extremes_and_large_constant_delay(self):
        times = np.array([-1e308, 0., 1e308])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ppm, offset = estimate_sro_ppm(times, times * 1e-4)
            self.assertTrue(math.isclose(ppm, 100., rel_tol=1e-12))
            self.assertEqual(offset, 0.)
            self.assertEqual(estimate_sro_ppm(times, np.full(3, 1.7e308)),
                             (0., 1.7e308))
            ppm, offset = estimate_sro_ppm(times, times)
            self.assertAlmostEqual(ppm, 1e6)
            self.assertEqual(offset, 0.)

    def test_sro_subnormal_range(self):
        times = np.array([0., 1e-320, 2e-320])
        ppm, offset = estimate_sro_ppm(times, times)
        self.assertAlmostEqual(ppm, 1e6)
        self.assertEqual(offset, 0.)

    def test_sro_rejects_unrepresentable_or_constant_fit(self):
        for times, delays in (([0., 1e-308], [0., 1e308]), ([2., 2.], [1., 2.]),
                              ([0., 0.], [1., 2.])):
            with self.subTest(times=times), self.assertRaises(ValueError):
                estimate_sro_ppm(np.array(times), np.array(delays))


class EngineeringRound2Exercises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_exercises()

    def test_pcm_units(self):
        self.assertEqual(self.results["E10-07"], {
            "scalar_samples": 640, "bytes_per_callback": 1280,
            "duration_ms": 10., "bytes_per_second": 128000,
        })

    def test_block_consumption_has_no_gaps_or_duplicates(self):
        result = self.results["E10-08"]
        self.assertEqual(result["common_multiple_samples"], 7680)
        for size, calls, first in ((240, 32, 2), (512, 15, 4)):
            consumer = result["consumers"][str(size)]
            self.assertEqual(len(consumer["events"]), calls)
            self.assertEqual(consumer["events"][0]["callback"], first)
            self.assertEqual(consumer["consumed"], 7680)
            self.assertEqual(consumer["leftovers"][-1], 0)
            # Independent expectation: event k consumes [k*size, (k+1)*size)
            # at the first callback with at least (k+1)*size input samples.
            for k, event in enumerate(consumer["events"]):
                self.assertEqual(event["start"], k * size)
                self.assertEqual(event["stop"], (k + 1) * size)
                self.assertEqual(event["callback"], ((k + 1) * size + 159) // 160)
            self.assertTrue(all(0 <= leftover < size for leftover in consumer["leftovers"]))

    def test_adapter_handles_multiple_calls_and_rejects_bad_counts(self):
        result = block_consumption(5, 2, 1)
        self.assertEqual(result["events"], [{"callback": 1, "start": 0, "stop": 2},
                                            {"callback": 1, "start": 2, "stop": 4}])
        self.assertEqual(result["leftovers"], [1])
        for args in ((0, 2, 1), (2, -1, 1), (2, 3, True), (2, 3., 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                block_consumption(*args)

    def test_polarity_changes_target_not_just_noise(self):
        result = self.results["E10-09"]
        self.assertAlmostEqual(result["correct_target_gain"], .95)
        self.assertAlmostEqual(result["reversed_target_gain"], .05)
        self.assertAlmostEqual(result["relative_target_level_db"], -25.57507201905658)

    def test_smoothing_is_tied_to_elapsed_time(self):
        updates = self.results["E10-10"]["updates"]
        self.assertAlmostEqual(updates["10"]["coefficient"], .09516258196404043)
        self.assertAlmostEqual(updates["32"]["coefficient"], .27385096292630906)
        for result in updates.values():
            self.assertAlmostEqual(result["remaining_error_fraction"], math.exp(-3.2))

    def test_displacement_projection(self):
        np.testing.assert_allclose(self.results["E10-11"]["projected_displacements_m"], [.0005, 0.])
        np.testing.assert_allclose(self.results["E10-11"]["phase_errors_deg"], [720 / 343, 0.])

    def test_known_clock_step_is_not_rate(self):
        result = self.results["E10-12"]
        self.assertAlmostEqual(result["whole_fit_ppm"], 2500 / 7)
        self.assertAlmostEqual(result["whole_fit_offset_ms"], -1 / 7)
        self.assertAlmostEqual(result["known_step_removed_ppm"], 100.)
        self.assertAlmostEqual(result["known_step_removed_offset_ms"], 0.)

    def test_wer_aggregation_and_quantile_convention(self):
        result = self.results["E11-03"]
        self.assertAlmostEqual(result["macro_average_wer"], .30)
        self.assertAlmostEqual(result["pooled_wer"], .46)
        result = self.results["E11-04"]
        self.assertEqual(result["nearest_ranks"], [10, 19, 20])
        self.assertEqual(result["nearest_rank_ms"], [10., 19., 100.])
        np.testing.assert_allclose(result["linear_interpolation_ms"], [10.5, 23.05, 84.61])


if __name__ == "__main__":
    unittest.main()
