"""Independent arithmetic and boundary checks for timestamp/control exercises."""
import math
import unittest

import numpy as np

from codes.array_tutorial.tracking import ConstantVelocityKalman
from codes.examples.tracking_time_exercises import (
    predict_timestamped_direction, run_exercises, vad_preroll_segments,
    zero_failure_upper_bound,
)


class TimeStateExercisesTests(unittest.TestCase):
    def test_particle_evidence_accumulates_without_resampling(self):
        result = run_exercises()["E09-07"]
        # Two independent observations: likelihood ratio e^.5 becomes e^1.
        np.testing.assert_allclose(result["first_weights"], [1 / (1 + math.exp(-.5)), 1 / (1 + math.exp(.5))])
        np.testing.assert_allclose(result["second_weights"], [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))])
        self.assertGreater(result["second_weights"][0], result["first_weights"][0])

    def test_timestamp_covariance_includes_cross_terms_and_process_noise(self):
        state, covariance = predict_timestamped_direction([30, 30], [[4, 1], [1, 9]], 1, 1.08)
        np.testing.assert_allclose(state, [32.4, 30])
        np.testing.assert_allclose(covariance, [[4.217941333333333, 1.7264], [1.7264, 9.16]])
        original = np.array([[4, 1], [1, 9]])
        _, unchanged = predict_timestamped_direction([30, 30], original, 1, 1)
        np.testing.assert_array_equal(unchanged, original)
        for consume in (.99, 1.2, float("nan")):
            with self.assertRaises(ValueError):
                predict_timestamped_direction([30, 30], original, 1, consume)

    def test_timestamp_lifetime_boundary_is_stable_under_clock_translation(self):
        for origin in (0.0, 1.0, 1000.0, 1_000_000.0):
            state, covariance = predict_timestamped_direction(
                [30, 30], [[4, 1], [1, 9]], origin, origin + .1)
            np.testing.assert_allclose(state, [33, 30], atol=1e-8, rtol=0)
            np.testing.assert_allclose(covariance, [[4.290666666666667, 1.91], [1.91, 9.2]], atol=1e-8, rtol=0)
            with self.assertRaises(ValueError):
                predict_timestamped_direction([30, 30], np.eye(2), origin, origin + .1001)

    def test_real_inputs_reject_complex_bool_text_before_conversion(self):
        for bad in (1 + 0j, True, "1", np.array([1 + 2j]), np.array([1], dtype=object)):
            for index in range(4):
                inputs = [1.0, 1.08, .1, 2.0]
                inputs[index] = bad
                with self.assertRaises(ValueError):
                    predict_timestamped_direction([30, 30], np.eye(2), *inputs)
            with self.assertRaises(ValueError):
                vad_preroll_segments(bad)
            with self.assertRaises(ValueError):
                zero_failure_upper_bound(20, bad)

    def test_preroll_has_no_trigger_duplication_and_correct_availability(self):
        result = vad_preroll_segments([0, .01, .04, .09, .04, 0, 0, 0, 0, 0])
        self.assertEqual(result["active_flags"], [False]*3 + [True]*4 + [False]*3)
        segment = result["segments"][0]
        self.assertEqual(segment["frame_indices"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(segment["sample_interval"], [160, 1120])
        self.assertEqual(segment["sample_count"], 960)
        self.assertEqual(segment["trigger_available_ms"], 40)
        self.assertEqual(segment["end_event_available_ms"], 80)

    def test_preroll_stream_boundaries(self):
        self.assertEqual(vad_preroll_segments([0]*5)["segments"], [])
        first = vad_preroll_segments([.09])["segments"][0]
        self.assertEqual(first["sample_interval"], [0, 160])
        self.assertIsNone(first["end_event_available_ms"])
        result = vad_preroll_segments([.09, 0, 0, 0, 0, .09, 0, 0, 0])
        self.assertEqual([s["frame_indices"] for s in result["segments"]], [[0, 1, 2], [3, 4, 5, 6, 7]])
        with self.assertRaises(ValueError):
            vad_preroll_segments([-.01])

    def test_zero_failure_bound_inverts_binomial_probability(self):
        for n in (20, 100, 1000):
            upper = zero_failure_upper_bound(n)
            self.assertAlmostEqual((1-upper)**n, .05, places=12)
        self.assertGreater(zero_failure_upper_bound(298), .01)
        self.assertLessEqual(zero_failure_upper_bound(299), .01)
        self.assertGreater(zero_failure_upper_bound(100, .99), zero_failure_upper_bound(100, .95))
        for invalid in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                zero_failure_upper_bound(invalid)
        for confidence in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                zero_failure_upper_bound(20, confidence)

    def test_covariance_validation_is_invariant_to_scale(self):
        for scale in (1e-200, 1, 1e200):
            for bad in (np.diag([-.1, 1]), np.array([[1, .2], [.20001, 1]])):
                for invalid_field in (0, 1):
                    matrices = [np.eye(2)*scale, np.eye(2)*scale]
                    matrices[invalid_field] = bad*scale
                    with self.assertRaises(ValueError):
                        ConstantVelocityKalman([0, 0], *matrices)
            valid = np.ones((2, 2))*scale
            tracker = ConstantVelocityKalman([0, 0], valid, np.zeros((2, 2)))
            np.testing.assert_array_equal(tracker.covariance, valid)
        ConstantVelocityKalman([0, 0], np.zeros((2, 2)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
