"""Independent fraction and analytic expectations for E06-11..E06-18."""
import json
import math
import unittest

from codes.examples.aec_advanced_exercises import run_exercises


class AdvancedAECExercisesTest(unittest.TestCase):
    def test_stable_ids_and_json(self):
        result = run_exercises()
        self.assertEqual(list(result), [f"E06-{n:02d}" for n in range(11, 19)])
        json.dumps(result, allow_nan=False)

    def test_haar_and_diagonal_model_are_distinct(self):
        result = run_exercises()
        self.assertEqual(result["E06-11"]["input_high_band"], [0, 0])
        self.assertAlmostEqual(result["E06-11"]["echo_high_band"][0], -math.sqrt(.5))
        self.assertAlmostEqual(result["E06-11"]["echo_high_band"][1], math.sqrt(.5))
        self.assertEqual(result["E06-12"]["missing_cross_terms"], [0, .5, 0, -.5])

    def test_ipnlms_floor_and_zero_start(self):
        row = run_exercises()["E06-13"]
        self.assertEqual(row["gain"], [.45, .3])
        self.assertEqual(row["denominator"], 1.)
        self.assertEqual(row["update"], [.09, .06])
        self.assertEqual(row["uniform_update"], [.08, .08])
        self.assertEqual(run_exercises()["E06-14"]["pure_proportionate_zero_start_update"], [0, 0])

    def test_rls_time_and_independent_batch_solution(self):
        result = run_exercises()
        self.assertAlmostEqual(result["E06-15"]["old_observation_weight_after_100_updates"], .3660323412732292)
        self.assertEqual(result["E06-15"]["elapsed_seconds_if_16khz_sample_updates"], .00625)
        self.assertEqual(result["E06-15"]["elapsed_seconds_if_10ms_block_updates"], 1.)
        self.assertAlmostEqual(result["E06-17"]["normal_determinant"], 19 / 16)
        for got, expected in zip(result["E06-17"]["batch_weights"], [18 / 19, 16 / 19]):
            self.assertAlmostEqual(got, expected)

    def test_kalman_known_near_end_and_cross_covariance(self):
        result = run_exercises()
        row = result["E06-16"]
        self.assertAlmostEqual(row["fixed_low_variance_gain"], 10 / 21)
        self.assertAlmostEqual(row["fixed_low_variance_weight"], 440 / 231)
        self.assertAlmostEqual(row["oracle_high_variance_gain"], 1 / 111)
        self.assertAlmostEqual(row["oracle_high_variance_weight"], 10 / 11 + 23 / 1221)
        self.assertEqual(result["E06-18"]["complete_innovation_variance"], 4.5)
        self.assertAlmostEqual(result["E06-18"]["diagonal_only_innovation_variance"], 19 / 6)


if __name__ == "__main__":
    unittest.main()
