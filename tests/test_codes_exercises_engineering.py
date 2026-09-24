"""Independent hand-calculated expectations for the engineering exercises."""
import json
import unittest

import numpy as np

from codes.examples.exercises_engineering import run_exercises


class EngineeringExerciseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_exercises()

    def test_ids_and_strict_json(self):
        expected = {f"E10-{i:02}" for i in range(1, 13)} | {"E10-14"}
        expected |= {f"E11-{i:02}" for i in range(1, 8)}
        expected |= {"E12-01", "E12-02", "E12-03", "E12-04", "E13-01"}
        self.assertEqual(set(self.results), expected)
        json.dumps(self.results, allow_nan=False)

    def test_sro_and_vad(self):
        self.assertAlmostEqual(self.results["E10-01"]["first_order_ppm"], 100)
        self.assertAlmostEqual(self.results["E10-01"]["offset_ms"], 2)
        self.assertAlmostEqual(self.results["E10-01"]["drift_600s_ms"], 60)
        self.assertEqual(self.results["E10-02"]["active"], [False, True, True, True, True, False])

    def test_queue_and_buffer(self):
        self.assertEqual(self.results["E10-03"], {"retained": [2., 3., 4., 5.], "dropped": 1})
        result = self.results["E10-04"]
        # Accepted jobs finish at 2, 28, 46, 48, 52 ms; arrival at 40 ms is lost.
        self.assertEqual((result["processed"], result["dropped"], result["deadline_misses"]), (5, 1, 3))
        self.assertEqual((result["queue_high_water"], result["max_wait_ms"]), (2, 16))
        self.assertAlmostEqual(result["offered_work_rtf"], 11 / 15)

    def test_quantizer_and_agc(self):
        self.assertEqual(self.results["E10-05"]["q15"], [-32768, 0, 0, 2, 32767])
        np.testing.assert_allclose(self.results["E10-06"]["gains"], [2.5, .5])
        np.testing.assert_allclose(self.results["E10-06"]["output_peaks"], [.5, 1.])

    def test_selection_constraints_and_energy(self):
        result = self.results["E11-01"]
        self.assertEqual(result["alias_boundary_hz"], 4287.5)
        self.assertAlmostEqual(result["drift_300s_ms"], 24)
        self.assertEqual(result["maximum_alignment_interval_s"], 1.25)
        self.assertEqual((result["base_latency_ms"], result["with_separator_ms"]), (121, 187))
        result = self.results["E11-02"]
        self.assertEqual(result["coverage"], .8)
        self.assertAlmostEqual(result["average_power_w"], 1.1)
        self.assertAlmostEqual(result["estimated_runtime_h"], 5.38181818181818)

    def test_coupled_resource_budget_has_independent_timeline(self):
        row = self.results["E10-14"]
        self.assertEqual([r["start_ms"] for r in row["timeline"]], [0, 10, 20, 45, 49, 53])
        self.assertEqual([r["finish_ms"] for r in row["timeline"]], [4, 14, 45, 49, 53, 57])
        self.assertEqual([r["response_ms"] for r in row["timeline"]], [4, 4, 25, 19, 13, 7])
        self.assertEqual((row["processed"], row["dropped"], row["deadline_misses"]), (6, 0, 3))
        self.assertEqual((row["queue_high_water"], row["max_wait_ms"]), (3, 15))
        self.assertEqual(row["offered_work_rtf"], .75)
        self.assertEqual(row["frame_kib"], 2.5)
        self.assertEqual(row["peak_memory_kib"], 667.5)
        self.assertEqual(row["average_power_w_if_schedule_repeats"], .95)
        self.assertAlmostEqual(row["estimated_runtime_h_if_schedule_repeats"], 5.92 / .95)

    def test_three_selection_cases_apply_hard_limits_before_scores(self):
        terminal = self.results["E11-05"]["candidates"]
        self.assertEqual([r["passes_adjacent_4khz_screen"] for r in terminal], [False, True])
        self.assertAlmostEqual(terminal[0]["nearest_neighbor_spacing_m"], .04 * 2**.5)
        self.assertAlmostEqual(terminal[1]["nearest_neighbor_spacing_m"], .04)
        meeting = self.results["E11-06"]["candidates"]
        self.assertEqual([r["wer"] for r in meeting], [.25, .20, None])
        self.assertEqual([r["errors_per_100_reference_words"] for r in meeting],
                         [25, 20, None])
        self.assertEqual([r["score_protocol"] for r in meeting[:2]],
                         ["same_reference_single_output_wer"] * 2)
        self.assertEqual(meeting[2]["score_protocol"],
                         "undefined_until_multistream_protocol_is_fixed")
        self.assertEqual([r["passes_hard_limits"] for r in meeting], [True, True, False])
        vehicle = self.results["E11-07"]
        self.assertEqual([r["drift_ms"] for r in vehicle["candidates"]], [.8, .08])
        self.assertEqual([r["passes_drift_limit"] for r in vehicle["candidates"]], [False, True])
        self.assertEqual(vehicle["maximum_interval_s"], 1.25)

    def test_linear_and_circular_convolution(self):
        result = self.results["E12-01"]
        np.testing.assert_allclose(result["linear"], [1., 2.5, 1.])
        np.testing.assert_allclose(result["fft_length_2"], [2., 2.5])
        np.testing.assert_allclose(result["fft_length_3"], [1., 2.5, 1.])

    def test_complex_covariance_and_rank_deficiency(self):
        result = self.results["E12-02"]
        np.testing.assert_array_equal(result["second_moment_real"], [[1, 0], [0, 1]])
        np.testing.assert_array_equal(result["centered_covariance_real"], [[0, 0], [0, 1]])
        np.testing.assert_array_equal(result["second_moment_imag"], np.zeros((2, 2)))
        np.testing.assert_array_equal(result["centered_covariance_imag"], np.zeros((2, 2)))
        result = self.results["E12-03"]
        np.testing.assert_allclose(result["solution"], [1., 1.])
        self.assertEqual(result["rank"], 1)
        self.assertLess(result["residual_norm"], 1e-12)

    def test_common_listening_gain(self):
        result = self.results["E13-01"]
        self.assertEqual(result["common_gain"], 2.)
        np.testing.assert_allclose(result["common_peaks"], [.8, .4])
        self.assertAlmostEqual(result["amplitude_difference_db"], 6.020599913279624)
        self.assertEqual(result["separately_normalized_difference"], 0.)

    def test_spatial_whitening_is_not_scalar_phat(self):
        row = self.results["E12-04"]
        np.testing.assert_allclose(row["noise_covariance"], [[4., 0.], [0., 1.]])
        np.testing.assert_allclose(row["whitening_matrix"], [[.5, 0.], [0., 1.]])
        np.testing.assert_allclose(row["whitened_covariance"], np.eye(2))
        np.testing.assert_allclose(row["whitened_steering"], [.5, 1.])
        np.testing.assert_allclose(row["raw_eigenvalues"], [1., 4.])
        np.testing.assert_allclose(row["whitened_eigenvalues"], [1., 1.])
        self.assertEqual(row["one_nonzero_cross_spectrum_phat"], 1.)
        self.assertEqual(row["theoretical_noise_cross_spectrum"], 0.)


if __name__ == "__main__":
    unittest.main()
