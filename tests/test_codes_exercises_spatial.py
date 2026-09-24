"""Independent hand-calculated expectations for exercises in Chapters 1--5."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.examples.exercises_spatial import run_exercises


class SpatialExerciseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_exercises()

    def test_registry_has_twenty_eight_finite_json_results(self):
        expected = {"E01-01", "E01-02", "E02-01", "E02-02", "E02-03", "E03-01",
                    "E03-02", "E04-01", "E04-02", "E04-03", "E05-01", "E05-02",
                    "E01-03", "E02-04", "E02-05", "E03-03", "E03-04", "E04-04",
                    "E05-03", "E05-04", "E04-05", "E02-06", "E03-05", "E03-06",
                    "E04-06", "E04-07", "E04-09", "E05-05"}
        self.assertEqual(set(self.results), expected)
        json.dumps(self.results, allow_nan=False)

    def test_mdl_scores_match_independent_hand_values(self):
        result = self.results["E04-05"]
        np.testing.assert_allclose([row["mdl"] for row in result["cases"]],
            [171.35547573266692, 86.43784729230299, 28.63605470127869, 34.53877639491069])
        np.testing.assert_allclose([row["arithmetic_mean"] for row in result["cases"]],
                                   [3.75, 2., 1., .9])
        self.assertEqual(result["selected_count"], 2)
        self.assertEqual(result["equal_spectrum_count"], 0)
        self.assertEqual(result["same_spectrum_n4_count"], 0)
        self.assertEqual(result["scaled_count"], 2)
        self.assertTrue(result["rank_deficient_rejected"])

    def test_noise_correlation_matches_covariance_sum(self):
        rows = self.results["E01-01"]["cases"]
        for row, power in zip(rows, [1 / 4, 5 / 8, 1]):
            self.assertAlmostEqual(row["noise_power"], power)
            self.assertAlmostEqual(row["gain_db"], -10 * math.log10(power))

    def test_half_amplitude_is_quarter_power(self):
        row = self.results["E01-02"]
        self.assertEqual(row["power_ratio"], 0.25)
        self.assertAlmostEqual(row["level_change_db"], -6.020599913279624)

    def test_frame_boundary_and_fft_padding_counts(self):
        row = self.results["E02-01"]
        self.assertEqual(row["uncentered_frames"], 122)
        self.assertEqual(row["centered_frames"], 126)
        self.assertEqual((row["rfft_bins_256"], row["rfft_bins_512"]), (129, 257))
        self.assertEqual(row["grid_spacing_hz"], [31.25, 15.625])

    def test_covariance_conjugacy_and_rank(self):
        row = self.results["E02-02"]
        np.testing.assert_allclose(row["matrix_real"], [[2, math.sqrt(3)], [math.sqrt(3), 2]])
        np.testing.assert_allclose(row["matrix_imag"], [[0, 1], [-1, 0]])
        np.testing.assert_allclose(row["eigenvalues"], [0, 4], atol=1e-14)
        self.assertAlmostEqual(row["swapped_offdiagonal_phase_deg"], -30)

    def test_hann_roundtrip_and_unsupported_boundary(self):
        row = self.results["E02-03"]
        self.assertLess(row["max_abs_error"], 1e-12)
        self.assertTrue(row["uncentered_hann_rejected"])

    def test_difference_lags_and_multiplicities(self):
        row = self.results["E03-01"]
        self.assertEqual(row["ula"]["lags"], list(range(-5, 6)))
        self.assertEqual(row["ula"]["multiplicities"], [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
        self.assertEqual(row["nested"]["lags"], list(range(-11, 12)))
        self.assertEqual(row["nested"]["multiplicities"][12:], [3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(row["nested"]["ordered_pair_count"], 36)

    def test_endfire_nonlinearity_does_not_clip_invalid_measurement(self):
        rows = self.results["E03-02"]["cases"]
        self.assertAlmostEqual(rows[1]["estimated_angle_deg"], math.degrees(math.asin(0.08575)))
        expected = math.degrees(math.asin(math.sin(math.radians(85)) - 0.08575))
        self.assertAlmostEqual(rows[2]["estimated_angle_deg"], expected)
        self.assertFalse(rows[3]["valid"])
        self.assertIsNone(rows[3]["estimated_angle_deg"])

    def test_known_direction_gain_calibration_and_unknown_direction_ambiguity(self):
        result = self.results["E03-06"]
        plus, minus = result["cases"]
        self.assertAlmostEqual(result["spacing_m"], 343 / 4000)
        self.assertAlmostEqual(plus["measured_ratio_phase_deg"], 75)
        self.assertAlmostEqual(minus["measured_ratio_phase_deg"], -15)
        for row, corrected_phase in ((plus, 45), (minus, -45)):
            self.assertAlmostEqual(row["estimated_relative_gain_real"], 0.6 * math.sqrt(3))
            self.assertAlmostEqual(row["estimated_relative_gain_imag"], 0.6)
            self.assertAlmostEqual(row["corrected_ratio_phase_deg"], corrected_phase)
        self.assertAlmostEqual(result["wrong_angle_deg_if_gain_phase_ignored"],
                               math.degrees(math.asin(5 / 6)))
        alternative = result["unknown_single_direction_alternative"]
        self.assertEqual(alternative["angle_deg"], 0)
        self.assertAlmostEqual(alternative["relative_gain_phase_deg"], 75)
        self.assertAlmostEqual(alternative["relative_gain_amplitude"], 1.2)

    def test_gcc_sign_silence_and_geometry(self):
        row = self.results["E04-01"]
        self.assertEqual(row["delay_pairs_samples"], [[3, -3], [3, -3]])
        self.assertTrue(row["silence_rejected"])
        self.assertEqual(row["minimum_spacing_for_3_samples_m"], 0.0643125)

    def test_four_mic_fractional_delay_has_independent_phasor_answer(self):
        row = self.results["E04-07"]
        adjacent_seconds = .04 * .5 / 343
        np.testing.assert_allclose(row["relative_arrival_us"],
                                   -1e6 * adjacent_seconds * np.arange(4), atol=1e-12)
        np.testing.assert_allclose(row["causal_alignment_samples"],
                                   16000 * adjacent_seconds * np.arange(4), atol=1e-12)
        phase = 2 * math.pi * 2298 * adjacent_seconds
        expected = abs(math.sin(2*phase) / (4*math.sin(phase/2)))
        self.assertAlmostEqual(row["ideal_unaligned_amplitude"], expected, places=13)
        self.assertAlmostEqual(row["ideal_aligned_amplitude"], 1., places=13)

    def test_esprit_least_squares_hand_matrix_and_perturbation(self):
        cases = self.results["E04-09"]["subspace_cases"]
        ideal, perturbed = cases
        self.assertEqual([row["label"] for row in cases], ["ideal", "perturbed"])
        self.assertEqual([row["first_conjugate_norm"] for row in cases], [2., 2.])
        np.testing.assert_allclose(
            [[row["first_conjugate_second_real"], row["first_conjugate_second_imag"]]
             for row in cases], [[0., 2.], [.1, 1.9]], atol=1e-15)
        np.testing.assert_allclose(
            [[row["rotation_real"], row["rotation_imag"]] for row in cases],
            [[0., 1.], [.05, .95]], atol=1e-15)
        self.assertAlmostEqual(ideal["residual_norm"], 0.)
        self.assertAlmostEqual(perturbed["residual_norm"], .1)
        self.assertAlmostEqual(ideal["angle_deg"], 30.)
        expected_angle = math.degrees(math.asin(math.atan2(.95, .05) / math.pi))
        self.assertAlmostEqual(perturbed["angle_deg"], expected_angle)
        for row in cases:
            self.assertAlmostEqual(row["api_angle_deg"], row["angle_deg"])

    def test_coherent_rank_preserves_physical_source_count(self):
        row = self.results["E04-02"]
        np.testing.assert_allclose(row["independent_eigenvalues"], [0, 0, 4, 4], atol=1e-13)
        np.testing.assert_allclose(row["coherent_eigenvalues"], [0, 0, 0, 8], atol=1e-13)
        self.assertEqual(row["physical_source_count"], 2)

    def test_aliased_directions_share_spectrum(self):
        row = self.results["E04-03"]
        self.assertLess(row["steering_max_abs_difference"], 1e-13)
        np.testing.assert_allclose(row["music_scores"], [1e15, 1e15], rtol=1e-12)

    def test_mwf_shortcut_requires_rank_one(self):
        full, rank_one = self.results["E05-01"]["cases"]
        np.testing.assert_allclose(full["general_weights"], [2 / 3, 0])
        np.testing.assert_allclose(full["trace_shortcut_weights"], [1 / 2, 0])
        np.testing.assert_allclose(rank_one["trace_shortcut_weights"], [2 / 3, 0])

    def test_mvdr_hand_weights_and_loading_tradeoff(self):
        plain, loaded = self.results["E05-02"]["cases"]
        np.testing.assert_allclose(plain["weight_imag"], [5 / 11, -5 / 11])
        self.assertAlmostEqual(plain["interference_amplitude"], math.sqrt(2) / 22)
        self.assertAlmostEqual(plain["wng_linear"], 242 / 221)
        self.assertAlmostEqual(loaded["interference_amplitude"], 3 * math.sqrt(2) / 11)
        self.assertAlmostEqual(loaded["wng_linear"], 242 / 146)
        self.assertLess(plain["target_response_error"], 1e-14)
        self.assertLess(loaded["target_response_error"], 1e-14)


if __name__ == "__main__":
    unittest.main()
