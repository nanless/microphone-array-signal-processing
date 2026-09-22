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

    def test_registry_has_twelve_finite_json_results(self):
        expected = {"E01-01", "E01-02", "E02-01", "E02-02", "E02-03", "E03-01",
                    "E03-02", "E04-01", "E04-02", "E04-03", "E05-01", "E05-02"}
        self.assertEqual(set(self.results), expected)
        json.dumps(self.results, allow_nan=False)

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

    def test_gcc_sign_silence_and_geometry(self):
        row = self.results["E04-01"]
        self.assertEqual(row["delay_pairs_samples"], [[3, -3], [3, -3]])
        self.assertTrue(row["silence_rejected"])
        self.assertEqual(row["minimum_spacing_for_3_samples_m"], 0.0643125)

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
