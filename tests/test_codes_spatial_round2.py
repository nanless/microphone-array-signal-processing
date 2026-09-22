"""Independent small-matrix checks for the second spatial review."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.array_tutorial.beamforming import mvdr_weights
from codes.array_tutorial.doa import capon_spectrum
from codes.array_tutorial.geometry import near_field_steering
from codes.examples.exercises_spatial import run_exercises


class SpatialRoundTwoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_exercises()

    def test_unequal_independent_noise_uses_inverse_variance(self):
        result = self.results["E01-03"]
        np.testing.assert_allclose(result["weights"], [4 / 5, 1 / 5])
        self.assertAlmostEqual(result["equal_noise_power"], 5 / 4)
        self.assertAlmostEqual(result["weighted_noise_power"], 4 / 5)
        self.assertAlmostEqual(result["gain_over_equal_db"], 10 * math.log10(25 / 16))

    def test_mask_support_rank_and_scale_are_separate(self):
        first, second = self.results["E02-04"]["cases"]
        np.testing.assert_allclose(first["matrix"], [[1, .5], [.5, .5]])
        np.testing.assert_allclose(second["matrix"], [[1, 0], [0, 0]])
        self.assertEqual((first["rank"], second["rank"]), (2, 1))
        self.assertEqual((first["weight_concentration_count"],
                          second["weight_concentration_count"]), (2, 1))
        self.assertEqual((first["rescale_error"], second["rescale_error"]), (0, 0))

    def test_recursive_startup_matches_geometric_weight_sum(self):
        for row, mass in zip(self.results["E02-05"]["cases"], [.5, .75, .875]):
            self.assertEqual(row["weight_mass"], mass)
            np.testing.assert_allclose(row["raw_matrix"], mass * np.array([[1, 2], [2, 4]]))
            np.testing.assert_allclose(row["normalized_matrix"], [[1, 2], [2, 4]])

    def test_azimuth_uses_positive_y_zero_and_roundtrips(self):
        result = self.results["E03-03"]
        self.assertAlmostEqual(result["azimuth_deg"], -144.73561031724535)
        self.assertAlmostEqual(result["elevation_deg"], 30)
        self.assertAlmostEqual(result["positive_x_convention_deg"], -125.26438968275465)
        np.testing.assert_allclose(result["recovered_direction"], [-.5, -math.sqrt(.5), .5], atol=1e-14)

    def test_near_field_distance_error_is_recomputed_from_radicals(self):
        rows = self.results["E03-04"]["cases"]
        for row, radius in zip(rows, [.2, .5, 2., 20.]):
            # Independent scalar triangle lengths, rather than a geometry helper.
            left = math.sqrt(radius**2 + .05 * radius + .0025)
            right = math.sqrt(radius**2 - .05 * radius + .0025)
            expected_distances = [(left - radius) * 1000, 0, (right - radius) * 1000]
            error_seconds = math.sqrt(((left - radius - .025)**2
                                       + (right - radius + .025)**2) / 3) / 343
            np.testing.assert_allclose(row["relative_distance_mm"], expected_distances, atol=1e-11)
            self.assertAlmostEqual(row["delay_rms_error_us"], error_seconds * 1e6, places=8)
            self.assertAlmostEqual(row["phase_rms_error_deg_at_4khz"], error_seconds * 1440000, places=8)
            np.testing.assert_allclose(row["relative_amplitudes"], [radius / left, 1, radius / right])
        self.assertGreater(rows[0]["delay_rms_error_us"], rows[-1]["delay_rms_error_us"])

    def test_spatial_smoothing_restores_rank_at_aperture_cost(self):
        row = self.results["E04-04"]
        np.testing.assert_allclose(row["original_eigenvalues"], [0, 0, 0, 8], atol=1e-13)
        np.testing.assert_allclose(row["smoothed_matrix_real"], [[2, 0, -2], [0, 2, 0], [-2, 0, 2]], atol=1e-13)
        np.testing.assert_allclose(row["smoothed_eigenvalues"], [0, 2, 4], atol=1e-13)
        self.assertEqual(row["original_aperture_in_spacings"], 3)
        self.assertEqual(row["smoothed_aperture_in_spacings"], 2)
        self.assertEqual((row["subarray_count"], row["subarray_channels"]), (2, 3))

    def test_postfilter_uses_beam_output_noise(self):
        row = self.results["E05-03"]
        self.assertAlmostEqual(row["true_output_noise"], 8 / 15)
        self.assertAlmostEqual(row["total_output_power"], 53 / 15)
        self.assertAlmostEqual(row["independent_model_output_noise_estimate"], 7 / 30)
        self.assertAlmostEqual(row["oracle_gain"], 45 / 53)
        self.assertAlmostEqual(row["independent_model_gain"], 99 / 106)

    def test_complex_lcmv_constraint_needs_conjugate_response(self):
        row = self.results["E05-04"]
        np.testing.assert_allclose(row["unconjugated_f_response_imag"], [0, -1])
        np.testing.assert_allclose(row["correct_weight_real"], [1, 0])
        np.testing.assert_allclose(row["correct_weight_imag"], [0, -1])
        np.testing.assert_allclose(row["correct_response_real"], [1, 0])
        np.testing.assert_allclose(row["correct_response_imag"], [0, 1])

    def test_indefinite_covariance_rejected_independently_of_scale(self):
        for scale in (1e-18, 1e-12, 1., 1e12):
            for solver in (mvdr_weights, capon_spectrum):
                with self.subTest(scale=scale, solver=solver.__name__):
                    with self.assertRaises(np.linalg.LinAlgError):
                        solver(scale * np.diag([1., -1.]), [1, 1])
                    with self.assertRaises(np.linalg.LinAlgError):
                        solver(np.zeros((2, 2)), [1, 1])

    def test_valid_covariance_scaling_preserves_weight_and_scales_capon(self):
        for scale in (1e-18, 1e-12, 1., 1e12):
            covariance = scale * np.diag([1., 2.])
            with self.subTest(scale=scale):
                np.testing.assert_allclose(mvdr_weights(covariance, [1, 1]), [2 / 3, 1 / 3])
                np.testing.assert_allclose(capon_spectrum(covariance, [1, 1]) / scale, 2 / 3)

    def test_complex_covariance_scaling_uses_relative_roundoff_tolerance(self):
        covariance = np.array([[2, 1j], [-1j, 3]])
        steering = np.array([1, 1 + .5j])
        # Hand inverse: R^-1 = [[3, -j], [j, 2]] / 5, hence
        # R^-1 a = [.7-.2j, .4+.4j] and a.H R^-1 a = 13/10.
        expected_weight = np.array([7 - 2j, 4 + 4j]) / 13
        for scale in (1e-18, 1e-12, 1., 1e12):
            with self.subTest(scale=scale):
                actual = mvdr_weights(scale * covariance, steering)
                np.testing.assert_allclose(actual, expected_weight, rtol=1e-13, atol=1e-15)
                np.testing.assert_allclose(np.vdot(actual, steering), 1., atol=1e-14)
                np.testing.assert_allclose(
                    capon_spectrum(scale * covariance, steering) / scale,
                    10 / 13, rtol=1e-13, atol=0.)

    def test_near_field_exclusion_radius_validation(self):
        microphones = [[0, 0], [.1, 0]]
        for radius in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                near_field_steering(microphones, [0, 1], [1000], minimum_distance=radius)
        with self.assertRaises(ValueError):
            near_field_steering(microphones, [0, 0], [1000], minimum_distance=0)
        with self.assertRaises(ValueError):
            near_field_steering(microphones, [0, 1e-7], [1000])
        self.assertTrue(np.all(np.isfinite(
            near_field_steering(microphones, [0, 1], [1000], minimum_distance=0))))

    def test_padding_linearizes_cc_but_does_not_fix_phat_values(self):
        x = np.array([1., 2., 3., 4.])
        y = np.array([1., 0., 1.])
        phat_zero_lags = []
        for length in (8, 16, 32):
            cross = np.fft.rfft(x, length) * np.fft.rfft(y, length).conj()
            cc = np.fft.irfft(cross, length)
            cropped = np.concatenate((cc[-2:], cc[:4]))
            np.testing.assert_allclose(cropped, np.correlate(x, y, mode="full"), atol=1e-13)
            phat = np.fft.irfft(cross / np.maximum(np.abs(cross), 1e-12), length)
            phat_zero_lags.append(phat[0])
        self.assertGreater(abs(phat_zero_lags[1] - phat_zero_lags[0]), .06)
        self.assertGreater(abs(phat_zero_lags[2] - phat_zero_lags[1]), .01)


if __name__ == "__main__":
    unittest.main()
