"""Malformed inputs must not become apparently valid spatial estimates."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.array_tutorial.beamforming import apply_beamformer, blocking_matrix, lcmv_weights, mvdr_weights, wiener_gain
from codes.array_tutorial.covariance import spatial_covariance
from codes.array_tutorial.doa import bartlett_spectrum, capon_spectrum, esprit_ula, gcc_phat, music_spectrum, srp_phat


class SpatialValidationTest(unittest.TestCase):
    def test_nonfinite_weights_and_steering_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    apply_beamformer(np.ones((2, 1, 1)), [1, bad])
                with self.assertRaises(ValueError):
                    mvdr_weights(np.eye(2), [1, bad])

    def test_invalid_gcc_and_srp_epsilon_rejected(self):
        for epsilon in (0, -1, np.nan, np.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    gcc_phat(np.zeros(8), np.zeros(8), 16000, epsilon=epsilon)
                with self.assertRaises(ValueError):
                    srp_phat(np.zeros((2, 1, 1)), [1000], [[0, 0], [.04, 0]], [0], epsilon=epsilon)

    def test_invalid_denominator_floors_rejected(self):
        for floor in (0, -1, np.nan, np.inf):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    music_spectrum(np.ones((2, 2)), [1, 1], source_count=1, denominator_floor=floor)
                with self.assertRaises(ValueError):
                    spatial_covariance(np.ones((2, 1, 2)), weights=np.zeros(2), denominator_floor=floor)
                with self.assertRaises(ValueError):
                    wiener_gain(0, 0, power_floor=floor)

    def test_nonfinite_esprit_physical_parameters_rejected(self):
        for parameter in ("spacing_m", "frequency_hz", "sound_speed"):
            for bad in (np.nan, np.inf, 0, -1):
                kwargs = {"spacing_m": .04, "frequency_hz": 1000, "sound_speed": 343}
                kwargs[parameter] = bad
                with self.subTest(parameter=parameter, bad=bad), self.assertRaises(ValueError):
                    esprit_ula(np.ones((2, 2)), source_count=1, **kwargs)

    def test_esprit_alias_tolerance_is_not_nan_or_negative(self):
        for value in (-1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                esprit_ula(np.ones((2, 2)), source_count=1, spacing_m=.04, frequency_hz=1000, alias_tolerance=value)

    def test_invalid_condition_limits_rejected(self):
        for limit in (0.5, -1, np.nan, np.inf):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    mvdr_weights(np.eye(2), [1, 1], condition_limit=limit)
                with self.assertRaises(ValueError):
                    capon_spectrum(np.eye(2), [1, 1], condition_limit=limit)

    def test_empty_or_nonfinite_covariances_rejected(self):
        for matrix in (np.empty((0, 0)), np.full((2, 2), np.nan)):
            with self.subTest(shape=matrix.shape):
                with self.assertRaises(ValueError):
                    bartlett_spectrum(matrix, [1, 1])
                with self.assertRaises(ValueError):
                    music_spectrum(matrix, [1, 1], source_count=1)

    def test_lcmv_nonfinite_constraints_and_responses_rejected(self):
        with self.assertRaises(ValueError):
            lcmv_weights(np.eye(2), [[1], [np.nan]], [1])
        with self.assertRaises(ValueError):
            lcmv_weights(np.eye(2), [[1], [1]], [np.nan])
        with self.assertRaises(ValueError):
            lcmv_weights(np.eye(2), np.empty((2, 0)), [])

    def test_blocking_tolerance_is_bounded(self):
        for value in (-1, 1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                blocking_matrix([1, 1], rtol=value)

    def test_valid_zero_cases_remain_supported(self):
        np.testing.assert_allclose(apply_beamformer(np.ones((2, 1, 1)), [0, 0]), [[0]])
        np.testing.assert_allclose(spatial_covariance(np.zeros((2, 1, 2))), 0)
        np.testing.assert_allclose(wiener_gain(0, 0), 1)
        self.assertEqual(blocking_matrix([1, 1], rtol=0).shape, (2, 1))
        np.testing.assert_allclose(mvdr_weights(np.eye(2), [1, 1], condition_limit=1), [.5, .5])


if __name__ == "__main__":
    unittest.main()
