"""Independent numerical boundary checks for spatial and tracking examples."""

import math
import unittest
import warnings

import numpy as np

from codes.array_tutorial.doa import (
    bartlett_spectrum,
    gcc_phat,
    music_spectrum,
    srp_phat,
)
from codes.array_tutorial.tracking import ConstantVelocityKalman, systematic_resample


class StableDoaTest(unittest.TestCase):
    def test_gcc_phat_two_sample_delay_is_gain_invariant(self):
        for gain in (1e-200, 1.0, 1e200):
            with self.subTest(gain=gain), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                earlier = np.zeros(16)
                later = np.zeros(16)
                earlier[1] = 2 * gain
                later[3] = 3 * gain
                tau, peak, _, correlation = gcc_phat(later, earlier, 16000)
                self.assertEqual(tau, 2 / 16000)
                self.assertAlmostEqual(peak, 1.0)
                self.assertTrue(np.all(np.isfinite(correlation)))

    def test_srp_true_azimuth_survives_extreme_channel_gains(self):
        frequency = 1000.0
        spacing = 343.0 / (2 * frequency)
        azimuths = np.deg2rad([-30.0, 30.0, 70.0])
        # Broadside 30 degrees gives adjacent phase +pi/2 in this book.
        steering = np.array([1.0, 1.0j])[:, None, None]
        positions = np.array([[0.0, 0.0], [spacing, 0.0]])
        expected = np.array([-1.0, 1.0, math.cos(math.pi * (math.sin(math.radians(70)) - 0.5))])
        for gains in ((1.0, 1.0), (1e200, 1e-200)):
            with self.subTest(gains=gains), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                spectra = steering * np.array(gains)[:, None, None]
                scores = srp_phat(spectra, [frequency], positions, azimuths)
                np.testing.assert_allclose(scores, expected, atol=1e-14)
                self.assertEqual(int(np.argmax(scores)), 1)

    def test_srp_rejects_no_informative_pair(self):
        positions = [[0.0, 0.0], [0.04, 0.0]]
        for spectra in (np.zeros((2, 2, 1), dtype=complex),
                        np.array([[[1.0], [1.0]], [[0.0], [0.0]]])):
            with self.subTest(spectra=spectra), self.assertRaisesRegex(ValueError, "informative"):
                srp_phat(spectra, [1000.0, 2000.0], positions, [0.0, 0.2])
        with self.assertRaisesRegex(ValueError, "informative"):
            srp_phat(np.ones((2, 1, 1)), [0.0], positions, [0.0, 0.2])

    def test_spectra_reject_negative_covariance_but_keep_rank_one_psd(self):
        impossible = np.diag([-1.0, 2.0])
        steering = [[1.0, 1.0]]
        for call in (lambda: bartlett_spectrum(impossible, steering),
                     lambda: music_spectrum(impossible, steering, source_count=1)):
            with self.assertRaisesRegex(np.linalg.LinAlgError, "positive semidefinite"):
                call()
        rank_one = np.ones((2, 2))
        np.testing.assert_allclose(bartlett_spectrum(rank_one, steering), [4.0])
        self.assertTrue(np.isfinite(music_spectrum(rank_one, steering, source_count=1)[0]))
        with self.assertRaisesRegex(ValueError, "no signal or noise energy"):
            music_spectrum(np.zeros((2, 2)), steering, source_count=1)


class StableTrackingTest(unittest.TestCase):
    def test_chapter_kalman_scalar_update_matches_hand_calculation(self):
        tracker = ConstantVelocityKalman([30.0, 0.5], np.diag([4.0, 0.25]), np.diag([0.1, 0.01]))
        np.testing.assert_allclose(tracker.predict(1.0), [30.5, 0.5])
        np.testing.assert_allclose(tracker.covariance, [[4.35, 0.25], [0.25, 0.26]])
        expected_state = np.array([30.5, 0.5]) + 1.5 * np.array([4.35, 0.25]) / 29.35
        expected_covariance = np.array([[4.35, 0.25], [0.25, 0.26]]) - np.array(
            [[4.35**2, 4.35 * 0.25], [4.35 * 0.25, 0.25**2]]
        ) / 29.35
        np.testing.assert_allclose(tracker.update(32.0, 25.0), expected_state)
        np.testing.assert_allclose(tracker.covariance, expected_covariance, rtol=1e-14)

    def test_update_avoids_large_variance_sum_overflow(self):
        tracker = ConstantVelocityKalman([0.0, 0.0], np.diag([1e308, 1.0]), np.zeros((2, 2)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            state = tracker.update(10.0, 1e308)
        np.testing.assert_allclose(state, [5.0, 0.0], rtol=0, atol=0)
        np.testing.assert_allclose(tracker.covariance, np.diag([5e307, 1.0]), rtol=1e-15)

    def test_predict_large_finite_covariance_stays_finite(self):
        tracker = ConstantVelocityKalman([0.0, 0.0], np.diag([1e308, 0.0]), np.zeros((2, 2)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            tracker.predict(1.0)
        np.testing.assert_array_equal(tracker.covariance, np.diag([1e308, 0.0]))

    def test_failed_prediction_does_not_change_filter(self):
        tracker = ConstantVelocityKalman([0.0, 1e308], np.eye(2), np.eye(2))
        before_state = tracker.state.copy()
        before_covariance = tracker.covariance.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(ValueError):
                tracker.predict(10.0)
        np.testing.assert_array_equal(tracker.state, before_state)
        np.testing.assert_array_equal(tracker.covariance, before_covariance)

    def test_failed_update_does_not_change_filter(self):
        # This rank-one PSD matrix permits an enormous velocity gain.  A
        # 179-degree innovation makes the updated velocity unrepresentable.
        prior = np.array([[1e-308, 1.0], [1.0, 1e308]])
        tracker = ConstantVelocityKalman([0.0, 1.0], prior, np.zeros((2, 2)))
        before_state = tracker.state.copy()
        before_covariance = tracker.covariance.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(ValueError):
                tracker.update(179.0, 1e-308)
        np.testing.assert_array_equal(tracker.state, before_state)
        np.testing.assert_array_equal(tracker.covariance, before_covariance)

    def test_systematic_resampling_accepts_large_equal_weights(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            indices = systematic_resample([1e308, 1e308], np.random.default_rng(0))
        np.testing.assert_array_equal(indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
