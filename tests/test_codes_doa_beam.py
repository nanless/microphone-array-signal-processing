from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codes.array_tutorial.beamforming import (
    apply_beamformer,
    blocking_matrix,
    diffuse_coherence,
    dsb_weights,
    lcmv_weights,
    mvdr_weights,
    superdirective_weights,
    wiener_gain,
)
from codes.array_tutorial.covariance import recursive_covariance, spatial_covariance
from codes.array_tutorial.doa import (
    bartlett_spectrum,
    capon_spectrum,
    esprit_ula,
    gcc_phat,
    music_spectrum,
    srp_phat,
)
from codes.array_tutorial.geometry import (
    near_field_steering,
    plane_wave_delays,
    plane_wave_steering,
)
from codes.array_tutorial.spectral import istft, stft


class SpectralAndCovarianceTest(unittest.TestCase):
    def test_stft_round_trip_and_shape(self):
        rng = np.random.default_rng(1234)
        waveform = rng.standard_normal((3, 1001))
        coefficients = stft(waveform, n_fft=128, hop_length=32)
        self.assertEqual(coefficients.shape[:2], (3, 65))
        reconstructed = istft(
            coefficients, n_fft=128, hop_length=32, length=waveform.shape[1]
        )
        np.testing.assert_allclose(reconstructed, waveform, atol=1e-12)

    def test_covariance_is_hermitian_psd_and_has_expected_rank(self):
        spectra = np.zeros((2, 1, 2), dtype=complex)
        spectra[:, 0, 0] = [1.0, 0.0]
        spectra[:, 0, 1] = [0.0, 1.0j]
        covariance = spatial_covariance(spectra)
        np.testing.assert_allclose(covariance[0], 0.5 * np.eye(2))
        np.testing.assert_allclose(covariance, covariance.swapaxes(1, 2).conj())
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance[0]) >= 0.0))

    def test_weighted_covariance_rejects_empty_mask(self):
        with self.assertRaises(ValueError):
            spatial_covariance(np.ones((2, 3, 4)), weights=np.zeros((3, 4)))

    def test_recursive_covariance_matches_definition(self):
        previous = np.zeros((1, 2, 2), dtype=complex)
        snapshot = np.array([[1.0], [1.0j]])
        updated = recursive_covariance(previous, snapshot, forgetting_factor=0.75)
        expected = 0.25 * np.outer(snapshot[:, 0], snapshot[:, 0].conj())
        np.testing.assert_allclose(updated[0], expected)


class GeometryAndDoaTest(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0.0, 0.0], [0.04, 0.0], [0.08, 0.0]])

    def test_positive_angle_sign_and_phase_convention(self):
        delays = plane_wave_delays(self.positions[:2], np.deg2rad(30.0))
        self.assertAlmostEqual(delays[0], 0.0)
        self.assertLess(delays[1], 0.0)
        steering = plane_wave_steering(
            self.positions[:2], [1000.0], np.deg2rad(30.0)
        )[0]
        expected_phase = 2.0 * np.pi * 1000.0 * 0.04 * 0.5 / 343.0
        np.testing.assert_allclose(steering, [1.0, np.exp(1.0j * expected_phase)])

        positions_3d = np.column_stack((self.positions[:2], np.zeros(2)))
        steering_3d = plane_wave_steering(
            positions_3d, [1000.0], np.deg2rad(30.0), elevation_rad=0.0
        )[0]
        np.testing.assert_allclose(steering_3d, steering)

    def test_near_field_steering_has_unit_reference_and_spherical_amplitude(self):
        source = np.array([0.2, 1.0])
        steering = near_field_steering(self.positions[:2], source, [1000.0])
        distances = np.linalg.norm(source - self.positions[:2], axis=1)
        self.assertAlmostEqual(steering[0, 0], 1.0 + 0.0j)
        self.assertAlmostEqual(abs(steering[0, 1]), distances[0] / distances[1])

    def test_gcc_phat_sign_swap_and_physical_lag_limit(self):
        sample_rate = 16_000
        x2 = np.zeros(64)
        x1 = np.zeros(64)
        x2[12] = 1.0
        x1[15] = 1.0
        tau12, _, _, _ = gcc_phat(x1, x2, sample_rate, max_tau=4 / sample_rate)
        tau21, _, _, _ = gcc_phat(x2, x1, sample_rate, max_tau=4 / sample_rate)
        self.assertAlmostEqual(tau12 * sample_rate, 3.0)
        self.assertAlmostEqual(tau21 * sample_rate, -3.0)
        with self.assertRaises(ValueError):
            gcc_phat(np.zeros(8), np.zeros(8), sample_rate)

    def test_srp_phat_true_grid_point_has_largest_score(self):
        frequencies = np.array([500.0, 1000.0, 1500.0, 2000.0])
        true_azimuth = np.deg2rad(20.0)
        steering = plane_wave_steering(self.positions, frequencies, true_azimuth)
        spectra = steering.T[:, :, None] * np.array([1.0, 0.7j])[None, None, :]
        candidates = np.deg2rad([-40.0, 20.0, 55.0])
        scores = srp_phat(spectra, frequencies, self.positions, candidates)
        self.assertEqual(int(np.argmax(scores)), 1)

    def test_bartlett_and_capon_reproduce_chapter_values(self):
        covariance = np.array([[1.25, 1.0], [1.0, 1.25]], dtype=complex)
        candidates = np.array([[1.0, 1.0], [1.0, 1.0j]])
        np.testing.assert_allclose(bartlett_spectrum(covariance, candidates), [4.5, 2.5])
        np.testing.assert_allclose(capon_spectrum(covariance, candidates), [1.125, 0.225])

    def test_music_and_esprit_find_the_analytic_ula_source(self):
        true = np.deg2rad(30.0)
        spacing = 0.5
        frequency = 343.0
        positions = np.column_stack((np.arange(3) * spacing, np.zeros(3)))
        steering = plane_wave_steering(positions, [frequency], true)[0]
        covariance = np.outer(steering, steering.conj()) + 0.1 * np.eye(3)
        candidates = np.vstack((np.ones(3), steering))
        spectrum = music_spectrum(covariance, candidates, source_count=1)
        self.assertGreater(spectrum[1], spectrum[0] * 1e6)
        estimate = esprit_ula(
            covariance, source_count=1, spacing_m=spacing, frequency_hz=frequency
        )
        np.testing.assert_allclose(estimate, [true], atol=1e-12)

    def test_capon_rejects_singular_covariance_without_loading(self):
        singular = np.ones((2, 2), dtype=complex)
        with self.assertRaises(np.linalg.LinAlgError):
            capon_spectrum(singular, [1.0, 1.0])
        loaded = capon_spectrum(
            singular, [1.0, 1.0], relative_diagonal_loading=1e-3
        )
        self.assertTrue(np.all(np.isfinite(loaded)))


class BeamformingTest(unittest.TestCase):
    def test_dsb_has_unit_target_response_and_two_channel_wng(self):
        steering = np.array([1.0, 1.0j])
        weights = dsb_weights(steering)
        self.assertAlmostEqual(np.vdot(weights, steering), 1.0 + 0.0j)
        self.assertAlmostEqual(1.0 / np.vdot(weights, weights).real, 2.0)
        spectra = steering[:, None, None]
        np.testing.assert_allclose(apply_beamformer(spectra, weights), [[1.0]])

    def test_mvdr_reproduces_chapter_weights_and_rejects_singular_input(self):
        target = np.array([1.0, 1.0], dtype=complex)
        interference = np.array([1.0, 1.0j], dtype=complex)
        covariance = 10.0 * np.outer(interference, interference.conj()) + np.eye(2)
        weights = mvdr_weights(covariance, target)
        expected = np.array([0.5 + 5.0j / 11.0, 0.5 - 5.0j / 11.0])
        np.testing.assert_allclose(weights, expected)
        self.assertAlmostEqual(np.vdot(weights, target), 1.0 + 0.0j)
        with self.assertRaises(np.linalg.LinAlgError):
            mvdr_weights(np.ones((2, 2)), target)
        with self.assertRaises(np.linalg.LinAlgError):
            mvdr_weights(np.diag([-1.0, 1.0]), np.array([0.0, 1.0]))

    def test_superdirective_loading_produces_finite_distortionless_weights(self):
        positions = np.array([[0.0, 0.0], [0.02, 0.0]])
        frequency = np.array([500.0])
        coherence = diffuse_coherence(positions, frequency)
        steering = np.ones((1, 2), dtype=complex)
        weights = superdirective_weights(
            coherence, steering, relative_diagonal_loading=1e-3
        )
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(np.vdot(weights[0], steering[0]), 1.0 + 0.0j)

    def test_lcmv_and_gsc_blocker_satisfy_constraints(self):
        target = np.array([1.0, 1.0], dtype=complex)
        interference = np.array([1.0, 1.0j], dtype=complex)
        constraints = np.column_stack((target, interference))
        responses = np.array([1.0, 0.0])
        weights = lcmv_weights(np.eye(2), constraints, responses)
        np.testing.assert_allclose(constraints.conj().T @ weights, responses, atol=1e-12)
        expected = np.array([0.5 + 0.5j, 0.5 - 0.5j])
        np.testing.assert_allclose(weights, expected)
        blocker = blocking_matrix(target)
        self.assertEqual(blocker.shape, (2, 1))
        np.testing.assert_allclose(blocker.conj().T @ target, 0.0, atol=1e-12)
        np.testing.assert_allclose(blocker.conj().T @ blocker, np.eye(1), atol=1e-12)

    def test_wiener_gain_matches_chapter_example_and_floor(self):
        self.assertAlmostEqual(wiener_gain(4.0, 0.7), 0.825)
        np.testing.assert_allclose(
            wiener_gain([1.0, 1.0], [2.0, 0.5], gain_floor=0.1), [0.1, 0.5]
        )


if __name__ == "__main__":
    unittest.main()
