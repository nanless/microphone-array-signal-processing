import unittest

import numpy as np

from codes.array_tutorial.aec import erle_db, nlms
from codes.array_tutorial.dereverberation import offline_wpe
from codes.array_tutorial.separation import (
    mask_mvdr_2x2,
    masked_spatial_covariance,
    pit_permutation,
    si_sdr,
)
from codes.array_tutorial.tracking import (
    CircularParticleFilter,
    ConstantVelocityKalman,
    systematic_resample,
    wrap_angle,
)


class TestAec(unittest.TestCase):
    def test_nlms_zero_energy_and_convergence(self):
        rng = np.random.default_rng(1)
        x = np.r_[np.zeros(20), rng.normal(size=3000)]
        path = np.array([0.7, -0.2, 0.1])
        d = np.convolve(x, path, mode="full")[: x.size]
        e, _, weights = nlms(x, d, 3, step_size=0.5, epsilon=1e-10)
        self.assertTrue(np.all(np.isfinite(e)))
        np.testing.assert_allclose(weights, path, atol=1e-5)

    def test_freeze_and_path_change(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=4000)
        first = np.convolve(x, [0.8, 0.1], mode="full")[: x.size]
        second = np.convolve(x, [-0.3, 0.6], mode="full")[: x.size]
        d = np.where(np.arange(x.size) < 2000, first, second)
        freeze = np.zeros(x.size, dtype=bool)
        freeze[2000:2500] = True
        e_frozen, _, _ = nlms(x, d, 2, freeze=freeze)
        e_active, _, weights = nlms(x, d, 2)
        self.assertGreater(np.mean(e_frozen[2100:2400] ** 2), np.mean(e_active[2100:2400] ** 2))
        np.testing.assert_allclose(weights, [-0.3, 0.6], atol=1e-5)

    def test_erle_excludes_double_talk(self):
        d = np.ones(8)
        e = np.r_[np.full(4, 0.1), np.full(4, 10.0)]
        double_talk = np.r_[np.zeros(4, dtype=bool), np.ones(4, dtype=bool)]
        self.assertAlmostEqual(erle_db(d, e, double_talk_mask=double_talk), 20.0, places=8)
        with self.assertRaises(ValueError):
            erle_db(np.zeros(8), np.zeros(8))
        with self.assertRaises(ValueError):
            erle_db(d, e, epsilon=0.0)


class TestWpe(unittest.TestCase):
    def test_passthrough_boundaries(self):
        x = np.ones((2, 4), dtype=np.complex128)
        np.testing.assert_array_equal(offline_wpe(x, taps=0, delay=2), x)
        np.testing.assert_array_equal(offline_wpe(x, taps=3, delay=3), x)
        zeros = np.zeros((1, 2, 20), dtype=np.complex128)
        np.testing.assert_array_equal(offline_wpe(zeros, taps=2, delay=2), zeros)

    def test_single_channel_complex_conjugate_prediction(self):
        # x[t] = conj(g) * x[t-1] is exactly representable for delay=1, taps=1.
        g = 0.5 + 0.25j
        x = np.empty(30, dtype=np.complex128)
        x[0] = 1.0 + 0.3j
        for t in range(1, x.size):
            x[t] = np.conj(g) * x[t - 1]
        result = offline_wpe(x[None, :], taps=1, delay=1, iterations=1, power_floor=1e-8)
        self.assertLess(np.mean(np.abs(result[0, 1:]) ** 2), 1e-10)

    def test_multichannel_shape_loading_and_scale(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 2, 40)) + 1j * rng.normal(size=(2, 2, 40))
        y = offline_wpe(x, taps=2, delay=2, iterations=2)
        scaled = offline_wpe(3.0 * x, taps=2, delay=2, iterations=2)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        np.testing.assert_allclose(scaled, 3.0 * y, rtol=1e-8, atol=1e-8)


class TestSeparation(unittest.TestCase):
    def test_pit_and_silent_reference(self):
        rng = np.random.default_rng(4)
        references = rng.normal(size=(2, 200))
        estimates = references[::-1] + 1e-3 * rng.normal(size=references.shape)
        permutation, score = pit_permutation(estimates, references)
        self.assertEqual(permutation, (1, 0))
        self.assertGreater(score, 50.0)
        with self.assertRaises(ValueError):
            si_sdr(np.ones(4), np.zeros(4))
        with self.assertRaises(ValueError):
            si_sdr(np.zeros(4), np.arange(4.0))
        with self.assertRaises(ValueError):
            si_sdr(np.ones(4), np.arange(4.0))  # zero after mean subtraction
        with self.assertRaises(ValueError):
            pit_permutation(np.empty((0, 4)), np.empty((0, 4)))

    def test_masked_scm_is_hermitian(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(3, 2, 10)) + 1j * rng.normal(size=(3, 2, 10))
        covariance = masked_spatial_covariance(x, np.ones((3, 10)))
        np.testing.assert_allclose(covariance, covariance.conj().transpose(0, 2, 1), atol=1e-12)

    def test_two_channel_mvdr_distortionless_response(self):
        rng = np.random.default_rng(6)
        target = rng.normal(size=80) + 1j * rng.normal(size=80)
        noise = rng.normal(size=80) + 1j * rng.normal(size=80)
        steering = np.array([1.0, np.exp(0.7j)])
        interference_steering = np.array([1.0, -1.0])
        x = steering[:, None] * target + 0.3 * interference_steering[:, None] * noise
        target_mask = np.r_[np.ones(40), np.zeros(40)][None, :]
        interference_mask = 1.0 - target_mask
        output, weights = mask_mvdr_2x2(x[None, :, :], target_mask, interference_mask)
        self.assertEqual(output.shape, (1, 80))
        estimated_steering = np.linalg.eigh(masked_spatial_covariance(x[None], target_mask)[0])[1][:, -1]
        estimated_steering /= estimated_steering[0]
        self.assertAlmostEqual(float(np.real(np.vdot(weights[0], estimated_steering))), 1.0, places=8)

    def test_mask_mvdr_falls_back_for_empty_statistics(self):
        x = np.ones((1, 2, 6), dtype=np.complex128)
        ones = np.ones((1, 6))
        zeros = np.zeros((1, 6))
        for target_mask, interference_mask in ((zeros, ones), (ones, zeros)):
            output, weights = mask_mvdr_2x2(x, target_mask, interference_mask)
            np.testing.assert_array_equal(weights, np.array([[1.0 + 0j, 0j]]))
            np.testing.assert_array_equal(output, x[:, 0, :])
        with self.assertRaises(ValueError):
            mask_mvdr_2x2(x, ones, ones, diagonal_loading=-1.0)

    def test_mask_mvdr_matches_independent_two_by_two_oracle(self):
        steering = np.array([1.0 + 0j, 1.0j])
        x = np.zeros((1, 2, 4), dtype=np.complex128)
        x[0, :, :2] = steering[:, None]
        x[0, :, 2] = np.array([np.sqrt(2.0), 0.0])
        x[0, :, 3] = np.array([0.0, np.sqrt(8.0)])
        target_mask = np.array([[1.0, 1.0, 0.0, 0.0]])
        interference_mask = 1.0 - target_mask
        _, weights = mask_mvdr_2x2(
            x,
            target_mask,
            interference_mask,
            diagonal_loading=0.0,
        )
        inverse_noise = np.diag([1.0, 0.25])
        expected = inverse_noise @ steering / np.vdot(steering, inverse_noise @ steering)
        np.testing.assert_allclose(weights[0], expected, atol=1e-12)


class TestTracking(unittest.TestCase):
    def test_wrap_and_joseph_update(self):
        self.assertEqual(wrap_angle(181.0), -179.0)
        tracker = ConstantVelocityKalman([179.0, 0.0], np.eye(2), np.diag([0.1, 0.01]))
        tracker.predict(1.0)
        tracker.update(-179.0, 4.0)
        self.assertGreater(tracker.state[0], 179.0)
        np.testing.assert_allclose(tracker.covariance, tracker.covariance.T, atol=1e-12)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(tracker.covariance)), -1e-12)

    def test_missing_measurement_increases_uncertainty(self):
        tracker = ConstantVelocityKalman([10.0, 1.0], np.eye(2), np.diag([0.1, 0.01]))
        before = tracker.covariance[0, 0]
        tracker.predict(1.0)
        self.assertGreater(tracker.covariance[0, 0], before)
        with self.assertRaises(ValueError):
            tracker.predict(0.0)
        with self.assertRaises(ValueError):
            tracker.update(float("nan"), 1.0)
        with self.assertRaises(ValueError):
            ConstantVelocityKalman([0.0, 0.0], np.array([[1.0, 2.0], [0.0, 1.0]]), np.eye(2))

    def test_systematic_resampling_and_particle_filter(self):
        rng = np.random.default_rng(9)
        indices = systematic_resample(np.array([0.0, 0.0, 1.0]), rng)
        np.testing.assert_array_equal(indices, [2, 2, 2])
        particle_filter = CircularParticleFilter(np.linspace(-180, 180, 200, endpoint=False))
        particle_filter.update(179.0, 2.0, clutter_probability=0.01)
        self.assertLess(abs(wrap_angle(particle_filter.estimate() - 179.0)), 1.0)
        self.assertTrue(particle_filter.resample_if_needed(rng, threshold=199.0))
        np.testing.assert_allclose(particle_filter.weights, 1.0 / 200)

    def test_particle_log_weights_and_undefined_circular_mean(self):
        particle_filter = CircularParticleFilter(np.array([0.0, 1.0]))
        particle_filter.update(180.0, 0.01, clutter_probability=0.0)
        self.assertGreater(particle_filter.weights[1], particle_filter.weights[0])

        symmetric = CircularParticleFilter(np.array([0.0, -180.0]))
        with self.assertRaises(ValueError):
            symmetric.estimate()
        with self.assertRaises(ValueError):
            CircularParticleFilter(np.array([0.0, float("nan")]))


if __name__ == "__main__":
    unittest.main()
