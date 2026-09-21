import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


figures = load_module()


class FigureAlgorithmTest(unittest.TestCase):
    def test_bartlett_matches_explicit_quadratic_form(self):
        covariance = np.array(
            [[2.0, 0.5 + 0.25j], [0.5 - 0.25j, 1.5]], dtype=complex)
        steering = np.array(
            [[1.0, 1.0], [1.0j, np.exp(0.4j)]], dtype=complex)
        actual = figures.bartlett_spectrum(covariance, steering)
        expected = np.array([
            (steering[:, k].conj() @ covariance @ steering[:, k]).real
            for k in range(steering.shape[1])
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertTrue(np.all(actual >= 0.0))

    def test_srp_tdoa_score_peaks_at_source_grid_point(self):
        mics = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0]])
        source = np.array([5.5, 3.8])
        gx = np.linspace(4.8, 6.2, 57)
        gy = np.linspace(3.1, 4.5, 57)
        grid_x, grid_y = np.meshgrid(gx, gy)
        score = figures.srp_tdoa_score(grid_x, grid_y, mics, source)
        peak = np.unravel_index(np.argmax(score), score.shape)
        self.assertAlmostEqual(grid_x[peak], source[0], delta=0.06)
        self.assertAlmostEqual(grid_y[peak], source[1], delta=0.06)
        self.assertGreaterEqual(float(score.min()), 0.0)
        self.assertLessEqual(float(score.max()), 1.0 + 1e-12)

    def test_wpe_regressor_has_exact_delay_and_length(self):
        samples = np.arange(12, dtype=float)[None, :]
        t0, past, current = figures.wpe_past_frames(samples, K=5, delay=3)
        self.assertEqual(t0, 7)
        self.assertEqual(past.shape, (1, 5, 5))
        np.testing.assert_array_equal(past[0, :, 0], [4, 3, 2, 1, 0])
        np.testing.assert_array_equal(past[0, :, -1], [8, 7, 6, 5, 4])
        np.testing.assert_array_equal(current[0], [7, 8, 9, 10, 11])

    def test_particle_filter_resamples_only_below_neff_threshold(self):
        observations = np.array([30.0, 31.0, 90.0, 32.0, 33.0])
        _, neff, resampled = figures.particle_filter_doa(
            observations, np.random.default_rng(123), n_particles=100,
            observation_std=2.0, resample_fraction=0.5)
        np.testing.assert_array_equal(resampled, neff < 50.0)
        self.assertTrue(np.all((neff >= 1.0) & (neff <= 100.0 + 1e-12)))

    def test_systematic_resampling_indices_stay_in_bounds(self):
        weights = np.array([1e-16, 1e-16, 1.0 - 2e-16])
        indices = figures.systematic_resample(
            weights, np.random.default_rng(321))
        self.assertTrue(np.all((indices >= 0) & (indices < len(weights))))

    def test_time_and_fft_correlation_are_independent_but_equivalent(self):
        rng = np.random.default_rng(456)
        x = rng.normal(size=37) + 1j * rng.normal(size=37)
        y = rng.normal(size=23) + 1j * rng.normal(size=23)
        lags_time, corr_time = figures.time_domain_correlation(x, y)
        lags_fft, corr_fft = figures.fft_correlation(x, y)
        np.testing.assert_array_equal(lags_time, lags_fft)
        np.testing.assert_allclose(corr_time, corr_fft, rtol=1e-12, atol=1e-12)

    def test_correlation_peak_recovers_zero_filled_delay(self):
        early = np.zeros(32)
        early[4:12] = [1, -2, 3, -1, 2, 0.5, -0.5, 1]
        delay = 5
        late = np.zeros_like(early)
        late[delay:] = early[:-delay]
        lags, values = figures.fft_correlation(late, early)
        self.assertEqual(int(lags[np.argmax(values)]), delay)


if __name__ == "__main__":
    unittest.main()
