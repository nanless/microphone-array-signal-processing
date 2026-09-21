import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


figures = load_module("make_figures", "make_figures.py")
aec_figures = load_module("make_aec_figures", "make_aec_figures.py")


class FigureAlgorithmTest(unittest.TestCase):
    def test_array_scale_indicators_use_exact_formulas(self):
        wng, pairs, covariance_entries = figures.array_scale_indicators(
            np.array([2, 4, 8]))
        np.testing.assert_allclose(wng, 10 * np.log10([2, 4, 8]))
        np.testing.assert_array_equal(pairs, [1, 6, 28])
        np.testing.assert_array_equal(covariance_entries, [4, 16, 64])

    def test_causal_delay_zero_fills_instead_of_wrapping(self):
        signal = np.arange(1.0, 7.0)
        delayed = figures.causal_delay(signal, 2)
        np.testing.assert_array_equal(delayed, [0, 0, 1, 2, 3, 4])

    def test_wilson_interval_handles_boundary_counts(self):
        low_zero, high_zero = figures.wilson_interval(0, 100)
        low_all, high_all = figures.wilson_interval(100, 100)
        self.assertEqual(low_zero, 0.0)
        self.assertGreater(high_zero, 0.0)
        self.assertLess(low_all, 1.0)
        self.assertAlmostEqual(high_all, 1.0, places=15)

    def test_fibonacci_sphere_is_deterministic_and_on_unit_sphere(self):
        points = figures.fibonacci_sphere(32)
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0,
                                   rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(points, figures.fibonacci_sphere(32))
        expected_z = 1.0 - 2.0 * (np.arange(32) + 0.5) / 32
        np.testing.assert_allclose(points[:, 2], expected_z)

    def test_random_decay_rir_extends_beyond_t60(self):
        fs = 16000
        t60 = 1.0
        rir = figures.random_decay_rir(
            t60, fs, np.random.default_rng(7))
        self.assertGreater(rir.size, fs)
        end_time = (rir.size - 1) / fs
        envelope_db = 20 * np.log10(np.exp(-6.91 * end_time / t60))
        self.assertLess(envelope_db, -60.0)
        self.assertEqual(rir[0], 1.0)

    def test_fft_convolve_prefix_matches_direct_linear_convolution(self):
        signal = np.array([1.0, -2.0, 0.5, 3.0])
        impulse = np.array([0.4, 1.0, -0.2])
        actual = figures.fft_convolve_prefix(signal, impulse, 5)
        expected = np.convolve(signal, impulse)[:5]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_gcc_peak_metrics_use_documented_sign_and_tolerance(self):
        self.assertTrue(figures.gcc_peak_is_correct(-9, 9))
        self.assertTrue(figures.gcc_peak_is_correct(-8, 9))
        self.assertFalse(figures.gcc_peak_is_correct(-7, 9))
        correlation = np.array([-1.0, 2.0, 4.0, -3.0])
        self.assertAlmostEqual(
            figures.gcc_peak_contrast(correlation),
            4.0 / (np.median(np.abs(correlation)) + 1e-12), places=12)

    def test_distortionless_weights_keep_unit_target_response(self):
        covariance = np.array(
            [[2.0, 0.3 - 0.1j], [0.3 + 0.1j, 1.0]], dtype=complex)
        steering = np.array([1.0, np.exp(-0.4j)])
        weights, loading = figures.distortionless_weights(
            covariance, steering, diagonal_loading=1e-3)
        self.assertAlmostEqual(float((weights.conj() @ steering).real), 1.0, places=12)
        self.assertAlmostEqual(float((weights.conj() @ steering).imag), 0.0, places=12)
        self.assertAlmostEqual(loading, 1e-3 * np.trace(covariance).real / 2)

    def test_interpolated_gcc_phat_recovers_fractional_delay(self):
        rng = np.random.default_rng(77)
        fs = 16000
        sample_delay = 0.7
        signal = rng.normal(size=2048)
        spectrum = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(signal.size, 1 / fs)
        spectrum[frequencies > 6000] = 0
        reference = np.fft.irfft(spectrum)
        delayed = np.fft.irfft(
            spectrum * np.exp(-2j * np.pi * frequencies * sample_delay / fs))
        _, _, estimate = figures.gcc_phat_interpolated(
            delayed, reference, fs, interp=32, max_tau=0.5e-3)
        self.assertAlmostEqual(estimate * fs, sample_delay, delta=1 / 32)

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

    def test_particle_filter_predicts_through_missing_observations(self):
        observations = np.array([30.0, 31.0, np.nan, np.nan, 34.0])
        estimates, neff, resampled = figures.particle_filter_doa(
            observations, np.random.default_rng(124), n_particles=300)
        self.assertTrue(np.all(np.isfinite(estimates)))
        self.assertTrue(np.all(np.isfinite(neff)))
        self.assertFalse(bool(resampled[2]))
        self.assertFalse(bool(resampled[3]))

    def test_phd_intensity_integrates_to_expected_target_count(self):
        grid = np.linspace(-90, 90, 2001)
        intensity = figures.normalized_phd_intensity(
            grid, centers=[-25, 40], stds=[4, 5], weights=[1, 1])
        self.assertAlmostEqual(float(np.trapezoid(intensity, grid)), 2.0, places=10)

    def test_erle_mask_removes_double_talk_interval_only(self):
        times = np.array([0.7, 0.8, 1.0, 1.2, 1.3])
        values = np.arange(5.0)
        masked = aec_figures.mask_metric_intervals(
            times, values, [(0.8, 1.2)])
        np.testing.assert_array_equal(np.isnan(masked), [False, True, True, False, False])
        np.testing.assert_array_equal(masked[[0, 3, 4]], values[[0, 3, 4]])

    def test_fig18_fixed_seed_steady_erle_matches_plotted_value(self):
        times, erle, plateau = figures.fig18_erle_simulation()
        steady = (times > 0.45) & (times < 0.8)
        steady_values = erle[steady & np.isfinite(erle)]
        self.assertGreater(steady_values.size, 0)
        self.assertAlmostEqual(plateau, float(np.mean(steady_values)), places=12)
        # 回归向量以当前样本 x[n] 开头；固定种子下应收敛到约 29.23 dB。
        self.assertAlmostEqual(plateau, 29.23, places=2)
        self.assertTrue(np.all(np.isnan(erle[(times >= 0.8) & (times < 1.2)])))

    def test_wpe_relative_loading_scales_with_covariance(self):
        covariance = np.array(
            [[2.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.0]], dtype=complex)
        cross = np.array([1.0 + 0.2j, 0.4 - 0.1j])
        filt, loading = figures.solve_wpe_filter(covariance, cross)
        scaled_filt, scaled_loading = figures.solve_wpe_filter(
            100.0 * covariance, 100.0 * cross)
        np.testing.assert_allclose(filt, scaled_filt, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(scaled_loading, 100.0 * loading, places=12)

    def test_wpe_all_zero_frequency_bins_are_bypassed(self):
        spectrum = np.zeros((5, 20), dtype=complex)
        output = figures.wpe_dereverb(spectrum, K=4, delay=2, iters=2)
        np.testing.assert_array_equal(output, spectrum)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_wpe_zero_order_empty_and_short_inputs_are_safe(self):
        spectrum = np.ones((2, 4), dtype=complex)
        np.testing.assert_array_equal(
            figures.wpe_dereverb(spectrum, K=0, delay=1, iters=2), spectrum)
        empty = np.empty((2, 0), dtype=complex)
        self.assertEqual(figures.wpe_dereverb(empty, K=1, delay=1).shape, (2, 0))
        for frames in (2, 3, 4):
            output = figures.wpe_dereverb(
                np.ones((2, frames), dtype=complex), K=1, delay=1, iters=1)
            self.assertEqual(output.shape, (2, frames))
            self.assertTrue(np.all(np.isfinite(output)))

    def test_wpe_power_smoothing_preserves_constant_edges(self):
        power = np.full((2, 9), 3.5)
        smoothed = figures.smooth_power_valid(power, width=5)
        np.testing.assert_allclose(smoothed, power, rtol=0, atol=1e-15)
        short = figures.smooth_power_valid(np.full((2, 3), 2.0), width=5)
        self.assertEqual(short.shape, (2, 3))
        np.testing.assert_allclose(short, 2.0, rtol=0, atol=1e-15)

    def test_scale_aligned_spectral_nmse_ignores_global_complex_gain(self):
        reference = np.array([[1 + 1j, 2 - 1j], [0.5, -0.2j]])
        estimate = (2.0 - 0.5j) * reference
        nmse_db = figures.scale_aligned_spectral_nmse_db(
            reference, estimate, np.array([True, True]))
        self.assertLess(nmse_db, -150.0)

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
