import importlib.util
import hashlib
import inspect
import re
import tempfile
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
    def capture_figure(self, builder):
        captured = []
        original_save = figures.save
        try:
            def capture(figure, _name):
                figures.finalize_figure(figure)
                figure.canvas.draw()
                captured.append(figure)
            figures.save = capture
            builder()
        finally:
            figures.save = original_save
        self.assertEqual(len(captured), 1)
        self.addCleanup(figures.plt.close, captured[0])
        return captured[0]

    def test_png_metadata_identifies_exact_source_without_timestamp(self):
        metadata = figures.figure_png_metadata()
        self.assertEqual(metadata["SourceScript"], "scripts/make_figures.py")
        expected = hashlib.sha256(
            (ROOT / "scripts" / "make_figures.py").read_bytes()).hexdigest()
        self.assertEqual(metadata["SourceScriptDigest"], expected)
        self.assertFalse(any("time" in key.lower() for key in metadata))

    def test_saved_png_contains_stable_source_metadata(self):
        original_out = figures.OUT
        with tempfile.TemporaryDirectory() as directory:
            try:
                figures.OUT = Path(directory)
                figure, axis = figures.plt.subplots(figsize=(1.0, 1.0))
                axis.plot([0, 1], [0, 1])
                figures.save(figure, "metadata-smoke.png")
                payload = (Path(directory) / "metadata-smoke.png").read_bytes()
            finally:
                figures.OUT = original_out
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"SourceScript", payload)
        self.assertIn(b"scripts/make_figures.py", payload)
        self.assertIn(b"SourceScriptDigest", payload)
        self.assertIn(figures.source_script_digest().encode("ascii"), payload)
        self.assertNotIn(b"Creation Time", payload)

    def test_ula_steering_obeys_positive_angle_sign_convention(self):
        steering = figures.ula_steering([0.0, 0.04], 30.0)[:, 0]
        expected = np.exp(2j * np.pi * 0.04 * np.sin(np.deg2rad(30)))
        np.testing.assert_allclose(steering, [1.0, expected], atol=1e-14)
        self.assertGreater(np.angle(steering[1]), 0.0)

    def test_positive_phase_steering_gives_documented_music_covariance(self):
        steering = figures.ula_steering([0.0, 0.5, 1.0], 30.0)[:, 0]
        expected_steering = np.array([1.0, 1.0j, -1.0], dtype=complex)
        expected_covariance = np.array([
            [1.0, -1.0j, -1.0],
            [1.0j, 1.0, -1.0j],
            [-1.0, 1.0j, 1.0],
        ], dtype=complex)
        np.testing.assert_allclose(steering, expected_steering, atol=1e-14)
        np.testing.assert_allclose(
            np.outer(steering, steering.conj()), expected_covariance,
            atol=1e-14)

    def test_plane_wave_steering_uses_array_to_source_direction(self):
        positions = np.array([[0.0, 0.0], [0.04, 0.0]])
        direction = np.array([0.5, np.sqrt(3) / 2])
        steering = figures.plane_wave_steering(
            positions, direction, frequency=1000.0, reference_position=positions[0])
        expected_phase = 2 * np.pi * 1000 * 0.04 * 0.5 / 343
        np.testing.assert_allclose(steering, [1.0, np.exp(1j * expected_phase)])

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

    def test_random_decay_rir_holds_drr_fixed_across_t60(self):
        for t60 in (0.1, 0.4, 0.8):
            rir = figures.random_decay_rir(
                t60, 16000, np.random.default_rng(17), drr_db=6.0)
            direct_energy = rir[0] ** 2
            tail_energy = np.sum(rir[1:] ** 2)
            self.assertAlmostEqual(
                10 * np.log10(direct_energy / tail_energy), 6.0, places=12)

    def test_gcc_reverb_statistics_is_two_factor_and_deterministic(self):
        args = ([0.1, 0.4], [6.0, 0.0, -6.0])
        first = figures.gcc_reverb_statistics(*args, trials=3)
        second = figures.gcc_reverb_statistics(*args, trials=3)
        for key in ("rate", "low", "high", "median", "q1", "q3"):
            self.assertEqual(first[key].shape, (3, 2))
            np.testing.assert_allclose(first[key], second[key])
            self.assertTrue(np.all(np.isfinite(first[key])))
        self.assertTrue(np.all((first["rate"] >= 0.0) & (first["rate"] <= 1.0)))
        self.assertTrue(np.all(first["low"] <= first["rate"]))
        self.assertTrue(np.all(first["rate"] <= first["high"]))
        self.assertTrue(np.all(first["q1"] <= first["median"]))
        self.assertTrue(np.all(first["median"] <= first["q3"]))

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

    def test_capon_relative_loading_is_scale_invariant(self):
        covariance = np.array(
            [[2.0, 0.4 + 0.2j], [0.4 - 0.2j, 1.2]], dtype=complex)
        steering = np.array(
            [[1.0, 1.0], [1.0j, np.exp(0.3j)]], dtype=complex)
        spectrum, loading = figures.capon_spectrum(covariance, steering)
        scaled_spectrum, scaled_loading = figures.capon_spectrum(
            100.0 * covariance, steering)
        np.testing.assert_allclose(
            spectrum / spectrum.max(), scaled_spectrum / scaled_spectrum.max(),
            rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(scaled_loading, 100.0 * loading, places=12)

    def test_capon_matches_independent_two_by_two_oracle(self):
        covariance = np.diag([2.0, 1.0])
        steering = np.array([[1.0, 0.0, 1.0],
                             [0.0, 1.0, 1.0]])
        spectrum, loading = figures.capon_spectrum(
            covariance, steering, relative_loading=0.0)
        self.assertEqual(loading, 0.0)
        np.testing.assert_allclose(spectrum, [2.0, 1.0, 2.0 / 3.0], atol=1e-12)

    def test_capon_rank_deficiency_requires_loading(self):
        covariance = np.ones((2, 2))
        steering = np.eye(2)
        with self.assertRaisesRegex(np.linalg.LinAlgError, "relative_loading"):
            figures.capon_spectrum(covariance, steering, relative_loading=0.0)
        spectrum, loading = figures.capon_spectrum(
            covariance, steering, relative_loading=1e-3)
        self.assertGreater(loading, 0.0)
        self.assertTrue(np.all(np.isfinite(spectrum)))

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

    def test_srp_candidate_distance_lines_end_at_candidate(self):
        microphones = np.array(
            [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0]])
        candidate = np.array([2.214, 2.1])
        segments = figures.candidate_distance_segments(microphones, candidate)
        np.testing.assert_allclose(segments[:, 0, :], microphones)
        np.testing.assert_allclose(
            segments[:, 1, :], np.broadcast_to(candidate, microphones.shape))

    def test_wpe_regressor_has_exact_delay_and_length(self):
        samples = np.arange(12, dtype=float)[None, :]
        t0, past, current = figures.wpe_past_frames(samples, K=5, delay=3)
        self.assertEqual(t0, 7)
        self.assertEqual(past.shape, (1, 5, 5))
        np.testing.assert_array_equal(past[0, :, 0], [4, 3, 2, 1, 0])
        np.testing.assert_array_equal(past[0, :, -1], [8, 7, 6, 5, 4])
        np.testing.assert_array_equal(current[0], [7, 8, 9, 10, 11])

    def test_stft_keeps_complete_frame_that_ends_at_last_sample(self):
        signal = np.arange(12, dtype=float)
        window = np.ones(4)
        spectrum = figures.stft_analysis(signal, window, hop=4)
        self.assertEqual(spectrum.shape, (3, 3))
        np.testing.assert_allclose(spectrum[:, -1], np.fft.rfft(signal[8:12]))

    def test_particle_filter_resamples_only_below_neff_threshold(self):
        observations = np.array([30.0, 31.0, 90.0, 32.0, 33.0])
        _, neff, resampled, reflection_fraction = figures.particle_filter_doa(
            observations, np.random.default_rng(123), n_particles=100,
            observation_std=2.0, resample_fraction=0.5)
        np.testing.assert_array_equal(resampled, neff < 50.0)
        self.assertTrue(np.all((neff >= 1.0) & (neff <= 100.0 + 1e-12)))
        self.assertEqual(reflection_fraction.dtype, np.dtype(float))
        self.assertTrue(np.all((reflection_fraction >= 0.0)
                               & (reflection_fraction <= 1.0)))

    def test_particle_filter_predicts_through_missing_observations(self):
        observations = np.array([30.0, 31.0, np.nan, np.nan, 34.0])
        estimates, neff, resampled, reflection_fraction = figures.particle_filter_doa(
            observations, np.random.default_rng(124), n_particles=300)
        self.assertTrue(np.all(np.isfinite(estimates)))
        self.assertTrue(np.all((estimates >= 0.0) & (estimates <= 120.0)))
        self.assertTrue(np.all(np.isfinite(neff)))
        self.assertFalse(bool(resampled[2]))
        self.assertFalse(bool(resampled[3]))
        self.assertEqual(reflection_fraction.shape, observations.shape)

    def test_particle_filter_prior_does_not_look_ahead_to_future_observation(self):
        first = figures.particle_filter_doa(
            [np.nan, 20.0], np.random.default_rng(125), n_particles=200)
        second = figures.particle_filter_doa(
            [np.nan, 100.0], np.random.default_rng(125), n_particles=200)
        self.assertEqual(first[0][0], second[0][0])
        self.assertEqual(first[1][0], second[1][0])

    def test_particle_filter_log_weights_survive_extreme_zero_clutter_case(self):
        estimates, neff, _, reflection_fraction = figures.particle_filter_doa(
            [1e6, -1e6], np.random.default_rng(126), n_particles=100,
            observation_std=1e-6, clutter_probability=0.0)
        self.assertTrue(np.all(np.isfinite(estimates)))
        self.assertTrue(np.all(np.isfinite(neff)))
        self.assertTrue(np.all(np.isfinite(reflection_fraction)))

    def test_particle_filter_survives_finite_extreme_and_boundary_observations(self):
        observations = np.array([1e300, -1e300, 0.0, 120.0])
        for clutter_probability in (0.0, 0.2):
            result = figures.particle_filter_doa(
                observations, np.random.default_rng(128), n_particles=128,
                observation_std=1.0, clutter_probability=clutter_probability)
            for values in (result[0], result[1], result[3]):
                self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((result[0] >= 0.0) & (result[0] <= 120.0)))

    def test_particle_filter_all_missing_observations_are_prediction_only(self):
        observations = np.full(6, np.nan)
        estimates, neff, resampled, reflection_fraction = figures.particle_filter_doa(
            observations, np.random.default_rng(129), n_particles=80)
        self.assertTrue(np.all(np.isfinite(estimates)))
        self.assertTrue(np.all(np.isfinite(neff)))
        self.assertFalse(np.any(resampled))
        self.assertTrue(np.all(np.isfinite(reflection_fraction)))

    def test_particle_filter_rejects_invalid_parameters(self):
        rng = np.random.default_rng(127)
        invalid = [
            ({"n_particles": 0}, [30.0]),
            ({"observation_std": 0.0}, [30.0]),
            ({"process_std": -1.0}, [30.0]),
            ({"resample_fraction": 1.1}, [30.0]),
            ({"angle_bounds": (20.0, 20.0)}, [30.0]),
            ({}, [np.inf]),
        ]
        for kwargs, observations in invalid:
            with self.subTest(kwargs=kwargs, observations=observations):
                with self.assertRaises(ValueError):
                    figures.particle_filter_doa(observations, rng, **kwargs)

    def test_reflect_interval_preserves_overshoot_and_reverses_velocity(self):
        position, velocity, hit = figures.reflect_interval(
            np.array([-10.0, 130.0, 250.0]),
            np.array([-3.0, 4.0, 5.0]), (0.0, 120.0))
        np.testing.assert_allclose(position, [10.0, 110.0, 10.0])
        np.testing.assert_allclose(velocity, [3.0, -4.0, 5.0])
        np.testing.assert_array_equal(hit, [True, True, True])

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

    def test_wpe_normal_equation_preserves_complex_conjugation(self):
        rng = np.random.default_rng(612)
        regressors = (rng.normal(size=(2, 20))
                      + 1j * rng.normal(size=(2, 20)))
        expected_filter = np.array([0.4 + 0.2j, -0.15 + 0.3j])
        current = expected_filter.conj() @ regressors
        covariance = regressors @ regressors.conj().T
        cross = regressors @ current.conj()
        actual_filter, loading = figures.solve_wpe_filter(
            covariance, cross, relative_loading=0.0)
        self.assertEqual(loading, 0.0)
        np.testing.assert_allclose(actual_filter, expected_filter, atol=1e-12)
        np.testing.assert_allclose(actual_filter.conj() @ regressors, current,
                                   atol=1e-12)

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

    def test_systematic_resampling_rejects_invalid_weights(self):
        for weights in ([], [0.0, 0.0], [0.5, -0.1], [0.5, np.nan],
                        [1e308, 1e308]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    figures.systematic_resample(
                        weights, np.random.default_rng(322))

    def test_pipeline_dependencies_keep_control_paths_separate(self):
        dependencies = figures.pipeline_dependency_spec()
        audio = dependencies["audio"]
        information = dependencies["information"]
        self.assertIn(("render", "render_tap"), audio)
        self.assertIn(("render", "speaker"), audio)
        self.assertIn(("render_tap", "aec"), audio)
        self.assertIn(("speaker", "capture"), audio)
        self.assertIn(("tracking", "bf"), information)
        self.assertNotIn(("tracking", "gss_mask"), information)
        self.assertIn(("diarization", "gss_mask"), information)
        self.assertIn(("gss_mask", "scm"), information)
        self.assertIn(("scm", "bf"), information)
        self.assertIn(("wpe", "neural_separator"), audio)
        self.assertEqual(
            dependencies["control"],
            {("activity_control", target)
             for target in ("aec", "wpe", "tracking", "bf", "backend")})

    def test_pipeline_draws_exactly_the_declared_dependencies(self):
        figure = self.capture_figure(figures.fig_pipeline)
        expected = {key: frozenset(value)
                    for key, value in figures.pipeline_dependency_spec().items()}
        self.assertEqual(figure._pipeline_edges, expected)
        self.assertLessEqual(figure.get_size_inches()[0], figures.MAX_FIGURE_WIDTH)
        text = " ".join(item.get_text() for item in figure.axes[0].texts)
        self.assertIn("未方向归一 STFT", text)

    def test_pipeline_node_text_stays_inside_boxes_and_layers_are_separate(self):
        figure = self.capture_figure(figures.fig_pipeline)
        renderer = figure.canvas.get_renderer()
        for name, rectangle in figure._pipeline_node_rectangles.items():
            with self.subTest(node=name):
                box = rectangle.get_window_extent(renderer).expanded(1.03, 1.08)
                text_box = figure._pipeline_node_texts[name].get_window_extent(renderer)
                self.assertGreaterEqual(text_box.x0, box.x0)
                self.assertLessEqual(text_box.x1, box.x1)
                self.assertGreaterEqual(text_box.y0, box.y0)
                self.assertLessEqual(text_box.y1, box.y1)
        positions = {name: rectangle.get_y()
                     for name, rectangle in figure._pipeline_node_rectangles.items()}
        self.assertGreater(positions["far_end"], positions["diarization"])
        self.assertGreater(positions["diarization"], positions["ssl"])
        self.assertGreater(positions["ssl"], positions["capture"])
        self.assertGreater(positions["capture"], positions["neural_separator"])

    def test_fig13_has_room_between_panel_titles_and_previous_xlabels(self):
        figure = self.capture_figure(figures.fig_gcc_reverb)
        self.assertGreater(figure._panel_vertical_gap, 0.055)
        renderer = figure.canvas.get_renderer()
        for upper, lower in zip(figure.axes[:-1], figure.axes[1:]):
            upper_xlabel = upper.xaxis.label.get_window_extent(renderer)
            lower_title = lower.title.get_window_extent(renderer)
            self.assertGreater(upper_xlabel.y0, lower_title.y1)

    def test_fig20_lower_responsibility_notes_use_distinct_lanes(self):
        figure = self.capture_figure(figures.fig_aec_pipeline)
        renderer = figure.canvas.get_renderer()
        first, second = figure._lower_annotation_lanes
        self.assertNotAlmostEqual(first.get_position()[1], second.get_position()[1])
        self.assertFalse(first.get_window_extent(renderer).overlaps(
            second.get_window_extent(renderer)))

    def test_fig21_footer_has_its_own_nonoverlapping_grid_row(self):
        figure = self.capture_figure(figures.fig_wpe)
        footer = figure._footer_axis.get_position()
        last_plot = figure.axes[2].get_position()
        self.assertLessEqual(footer.y1, last_plot.y0)
        renderer = figure.canvas.get_renderer()
        for text_item in figure._footer_axis.texts:
            self.assertTrue(figure.bbox.contains(*text_item.get_window_extent(renderer).get_points()[0]))
            self.assertTrue(figure.bbox.contains(*text_item.get_window_extent(renderer).get_points()[1]))

    def test_fig6_has_no_floating_interrow_explanation_box(self):
        figure = self.capture_figure(figures.fig_stft_cov)
        all_text = " ".join(text_item.get_text()
                            for axis in figure.axes for text_item in axis.texts)
        all_text += " " + " ".join(text_item.get_text() for text_item in figure.texts)
        self.assertNotIn("上排：", all_text)

    def test_fig14_short_title_fits_inside_figure(self):
        figure = self.capture_figure(figures.fig_srp_grid)
        renderer = figure.canvas.get_renderer()
        title = figure.axes[1].title
        self.assertEqual(title.get_text(), "(b) SRP-PHAT 累积分数与峰值位置")
        self.assertLessEqual(title.get_window_extent(renderer).x1, figure.bbox.x1)

    def test_fig24_current_note_does_not_overlap_formula(self):
        figure = self.capture_figure(figures.fig_wpe_frames)
        renderer = figure.canvas.get_renderer()
        current = figure._wpe_current_note.get_window_extent(renderer)
        formula = figure._wpe_formula_note.get_window_extent(renderer)
        self.assertFalse(current.overlaps(formula))

    def test_all_declared_main_figure_widths_fit_final_page(self):
        source = inspect.getsource(figures)
        widths = [float(value) for value in re.findall(
            r"figsize=\(\s*([0-9]+(?:\.[0-9]+)?)\s*,", source)]
        self.assertGreaterEqual(len(widths), 26)
        self.assertLessEqual(max(widths), figures.MAX_FIGURE_WIDTH)
        numeric_font_sizes = [float(value) for value in re.findall(
            r"fontsize\s*=\s*([0-9]+(?:\.[0-9]+)?)", source)]
        self.assertTrue(numeric_font_sizes)
        self.assertGreaterEqual(min(numeric_font_sizes), 11.0)
        self.assertGreaterEqual(
            min(figures.FS_SUP, figures.FS_TITLE, figures.FS_LABEL,
                figures.FS_SMALL, figures.FS_TINY), 11.0)

    def test_finalize_figure_enforces_key_text_sizes(self):
        figure, axis = figures.plt.subplots(figsize=(12.0, 3.0))
        axis.set_title("标题", fontsize=7)
        axis.set_xlabel("横轴", fontsize=7)
        axis.set_ylabel("纵轴", fontsize=7)
        axis.text(0.5, 0.5, "关键注释", fontsize=7)
        axis.plot([0, 1], [0, 1], label="图例")
        axis.legend(fontsize=7)
        figures.finalize_figure(figure)
        self.assertLessEqual(figure.get_size_inches()[0], figures.MAX_FIGURE_WIDTH)
        self.assertGreaterEqual(axis.title.get_fontsize(), figures.FS_TITLE)
        self.assertGreaterEqual(axis.xaxis.label.get_fontsize(), figures.FS_LABEL)
        self.assertTrue(all(text.get_fontsize() >= figures.FS_LABEL
                            for text in axis.texts + axis.get_legend().get_texts()))
        figures.plt.close(figure)

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
