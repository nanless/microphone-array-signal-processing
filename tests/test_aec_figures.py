import hashlib
from pathlib import Path
import tempfile
import numpy as np
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image
import scripts.make_aec_figures as aec_figures

from scripts.make_aec_figures import (
    FS,
    SOURCE_SCRIPT,
    block_erle,
    causal_delay,
    colored_x,
    figure_metadata,
    mask_metric_intervals,
    nlms_adaptation_trace,
    nlms_run,
    save,
)


class AecFiguresTest(unittest.TestCase):
    def test_figures_26_to_32_fit_final_width_and_keep_key_text_readable(self):
        builders = [
            aec_figures.fig_problem,
            aec_figures.fig_concept,
            aec_figures.fig_nlms,
            aec_figures.fig_erle,
            aec_figures.fig_delay_dtd,
            aec_figures.fig_nonlinear,
            aec_figures.fig_hybrid,
        ]
        figures = {}

        def capture(figure, name):
            figures[name] = figure

        try:
            with patch.object(aec_figures, "save", side_effect=capture):
                for builder in builders:
                    builder()

            self.assertEqual(len(figures), 7)
            for name, figure in figures.items():
                with self.subTest(figure=name):
                    self.assertLessEqual(figure.get_size_inches()[0], 10.0)
                    ticks = [
                        tick
                        for axis in figure.axes
                        for tick in (
                            *axis.xaxis.get_major_ticks(),
                            *axis.xaxis.get_minor_ticks(),
                            *axis.yaxis.get_major_ticks(),
                            *axis.yaxis.get_minor_ticks(),
                        )
                    ]
                    tick_text_ids = {
                        id(label)
                        for tick in ticks
                        for label in (tick.label1, tick.label2)
                    }
                    key_texts = [
                        artist
                        for artist in figure.findobj(match=Text)
                        if artist.get_text().strip() and id(artist) not in tick_text_ids
                    ]
                    self.assertTrue(key_texts)
                    self.assertGreaterEqual(
                        min(text.get_fontsize() for text in key_texts),
                        11.0,
                    )

            figure26 = figures["fig26_aec_problem.png"]
            self.assertEqual(figure26.axes[0].get_xlabel(), "抽头索引 ℓ（采样）")
            self.assertEqual(len(figure26.axes), 3)
            self.assertEqual(figure26.axes[0].get_subplotspec().colspan.stop, 2)

            figure28 = figures["fig28_aec_nlms.png"]
            path_axis = next(axis for axis in figure28.axes if axis.get_title().startswith("(a)"))
            trend_axis = next(axis for axis in figure28.axes if axis.get_title().startswith("(b)"))
            step_axis = next(axis for axis in figure28.axes if axis.get_title().startswith("(c)"))
            trend_twin = next(
                axis for axis in figure28.axes
                if not axis.get_title() and axis.get_ylabel() == "ERLE (dB)"
            )
            self.assertEqual(path_axis.get_ylabel(), "滤波器系数（无量纲）")
            self.assertNotEqual(
                trend_axis.get_lines()[0].get_linestyle(),
                trend_twin.get_lines()[0].get_linestyle(),
            )
            self.assertEqual(
                len({line.get_linestyle() for line in step_axis.get_lines()}),
                3,
            )
            self.assertTrue(all(line.get_marker() != "None" for line in step_axis.get_lines()))

            figure31 = figures["fig31_aec_nonlinear.png"]
            erle_axis, mapping_axis = figure31.axes
            self.assertNotEqual(
                erle_axis.get_lines()[0].get_linestyle(),
                erle_axis.get_lines()[1].get_linestyle(),
            )
            self.assertEqual(mapping_axis.get_xlabel(), "归一化输入幅度（无量纲）")
            self.assertEqual(mapping_axis.get_ylabel(), "归一化输出幅度（无量纲）")
            mapping_note = next(
                text for text in mapping_axis.texts
                if text.get_text().startswith("输入幅度增大后")
            )
            self.assertIn("\n", mapping_note.get_text())
            self.assertGreaterEqual(mapping_note.get_fontsize(), 11.0)
            self.assertIs(mapping_note.get_transform(), mapping_axis.transAxes)
            self.assertLessEqual(mapping_note.get_position()[1], 0.10)
            figure31.canvas.draw()
            note_bounds = mapping_note.get_window_extent()
            axis_bounds = mapping_axis.get_window_extent()
            self.assertGreaterEqual(note_bounds.x0, axis_bounds.x0)
            self.assertLessEqual(note_bounds.x1, axis_bounds.x1)
            self.assertGreaterEqual(note_bounds.y0, axis_bounds.y0)
            self.assertLessEqual(note_bounds.y1, axis_bounds.y1)

            figure32 = figures["fig32_aec_hybrid_select.png"]
            self.assertEqual(len(figure32.axes), 1)
            self.assertFalse(figure32.axes[0].tables)
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_causal_delay_zero_fills_instead_of_wrapping(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        delayed = causal_delay(x, 2)

        np.testing.assert_array_equal(delayed, [0.0, 0.0, 1.0, 2.0])

    def test_causal_delay_handles_zero_and_delay_past_signal(self):
        x = np.array([1, 2, 3])

        np.testing.assert_array_equal(causal_delay(x, 0), x)
        np.testing.assert_array_equal(causal_delay(x, len(x)), np.zeros_like(x))
        with self.assertRaises(ValueError):
            causal_delay(x, -1)

    def test_block_erle_uses_power_ratio_in_decibels(self):
        echo = np.ones(1200)
        residual = np.full(1200, 0.1)

        times, erle = block_erle(echo, residual, blk=400)

        np.testing.assert_array_equal(times, np.array([0.0, 0.025, 0.05]))
        np.testing.assert_allclose(erle, 20.0, atol=1e-12)

    def test_block_erle_is_invariant_to_common_amplitude_scale(self):
        echo = np.linspace(0.2, 1.2, 1200)
        residual = 0.1 * echo
        _, reference = block_erle(echo, residual, blk=400)

        for scale in (1e-9, 1e9):
            _, scaled = block_erle(scale * echo, scale * residual, blk=400)
            np.testing.assert_allclose(scaled, reference, atol=1e-12)

    def test_block_erle_preserves_zero_power_semantics(self):
        echo = np.r_[np.ones(400), np.zeros(800)]
        residual = np.r_[np.zeros(400), np.ones(400), np.zeros(400)]

        _, erle = block_erle(echo, residual, blk=400)

        self.assertTrue(np.isposinf(erle[0]))
        self.assertTrue(np.isneginf(erle[1]))
        self.assertTrue(np.isnan(erle[2]))

    def test_block_erle_rejects_incompatible_inputs(self):
        with self.assertRaises(ValueError):
            block_erle(np.ones(800), np.ones(400), blk=400)
        with self.assertRaises(ValueError):
            block_erle(np.ones(800), np.ones(800), blk=0)

    def test_colored_signal_seed_is_repeatable_and_independent(self):
        first = colored_x(128, seed=101)
        _ = colored_x(128, seed=202)
        repeated = colored_x(128, seed=101)

        np.testing.assert_array_equal(first, repeated)

    def test_nlms_trace_records_initial_state_before_first_update(self):
        x = np.array([1.0, 0.0, 0.0])
        true_path = np.array([1.0])
        d = np.convolve(x, true_path, mode="full")[: len(x)]

        _e, snapshots, _w, mismatch, samples = nlms_adaptation_trace(
            x,
            d,
            taps=1,
            mu=0.5,
            snapshot_interval=1,
            true_path=true_path,
        )

        np.testing.assert_array_equal(samples, [0, 1, 2])
        np.testing.assert_array_equal(snapshots[0], [0.0])
        self.assertAlmostEqual(mismatch[0], 0.0, places=12)
        self.assertGreater(snapshots[1, 0], 0.0)
        self.assertLess(mismatch[1], mismatch[0])

    def test_nlms_trace_validates_freeze_and_true_path_shapes(self):
        x = np.ones(4)
        d = np.ones(4)

        with self.assertRaises(ValueError):
            nlms_adaptation_trace(x, d, taps=2, freeze=np.zeros(3, dtype=bool))
        with self.assertRaises(ValueError):
            nlms_adaptation_trace(x, d, taps=2, true_path=np.ones(3))

    def test_nlms_multitap_current_first_regressor_converges(self):
        rng = np.random.default_rng(6101)
        x = rng.standard_normal(3000)
        true_path = np.array([0.7, -0.2, 0.1])
        d = np.convolve(x, true_path, mode="full")[: len(x)]

        residual, _snapshots, estimate, _mismatch, _samples = nlms_adaptation_trace(
            x, d, taps=3, mu=0.5, true_path=true_path
        )

        np.testing.assert_allclose(estimate, true_path, atol=1e-6)
        self.assertLess(np.mean(residual[-500:] ** 2), 1e-12)

    def test_nlms_freeze_keeps_coefficients_unchanged(self):
        x = np.array([1.0, 0.5, -0.25, 0.75])
        d = np.array([0.8, -0.4, 0.2, 0.1])
        freeze = np.array([False, True, True, False])

        _e, snapshots, _w, _mismatch, _samples = nlms_adaptation_trace(
            x, d, taps=2, mu=0.5, freeze=freeze, snapshot_interval=1
        )

        self.assertGreater(np.linalg.norm(snapshots[1]), 0.0)
        np.testing.assert_array_equal(snapshots[1], snapshots[2])
        np.testing.assert_array_equal(snapshots[2], snapshots[3])

    def test_mask_metric_intervals_is_half_open_and_non_mutating(self):
        times = np.array([0.0, 0.5, 1.0, 1.5])
        values = np.arange(4.0)

        masked = mask_metric_intervals(times, values, [(0.5, 1.5)])

        np.testing.assert_array_equal(values, np.arange(4.0))
        self.assertFalse(np.isnan(masked[0]))
        self.assertTrue(np.isnan(masked[1:3]).all())
        self.assertFalse(np.isnan(masked[3]))
        with self.assertRaises(ValueError):
            mask_metric_intervals(times, values, [(1.0, 1.0)])

    def test_figure30_configuration_has_expected_peak_and_masked_double_talk(self):
        sample_count = int(1.6 * FS)
        taps = 128
        delay = 300
        path = np.exp(-np.arange(taps) / 25) * np.random.default_rng(3001).standard_normal(taps)
        reference = colored_x(sample_count, seed=3002)
        delayed = causal_delay(reference, delay)
        echo = np.convolve(delayed, path, mode="full")[:sample_count]
        near_end = np.zeros(sample_count)
        near_end[int(0.8 * FS):int(1.2 * FS)] = 0.7 * np.random.default_rng(3003).standard_normal(int(0.4 * FS))
        microphone = echo + near_end
        freeze = np.zeros(sample_count, dtype=bool)
        freeze[int(0.8 * FS):int(1.2 * FS)] = True

        aligned, _, _ = nlms_run(delayed, microphone, taps, 0.5, freeze)
        misaligned, _, _ = nlms_run(reference, microphone, taps, 0.5, freeze)
        times, aligned_erle = block_erle(echo, aligned)
        _, misaligned_erle = block_erle(echo, misaligned)
        segment = 4000
        correlation = np.correlate(microphone[:segment], reference[:segment], mode="full")
        lags = np.arange(correlation.size) - (segment - 1)
        peak_lag = int(lags[np.argmax(correlation)])
        aligned_plateau = float(np.mean(aligned_erle[(times > 0.45) & (times < 0.8)]))
        misaligned_plateau = float(np.mean(misaligned_erle[(times > 0.45) & (times < 0.8)]))
        view = mask_metric_intervals(times, aligned_erle, [(0.8, 1.2)])

        self.assertEqual(peak_lag, 331)
        self.assertAlmostEqual(aligned_plateau, 27.949, delta=0.05)
        self.assertAlmostEqual(misaligned_plateau, 1.855, delta=0.05)
        self.assertGreater(aligned_plateau, misaligned_plateau + 20.0)
        self.assertTrue(np.isnan(view[(times >= 0.8) & (times < 1.2)]).all())

    def test_figure30_uses_readable_two_row_layout_and_labels_both_erle_axes(self):
        captured = {}

        def capture(figure, name):
            captured["figure"] = figure
            captured["name"] = name

        with patch.object(aec_figures, "save", side_effect=capture):
            aec_figures.fig_delay_dtd()

        figure = captured["figure"]
        try:
            self.assertEqual(captured["name"], "fig30_aec_delay_dtd.png")
            self.assertEqual(len(figure.axes), 3)
            correlation_axis, alignment_axis, double_talk_axis = figure.axes

            correlation_spec = correlation_axis.get_subplotspec()
            alignment_spec = alignment_axis.get_subplotspec()
            double_talk_spec = double_talk_axis.get_subplotspec()
            self.assertEqual(
                (correlation_spec.rowspan.start, correlation_spec.rowspan.stop),
                (0, 1),
            )
            self.assertEqual(
                (correlation_spec.colspan.start, correlation_spec.colspan.stop),
                (0, 2),
            )
            self.assertEqual(
                (alignment_spec.rowspan.start, alignment_spec.colspan.start),
                (1, 0),
            )
            self.assertEqual(
                (double_talk_spec.rowspan.start, double_talk_spec.colspan.start),
                (1, 1),
            )
            self.assertEqual(alignment_axis.get_ylabel(), "ERLE (dB)")
            self.assertEqual(double_talk_axis.get_ylabel(), "ERLE (dB)")
        finally:
            plt.close(figure)

    def test_nonlinear_path_has_lower_late_erle_than_matched_linear_path(self):
        sample_count = int(3.0 * FS)
        taps = 256
        rng = np.random.default_rng(3101)
        path = np.exp(-np.arange(taps) / 60) * rng.standard_normal(taps)
        reference = colored_x(sample_count, seed=3102)
        reference = reference / np.std(reference) * 0.5
        linear_echo = np.convolve(reference, path, mode="full")[:sample_count]
        nonlinear_echo = np.convolve(
            np.tanh(1.1 * reference) / np.tanh(1.1), path, mode="full"
        )[:sample_count]

        linear_residual, _, _ = nlms_run(reference, linear_echo, taps, 0.3)
        nonlinear_residual, _, _ = nlms_run(reference, nonlinear_echo, taps, 0.3)
        linear_times, linear_erle = block_erle(linear_echo, linear_residual, blk=800)
        nonlinear_times, nonlinear_erle = block_erle(nonlinear_echo, nonlinear_residual, blk=800)

        linear_late = float(np.mean(linear_erle[linear_times > 2.2]))
        nonlinear_late = float(np.mean(nonlinear_erle[nonlinear_times > 2.2]))
        self.assertAlmostEqual(linear_late, 34.024, delta=0.05)
        self.assertAlmostEqual(nonlinear_late, 17.747, delta=0.05)
        self.assertGreater(linear_late, nonlinear_late + 5.0)

    def test_png_metadata_tracks_full_source_digest_without_timestamp(self):
        metadata = figure_metadata()
        source_path = Path(__file__).parents[1] / SOURCE_SCRIPT
        expected = hashlib.sha256(source_path.read_bytes()).hexdigest()

        self.assertEqual(metadata["SourceScript"], SOURCE_SCRIPT)
        self.assertEqual(metadata["SourceScriptDigest"], expected)
        self.assertEqual(len(metadata["SourceScriptDigest"]), 64)
        self.assertNotIn("Date", metadata)
        self.assertNotIn("Creation Time", metadata)

    def test_save_embeds_source_metadata_in_png(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1])
            with patch.object(aec_figures, "OUT", Path(temporary_directory)):
                save(figure, "metadata.png")

            with Image.open(Path(temporary_directory) / "metadata.png") as image:
                self.assertEqual(image.info["SourceScript"], SOURCE_SCRIPT)
                self.assertEqual(image.info["SourceScriptDigest"], figure_metadata()["SourceScriptDigest"])


if __name__ == "__main__":
    unittest.main()
