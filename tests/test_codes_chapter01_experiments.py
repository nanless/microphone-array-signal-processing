"""Chapter 1 checks from rational sums, half-angle identities and geometry."""
import json
import math
import hashlib
import io
from pathlib import Path
import unittest
import wave

import numpy as np

from codes.chapters.ch01.chapter01_experiments import run_exercises
from codes.chapters.ch00.core.audio_samples import alignment_error_case, prepare_exports, read_pcm16


class Chapter01ExperimentsTest(unittest.TestCase):
    def test_result_ids_and_finite_json(self):
        results = run_exercises()
        self.assertEqual(set(results), {'E01-04', 'E01-05', 'E01-06', 'E01-07', 'E01-08', 'E01-09', 'E01-10'})
        json.dumps(results, allow_nan=False)

    def test_short_record_cross_terms_are_not_assumed_zero(self):
        full, cropped = run_exercises()['E01-04']['cases']
        self.assertEqual(full['products'], [1, -1, -1, 1])
        self.assertEqual(full['average_squared'], [1, 0, 0, 1])
        self.assertEqual(full['output_mean_square'], 1 / 2)
        self.assertEqual(full['cross_second_moment'], 0)
        self.assertEqual(cropped['average_squared'], [1, 0, 0])
        self.assertEqual(cropped['cross_second_moment'], -1 / 3)
        self.assertEqual(cropped['diagonal_contribution'], 1 / 2)
        self.assertEqual(cropped['cross_contribution'], -1 / 6)
        self.assertEqual(cropped['output_mean_square'], 1 / 3)
        self.assertAlmostEqual(cropped['gain_relative_to_channel_1_db'], 10 * math.log10(3))

    def test_residual_delay_has_independent_half_angle_answers(self):
        result = run_exercises()['E01-05']
        low, high = result['cases']
        # cos(pi/16)^2=(2+sqrt(2+sqrt(2)))/4, cos(pi/4)^2=1/2.
        low_power = (2 + math.sqrt(2 + math.sqrt(2))) / 4
        for row, expected in ((low, low_power), (high, .5)):
            self.assertAlmostEqual(row['target_power_ratio'], expected)
            self.assertAlmostEqual(row['projected_target_amplitude_ratio'], math.sqrt(expected), places=12)
            self.assertAlmostEqual(row['target_level_change_db'], 10 * math.log10(expected))
            self.assertEqual(row['aligned_target_amplitude_ratio'], 1)
        self.assertEqual(result['residual_delay_seconds'], 1 / 16000)
        self.assertEqual(result['unaligned_common_delay_samples'], .5)
        self.assertEqual(result['aligned_common_delay_samples'], 1)

    def test_woodworth_zero_side_and_left_right_symmetry(self):
        rows = {row['azimuth_deg']: row for row in run_exercises()['E01-06']['cases']}
        self.assertEqual(rows[0]['sphere_itd_us'], 0)
        self.assertEqual(rows[0]['free_point_itd_us'], 0)
        # At the side, the extra ray length is quarter circumference + radius.
        side_length = .0875 * math.pi / 2 + .0875
        self.assertAlmostEqual(rows[90]['sphere_itd_us'], side_length / 343 * 1e6)
        self.assertAlmostEqual(rows[90]['free_point_itd_us'], .175 / 343 * 1e6)
        self.assertAlmostEqual(rows[30]['free_point_itd_us'], .0875 / 343 * 1e6)
        self.assertAlmostEqual(rows[1]['sphere_itd_us'], 8.90451504010888)
        for angle in (1, 30, 90):
            for key in ('sphere_itd_us', 'free_point_itd_us'):
                self.assertAlmostEqual(rows[-angle][key], -rows[angle][key])

    def test_audio_alignment_is_causal_and_preserves_reference(self):
        case = alignment_error_case()
        signal = case['signals']
        early, later = signal['alignment_array']
        self.assertEqual(later[0], 0)
        np.testing.assert_array_equal(later[1:], early[:-1])
        np.testing.assert_array_equal(signal['alignment_aligned'], later)
        np.testing.assert_array_equal(signal['alignment_reference'], later)
        self.assertEqual(case['parameters']['seed'], None)
        self.assertEqual(signal['alignment_array'].shape, (2, 32000))

    def test_pcm_tone_amplitudes_and_gain_match_independent_identities(self):
        files, groups = prepare_exports({'alignment_error': alignment_error_case()})
        self.assertEqual(len(files), 4)
        self.assertEqual(groups['alignment_error']['common_export_gain'], 1.)
        expected_low = .18 * math.sqrt((2 + math.sqrt(2 + math.sqrt(2))) / 4)
        expected_high = .18 / math.sqrt(2)
        index = np.arange(1600, 30400)
        for stem, expected in [('alignment_reference', [.18, .18]),
                               ('alignment_aligned', [.18, .18]),
                               ('alignment_unaligned', [expected_low, expected_high])]:
            fs, decoded = read_pcm16(files[stem + '.wav'][0])
            self.assertEqual(fs, 16000)
            self.assertEqual(decoded.shape, (1, 32000))
            amplitudes = [2 * abs(np.mean(decoded[0, index] * np.exp(-2j * np.pi * f * index / fs)))
                          for f in (1000, 4000)]
            # DFT magnitude error <= twice the pointwise PCM rounding bound.
            np.testing.assert_allclose(amplitudes, expected, atol=1/32768, rtol=0)
        self.assertEqual(files['alignment_reference.wav'][0], files['alignment_aligned.wav'][0])
        _, stereo = read_pcm16(files['alignment_array.wav'][0])
        self.assertEqual(stereo.shape, (2, 32000))

    def test_published_alignment_scores_match_actual_pcm_real_projection(self):
        result = run_exercises()['E01-05']['published_audio']
        root = Path(__file__).resolve().parents[1]
        index = np.arange(1600, 30400)
        measured = {}
        for stem, record in result['assets'].items():
            raw = (root / record['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record['sha256'])
            with wave.open(io.BytesIO(raw), 'rb') as stream:
                self.assertEqual(stream.getframerate(), 16000)
                self.assertEqual(stream.getnframes(), 32000)
                self.assertEqual(stream.getsampwidth(), 2)
                channels = stream.getnchannels()
                samples = np.frombuffer(stream.readframes(32000), dtype='<i2').reshape(-1, channels).T / 32768
            # Orthogonal integer-period real projections, rather than the
            # implementation's multi-column least-squares solver.
            amplitudes = []
            for frequency in (1000, 4000):
                phase = 2 * np.pi * frequency * index / 16000
                a = 2 * np.mean(samples[0, index] * np.sin(phase))
                b = 2 * np.mean(samples[0, index] * np.cos(phase))
                amplitudes.append(float(np.hypot(a, b)))
            measured[stem] = amplitudes
            np.testing.assert_allclose(record['tone_amplitudes_by_channel'][0], amplitudes, atol=1e-14, rtol=0)
        float_cases = result['float_measurements']['cases']
        pcm_cases = result['pcm_measurements']['cases']
        expected = [math.sqrt((2 + math.sqrt(2 + math.sqrt(2))) / 4), 1 / math.sqrt(2)]
        for i, (float_case, pcm_case) in enumerate(zip(float_cases, pcm_cases)):
            self.assertAlmostEqual(float_case['unaligned_amplitude_ratio'], expected[i], places=12)
            self.assertAlmostEqual(pcm_case['reference_amplitude'], measured['alignment_reference'][i], places=13)
            self.assertAlmostEqual(pcm_case['unaligned_amplitude_ratio'],
                                   measured['alignment_unaligned'][i] / measured['alignment_reference'][i], places=13)
            self.assertEqual(pcm_case['aligned_amplitude_ratio'], 1.)
        # Quantization has an observable effect; do not silently substitute
        # analytic floating-point values for actual published PCM measurements.
        self.assertGreater(abs(pcm_cases[0]['unaligned_amplitude_ratio'] - expected[0]), 1e-6)
        self.assertEqual(result['scoring_samples'], 28800)

    def test_cross_terms_and_same_mixture_have_independent_rational_answers(self):
        result = run_exercises()['E01-07']
        for row, cross, total in zip(result['illustrative_cross_terms'], (0, 1, -1), (2, 4, 0)):
            self.assertEqual(row['signal_mean_square'], 1)
            self.assertEqual(row['noise_mean_square'], 1)
            self.assertEqual(row['cross_second_moment'], cross)
            self.assertEqual(row['mixture_mean_square'], total)
            self.assertEqual(row['snr_db'], 0)
        first, second = result['cases']
        self.assertEqual(first['mixture'], second['mixture'])
        self.assertEqual(first['mixture'], [2, 0, -2, 0])
        for row, ps, pv, cross, snr in ((first, 1/2, 1/2, 1/2, 0),
                                       (second, 9/8, 1/8, 3/8, 10 * math.log10(9))):
            self.assertEqual(row['signal_mean_square'], ps)
            self.assertEqual(row['noise_mean_square'], pv)
            self.assertEqual(row['cross_second_moment'], cross)
            self.assertEqual(row['mixture_mean_square'], 2)
            self.assertEqual(ps + pv + 2 * cross, 2)
            self.assertAlmostEqual(row['snr_db'], snr)

    def test_white_noise_gain_counts_actual_noise_and_preserves_target(self):
        equal, aggressive = run_exercises()['E01-09']['cases']
        for row, weights, expected_noise, expected_gain in ((equal, [.5, .5], .5, 2),
                                                         (aggressive, [2, -1], 5, .2)):
            self.assertEqual(row['weights'], weights)
            self.assertEqual(row['target_amplitude_gain'], 1)
            self.assertEqual(row['output_noise_mean_square'], expected_noise)
            self.assertEqual(row['white_noise_gain_linear'], expected_gain)
            self.assertAlmostEqual(row['white_noise_gain_db'], 10 * math.log10(expected_gain))

    def test_binaural_exercise_reads_all_five_actual_assets(self):
        result = run_exercises()['E01-08']
        self.assertEqual(set(result['samples']), {'reference', 'itd_only', 'ild_only', 'consistent', 'conflicting'})
        self.assertEqual(result['common_export_gain'], 1.)
        for name, expected_lag, sign in (('reference', 0, 0), ('itd_only', 8, 0),
                                         ('ild_only', 0, 1), ('consistent', 8, 1), ('conflicting', 8, -1)):
            row = result['samples'][name]
            pcm, floats = row['pcm_measurements'], row['float_measurements']
            self.assertEqual(pcm['estimated_itd_samples'], expected_lag)
            self.assertEqual(pcm['truth_itd_samples'], expected_lag)
            self.assertAlmostEqual(floats['ild_right_minus_left_db'], sign * 20 * math.log10(2))
            self.assertAlmostEqual(pcm['ild_right_minus_left_db'], sign * 20 * math.log10(2), delta=.0002)
            self.assertEqual(pcm['correlation_window_samples'], 28800)
            self.assertEqual(pcm['power_window_samples'], 28800)

    def test_spectral_exercise_reads_four_assets_and_keeps_nonzero_conditions(self):
        result = run_exercises()['E01-10']
        manifest = result['published_audio']
        self.assertEqual(set(manifest['files']), {'flat_source.wav', 'flat_stereo.wav',
                                                'tilted_source.wav', 'tilted_stereo.wav'})
        self.assertEqual(manifest['common_export_gain'], 1.)
        self.assertEqual(set(manifest['samples']), {'flat', 'tilted'})
        cosine = math.sqrt(2 + math.sqrt(2)) / 2
        for row, sign in zip(result['analytic_spectral'], (1, -1)):
            self.assertAlmostEqual(row['left_power_gain'], 1.25 + sign*cosine)
            self.assertAlmostEqual(row['right_power_gain'], 1.25 - sign*cosine)
        self.assertTrue(all(row['finite_metrics_rejected'] for row in result['zero_support_controls']))


if __name__ == '__main__':
    unittest.main()
