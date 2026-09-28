"""Chapter 1 checks from rational sums, half-angle identities and geometry."""
import json
import math
import unittest

import numpy as np

from codes.examples.chapter01_experiments import run_exercises
from codes.array_tutorial.audio_samples import alignment_error_case, prepare_exports, read_pcm16


class Chapter01ExperimentsTest(unittest.TestCase):
    def test_result_ids_and_finite_json(self):
        results = run_exercises()
        self.assertEqual(set(results), {'E01-04', 'E01-05', 'E01-06'})
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


if __name__ == '__main__':
    unittest.main()
