"""Independent arithmetic and PCM oracles for Appendix A E12-06..12."""

import json
import unittest

import numpy as np

from codes.chapters.ch00.core.audio_samples import math_block_case, pcm16_bytes, read_pcm16
from codes.chapters.appendix_a.core.math_foundations import (
    blockwise_circular_convolution, fft_overlap_add,
)
from codes.chapters.appendix_a.appendix_a_experiments import run_experiments


class AppendixAMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run_experiments()

    def test_exact_ids_and_json(self):
        self.assertEqual(set(self.rows), {f'E12-{number:02d}' for number in range(6, 13)})
        json.dumps(self.rows, allow_nan=False)

    def test_signed_bins_and_complex_norm(self):
        row = self.rows['E12-06']
        self.assertEqual(row['signed_frequency_hz_by_bin'],
                         [0, 1000, 2000, 3000, -4000, -3000, -2000, -1000])
        self.assertEqual(row['real_cosine_nonzero_bins'], [1, 7])
        self.assertEqual(row['real_cosine_nonzero_coefficients'], [4, 4])
        self.assertEqual(row['nyquist_bin']['frequency_magnitude_hz'], 4000)
        row = self.rows['E12-07']
        self.assertEqual(row['hermitian_norm_squared'], 2)
        self.assertEqual(row['transpose_square'], 0)

    def test_overlap_add_and_wrong_circular_outputs(self):
        row = self.rows['E12-08']
        self.assertEqual(row['linear_output'], [1, 2.5, 4, 5.5, 2])
        self.assertEqual(row['block_circular_output'], [2, 2.5, 5, 5.5])
        # Independent direct causal convolution, including the final tail.
        x = [1, 2, 3, 4]
        expected = [0.] * 5
        for n, value in enumerate(x):
            expected[n] += value
            expected[n + 1] += .5 * value
        np.testing.assert_allclose(fft_overlap_add(x, [1, .5], 3), expected,
                                   atol=2e-15)
        np.testing.assert_allclose(fft_overlap_add([3], [1, .5], 3), [3, 1.5],
                                   atol=2e-15)
        np.testing.assert_allclose(blockwise_circular_convolution([1, 2], [1, .5], 2),
                                   [2, 2.5], atol=2e-15)

    def test_convolution_rejects_complex_nonfinite_or_illegal_block(self):
        for function in (fft_overlap_add, blockwise_circular_convolution):
            for x, h, size in (([1+1j], [1], 2), ([1], [1j], 2),
                               ([], [1], 2), ([1], [], 2),
                               ([np.nan], [1], 2), ([1], [np.inf], 2),
                               ([1], [1], True), ([1], [1], 0)):
                with self.subTest(function=function.__name__, x=x, h=h, size=size):
                    with self.assertRaises(ValueError):
                        function(x, h, size)
        with self.assertRaises(ValueError):
            blockwise_circular_convolution([1, 2], [1, 0, 1], 2)

    def test_correlation_covariance_conditioning_and_sdw(self):
        row = self.rows['E12-09']
        self.assertEqual(row['r12'], [0, 0, 0, 1, 0])
        self.assertEqual(row['r21'], [0, 1, 0, 0, 0])
        self.assertEqual((row['r12_peak_lag_samples'], row['r21_peak_lag_samples']), (1, -1))
        row = self.rows['E12-10']
        np.testing.assert_array_equal(row['uncentered_second_moment'],
                                      np.diag([.5, .5, 0]))
        np.testing.assert_array_equal(row['centered_covariance'],
                                      [[.25, -.25, 0], [-.25, .25, 0], [0, 0, 0]])
        self.assertEqual((row['uncentered_rank'], row['centered_rank']), (2, 1))
        self.assertEqual(row['centered_eigenvalues'], [0, 0, .5])
        row = self.rows['E12-11']
        self.assertEqual((row['condition_A'], row['condition_normal']), (1e4, 1e8))
        self.assertEqual(row['recovered_x'], [1, 1])
        row = self.rows['E12-12']
        self.assertEqual(row['singular_noise_weights'], [[1, 0]] * 3)
        for mu, weights in zip(row['mus'], row['identity_noise_weights']):
            self.assertAlmostEqual(weights[0], 1/(1+mu), places=14)
            self.assertEqual(weights[1], 0)

    def test_audio_impulses_and_pcm_readback(self):
        case = math_block_case()
        p = case['parameters']
        self.assertEqual(p['pulse_positions_samples'], [500 + 3072*i for i in range(10)])
        self.assertEqual((p['block_size'], p['filter_nonzero_samples']), (512, [0, 120]))
        self.assertEqual(case['export_gain_override'], 1)
        signals = case['signals']
        expected = {name: np.zeros(32000) for name in signals}
        for position in p['pulse_positions_samples']:
            expected['math_block_dry'][position] = .3
            expected['math_block_linear'][position] = .3
            expected['math_block_linear'][position + 120] = .18
            expected['math_block_circular'][position] = .3
            expected['math_block_circular'][position - 392] = .18
        for name, signal in signals.items():
            np.testing.assert_allclose(signal, expected[name], atol=1e-14)
            rate, actual_pcm = read_pcm16(pcm16_bytes(signal))
            self.assertEqual(rate, 16000)
            self.assertEqual(actual_pcm.shape, (1, 32000))
            expected_pcm = np.rint(expected[name] * 32768) / 32768
            np.testing.assert_array_equal(actual_pcm[0], expected_pcm)
        pcm = case['pcm_analysis']
        self.assertEqual(pcm['first_dry_sample_500'], round(.3 * 32768) / 32768)
        self.assertEqual(pcm['first_linear_echo_sample_620'], round(.18 * 32768) / 32768)
        self.assertEqual(pcm['first_wrong_wrap_sample_108'], round(.18 * 32768) / 32768)
        self.assertEqual(pcm['first_wrong_sample_620'], 0)
        self.assertEqual(pcm['wrong_minus_linear_nonzero_count'], 20)


if __name__ == '__main__':
    unittest.main()
