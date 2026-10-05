"""Independent arithmetic and real temporary PCM oracles for E12-06..20."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codes.chapters.ch00.core.audio_samples import math_block_case, pcm16_bytes, read_pcm16
from codes.chapters.appendix_a.core.math_foundations import (
    blockwise_circular_convolution, fft_overlap_add,
)
from codes.chapters.appendix_a.appendix_a_experiments import run_experiments
from codes.chapters.appendix_a.examples.generate_weighted_audio import generate
from tests.test_codes_appendix_a_boundaries import main_fixture


class AppendixAMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = main_fixture(Path(cls.temporary.name)/'repo')
        directory = Path(cls.temporary.name)/'weighted'
        generate(directory)
        cls.rows = run_experiments(repo_root=root, weighted_directory=directory)

    def test_exact_ids_and_json(self):
        self.assertEqual(set(self.rows), {f'E12-{number:02d}' for number in range(6, 21)})
        json.dumps(self.rows, allow_nan=False)

    def test_signed_bins_and_complex_norm(self):
        row = self.rows['E12-06']
        self.assertEqual(row['signed_frequency_hz_by_bin'],
                         [0, 1000, 2000, 3000, -4000, -3000, -2000, -1000])
        # Analytically, f_k = k f_s/N for the five retained real-FFT bins.
        self.assertEqual(row['one_sided_frequency_hz_by_bin'],
                         [0, 1000, 2000, 3000, 4000])
        self.assertEqual(row['real_cosine_nonzero_bins'], [1, 7])
        self.assertEqual(row['real_cosine_nonzero_coefficients'], [4, 4])
        # (-1)^n = exp(j 2*pi*4*n/8), so its unnormalized DFT is 8 at k=4 only.
        self.assertEqual(row['alternating_signal_nonzero_bins'], [4])
        self.assertEqual(row['alternating_signal_nonzero_coefficients'], [8])
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

    def test_absolute_and_trace_relative_loading_scales(self):
        row = self.rows['E12-13']
        self.assertEqual(row['covariance'], [[4, 0], [0, 1]])
        self.assertEqual(row['steering'], [1, 1])
        self.assertEqual((row['microphones'], row['absolute_loading'],
                          row['relative_coefficient'], row['scale_factor']),
                         (2, 1, .4, 10))

        # For diagonal d1,d2 and a=[1,1], w=[d2,d1]/(d1+d2).
        expected = (
            (1, 'absolute', 1, [[5, 0], [0, 2]], [2/7, 5/7], 5/2),
            (1, 'relative', 1, [[5, 0], [0, 2]], [2/7, 5/7], 5/2),
            (10, 'absolute', 1, [[41, 0], [0, 11]], [11/52, 41/52], 41/11),
            (10, 'relative', 10, [[50, 0], [0, 20]], [2/7, 5/7], 5/2),
        )
        self.assertEqual(len(row['cases']), len(expected))
        for case, (scale, kind, addition, matrix, weights, condition) in zip(row['cases'], expected):
            with self.subTest(scale=scale, kind=kind):
                self.assertEqual((case['input_scale'], case['kind'],
                                  case['effective_diagonal_addition']),
                                 (scale, kind, addition))
                np.testing.assert_array_equal(case['loaded_covariance'], matrix)
                np.testing.assert_allclose(case['weights'], weights, rtol=0, atol=1e-15)
                self.assertAlmostEqual(case['condition_2'], condition, places=12)
                self.assertAlmostEqual(case['target_response'], 1, places=12)

    def test_complex_ls_residual_and_conjugate(self):
        row=self.rows['E12-14']
        np.testing.assert_allclose(row['solution']['real'],[.5],atol=2e-16)
        np.testing.assert_allclose(row['solution']['imag'],[-.5],atol=2e-16)
        np.testing.assert_allclose(row['residual']['real'],[-.5,-.5],atol=2e-16)
        np.testing.assert_allclose(row['residual']['imag'],[-.5,.5],atol=2e-16)
        np.testing.assert_allclose(row['hermitian_orthogonality']['real'],[0],atol=1e-15)
        self.assertEqual(row['transpose_gram'],{'real':[[0.]],'imag':[[0.]]})
        self.assertAlmostEqual(row['residual_squared_sum'],1)

    def test_weighted_ls_is_orthogonal_in_its_actual_metric(self):
        row=self.rows['E12-15']['cases']
        self.assertEqual(row['ols']['solution'],[1])
        self.assertAlmostEqual(row['gls']['solution'][0],2/5)
        np.testing.assert_allclose(row['gls']['residual'],[2/5,-8/5],atol=1e-16)
        self.assertAlmostEqual(row['gls']['weighted_orthogonality'][0],0)
        self.assertAlmostEqual(row['gls']['ordinary_orthogonality'][0],-6/5)
        self.assertAlmostEqual(row['gls']['weighted_residual_squared_sum'],4/5)
        self.assertAlmostEqual(row['gls']['parameter_variance'],4/5)
        self.assertEqual(row['ols']['parameter_variance'],5/4)
        # Evaluate both residuals with the same known C^-1 metric:
        # OLS: 1 + 1/4; GLS: (2/5)^2 + (1/4)*(-8/5)^2.
        self.assertEqual(row['ols']['weighted_residual_squared_sum'],2)
        self.assertEqual(row['ols']['common_noise_precision_cost'],5/4)
        self.assertAlmostEqual(row['gls']['common_noise_precision_cost'],4/5)

    def test_rcond_rank_and_ridge_are_different_rules(self):
        row=self.rows['E12-16'];first,second=row['cases']
        self.assertEqual(row['numpy_version'],np.__version__)
        self.assertEqual(first['effective_singular_value_threshold'],1e-10)
        self.assertEqual(second['effective_singular_value_threshold'],1e-6)
        self.assertEqual((first['rank'],second['rank']),(2,1))
        self.assertEqual(first['solution'],[1,1]);self.assertEqual(second['solution'],[1,0])
        self.assertEqual(first['returned_residual_array'],[])
        self.assertEqual(second['returned_residual_array'],[])
        self.assertEqual(first['actual_residual_norm'],0)
        self.assertEqual(second['actual_residual_norm'],1e-8)
        self.assertAlmostEqual(row['ridge_solution'][0],1)
        self.assertAlmostEqual(row['ridge_solution'][1],.5)
        self.assertAlmostEqual(row['ridge_residual_squared_sum']/1e-16,.25)
        self.assertAlmostEqual(row['ridge_penalty']/1e-16,1.25)

    def test_complex_correlation_phase_and_conjugate(self):
        row=self.rows['E12-17']
        self.assertEqual(row['r12'],{'real':[0,0,0,2,0],'imag':[0,0,-1,0,1]})
        self.assertEqual(row['without_conjugate'],{'real':[0]*5,'imag':[0,0,1,0,1]})
        self.assertEqual(row['magnitude_peak_lag_samples'],1)
        self.assertEqual(row['r21'],{'real':[0,2,0,0,0],'imag':[-1,0,1,0,0]})
        self.assertEqual(row['r21_magnitude_peak_lag_samples'],-1)

    def test_eigh_uses_triangle_not_original_asymmetric_input(self):
        lower,upper,diagonal=self.rows['E12-18']['cases']
        self.assertEqual(lower['eigenvalues'],[1,2])
        # Upper Hermitian reconstruction [[1,100],[100,2]] has trace3
        # and gap sqrt(1+4*100^2), independently from the called solver.
        np.testing.assert_allclose(upper['eigenvalues'],[(3-np.sqrt(40001))/2,(3+np.sqrt(40001))/2],atol=2e-14)
        for case in (lower,upper):
            self.assertAlmostEqual(case['original_matrix_residual_frobenius'],100)
            self.assertLess(case['effective_matrix_residual_frobenius'],1e-12)
        self.assertEqual(diagonal['eigenvalues'],[1,2])
        self.assertAlmostEqual(diagonal['original_matrix_residual_frobenius'],np.sqrt(34))
        for case in (lower,diagonal):
            np.testing.assert_array_equal(case['eigenvectors']['real'],np.eye(2))
            np.testing.assert_array_equal(case['eigenvectors']['imag'],np.zeros((2,2)))
        # Independently solve each two-row eigen-equation. For eigenvalue l,
        # [100,l-1] is a real eigenvector; its overall sign is not unique.
        vectors=np.array(upper['eigenvectors']['real'])
        np.testing.assert_array_equal(upper['eigenvectors']['imag'],np.zeros((2,2)))
        for column,value in enumerate(((3-np.sqrt(40001))/2,(3+np.sqrt(40001))/2)):
            expected=np.array([100.,value-1.])
            expected/=np.sqrt(100**2+(value-1)**2)
            actual=vectors[:,column]
            self.assertAlmostEqual(abs(float(actual@expected)),1,places=14)
            self.assertAlmostEqual(float(actual@actual),1,places=14)

    def test_weighted_audio_anchor_reads_integer_report(self):
        row=self.rows['E12-19']
        self.assertEqual(row['parameters']['source_score'],[1600,30400])
        pcm=row['pcm_analysis']
        self.assertEqual(pcm['integer_reference_squared_sum'],618489148320)
        for name,E in [('weighted_ols.wav',17394760800),('weighted_gls.wav',11131529760),('weighted_reversed.wav',36180064800)]:
            candidate=pcm['candidates'][name]
            self.assertEqual(candidate['integer_error_squared_sum'],E)
            self.assertEqual(candidate['nmse'],E/618489148320)


if __name__ == '__main__':
    unittest.main()
