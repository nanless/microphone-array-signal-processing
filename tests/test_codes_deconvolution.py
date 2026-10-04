"""Independent FIR/DFT controls; never download, generate WAVs or rewrite assets."""
import cmath
import math
import unittest
from fractions import Fraction

import numpy as np

from codes.chapters.ch02.core.deconvolution import (
    MAX_FFT_LENGTH, build_sweep_cases, regularized_inverse, score_impulse_response)


class DeconvolutionTest(unittest.TestCase):
    def test_four_bin_regularized_answer_and_exact_normal_equations(self):
        result = regularized_inverse([1, .5], [1, .5, .5, .25], n_fft=4, regularization=.25)
        expected = [Fraction(11, 15), Fraction(3, 20), Fraction(19, 60), Fraction(3, 20)]
        np.testing.assert_allclose(result['impulse_response'], [float(v) for v in expected], atol=2e-15)
        # Independent circular convolution matrix and exact rational stationarity.
        C = [[Fraction(1) if i == j else Fraction(1, 2) if i == (j+1) % 4
              else Fraction(0) for j in range(4)] for i in range(4)]
        y = [Fraction(1), Fraction(1, 2), Fraction(1, 2), Fraction(1, 4)]
        residual = [sum(C[i][j]*expected[j] for j in range(4))-y[i] for i in range(4)]
        for j in range(4):
            self.assertEqual(sum(C[i][j]*residual[i] for i in range(4))+expected[j]/4, 0)

    def test_noiseless_full_tail_and_distinct_known_support_problem(self):
        h = [1, 0, .5]
        result = regularized_inverse([1, .5], [1, .5, .5, .25], n_fft=4, regularization=0)
        np.testing.assert_allclose(result['impulse_response'], [1, 0, .5, 0], atol=1e-15)
        # Short-support Tikhonov is a different 4x3 linear problem.
        C = np.array([[1, 0, 0], [.5, 1, 0], [0, .5, 1], [0, 0, .5]])
        q = np.array([67/84, 3/28, 8/21])
        np.testing.assert_allclose(C.T@(C@q-np.array([1, .5, .5, .25]))+.25*q, 0, atol=3e-16)
        circular_regularized = regularized_inverse([1, .5], [1, .5, .5, .25],
                                                  n_fft=4, regularization=.25)
        np.testing.assert_allclose(q-circular_regularized['impulse_response'][:3],
                                   [9/140, -3/70, 9/140], atol=2e-16)
        score = score_impulse_response(result['impulse_response'], h, support_length=3)
        self.assertLess(score['full_ir_relative_error_energy'], 1e-28)

    def test_spectral_zero_not_a_full_linear_toeplitz_nullspace(self):
        with self.assertRaisesRegex(ValueError, 'zero excitation'):
            regularized_inverse([1, 1], [1, 1, .5, .5], n_fft=4, regularization=0)
        C = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
        self.assertEqual(round(np.linalg.det(C[:3])), 1)
        np.testing.assert_array_equal(C@np.array([1, 0, .5]), [1, 1, .5, .5])
        np.testing.assert_array_equal(np.convolve([1, 1], [1, -1, 1, -1]), [1, 0, 0, 0, -1])

    def test_positive_regularization_zero_excitation_has_explicit_status(self):
        result = regularized_inverse([0, 0], [1, 2, 3], n_fft=4, regularization=.25)
        self.assertTrue(result['no_excitation'])
        self.assertEqual(result['zero_excitation_bins'], [0, 1, 2])
        np.testing.assert_array_equal(result['impulse_response'], np.zeros(4))

    def test_extreme_signal_scale_does_not_confuse_power_underflow_with_zero(self):
        for scale in (1e-200, 1., 1e200):
            with self.subTest(scale=scale):
                result = regularized_inverse([scale], [2*scale], n_fft=4, regularization=0)
                np.testing.assert_allclose(result['impulse_response'], [2, 0, 0, 0], atol=1e-15)
                self.assertFalse(result['no_excitation'])
                self.assertEqual(result['zero_excitation_bins'], [])
        self.assertTrue(regularized_inverse([1e-200], [2e-200], n_fft=4,
                                          regularization=0)['excitation_power_underflow'])
        result = regularized_inverse([1e200], [2e200], n_fft=4, regularization=1e300)
        np.testing.assert_allclose(result['impulse_response'], [2, 0, 0, 0], atol=1e-15)
        self.assertFalse(result['excitation_power_representable'])
        result = regularized_inverse([1e-200], [2e-200], n_fft=4, regularization=1e-300)
        self.assertAlmostEqual(result['impulse_response'][0]/2e-100, 1., places=14)

    def test_nonzero_numerator_underflow_is_rescaled_before_division(self):
        result = regularized_inverse([1e-150], [1e-200], n_fft=4, regularization=1e-300)
        np.testing.assert_allclose(result["impulse_response"], [5e-51, 0, 0, 0], rtol=2e-15, atol=0)

    def test_finite_power_plus_regularization_overflow_is_rescaled(self):
        result = regularized_inverse([1e154], [1e154], n_fft=4, regularization=1e308)
        np.testing.assert_allclose(result['impulse_response'], [.5, 0, 0, 0], atol=2e-16)

    def test_invalid_types_shapes_lengths_and_regularization(self):
        for invalid in ([True], ['1'], [1j], [np.nan], [np.inf], [], [[1.]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                regularized_inverse(invalid, [1], n_fft=4, regularization=.25)
            with self.subTest(response=invalid), self.assertRaises(ValueError):
                regularized_inverse([1], invalid, n_fft=4, regularization=.25)
        for n in (True, 1, 2.5, MAX_FFT_LENGTH+1):
            with self.subTest(n_fft=n), self.assertRaises(ValueError):
                regularized_inverse([1], [1, 2, 3], n_fft=n, regularization=.25)
        for epsilon in (True, '1', 1j, -1., np.inf, np.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                regularized_inverse([1], [1], n_fft=4, regularization=epsilon)

    def test_input_arrays_remain_unchanged_and_odd_fft_is_explicit(self):
        u, y = np.array([1., .5]), np.array([1., .5, .5, .25])
        u0, y0 = u.copy(), y.copy()
        result = regularized_inverse(u, y, n_fft=5, regularization=0)
        np.testing.assert_allclose(result['impulse_response'], [1, 0, .5, 0, 0], atol=1e-15)
        np.testing.assert_array_equal(u, u0)
        np.testing.assert_array_equal(y, y0)

    def test_full_score_preserves_outside_support_error(self):
        result = score_impulse_response([1, .1, .4, .2], [1, 0, .5], support_length=3)
        self.assertAlmostEqual(result['reference_energy'], 1.25)
        self.assertAlmostEqual(result['full_ir_relative_error_energy'], .048)
        self.assertAlmostEqual(result['true_support_relative_error_energy'], .016)
        self.assertAlmostEqual(result['outside_support_energy'], .04)
        self.assertFalse(result['delay_fit'])
        self.assertFalse(result['gain_fit'])
        self.assertEqual(result['full_interval_samples'], [0, 4])

    def test_score_rejects_zero_nonrepresentable_energy_or_hidden_reference_tail(self):
        for ref in ([0.], [1e-200], [1e200]):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                score_impulse_response([0., 0.], ref, support_length=1)
        for support in (0, 4, True, 1.5):
            with self.subTest(support=support), self.assertRaises(ValueError):
                score_impulse_response([1, 0, 1], [1, 0, 1], support_length=support)
        with self.assertRaises(ValueError):
            score_impulse_response([1, 0, 1], [1, 0, 1], support_length=2)
        with self.assertRaises(ValueError):
            score_impulse_response([1], [1, 0], support_length=1)
        with self.assertRaisesRegex(ValueError, "relative error"):
            score_impulse_response([1e150], [1e-150], support_length=1)
        with self.assertRaisesRegex(ValueError, "underflows"):
            score_impulse_response([1e-150, 1e-200], [1e-150, 0.], support_length=2)

    def test_score_rejects_positive_ratio_and_component_energy_underflow(self):
        # D=10**200 and E=10**-200 are individually representable, but
        # the strictly positive exact ratio 10**-400 cannot be float64.
        with self.assertRaisesRegex(ValueError, "positive IR relative error underflows"):
            score_impulse_response([1e100, 1e-100], [1e100], support_length=1)
        # A unit error in one partition must not hide the other partition's
        # nonzero 10**-200 sample whose square underflows independently.
        for estimate in ([1., 1e-200, 1.], [1., 1., 1e-200]):
            with self.subTest(estimate=estimate), self.assertRaisesRegex(ValueError, "error energy underflows"):
                score_impulse_response(estimate, [1.], support_length=2)

    def test_sweep_samples_against_scalar_phase_and_full_sparse_convolution(self):
        case = build_sweep_cases()
        x, signals = case['source'], case['signals']
        self.assertEqual(x.size, 32000)
        self.assertTrue(all(v.size == 32160 for v in signals.values()))
        for n in (0, 1, 159, 160, 4321, 15999, 31839, 31840, 31998, 31999):
            phase = 2*math.pi*100*2/math.log(60)*math.expm1(n/16000*math.log(60)/2)
            envelope = min(1, n/160, (31999-n)/160)
            self.assertAlmostEqual(x[n], .2*math.sin(phase)*envelope, places=15)
        expected = np.array([(.8*x[n] if n < 32000 else 0)
                             +(.25*x[n-160] if 0 <= n-160 < 32000 else 0)
                             for n in range(32160)])
        np.testing.assert_allclose(signals['sweep_complete'], expected, atol=3e-17, rtol=1e-15)
        np.testing.assert_array_equal(signals['sweep_cut'][:32000], expected[:32000])
        np.testing.assert_array_equal(signals['sweep_cut'][32000:], np.zeros(160))
        np.testing.assert_allclose(signals['sweep_noisy']-signals['sweep_complete'],
                                   case['noise'], atol=2e-17)
        self.assertGreater(float(np.dot(expected[32000:], expected[32000:])), .06)
        self.assertAlmostEqual(case['parameters']['last_instantaneous_frequency_hz'],
                               100*math.exp(31999/16000*math.log(60)/2), places=10)

    def test_sweep_bias_noise_and_full_frequency_parseval_identity(self):
        case = build_sweep_cases()
        truth = np.pad(case['impulse_response'], (0, 65536-161))
        U = np.fft.rfft(case['source'], n=65536)
        weights = np.full(U.size, 2.); weights[0] = weights[-1] = 1.
        for name, signal in case['signals'].items():
            for rho in (0., 1e-8, 1e-6, 1e-4):
                with self.subTest(case=name, rho=rho):
                    epsilon = rho*float(max(abs(U)**2))
                    result = regularized_inverse(case['source'], signal, n_fft=65536, regularization=epsilon)
                    score = score_impulse_response(result['impulse_response'], case['impulse_response'], support_length=161)
                    spectral_error = np.sum(weights*abs(result['estimated_spectrum']-np.fft.rfft(truth))**2)/65536
                    self.assertAlmostEqual(score['error_energy']/max(1., score['error_energy']),
                                           float(spectral_error)/max(1., score['error_energy']), places=12)
                    self.assertAlmostEqual(score['error_energy'], score['support_error_energy']+score['outside_support_energy'], delta=max(1e-14, score['error_energy']*1e-12))
                    if name == 'sweep_complete' and rho == 0:
                        self.assertLess(score['full_ir_relative_error_energy'], 1e-20)
                    if name == 'sweep_complete' and rho == 1e-4:
                        self.assertGreater(score['full_ir_relative_error_energy'], .24)
                    if name == 'sweep_noisy' and rho == 0:
                        self.assertGreater(score['full_ir_relative_error_energy'], 10000)


if __name__ == '__main__':
    unittest.main()
