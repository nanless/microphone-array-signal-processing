"""Independent extreme-scale and versioned-SCM controls for Chapter 15."""
import unittest
import numpy as np
from codes.chapters.ch15.core.distributed import (
    mwf_weights, mse_components, compressed_mwf, distributed_updates,
    broadcast_statistics_control,
)


class DistributedNumericalDomainTests(unittest.TestCase):
    def test_nonzero_normalized_power_underflow_is_not_exact_zero(self):
        # The physical error is 1e-100 and the reference power 1e300.
        # Their positive ratio 1e-400 is outside binary64, not exact zero.
        with self.assertRaisesRegex(ValueError, 'normalized MSE'):
            mse_components([[1e300]], [[1e-100]], [1.])
        zero = mse_components([[1e300]], [[0.]], [1.])
        self.assertEqual(zero['total_mse'], 0.)
        self.assertEqual(zero['normalized_mse'], 0.)

    def test_negative_reference_variance_is_not_a_valid_denominator(self):
        # A shared PSD tolerance may admit the small negative eigenvalue.
        # The selected physical variance still cannot be negative.
        with self.assertRaisesRegex(ValueError, 'reference target power'):
            mse_components(np.diag([1., -1e-16]), np.eye(2), [0., 1.], 1)

    def test_arbitrary_weight_cost_preserves_small_target_beside_large_noise(self):
        # With w=0, noise does not contribute. Both physical target values
        # remain representable; joint normalization formerly erased them.
        for target, noise in ((1e-100, 1e300), (1e300, 1e-100)):
            value = mse_components([[target]], [[noise]], [0.])
            self.assertEqual(value['total_mse'], target)
            self.assertEqual(value['reference_target_power'], target)
            self.assertEqual(value['normalized_mse'], 1.)
        with self.assertRaisesRegex(ValueError, 'erases a nonzero'):
            mwf_weights([[1e-100]], [[1e300]])
        with self.assertRaisesRegex(ValueError, 'erases a nonzero'):
            compressed_mwf([[1e-100]], [[1e300]], [[1.]])

    def test_separate_complex_psd_scales_and_explicit_unsupported_power(self):
        # Independent rank-one identity: (w-e)^H aa^H(w-e)=|a^H(w-e)|².
        a = np.array([1., 1j]); rs = 1e-100*np.outer(a, a.conj())
        value = mse_components(rs, 1e300*np.eye(2), np.zeros(2))
        self.assertEqual(value['total_mse'], 1e-100)
        self.assertEqual(value['normalized_mse'], 1.)
        with self.assertRaisesRegex(ValueError, 'physical power'):
            mse_components([[1e-300]], [[1e-300]], [.5e-100])
        with self.assertRaisesRegex(ValueError, 'physical power'):
            mse_components([[1.]], [[1.]], [1e308])

    def test_huge_null_coordinate_cannot_hide_nonzero_material_power(self):
        rs = np.diag([0., 1e-100]); rn = np.diag([0., 1e300])
        # Explicit physical scalar powers, not production normalization.
        value = mse_components(rs, rn, [1e300, 1e-100], 1)
        self.assertAlmostEqual(value['noise_power']/1e100, 1.)
        self.assertAlmostEqual(value['target_distortion']/1e-100, 1.)
        self.assertAlmostEqual(value['normalized_mse']/1e200, 1.)
        # A genuinely null direction may carry a huge coefficient and still
        # have exact zero target/noise power: reject neither its magnitude nor
        # exact zero without checking its action.
        zero = mse_components(rs, rn, [1e300, 0.], 0)
        self.assertEqual(zero['total_mse'], 0.)
        self.assertIsNone(zero['normalized_mse'])
        # Nonzero rows remain: a tiny component erased in vector scaling and
        # a nonzero action with an unrepresentable reduced square are refused.
        for matrix, vector in ((np.eye(2), [1e300, 1e-100]),
                               (np.diag([1., 1e-200]), [1e200, 1.])):
            with self.assertRaisesRegex(ValueError, 'quadratic'):
                mse_components(np.zeros((2, 2)), matrix, vector)

    def test_nonzero_action_underflow_is_not_a_true_nullspace(self):
        matrix = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
        # The first two coefficients cancel exactly, the third gives true
        # physical power 1. Its reduced squared value is below float64.
        with self.assertRaisesRegex(ValueError, 'reduced power'):
            mse_components(np.zeros((3, 3)), matrix, [1e200, -1e200, 1.])
        true_null = mse_components(np.zeros((3, 3)), matrix, [1e200, -1e200, 0.])
        self.assertEqual(true_null['total_mse'], 0.)
        self.assertIsNone(true_null['normalized_mse'])

    def test_exact_hermitian_minimum_subnormal_is_retained_after_psd_check(self):
        tiny = np.nextafter(0., 1.)
        # True zero, a positive subnormal, and exact complex Hermitian input.
        zero = mse_components([[0.]], [[tiny]], [0.])
        self.assertEqual(zero['total_mse'], 0.)
        self.assertIsNone(zero['normalized_mse'])
        positive = mse_components([[tiny]], [[0.]], [0.])
        self.assertEqual(positive['total_mse'], tiny)
        self.assertEqual(positive['reference_target_power'], tiny)
        self.assertEqual(positive['normalized_mse'], 1.)
        np.testing.assert_array_equal(mwf_weights([[tiny]], [[0.]]), [1.])
        rs = np.array([[2*tiny, 1j*tiny], [-1j*tiny, 2*tiny]])
        value = mse_components(rs, np.zeros((2, 2)), [0., 0.])
        self.assertEqual(value['total_mse'], 2*tiny)
        for bad in (np.array([[0., tiny], [tiny, 0.]]),
                    np.array([[tiny, 1j*tiny], [1j*tiny, tiny]])):
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                mse_components(bad, np.eye(2), [0., 0.])
        # Positive powers below the representable domain are rejected, not 0.
        with self.assertRaisesRegex(ValueError, 'physical power'):
            mse_components([[tiny]], [[tiny]], [.5])

    def test_relative_residual_does_not_square_tiny_rhs_before_normalizing(self):
        a = np.array([1., .5, 2., -.5]); rs = np.outer(a, a)
        rn = np.eye(4); rn[0, 2] = rn[2, 0] = .2; rn[0, 3] = rn[3, 0] = .8
        run = distributed_updates(rs, 1e200*rn, ((0, 1), (2, 3)), (0, 2),
                                  [[1., 0.], [1., 0.]], max_updates=2)
        self.assertEqual(run['status'], 'budget_exhausted')
        # Cancel the 1e-200 scale BEFORE forming each squared norm.
        independent = []
        for k, w in enumerate(run['cached_outputs']):
            p = rs[:, 2*k]
            rho = (rn+1e-200*rs)@(w*1e200)-p
            independent.append(np.sqrt(sum(abs(v)**2 for v in rho))
                               / np.sqrt(sum(abs(v)**2 for v in p)))
        np.testing.assert_allclose(run['history'][-1]['normal_equation_relative_residuals'],
                                   independent, rtol=1e-14, atol=0.)
        np.testing.assert_allclose(independent, [.9201037490907801, .3640401798225748], rtol=1e-14)
        self.assertEqual(run['history'][-1]['normal_equation_residual_modes'],
                         ['relative_nonzero_rhs']*2)

    def test_genuinely_zero_rhs_has_declared_absolute_scope(self):
        run = distributed_updates(np.zeros((2, 2)), np.eye(2), ((0,), (1,)), (0, 1),
                                  [[1.], [1.]], zero_policy='local', max_updates=2)
        self.assertEqual(run['history'][-1]['normal_equation_relative_residuals'], [0., 0.])
        self.assertEqual(run['history'][-1]['normal_equation_residual_modes'],
                         ['absolute_scaled_coordinates_zero_rhs']*2)
        self.assertEqual(run['status'], 'known_covariance_residual_reached')

    def test_versioned_scm_rational_solution_and_original_physical_cost(self):
        case = broadcast_statistics_control(); cases = case['cases']
        np.testing.assert_array_equal(case['current_statistics']['observation_covariance'], [[2, 2], [2, 8]])
        np.testing.assert_array_equal(case['unconverted_mixture_statistics']['observation_covariance'],
                                      [[2, 1.5], [1.5, 5]])
        # 2h1+1.5h2=1 and 1.5h1+5h2=1.5 give 11/31,6/31.
        np.testing.assert_allclose(cases['unconverted_mixture']['receiver_weights'], [11/31, 6/31], atol=1e-16)
        w = np.array([11/31, 12/31]); H = sum(w)
        self.assertAlmostEqual(cases['unconverted_mixture']['physical_components']['target_distortion'], 64/961)
        self.assertAlmostEqual(cases['unconverted_mixture']['physical_components']['noise_power'], 265/961)
        self.assertAlmostEqual((H-1)**2+sum(w*w), 329/961)
        self.assertAlmostEqual(cases['unconverted_mixture']['physical_components']['total_mse'], 329/961)
        for name in ('current_reset', 'converted_mixture'):
            np.testing.assert_allclose(cases[name]['receiver_weights'], [1/3, 1/6], atol=1e-16)
            np.testing.assert_allclose(cases[name]['effective_weights'], [1/3, 1/3], atol=1e-16)
            self.assertAlmostEqual(cases[name]['physical_components']['total_mse'], 1/3)
        self.assertGreater(np.linalg.det(case['unconverted_mixture_statistics']['observation_covariance']).real, 0.)

    def test_complex_coordinate_change_uses_hermitian_not_plain_transpose(self):
        rs = np.ones((2, 2), complex); rn = np.eye(2); c = np.diag([1., 2j])
        h = mwf_weights(c@rs@c.conj().T, c@rn@c.conj().T)
        np.testing.assert_allclose(h, [1/3, 1j/6], atol=1e-16)
        np.testing.assert_allclose(c.conj().T@h, [1/3, 1/3], atol=1e-16)
        self.assertGreater(np.max(abs(c.T@h-np.array([1/3, 1/3]))), .6)
        # A changed row space has no invertible transport: old B loses e3.
        old = np.eye(3)[:2]; new = np.eye(3)[[0, 2]]; v = np.array([0., 0., 1.])
        np.testing.assert_array_equal(old@v, [0., 0.])
        np.testing.assert_array_equal(new@v, [0., 1.])


if __name__ == '__main__':
    unittest.main()
