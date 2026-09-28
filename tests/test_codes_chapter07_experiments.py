"""Independent rational, polynomial and index anchors for E07-08..17."""
import json
import unittest
from fractions import Fraction as F
import numpy as np
from codes.chapters.ch07 import chapter07_experiments as ex


def z(encoded):
    return np.asarray(encoded['real']) + 1j*np.asarray(encoded['imag'])


class Chapter07Experiments(unittest.TestCase):
    def test_complex_fit_explicit_matrix_and_rational_solution(self):
        result = ex.complex_weighted_fit()
        np.testing.assert_array_equal(z(result['correlation']), [[2,-1j],[1j,3]])
        np.testing.assert_array_equal(z(result['cross']), [3,4j])
        plain = result['absolute_loading_cases']['0']
        np.testing.assert_allclose(z(plain['coefficients']), [1,1j], atol=1e-15)
        np.testing.assert_allclose(z(plain['residual']), [-1,.5j,1], atol=1e-15)
        self.assertAlmostEqual(plain['weighted_residual_cost'], 2.5)
        np.testing.assert_allclose(z(plain['weighted_history_residual_cross']), [0,0], atol=1e-15)
        loaded = result['absolute_loading_cases']['1']
        expected = np.array([8/11,9j/11])
        np.testing.assert_allclose(z(loaded['coefficients']), expected, atol=1e-15)
        np.testing.assert_allclose(z(loaded['weighted_history_residual_cross']), expected, atol=1e-15)
        # Full six-frame observation must generate these histories and targets.
        obs = z(result['observations'])
        np.testing.assert_array_equal(obs[:, [0,1,2]].T, [[1,0],[0,1],[1,1j]])
        np.testing.assert_array_equal(obs[0, [3,4,5]], [0,-.5j,3])

    def test_power_objective_floor_and_smoothing_are_distinct(self):
        r = ex.power_floor_and_smoothing()['cases']
        self.assertAlmostEqual(r['unconstrained']['objective_without_constants'],2+np.log(9))
        self.assertAlmostEqual(r['floor_2']['objective_without_constants'],1.5+np.log(18))
        self.assertAlmostEqual(r['mean_smoothed']['objective_without_constants'],2+2*np.log(5))
        self.assertGreater(r['mean_smoothed']['objective_without_constants'],r['unconstrained']['objective_without_constants'])
        zero = ex.power_floor_and_smoothing()['zero_energy']['objective_without_constants']
        np.testing.assert_allclose(zero, [0,-np.log(10),-2*np.log(10)])

    def test_permutation_preserves_inner_product_only_when_both_move(self):
        r = ex.history_permutation()
        self.assertEqual(r['lag_major_history'],[3,13,2,12,1,11])
        self.assertEqual(r['channel_major_history'],[3,2,1,13,12,11])
        self.assertEqual(z(r['prediction_lag_major']),18-1j)
        self.assertEqual(z(r['prediction_both_permuted']),18-1j)
        self.assertEqual(z(r['prediction_history_only_permuted']),16+11j)

    def test_frame_count_does_not_identify_collinear_parameters(self):
        cases = ex.frame_count_and_rank()['cases']
        np.testing.assert_allclose(cases['repeated']['eigenvalues'],[0,8],atol=1e-15)
        np.testing.assert_allclose(cases['repeated']['minimum_norm_coefficients'],[.75,.75],atol=1e-15)
        np.testing.assert_allclose(cases['repeated']['absolute_loading_1_coefficients'],[2/3,2/3],atol=1e-15)
        np.testing.assert_array_equal(cases['independent']['eigenvalues'],[2,2])
        np.testing.assert_allclose(cases['independent']['minimum_norm_coefficients'],[1,2],atol=1e-15)
        # Null direction changes coefficients but leaves every training value.
        np.testing.assert_array_equal(np.ones((4,2)) @ [1.,-1.],np.zeros(4))

    def test_window_overlap_by_explicit_sample_sets(self):
        r = ex.frame_window_overlap()
        for item, expected in zip(r['cases'], [128,0]):
            a,b = item['nearest_history_support_samples']
            self.assertEqual(len(set(range(512)) & set(range(a,b))),expected)
            self.assertEqual(item['overlap_samples'],expected)
        self.assertEqual([a['oldest_frame_start_offset_samples'] for a in r['cases']],[-896,-1024])

    def test_exponential_statistics_independent_fraction_sum(self):
        r = ex.exponential_statistics()
        # Expand all three weighted terms plus the decayed initial statistics.
        R = F(1,8)*2+F(1,4)*1+F(1,2)*4+1
        cross = F(1,8)+F(1,4)*F(4,5)+F(1,2)*2*F(8,5)+1
        self.assertEqual((R,cross),(F(7,2),F(117,40)))
        self.assertAlmostEqual(r['batch_correlation'],float(R))
        self.assertAlmostEqual(r['batch_cross'],float(cross))
        self.assertAlmostEqual(r['steps'][-1]['posterior_coefficient'],float(F(117,140)))
        np.testing.assert_allclose([s['prior_output'] for s in r['steps']],[.3,.3,.23],atol=1e-15)
        self.assertAlmostEqual(r['two_zero_history_steps_forgetting']['correlation'],7/8)
        self.assertAlmostEqual(r['two_zero_history_steps_forgetting']['cross'],117/160)
        self.assertEqual(r['two_fully_frozen_statistics_steps']['correlation'],3.5)

    def test_mint_polynomial_identity_and_post_path_noise(self):
        a,b = ex.mint_near_common_zero()['cases']
        np.testing.assert_allclose(a['constant_inverse_weights'],[.5,.5],atol=1e-14)
        np.testing.assert_allclose(b['constant_inverse_weights'],[-49,50],rtol=1e-14)
        np.testing.assert_allclose(a['summed_impulse_response'],[1,0],atol=1e-14)
        np.testing.assert_allclose(b['summed_impulse_response'],[1,0],atol=1e-14)
        self.assertAlmostEqual(a['post_path_independent_unit_noise_variance'],.5)
        self.assertAlmostEqual(b['post_path_independent_unit_noise_variance'],4901,places=9)
        self.assertAlmostEqual(b['perturbed_second_response_tap'],-.049,places=12)
        self.assertFalse(ex.mint_near_common_zero()['common_zero_case']['constant_inverse_exists'])

    def test_resource_bytes_not_timing(self):
        a,b = ex.wpe_resource_budget()['cases']
        self.assertEqual((a['channels'],a['taps'],a['history_dimension']),(8,10,80))
        self.assertEqual(a['one_correlation_array_bytes'],26316800)
        self.assertEqual(a['one_predictor_array_bytes'],2631680)
        self.assertEqual(b['one_correlation_array_bytes'],4*a['one_correlation_array_bytes'])
        self.assertEqual(b['one_predictor_array_bytes'],4*a['one_predictor_array_bytes'])
        self.assertEqual(b['dense_factorization_cubic_dimension_proxy'],8*a['dense_factorization_cubic_dimension_proxy'])

    def test_loaded_wpd_missing_term_against_rational_solution(self):
        r = ex.loaded_wpd_factorization()
        np.testing.assert_allclose(r['G'],np.diag([1/3,0]),atol=1e-15)
        np.testing.assert_allclose(r['schur'],np.diag([8/3,2]),atol=1e-15)
        np.testing.assert_allclose(r['residual_covariance'],np.diag([14/9,1]),atol=1e-15)
        np.testing.assert_allclose(r['required_correction'],np.diag([10/9,1]),atol=1e-15)
        np.testing.assert_allclose(r['direct_loaded_filter'],[3/7,4/7,-1/7,0],atol=1e-15)
        self.assertAlmostEqual(r['direct_loaded_objective'],8/7)
        np.testing.assert_allclose(r['omitted_GHG_correction_spatial_filter'],[18/41,23/41],atol=1e-15)
        np.testing.assert_allclose(r['omitted_GHG_correction_history_filter'],[-6/41,0],atol=1e-15)

    def test_stable_ids_and_finite_json(self):
        r = ex.run_experiments()
        self.assertEqual(set(r),{f'E07-{i:02d}' for i in range(8,18)})
        self.assertEqual(len(r['E07-17']['files']),4)
        json.dumps(r,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
