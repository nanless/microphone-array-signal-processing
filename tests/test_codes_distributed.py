"""Independent algebra, input boundaries and update-time contracts for ch15."""
import json
import unittest
import numpy as np
from codes.chapters.ch15.core.distributed import (
    known_models, mwf_weights, mse_components, compressed_mwf, distributed_updates,
    jacobi_control, step_size_control, tree_sum_control, gevd_control,
)
from codes.chapters.ch15.chapter15_exercises import run_experiments, resource_control


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1., .5, 2., -.5])
        self.rs = np.outer(self.a, self.a)
        self.rn = np.eye(4)
        self.rn[0, 2] = self.rn[2, 0] = .2
        self.rn[0, 3] = self.rn[3, 0] = .8

    def test_white_and_correlated_rational_anchors(self):
        for noise, expected, mse in [(np.eye(4), np.array([2, 1, 4, -1])/13, 2/13),
                                     (self.rn, np.array([25, 4, 11, -24])/69, 8/69)]:
            w = mwf_weights(self.rs, noise)
            np.testing.assert_allclose(w, expected, atol=5e-15)
            self.assertAlmostEqual(mse_components(self.rs, noise, w)['total_mse'], mse)
            # Independent expansion for a nonoptimal half-sized weight.
            q = expected/2
            scalar = (q@self.a-1)**2+q@noise@q
            self.assertAlmostEqual(mse_components(self.rs, noise, q)['total_mse'], scalar)

    def test_complete_arbitrary_weight_cost(self):
        result = mse_components(np.diag([2., 1.]), np.eye(2), [2., 3.])
        self.assertEqual(result['target_distortion'], 11.)
        self.assertEqual(result['noise_power'], 13.)
        self.assertEqual(result['total_mse'], 24.)
        zero = mse_components(self.rs, np.eye(4), np.zeros(4))
        self.assertEqual(zero['total_mse'], 1.)

    def test_complex_reference_conjugation(self):
        a = np.array([1., .5j, 2., -.5])
        rs = np.outer(a, a.conj())
        expected = np.array([-1j, .5, -2j, .5j])/13
        w = mwf_weights(rs, np.eye(4), 1)
        np.testing.assert_allclose(w, expected, atol=2e-16)
        self.assertAlmostEqual(np.vdot(w, a).imag, 11/26)
        self.assertAlmostEqual(mse_components(rs, np.eye(4), w, 1)['total_mse'], 1/26)

    def test_compression_and_independent_kkt(self):
        t = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 2., -.5]])
        result = compressed_mwf(self.rs, self.rn, t)
        normal = np.array([0., 0., .5, 2.])
        kkt = np.block([[self.rs+self.rn, normal[:, None]], [normal[None, :], np.zeros((1, 1))]])
        independent = np.linalg.solve(kkt, np.r_[self.a, 0])[:4]
        np.testing.assert_allclose(result['weights'], independent, atol=5e-16)
        self.assertAlmostEqual(result['components']['total_mse'], 2/13)
        true_w = np.array([25, 4, 11, -24])/69
        t[2, 2:] = true_w[2:]
        np.testing.assert_allclose(compressed_mwf(self.rs, self.rn, t)['weights'], true_w, atol=5e-16)

    def test_normalized_nonzero_scale_and_complex_phase(self):
        expected = np.array([2, 1, 4, -1])/13
        for scale in (1e-300, 1e300, 2j):
            t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, -.5]], complex)
            t[2] *= scale
            result = compressed_mwf(self.rs, np.eye(4), t)
            np.testing.assert_allclose(result['weights'], expected, atol=3e-15)
            np.testing.assert_allclose(result['normalized_projection'].conj().T@result['compressed_weights'], expected, atol=3e-15)

    def test_complex_projection_uses_conjugated_weights(self):
        a = np.array([1., .5j, 2j, -.5])
        rs = np.outer(a, a.conj())
        t = np.zeros((3, 4), complex); t[:2, :2] = np.eye(2); t[2, 2:] = a[2:].conj()
        expected = a/6.5
        np.testing.assert_allclose(compressed_mwf(rs, np.eye(4), t)['weights'], expected, atol=5e-16)

    def test_invalid_projection_shapes_zero_and_dependent_rows(self):
        for t in (np.zeros((3, 4)), np.ones((2, 4)), np.eye(5), np.ones(4), [[True]*4]):
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                compressed_mwf(self.rs, np.eye(4), t)

    def test_covariance_validation_precedes_solve(self):
        for rs, rn in [(np.eye(2), [[1, 2], [2, 1]]),
                       ([[1, 1], [0, 1]], np.eye(2)), (np.zeros((2, 2)), np.zeros((2, 2))),
                       (np.diag([1., 0.]), np.diag([1., 1e-14])), (np.eye(2), np.eye(3))]:
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                mwf_weights(rs, rn)
        for bad in (True, -1, 4, 1.5):
            with self.assertRaises(ValueError):
                mwf_weights(self.rs, np.eye(4), bad)

    def test_covariance_scale_invariance_and_physical_units(self):
        expected = np.array([25, 4, 11, -24])/69
        for scale in (1e-300, 1e300):
            w = mwf_weights(self.rs*scale, self.rn*scale)
            np.testing.assert_allclose(w, expected, atol=5e-15)
            result = mse_components(self.rs*scale, self.rn*scale, w)
            self.assertAlmostEqual(result['total_mse']/scale, 8/69)
        with self.assertRaises(ValueError):
            mse_components(np.eye(2), np.eye(2), [1e308, 1e308])

    def test_zero_reference_is_explicit(self):
        result = mse_components(np.diag([1., 0.]), np.eye(2), [0., 0.], 1)
        self.assertEqual(result['reference_target_power'], 0.)
        self.assertIsNone(result['normalized_mse'])

    def _updates(self, **kwargs):
        return distributed_updates(self.rs, self.rn, ((0, 1), (2, 3)), (0, 2),
                                   [self.a[:2], self.a[2:]], **kwargs)

    def test_round_robin_preserves_stale_receivers_and_rebuilds_current_outputs(self):
        result = self._updates(max_updates=6, tolerance=0.)
        history = result['history']
        expected_j = [2/13, .1251372118551044, .1180174009356284,
                      .11605609650078663, .11594737978282565, .1159422132581912]
        for i, entry in enumerate(history):
            k = i % 2
            self.assertEqual(entry['active_nodes'], [k])
            self.assertEqual(entry['total_solve_count'], i+3)
            self.assertAlmostEqual(entry['solutions'][0]['components']['normalized_mse'], expected_j[i], places=12)
            if i:
                np.testing.assert_array_equal(entry['receiver_coefficients_raw_coordinates'][1-k],
                                              history[i-1]['receiver_coefficients_raw_coordinates'][1-k])
                np.testing.assert_array_equal(entry['outputs_at_last_solve'][1-k], history[i-1]['outputs_at_last_solve'][1-k])
                self.assertGreater(entry['output_age_in_update_solves'][1-k], 0)
            for peer in (0, 1):
                expected = entry['current_receiver_projections'][peer].conj().T@entry['receiver_coefficients_raw_coordinates'][peer]
                np.testing.assert_array_equal(entry['cached_outputs'][peer], expected)
        saved = history[0]['compressions_after'][0].copy()
        result['compressions'][0][:] = 100
        np.testing.assert_array_equal(history[0]['compressions_after'][0], saved)

    def test_independent_white_two_step_current_broadcast_rational_anchor(self):
        result = distributed_updates(self.rs, np.eye(4), ((0, 1), (2, 3)), (0, 2),
                                     [[1., 0.], [1., 0.]], max_updates=2, tolerance=0.)
        first, second = result['history']
        np.testing.assert_allclose(first['cached_outputs'][0], np.array([4, 2, 8, 0])/25, atol=1e-16)
        np.testing.assert_allclose(second['cached_outputs'][0], np.array([52, 26, 64, -16])/325, atol=2e-16)
        np.testing.assert_array_equal(first['receiver_coefficients_raw_coordinates'][0], second['receiver_coefficients_raw_coordinates'][0])
        np.testing.assert_array_equal(first['outputs_at_last_solve'][0], second['outputs_at_last_solve'][0])
        self.assertAlmostEqual(second['cached_components'][0]['total_mse'], 23108/105625, places=14)
        self.assertEqual(second['total_solve_count'], 4)

    def test_simultaneous_snapshot_and_budget(self):
        result = self._updates(schedule='simultaneous', max_updates=3, tolerance=0.)
        self.assertEqual(result['update_solve_count'], 2)
        self.assertEqual(result['unused_update_budget'], 1)
        self.assertEqual(result['total_solve_count'], 4)
        entry = result['history'][0]
        self.assertEqual(entry['active_nodes'], [0, 1])
        for k in (0, 1):
            for peer in (0, 1):
                np.testing.assert_array_equal(entry['solutions'][k]['incoming_compressions'][peer], entry['compressions_before'][peer])
            expected = entry['current_receiver_projections'][k].conj().T@entry['receiver_coefficients_raw_coordinates'][k]
            np.testing.assert_array_equal(entry['cached_outputs'][k], expected)
            np.testing.assert_array_equal(entry['solutions'][k]['effective_weights_after_broadcast'], expected)
        self.assertFalse(np.allclose(entry['cached_outputs'][0], entry['solutions'][0]['weights']))

    def test_actual_residual_stopping_and_count(self):
        result = self._updates(max_updates=40, tolerance=1e-8)
        self.assertEqual(result['status'], 'known_covariance_residual_reached')
        self.assertEqual(result['update_solve_count'], 14)
        self.assertEqual(result['total_solve_count'], 16)
        self.assertLessEqual(max(result['history'][-1]['normal_equation_relative_residuals']), 1e-8)
        self.assertGreater(max(result['history'][-2]['normal_equation_relative_residuals']), 1e-8)

    def test_zero_peer_only_explicit_local_policy(self):
        arguments = (self.rs, np.eye(4), ((0, 1), (2, 3)), (0, 2), [np.zeros(2), np.zeros(2)])
        with self.assertRaises(np.linalg.LinAlgError):
            distributed_updates(*arguments)
        result = distributed_updates(*arguments, zero_policy='local', max_updates=1, tolerance=0.)
        solve = result['history'][0]['solutions'][0]
        self.assertEqual(solve['missing_peers'], [1])
        np.testing.assert_allclose(solve['weights'], [4/9, 2/9, 0, 0])
        self.assertEqual(result['receiver_peer_maps'][1], [])
        np.testing.assert_allclose(result['cached_outputs'][1], [0, 0, 16/21, -4/21])
        for kwargs in ({'relaxation': 0}, {'max_updates': True}, {'tolerance': -1}, {'schedule': 'unknown'}):
            with self.assertRaises(ValueError):
                self._updates(**kwargs)

    def test_target_rank_counterexample(self):
        a = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], float)
        rs = a@a.T; rn = np.eye(4); t = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]
        full = np.column_stack([mwf_weights(rs, rn, r) for r in (0, 1)])
        np.testing.assert_allclose(full, a/4)
        self.assertEqual(np.linalg.matrix_rank(full[2:]), 2)
        self.assertAlmostEqual(compressed_mwf(rs, rn, t, 0)['components']['total_mse'], 1/4)
        self.assertAlmostEqual(compressed_mwf(rs, rn, t, 1)['components']['total_mse'], 1/2)

    def test_jacobi_controls(self):
        bad = jacobi_control(); good = jacobi_control(.5, 40)
        np.testing.assert_allclose(bad['trace'][1:, 0], [1., -.8, 2.44, -3.392], atol=1e-14)
        self.assertAlmostEqual(bad['spectral_radius'], 1.8)
        self.assertAlmostEqual(good['spectral_radius'], .95)
        # Symmetric rhs excites only the all-ones eigenvector, multiplier -.4.
        np.testing.assert_allclose(good['trace'][-1], np.full(3, 5/14), atol=1e-14)

    def test_infinite_step_conditions_are_not_empirical(self):
        result = step_size_control(8)
        np.testing.assert_allclose(result['harmonic']['finite_values'], [1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8])
        self.assertTrue(result['harmonic']['limit_zero'])
        self.assertFalse(result['constant_half']['limit_zero'])
        self.assertEqual(result['geometric_half']['infinite_sum'], '1')
        self.assertEqual(result['geometric_half']['infinite_square_sum'], '1/3')

    def test_actual_two_pass_tree_messages(self):
        result = tree_sum_control()
        self.assertEqual([(m['from'], m['to'], m['value']) for m in result['edge_messages']],
                         [(3, 2, 3), (2, 1, 5), (1, 2, 6), (2, 3, 6)])
        np.testing.assert_array_equal(result['totals'], [6, 6, 6])
        np.testing.assert_array_equal(result['subtract_own'], [5, 4, 3])
        np.testing.assert_array_equal(result['neighbors_self_only'], [2, 4, 2])

    def test_gevd_independent_analytic_values_and_metric(self):
        result = gevd_control()
        np.testing.assert_allclose(result['generalized_eigenvalues'], [5, 1], atol=3e-15)
        np.testing.assert_allclose(result['rank1_weights'], [0, 2/5], atol=5e-16)
        self.assertAlmostEqual(result['components']['total_mse'], 1/5)
        np.testing.assert_allclose(result['noise_metric_gram'], np.eye(2), atol=8e-16)
        np.testing.assert_allclose(result['generalized_residual'], 0, atol=3e-15)
        np.testing.assert_allclose(result['ordinary_eigenvalues'], [(7+np.sqrt(34))/2, (7-np.sqrt(34))/2])

    def test_resource_payload_arithmetic(self):
        result = resource_control()
        self.assertEqual(result['two_nodes_PCM16_raw_upload_bps'], 1024000)
        self.assertEqual(result['two_nodes_PCM16_scalar_exchange_bps'], 512000)
        self.assertEqual(result['two_nodes_float32_scalar_exchange_bps'], 1024000)
        self.assertEqual(result['two_nodes_STFT_complex_float32_exchange_bps'], 4112000)
        self.assertEqual(result['network'][1]['compressed_full_mesh_unicast_PCM16_bits_per_second'], 3072000)
        self.assertEqual(result['network'][1]['network_full_complex128_covariance_bytes'], 1600)

    def test_independent_exercise_id_set_and_finite_json(self):
        result = run_experiments()
        expected = {'E15-'+str(i).zfill(2) for i in range(1, 26)}
        self.assertEqual(set(result['exercises']), expected)
        self.assertNotIn('schema_version', result['exercises'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
