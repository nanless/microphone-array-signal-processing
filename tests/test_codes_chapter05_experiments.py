"""Independent small-matrix oracles for Chapter 5, including failure boundaries."""
import json
import math
import unittest
import warnings
import numpy as np

from codes.chapters.ch05.core.beamforming import lcmv_weights, mvdr_weights
from codes.chapters.ch04.core.doa import capon_spectrum
from codes.chapters.ch05 import chapter05_experiments as ex
from codes.chapters.ch05.beamformer_common_input_demo import run_experiment


class ChapterFiveExamplesTest(unittest.TestCase):
    def test_catalog_is_complete_and_strict_json(self):
        results = ex.run_exercises()
        self.assertEqual(set(results), {f'E05-{i:02d}' for i in range(8, 25)})
        json.dumps(results, allow_nan=False)

    def test_gsc_solution_matches_inverse_variance_weights(self):
        r = ex.gsc_lcmv_equivalence()
        np.testing.assert_allclose(r['reference_covariance'], [[3, 1], [1, 5]])
        np.testing.assert_allclose(r['adaptive_coefficient'], [-1/21, -4/21])
        for key in ['gsc_weights', 'lcmv_weights']:
            np.testing.assert_allclose(r[key], [4/7, 2/7, 1/7])
        self.assertAlmostEqual(r['output_noise_power'], 4/7)
        self.assertAlmostEqual(r['arbitrary_h_target_response'], 1)

    def test_single_snapshot_and_correlated_waveform_are_different_objectives(self):
        r = ex.gsc_statistical_boundary()
        a = r['sixty_degree_case']
        # (1+e^jphi)/(1-e^jphi) = j*cot(phi/2).
        cotangent = 1/math.tan(math.pi*math.sqrt(3)/4)
        self.assertAlmostEqual(a['oracle_h'], -.5j*cotangent)
        self.assertAlmostEqual(a['single_snapshot_ls_h'], .5-1j*cotangent)
        self.assertAlmostEqual(a['oracle_residual'], 1)
        self.assertLess(abs(a['single_snapshot_ls_residual']), 1e-14)
        for row, power in zip(r['correlated_cases'], [1., .75, 0.]):
            self.assertAlmostEqual(row['target_response'], 1)
            self.assertAlmostEqual(row['joint_output_power'], power)
            self.assertAlmostEqual(row['interference_response'], -row['rho'])

    def test_frost_real_fir_and_complex_conjugates_have_hand_answers(self):
        report = ex.frost_finite_step()
        real = report['real_two_mic_two_tap']
        self.assertEqual(real['pre_update_output'], 2)
        np.testing.assert_allclose(real['ordinary_lms_weights'], [.3, -.1, -.4, 0], atol=1e-15)
        np.testing.assert_allclose(real['frost_weights'], [.7, .3, -.2, .2])
        np.testing.assert_allclose(real['frost_constraint'], [1, 0], atol=1e-15)
        comp = report['complex_algebra_check']
        self.assertEqual(comp['pre_update_output'], 1.5)
        np.testing.assert_allclose(comp['frost_weights'], [.575, .425j, -.15+.15j])
        self.assertAlmostEqual(comp['ordinary_lms_constraint'][0], .55)
        for row in (real, comp):
            p, c = row['projection'], row['constraints']
            np.testing.assert_allclose(p@p, p, atol=1e-15)
            np.testing.assert_allclose(c.conj().T@p, 0, atol=1e-15)

    def test_pair_differences_are_nonnegative_but_may_be_target_energy(self):
        r = ex.zelinski_pair_difference()
        self.assertAlmostEqual(r['estimated_input_noise'], 16/9)
        self.assertAlmostEqual(r['pair_expression'], 16/9)
        np.testing.assert_allclose(r['pair_difference_powers'], [11/3, 14/3, 7/3])
        np.testing.assert_allclose([x['estimated_noise_power'] for x in r['noiseless_mismatch_cases']], [.02, 1])

    def test_coherence_model_inverts_two_equations_not_a_general_postfilter(self):
        r = ex.coherent_noise_pair()
        for row in r['cases']:
            self.assertAlmostEqual(row['estimated_target_power'], 3)
            self.assertAlmostEqual(row['estimated_noise_power'], 1)
        self.assertAlmostEqual(r['cases'][0]['noise_estimate_change'], -1/70)
        self.assertAlmostEqual(r['cases'][1]['noise_estimate_change'], -1)
        self.assertAlmostEqual(r['dsb_output_noise_at_rho03'], 13/20)
        self.assertAlmostEqual(r['wiener_gain_at_rho03'], 60/73)

    def test_matched_target_leakage_is_invariant_but_wrong_constraint_is_not(self):
        for row in ex.target_covariance_contamination()['cases']:
            eta = row['target_power_in_noise_scm']
            if row['case'] == 'matched':
                np.testing.assert_allclose(row['weights'], [.5, .5])
                self.assertAlmostEqual(row['actual_target_response'], 1)
            else:
                self.assertAlmostEqual(row['actual_target_response'], (1+1j)/(2*(eta+1)))
                self.assertAlmostEqual(row['actual_white_noise_power'], .5+eta**2/(2*(eta+1)**2))

    def test_gev_scale_and_mwf_mse_use_independent_rational_answers(self):
        r = ex.gev_mwf_scale()
        self.assertAlmostEqual(r['largest_generalized_eigenvalue'], 4.5)
        rows = {x['method']: x for x in r['cases']}
        np.testing.assert_allclose(rows['GEV']['weights'], [.5, 1])
        self.assertAlmostEqual(rows['GEV']['target_response'], 1.5)
        self.assertAlmostEqual(rows['GEV_times_two']['target_response'], 3)
        for row in rows.values():
            self.assertAlmostEqual(row['output_snr_linear'], 4.5)
        mwf = rows['MWF']
        np.testing.assert_allclose(mwf['weights'], [3/11, 6/11])
        self.assertAlmostEqual(mwf['output_noise_power'], 54/121)
        self.assertAlmostEqual(mwf['target_distortion_power'], 12/121)
        self.assertAlmostEqual(mwf['reference_mse'], 6/11)

    def test_equal_sensor_counts_do_not_imply_equal_sampling_rank(self):
        equator, tetra = ex.first_order_spatial_rank()['cases']
        self.assertEqual((equator['rank'], tetra['rank']), (3, 4))
        np.testing.assert_allclose(equator['gram'], np.diag([4, 2, 2, 0]))
        np.testing.assert_allclose(tetra['gram'], np.diag([4, 4/3, 4/3, 4/3]), atol=1e-15)
        self.assertIsNone(equator['condition_number'])
        self.assertAlmostEqual(tetra['condition_number'], math.sqrt(3))
        self.assertAlmostEqual(abs(np.linalg.det(tetra['sampling_matrix'])), 16/(3*math.sqrt(3)))

    def test_signed_masks_are_rejected_and_floor_breaks_tiny_scale_invariance(self):
        r = ex.nonnegative_mask_model()
        self.assertTrue(r['negative_mask_rejected_by_real_interface'])
        np.testing.assert_array_equal(r['unguarded_signed_weight_result'], [[2, 0], [0, -1]])
        rows = r['cases']
        for row, expected in zip(rows, [np.eye(2)/2, np.diag([1., 0]), 1e-288*np.eye(2)]):
            np.testing.assert_allclose(row['covariance'], expected, atol=0, rtol=1e-14)
        self.assertEqual([r['weight_concentration_count'] for r in rows], [2, 1, 2])

    def test_derivative_constraint_has_rational_noise_and_local_complex_flatness(self):
        rows = {r['method']: r for r in ex.derivative_lcmv()['cases']}
        np.testing.assert_allclose(rows['single']['weights'], [4/7, 2/7, 1/7])
        np.testing.assert_allclose(rows['constrained']['weights'], [4/13, 5/13, 4/13])
        for key, power, wng, derivative in [('single', 4/7, 7/3, -3j/7),
                                           ('constrained', 10/13, 169/57, 0), ('DSB', 7/9, 3, 0)]:
            row = rows[key]
            self.assertAlmostEqual(row['output_noise_power'], power)
            self.assertAlmostEqual(row['wng_linear'], wng)
            self.assertAlmostEqual(row['complex_response_derivative_at_zero'], derivative)
            self.assertEqual([r['phi_rad'] for r in row['responses']], [0., .1, .2, .3])
            for response in row['responses']:
                phi = response['phi_rad']
                expected = ((2+5*math.cos(phi)-3j*math.sin(phi))/7 if key == 'single' else
                            (5+8*math.cos(phi))/13 if key == 'constrained' else (1+2*math.cos(phi))/3)
                self.assertAlmostEqual(response['complex_response'], expected)

    def test_norm_ball_optimum_has_nonunit_nominal_response_and_general_wng(self):
        nominal, robust = ex.norm_ball_robust()['cases']
        t = 1/(2-math.sqrt(2)/5)
        np.testing.assert_allclose(robust['weights'], [t, t])
        self.assertAlmostEqual(robust['nominal_response'], 2*t)
        self.assertAlmostEqual(robust['worst_case_amplitude'], 1)
        self.assertAlmostEqual(nominal['worst_case_amplitude'], 1-math.sqrt(2)/10)
        self.assertAlmostEqual(robust['general_wng_linear'], 2)
        self.assertNotAlmostEqual(robust['unit_response_only_shortcut'], 2)
        # Cauchy: every feasible w has ||w|| >= 1/(sqrt(2)-epsilon).
        self.assertAlmostEqual(robust['weight_norm_squared'], 1/(math.sqrt(2)-.2)**2)

    def test_souden_selects_complex_reference_not_unit_source_response(self):
        report = ex.souden_reference_channel()
        self.assertEqual(report['trace_normalizer'], 12)
        for row, expected, reference in zip(report['cases'], ([.5, .5+.5j], [.25-.25j, .5]), (2, 1+1j)):
            np.testing.assert_allclose(row['souden_weights'], expected)
            np.testing.assert_allclose(row['rtf_mvdr_weights'], expected)
            self.assertAlmostEqual(row['souden_measurements']['target_response'], reference)
            self.assertAlmostEqual(row['mwf_measurements']['target_response'], reference*12/13)
        second = report['cases'][1]['mwf_measurements']
        self.assertAlmostEqual(second['output_noise_power'], 72/169)
        self.assertAlmostEqual(second['reference_target_distortion_power'], 6/169)
        self.assertAlmostEqual(second['reference_mse'], 6/13)
        control = report['full_rank_control']
        np.testing.assert_array_equal(control['target_covariance'], [[2, 0], [0, 1]])
        self.assertEqual(control['trace_normalizer'], 3)
        np.testing.assert_allclose(control['souden_weights'], [2/3, 0], atol=1e-15)
        np.testing.assert_allclose(control['general_mwf_weights'], [2/3, 0], atol=1e-15)
        np.testing.assert_allclose(control['rank_one_trace_shortcut_weights'], [1/2, 0], atol=1e-15)
        self.assertAlmostEqual(control['reference_target_distortion_power'], 2/9)
        self.assertAlmostEqual(control['output_noise_power'], 4/9)

    def test_om_lsa_is_bayes_and_geometric_combination_only(self):
        report = ex.om_lsa_probability_combination()
        self.assertEqual(report['nu'], 3)
        for row, odds in zip(report['cases'][:2], (4*math.exp(-3), math.exp(-3))):
            p = 1/(1+odds)
            self.assertAlmostEqual(row['posterior_presence_probability'], p)
            self.assertAlmostEqual(row['geometric_gain'], .8**p*.1**(1-p))
            self.assertLess(row['geometric_gain'], row['arithmetic_mix_for_comparison'])
        self.assertEqual(report['cases'][2]['posterior_presence_probability'], 1)
        self.assertEqual(report['cases'][3]['posterior_presence_probability'], 0)

    def test_common_input_comparison_has_independent_rank_one_inverse(self):
        phase = 2*math.pi*2000*.04/343
        a = np.ones(4)
        b = np.exp(1j*phase*np.arange(4)*math.sin(math.radians(40)))
        true = np.exp(1j*phase*np.arange(4)*math.sin(math.radians(10)))
        for row in run_experiment()['results']:
            method = row['method']
            if method == 'DSB':
                w = a/4
            else:
                coefficient = 1/4 if method == 'LCMV' else 10/(40+(12 if method == 'MVDR_loaded' else 1))
                v = a-coefficient*b*np.vdot(b, a)
                w = v/np.vdot(a, v)
            response = np.vdot(w, true)
            power = 10*abs(np.vdot(w, b))**2+np.vdot(w, w).real
            self.assertAlmostEqual(row['true_target_response_abs'], abs(response))
            self.assertAlmostEqual(row['interference_plus_noise_output_power'], power)
            self.assertAlmostEqual(row['true_target_output_sinr_db'], 10*math.log10(abs(response)**2/power))


class BeamCovarianceBoundaryTest(unittest.TestCase):
    def test_tiny_negative_direction_cannot_be_a_minimum(self):
        # Feasible w=[t,1] has cost 1-1e-11*t² and is unbounded below.
        for solver in (mvdr_weights, capon_spectrum):
            for scale in (1e-18, 1., 1e18):
                with self.subTest(solver=solver.__name__, scale=scale):
                    with self.assertRaises(np.linalg.LinAlgError):
                        solver(scale*np.diag([-1e-11, 1]), [0, 1])
                    repaired = solver(scale*np.diag([-1e-11, 1]), [0, 1], relative_diagonal_loading=1e-8)
                    self.assertTrue(np.all(np.isfinite(repaired)))
                    with self.assertRaises(np.linalg.LinAlgError):
                        solver(scale*np.diag([-1., 3]), [1, 1], relative_diagonal_loading=2)

    def test_large_nonhermitian_part_is_not_silently_discarded(self):
        for solver in (mvdr_weights, capon_spectrum):
            for scale in (1e-18, 1., 1e18):
                with self.assertRaisesRegex(ValueError, 'Hermitian'):
                    solver(scale*np.array([[1, 1], [-1, 1]]), [1, 1])
                result = solver(scale*np.array([[1., 1e-12], [0, 2]]), [1, 1])
                self.assertTrue(np.all(np.isfinite(result)))

    def test_lcmv_scaling_preserves_representable_solutions(self):
        for scale in (1e-300, 1e-10, 1., 1e100):
            actual = lcmv_weights(np.eye(2), scale*np.eye(2), scale*np.array([1., 1j]))
            np.testing.assert_allclose(actual, [1, 1j], rtol=1e-14, atol=0)
        large = lcmv_weights(np.eye(2), 1e-10*np.eye(2), [1e298, 1e298])
        np.testing.assert_allclose(large/1e308, [1, 1], rtol=1e-14)
        complex_large = lcmv_weights(np.eye(2), 1e-10*np.eye(2), [1e298+1e298j, 1e298-1e298j])
        np.testing.assert_allclose(complex_large.real/1e308, [1, 1], rtol=1e-14)
        np.testing.assert_allclose(complex_large.imag/1e308, [1, -1], rtol=1e-14)
        # Independent constraints with wildly different units must remain feasible.
        np.testing.assert_allclose(lcmv_weights(np.eye(2), np.diag([1e-200, 1e200]), [1e-200, 1e200]), [1, 1])

    def test_covariance_options_reject_non_scalar_and_complex_inputs(self):
        for solver in (mvdr_weights, capon_spectrum):
            for keyword in ('relative_diagonal_loading', 'condition_limit'):
                for bad in (True, 1+0j, [1.], '1', float('nan')):
                    with self.subTest(solver=solver.__name__, keyword=keyword, bad=bad):
                        with self.assertRaises(ValueError):
                            solver(np.eye(2), [1, 1], **{keyword: bad})

    def test_lcmv_nonrepresentable_result_and_rank_failure_are_explicit(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaisesRegex(ValueError, 'floating-point range'):
                lcmv_weights(np.eye(2), 1e-10*np.eye(2), [1e308, 1e308])
        for c in (np.ones((2, 2)), np.zeros((2, 1))):
            with self.assertRaises(np.linalg.LinAlgError):
                lcmv_weights(np.eye(2), c, np.ones(c.shape[1]))


if __name__ == '__main__':
    unittest.main()
