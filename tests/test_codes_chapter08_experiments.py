"""Analytic, fractional expectations for the twelve chapter-eight exercises."""
import json
import unittest
import numpy as np
from codes.examples.chapter08_experiments import run_experiments


def complex_array(value):
    return np.array(value['real']) + 1j * np.array(value['imag'])


class Chapter08Exercises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_experiments()

    def test_ids_and_finite_json(self):
        self.assertEqual(set(self.r), {f'E08-{i:02}' for i in range(12, 24)})
        json.dumps(self.r, allow_nan=False)

    def test_sequential_complex_rows(self):
        r = self.r['E08-12']
        expected = [[np.sqrt(2/3), -1j/np.sqrt(6)], [0, 1/np.sqrt(2)]]
        np.testing.assert_allclose(complex_array(r['sequential_rows']), expected, atol=1e-15)
        np.testing.assert_allclose(complex_array(r['simultaneous_weighted_gram']), [[1, -.5j], [.5j, 1]], atol=1e-15)

    def test_joint_vs_scalar_regression(self):
        r = self.r['E08-13']
        np.testing.assert_allclose(r['joint_gains'], [1, 2])
        np.testing.assert_allclose(r['individual_gains'], [2, 5/2])
        np.testing.assert_allclose(r['individual_sum'], [2, 5/2, 9/2])
        np.testing.assert_allclose(complex_array(r['complex_application_gain']), [2+1j])

    def test_zero_reference_image(self):
        r = self.r['E08-14']
        self.assertEqual(r['reference0_images'][1], [0, 0, 0, 0])
        self.assertEqual(r['reference1_images'][1], [0, 0, 1, -1])
        self.assertEqual(r['second_reference0_score'], 'undefined_zero_reference_and_estimate')

    def test_mixture_consistency_not_independence(self):
        r = self.r['E08-15']
        self.assertEqual(r['sum_error'], 0)
        self.assertEqual(r['output_rank'], 1)
        np.testing.assert_allclose(r['fixed_si_sdr_db'], [0, 0], atol=1e-14)

    def test_direction_vs_power(self):
        r = self.r['E08-16']
        np.testing.assert_allclose(r['identity_shape_density'], [1, 1])
        np.testing.assert_allclose(r['original_scm'], np.diag([.5, .5]))
        np.testing.assert_allclose(r['radially_changed_scm'], np.diag([.5, 4.5]))

    def test_symmetric_guided_classes(self):
        r = self.r['E08-17']
        np.testing.assert_allclose(r['posterior'], 1/3, atol=1e-15)
        self.assertEqual(r['resets'], 0)
        self.assertEqual(r['beam_diagnostics']['target_relative_eigenvalue_gap'], [0.])

    def test_conditional_prior_counterexample(self):
        r = self.r['E08-18']
        np.testing.assert_allclose(r['posterior'], [[9/10, 1/10], [0, 1], [0, 1]])
        np.testing.assert_allclose(r['mean_posterior_priors'], [.3, .7])
        self.assertAlmostEqual(r['change'], np.log(17/25))
        self.assertLess(r['change'], 0)

    def test_nmf_substep_objectives(self):
        r = self.r['E08-19']
        np.testing.assert_allclose(r['updated_basis'], [[1], [3]])
        np.testing.assert_allclose(r['updated_activation'], [[np.sqrt(2), np.sqrt(2)]])
        np.testing.assert_allclose(r['objectives_initial_basis_activation'], [20, 8+2*np.log(3), 4*np.sqrt(2)+2*np.log(2)+2*np.log(3)])

    def test_underdetermined_covariance(self):
        r = self.r['E08-20']
        np.testing.assert_allclose(r['observation_scm_a'], np.eye(2))
        np.testing.assert_allclose(r['observation_scm_b'], np.eye(2))
        np.testing.assert_allclose(r['first_wiener_image_a'], [.5, 1])
        np.testing.assert_allclose(r['first_wiener_image_b'], [.6, .4])

    def test_congruence_wiener_fractions(self):
        r = self.r['E08-21']
        np.testing.assert_allclose(r['commutator'], [[0, -5], [5, 0]])
        np.testing.assert_allclose(r['microphone_images'], [[11/12, 2/3], [13/12, 1/3]])
        np.testing.assert_allclose(r['sum_images'], [2, 1])

    def test_bounded_real_masks(self):
        r = self.r['E08-22']
        np.testing.assert_allclose(complex_array(r['complex_oracle_mask']), [5, (1+1j)/2])
        np.testing.assert_allclose(r['squared_errors'], [16/25, 1/2])

    def test_overlap_matching_and_ambiguities(self):
        r = self.r['E08-23']
        m = r['audio_parameters']['matching']
        np.testing.assert_allclose(m['absolute_centered_correlation'], [[20/101, 1], [1, 20/101]], atol=1e-13)
        self.assertEqual(m['current_indices_for_previous'], [1, 0])
        for key in ('silence', 'identical_slots'):
            self.assertEqual(r[key]['status'], 'ambiguous')
            self.assertIsNone(r[key]['current_indices_for_previous'])
