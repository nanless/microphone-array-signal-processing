"""Hand fractions and independent physical/metric anchors for E06-22..33."""
from fractions import Fraction as F
import json
import math
import unittest
import numpy as np
from codes.chapters.ch06.chapter06_experiments import run_experiments


class Chapter06Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_experiments()

    def test_ids_and_finite_serialization(self):
        self.assertEqual(set(self.results), {f'E06-{n}' for n in range(22,34)})
        json.dumps(self.results, allow_nan=False)

    def test_shared_normalization_independent_error_recurrence(self):
        r = self.results['E06-22']['normalization']
        self.assertEqual(r['joint']['prior_errors'], [1., -.5, .25])
        self.assertEqual(r['separate']['prior_errors'], [1., -2., 4.])
        self.assertEqual(r['joint']['next_error'], -.125)
        self.assertEqual(r['separate']['next_error'], -8.)

    def test_program_change_exposes_unidentified_difference(self):
        r = self.results['E06-23']
        self.assertEqual(r['correlation_eigenvalues'], [0., 2.])
        self.assertEqual(r['true_training_echo'], [3., -3.])
        self.assertEqual(r['alternative_training_echo'], [3., -3.])
        self.assertEqual(r['true_holdout_echo'], -1.)
        self.assertEqual(r['alternative_holdout_echo'], 0.)

    def test_delay_support_and_phase_from_quarter_period(self):
        c = self.results['E06-24']['cases']
        self.assertEqual([v['remaining_path_delay_samples'] for v in c], [2,0,-1])
        self.assertEqual([v['representable_by_causal_fir'] for v in c], [True,True,False])
        r = self.results['E06-25']
        self.assertEqual(r['uncompensated_delay_samples'], [0,4,8])
        # 1 kHz has 16 samples per cycle; 4 and 8 samples are quarter/half cycle.
        np.testing.assert_allclose(r['residual_to_echo_power_ratio'], [0,2,4], atol=1e-15)

    def test_dropout_tail_uses_delayed_missing_samples(self):
        r = self.results['E06-26']
        self.assertEqual(r['echo'], [1,2.5,4,5.5,7,8.5])
        self.assertEqual(r['prior_residual'], [0,0,3,5.5,2,0])
        self.assertEqual(r['first_fully_recovered_output_sample'], 5)

    def test_ncc_quiet_microphone_and_negative_polarity(self):
        r = self.results['E06-27']
        self.assertEqual(r['states'], [0,1,2,3,1,1])
        np.testing.assert_allclose(r['absolute_ncc'], [0,1,0,0,0,1], atol=1e-15)
        self.assertEqual(r['buffering']['first_sample_wait_seconds'], 159/16000)

    def test_prior_is_the_scored_output(self):
        r = self.results['E06-28']
        self.assertEqual((r['gain'],r['posterior_weight'],r['prior_error'],r['posterior_error']),
                         (.5,1.,2.,1.))
        self.assertAlmostEqual(r['same_observation_fit_power_difference_db'], 10*math.log10(4))

    def test_freeze_prediction_and_zero_uncertainty(self):
        r = self.results['E06-29']
        self.assertEqual(r['kalman']['weights'], [1.,.5])
        self.assertEqual(r['kalman']['variances'], [float(F(7,4)),float(F(23,16))])
        self.assertEqual(r['rls']['weight'], 2.)
        self.assertEqual(r['rls']['inverse_reference_correlation'], 3.)
        self.assertEqual(r['zero_uncertainty_wrong_prior']['gain'], 0.)
        self.assertEqual(r['zero_uncertainty_wrong_prior']['posterior_weight'], 0.)

    def test_psd_is_not_positive_definiteness(self):
        r = self.results['E06-30']['outcomes']
        self.assertTrue(r['singular_psd']['accepted'])
        self.assertFalse(any(r[k]['accepted'] for k in ('indefinite','small_asymmetric','negative_variance')))

    def test_aggregate_score_is_not_mean_db(self):
        r = self.results['E06-31']
        self.assertEqual(r['arithmetic_mean_block_db'], 10.)
        self.assertAlmostEqual(r['combined_energy_ratio_db'], 10*math.log10(float(F(10100,10001))))

    def test_normalized_gradient_has_different_expectation(self):
        r = self.results['E06-32']
        self.assertEqual(r['cross_second_moment'], 0.)
        self.assertEqual(r['mean_near_over_reference'], float(F(3,8)))
        self.assertAlmostEqual(r['batch_least_squares_path'], float(F(4,5)))
        self.assertAlmostEqual(r['expected_nlms_equilibrium'], float(F(47,40)))

    def test_increment_error_pythagorean_identity(self):
        r = self.results['E06-33']
        self.assertAlmostEqual(r['projection_gain'], float(F(4,5)))
        self.assertAlmostEqual(r['target_remainder_inner_product'], 0.)
        self.assertAlmostEqual(r['gain_error_term'], float(F(1,25)))
        self.assertAlmostEqual(r['orthogonal_error_term'], float(F(9,100)))
        self.assertAlmostEqual(r['total_relative_squared_error'], float(F(13,100)))


if __name__ == '__main__':
    unittest.main()
