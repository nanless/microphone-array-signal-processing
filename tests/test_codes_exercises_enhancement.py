"""Independent hand-calculated fixtures for the chapter 6--9 exercises."""

import json
import unittest
import numpy as np

from codes.examples.exercises_enhancement import run_exercises
from codes.array_tutorial.aec import nlms
from codes.array_tutorial.dereverberation import offline_wpe
from codes.array_tutorial.separation import masked_spatial_covariance, si_sdr


class TestEnhancementExercises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_exercises()

    def test_stable_ids_and_serialization(self):
        self.assertEqual(set(self.results),{f'E{chapter:02d}-{exercise:02d}' for chapter in range(6,10) for exercise in range(1,6)})
        json.dumps(self.results,allow_nan=False)
        self.assertEqual(self.results,run_exercises())

    def test_nlms_impulse_and_memory(self):
        np.testing.assert_allclose(self.results['E06-01']['weights'],[.4,-.1,.05])
        np.testing.assert_allclose(self.results['E06-01']['residual'],[.8,-.2,.1])
        np.testing.assert_allclose(self.results['E06-02']['short_residual'],[0,0,1,0])
        np.testing.assert_allclose(self.results['E06-02']['long_weights'],[0,0,1])

    def test_double_talk_control(self):
        result=self.results['E06-03']
        np.testing.assert_allclose(result['frozen_residual'],[.1,.05,1.025,1.025])
        np.testing.assert_allclose(result['frozen_weights'],[.975])
        np.testing.assert_allclose(result['active_weights'],[1.74375])
        self.assertAlmostEqual(result['valid_erle_db'],10*np.log10(160),places=9)

    def test_wpe_index_and_conjugate(self):
        result=self.results['E07-01']
        self.assertEqual(result['first_valid_frame'],4)
        self.assertEqual(result['valid_frames'],8)
        self.assertEqual(result['first_history_frames'],[1,0])
        self.assertEqual(result['regression_dimension'],4)
        self.assertLess(self.results['E07-02']['maximum_valid_residual'],1e-12)
        np.testing.assert_allclose(self.results['E07-02']['first_output_real_imag'],[1,0])

    def test_loading_eigenvalues(self):
        result=self.results['E07-03']
        np.testing.assert_allclose(result['raw_eigenvalues'],[0,2])
        np.testing.assert_allclose(result['loaded_eigenvalues'],[.1,2.1])
        self.assertAlmostEqual(result['loaded_condition_number'],21)
        np.testing.assert_allclose(result['loaded_solution'],[10/21,10/21])

    def test_si_sdr_and_pit(self):
        result=self.results['E08-01']
        self.assertAlmostEqual(result['si_sdr_db'],20)
        self.assertAlmostEqual(result['quiet_si_sdr_db'],20)
        self.assertAlmostEqual(result['one_sample_circular_shift_db'],-120)
        self.assertAlmostEqual(result['perfect_db'],120)
        self.assertEqual(self.results['E08-02']['output_to_reference'],[1,0])
        self.assertAlmostEqual(self.results['E08-02']['mean_si_sdr_db'],20)
        self.assertAlmostEqual(self.results['E08-02']['identity_fixed_mean_db'],-20)

    def test_known_mixing_is_not_blind(self):
        result=self.results['E08-03']
        self.assertFalse(result['blind'])
        self.assertAlmostEqual(result['condition_number'],3)
        self.assertAlmostEqual(result['near_singular_condition_number'],199,places=9)
        np.testing.assert_allclose(result['recovered_error'],[.02,-.02])
        np.testing.assert_allclose(result['near_singular_recovered_error'],[1,-1])
        self.assertLess(result['reconstruction_max_error'],1e-12)

    def test_kalman_seconds_and_angle_wrap(self):
        history=self.results['E09-01']['predictions']
        np.testing.assert_allclose(history[0]['state'],[30.5,5])
        np.testing.assert_allclose(history[1]['state'],[31,5])
        np.testing.assert_allclose(history[1]['covariance'],[[4.2401,.201],[.201,1.02]])
        self.assertEqual(self.results['E09-02']['wrapped_innovation_degrees'],2)
        self.assertEqual(self.results['E09-02']['updated_angle_degrees'],-180)
        self.assertAlmostEqual(self.results['E09-02']['updated_angle_variance'],.5)

    def test_particle_cdf(self):
        result=self.results['E09-03']
        self.assertAlmostEqual(result['effective_sample_size'],1/.42)
        self.assertFalse(result['threshold_would_resample'])
        self.assertTrue(.15 < result['first_position'] < .16)
        self.assertEqual(result['ancestors_zero_based'],[1,2,2,3])


class TestEnhancementInvalidInputs(unittest.TestCase):
    def test_nlms_rejects_nonfinite(self):
        for value in [np.nan,np.inf,-np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):nlms([1,value],[1,1],1)
                with self.assertRaises(ValueError):nlms([1,1],[1,value],1)
                with self.assertRaises(ValueError):nlms([1,1],[1,1],1,epsilon=value)
                with self.assertRaises(ValueError):nlms([1,1],[1,1],1,initial_weights=[value])

    def test_nlms_integer_length_and_zero_energy(self):
        for length in [True,1.5,0]:
            with self.assertRaises(ValueError):nlms([1],[1],length)
        np.testing.assert_array_equal(nlms([0,0],[1,2],1,epsilon=0)[0],[1,2])

    def test_wpe_rejects_invalid_even_when_bypassed(self):
        for value in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError):offline_wpe(np.array([[value+0j]]),taps=0,delay=1)
            for name in ['power_floor','diagonal_loading']:
                with self.assertRaises(ValueError):offline_wpe(np.ones((1,8),complex),taps=1,delay=1,**{name:value})
        for name in ['taps','delay','iterations']:
            options=dict(taps=1,delay=1,iterations=1);options[name]=1.5
            with self.assertRaises(ValueError):offline_wpe(np.ones((1,8),complex),**options)
        with self.assertRaises(ValueError):offline_wpe(np.ones((1,0),complex),taps=1,delay=1)

    def test_scm_rejects_invalid_floor(self):
        for value in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):masked_spatial_covariance(np.ones((1,2,3),complex),np.zeros((1,3)),epsilon=value)
        np.testing.assert_array_equal(masked_spatial_covariance(np.ones((1,2,3),complex),np.zeros((1,3))),np.zeros((1,2,2)))

    def test_si_sdr_independent_scaling_and_nonorthogonal_error(self):
        reference=np.array([1.,-1.,1.,-1.])
        orthogonal=np.array([1.,1.,-1.,-1.])
        estimate=1.5*reference+.2*orthogonal
        expected=10*np.log10(1.5**2/.2**2)
        for left,right in [(1.,1.),(1e-200,1e200),(-1e100,1e-100)]:
            self.assertAlmostEqual(si_sdr(left*estimate,right*reference),expected,places=10)
        self.assertAlmostEqual(si_sdr(reference,reference),120)
        self.assertAlmostEqual(si_sdr(orthogonal,reference),-120)

    def test_si_sdr_degeneracy(self):
        for estimate,reference in [([0,0],[1,-1]),([1,-1],[0,0]),([1,1],[1,-1]),([1,-1],[1,1])]:
            with self.assertRaises(ValueError):si_sdr(estimate,reference)
        for epsilon in [0,-1,1,np.inf,np.nan]:
            with self.assertRaises(ValueError):si_sdr([1,-1],[1,-1],epsilon=epsilon)

    def test_si_sdr_subnormal_epsilon_keeps_finite_caps(self):
        # The formula has finite dB caps even when the linear energy ratio
        # cannot be represented.  The near-constant case also catches an
        # underflow of epsilon times the centered estimate energy.
        reference = np.array([1., -1., 1., -1.])
        orthogonal = np.array([1., 1., -1., -1.])
        for epsilon in [1e-320, np.nextafter(0., 1.)]:
            expected = -10 * np.log10(epsilon)
            with self.subTest(epsilon=epsilon), np.errstate(all='raise'):
                self.assertAlmostEqual(si_sdr(reference, reference, epsilon=epsilon), expected)
                self.assertAlmostEqual(si_sdr(orthogonal, reference, epsilon=epsilon), -expected)
                near_constant = np.array([1., 1. + 1e-8])
                self.assertAlmostEqual(si_sdr(near_constant, near_constant, epsilon=epsilon), expected)
                self.assertAlmostEqual(si_sdr(reference + .1 * orthogonal, reference, epsilon=epsilon), 20.)


if __name__=='__main__':
    unittest.main()
