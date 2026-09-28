"""Offline tracking report oracles: no external checkout, SciPy or network."""
import json
from pathlib import Path
import unittest

import numpy as np

from codes.chapters.ch09.examples import audit_tracking_upstream_interfaces as audit

ROOT = Path(__file__).resolve().parents[1]


class TrackingUpstreamReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'codes/chapters/ch09/reports/tracking_upstream_interfaces.json').read_text(),
                                parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

    def test_binding_and_execution_scope(self):
        r = self.report
        self.assertEqual(r['harness_sha256'], audit.sha256(audit.__file__))
        self.assertEqual(r['source_config_sha256'], audit.binding_sha256())
        self.assertEqual(r['sources'], audit.SOURCES)
        self.assertEqual(r['config'], audit.CONFIG)
        self.assertFalse(r['stonesoup_reducer']['stonesoup_package_executed'])
        self.assertIn('original package', r['filterpy']['execution'])

    def test_imm_first_prediction_against_joint_moments(self):
        records = self.report['filterpy']['imm_predict_only']
        mu = np.array([.75,.25])
        transition = np.array([[.9,.1],[.2,.8]])
        joint = mu[:,None]*transition
        new_mu = joint.sum(axis=0)
        conditional_means = (joint*np.array([0.,10.])[:,None]).sum(axis=0)/new_mu
        conditional_second = (joint*np.array([1.,104.])[:,None]).sum(axis=0)/new_mu
        conditional_var = conditional_second-conditional_means**2
        np.testing.assert_allclose(records[1]['component_means'], conditional_means)
        np.testing.assert_allclose(records[1]['component_variances'], conditional_var)
        np.testing.assert_allclose(records[1]['cbar'], new_mu)
        # No physical motion or new observation: the unconditional moments stay fixed.
        self.assertAlmostEqual(sum(new_mu*conditional_means), 2.5)
        self.assertAlmostEqual(sum(new_mu*conditional_second), 26.75)
        self.assertAlmostEqual(26.75-2.5**2, 20.5)
        self.assertAlmostEqual(records[1]['mean'], (.75*(20/29)+.25*(80/11)), places=12)
        self.assertGreater(abs(records[1]['variance']-20.5), .8)
        for rec in records[1:]:
            np.testing.assert_allclose(rec['mu'], mu)
            self.assertAlmostEqual(rec['second_raw_moment'], rec['variance']+rec['mean']**2)
        self.assertNotEqual(records[1]['mean'], records[2]['mean'])

    def test_none_uses_zero_residual_with_old_innovation_covariance(self):
        r = self.report['filterpy']['imm_update_none_after_measurement']
        likelihood = 1/np.sqrt(2*np.pi*np.array(r['retained_innovation_variances']))
        np.testing.assert_allclose(likelihood, r['likelihoods_used_by_none'])
        expected_no_observation = np.array(r['before']['cbar'])
        weighted = expected_no_observation*likelihood
        np.testing.assert_allclose(r['after']['mu'], weighted/weighted.sum())
        self.assertGreater(np.max(abs(np.array(r['after']['mu'])-expected_no_observation)), .001)
        fresh = self.report['filterpy']['imm_update_none_fresh']
        np.testing.assert_allclose(fresh['after']['mu'], fresh['before']['cbar'])
        self.assertNotEqual(fresh['before']['mu'], fresh['after']['mu'])

    def test_ekf_calls_and_independent_scalar_joseph_variance(self):
        r = self.report['filterpy']['ekf']
        for name,h in [('predict_update',2.),('predict_then_update',4.)]:
            observed = r[name]
            S = h*h*4+1
            K = 4*h/S
            variance = (1-K*h)**2*4+K*K
            self.assertAlmostEqual(observed['gain'], K)
            self.assertAlmostEqual(observed['variance'], variance)
            self.assertAlmostEqual(observed['innovation_variance'], S)
            self.assertEqual(observed['calls'][1], {'function':'Hx','x':2.})
        self.assertEqual(r['predict_update']['calls'][0]['x'], 1.)
        self.assertEqual(r['predict_then_update']['calls'][0]['x'], 2.)
        self.assertAlmostEqual(r['predict_then_update']['variance'], 4/65)
        # Original combined path also records the old state as *_prior.
        self.assertEqual(r['predict_update']['stored_x_prior'], 1.)
        self.assertEqual(r['predict_then_update']['stored_x_prior'], 2.)

    def test_angle_residual_and_unscented_polynomial(self):
        a = self.report['filterpy']['angle']
        self.assertEqual(a['merged_default']['innovation_degrees'], -358.)
        self.assertEqual(a['separate_wrapped']['innovation_degrees'], 2.)
        self.assertEqual(a['separate_wrapped']['unwrapped_mean_degrees'], 180.)
        u = self.report['filterpy']['unscented_transform']
        # For X~N(0,1), E[X²]=1, Var[X²]=E[X⁴]-1=3-1=2.
        self.assertAlmostEqual(u['mean'], 1.)
        self.assertAlmostEqual(u['variance'], 2.)
        np.testing.assert_allclose(u['Wm'], [0.,.5,.5])
        np.testing.assert_allclose(u['Wc'], [2.,.5,.5])

    def test_reducer_mass_and_second_moment(self):
        records = self.report['stonesoup_reducer']['records']
        np.testing.assert_allclose(records[0]['output_weights'], [.45,.55])
        self.assertEqual(records[0]['mass_after'], 1.)
        self.assertEqual(records[1]['output_weights'], [])
        self.assertEqual(records[1]['mass_after'], 0.)
        self.assertEqual(records[2]['output_weights'], [1.])
        r = records[3]
        self.assertEqual(r['mass_before'], 1.5)
        self.assertEqual(r['mass_after'], 1.)
        mean = 7/15
        second = (8/15)*1+(7/15)*2
        self.assertAlmostEqual(r['output_means'][0], mean)
        self.assertAlmostEqual(r['output_variances'][0], second-mean*mean)


if __name__ == '__main__':
    unittest.main()
