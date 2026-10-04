"""Independent scalar Bayes, sphere integrals and differential controls."""
import copy
import math
import unittest
import numpy as np
from codes.chapters.ch09.core.imm_teaching import ScalarRandomWalkIMM, imm_missing_observation_example
from codes.chapters.ch09.core.spherical_tracking import (
    direction_from_angles, angles_from_direction, rotate_direction,
    normalized_direction_covariance, vmf_log_density, vmf_static_update, vmf_resultant_length,
)


class IMMTeachingTests(unittest.TestCase):
    def test_gaussian_evidence_and_missing_moments(self):
        rows = imm_missing_observation_example()['steps']
        # Direct scalar Gaussian product: P/(P+R), then density at z=3.
        densities = [math.exp(-9/4)/math.sqrt(4*math.pi), math.exp(-9/12)/math.sqrt(12*math.pi)]
        weights = np.array([.76,.24])*densities; weights /= sum(weights)
        np.testing.assert_allclose(rows[0]['mode_probabilities'], weights)
        np.testing.assert_allclose(np.exp(rows[0]['log_likelihood']), densities)
        np.testing.assert_allclose(rows[0]['means'], [1.5,2.5])
        np.testing.assert_allclose(rows[0]['variances'], [.5,5/6])
        overall = weights@np.array([1.5,2.5])
        variance = weights@np.array([.5+1.5**2,5/6+2.5**2])-overall**2
        for row in rows[1:]:
            # With F=1 interaction preserves unconditional first/second moments;
            # only new mode-weighted process variance increases the variance.
            weights = weights@np.array([[.9,.1],[.2,.8]])
            variance += 4*weights[1]
            np.testing.assert_allclose(row['mode_probabilities'], weights)
            self.assertAlmostEqual(row['fused_mean'], overall)
            self.assertAlmostEqual(row['fused_variance'], variance)
            self.assertIsNone(row['log_likelihood'])
            self.assertIsNone(row['innovation'])

    def test_equal_residual_different_density_and_recoverable_log_support(self):
        model = ScalarRandomWalkIMM([.5,.5],[0.,0.],[1.,5.],np.eye(2),[0.,0.],1.)
        row = model.step(0.)
        self.assertAlmostEqual(row['mode_probabilities'][0]/row['mode_probabilities'][1], math.sqrt(3))
        narrow = ScalarRandomWalkIMM([.5,.5],[0.,10.],[0.,0.],np.eye(2),[0.,0.],.0001)
        first = narrow.step(10.)
        self.assertEqual(first['display_underflow_indices'], [0])
        self.assertTrue(np.isfinite(narrow.log_probabilities[0]))
        second = narrow.step(0.)
        np.testing.assert_allclose(second['mode_probabilities'], [.5,.5], atol=1e-10)
        # Genuine zero old mass with identity transition cannot reach mode0.
        literal_zero = ScalarRandomWalkIMM([0.,1.],[0.,10.],[0.,0.],np.eye(2),[0.,0.],1.)
        with self.assertRaisesRegex(ValueError,'reachable'):
            literal_zero.step(0.)

    def test_invalid_inputs_and_atomic_failure(self):
        args = [[.5,.5],[0.,0.],[1.,1.],[[.9,.1],[.2,.8]],[0.,4.],1.]
        for index, value in [(0,[True,False]),(2,[-1.,1.]),(3,[[.9,.2],[.1,.8]]),(4,[0.,-1.]),(5,0.)]:
            invalid=copy.deepcopy(args); invalid[index]=value
            with self.assertRaises(ValueError):
                ScalarRandomWalkIMM(*invalid)
        model=ScalarRandomWalkIMM(*args); before=copy.deepcopy(model.__dict__)
        with self.assertRaises(ValueError):
            model.step(1e308)
        for key in before:
            np.testing.assert_array_equal(model.__dict__[key],before[key])


class SphericalTeachingTests(unittest.TestCase):
    def test_coordinate_orientation_pose_inverse_and_pole(self):
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        u=direction_from_angles(30.,0.)
        np.testing.assert_allclose(u,[.5,math.sqrt(3)/2,0.],atol=1e-15)
        world=rotate_direction(u,r)
        np.testing.assert_allclose(world,[-math.sqrt(3)/2,.5,0.],atol=1e-15)
        self.assertAlmostEqual(angles_from_direction(world)['azimuth_deg'],-60.)
        np.testing.assert_allclose(rotate_direction(world,r.T),u,atol=1e-15)
        for angle in (0.,120.,-180.):
            np.testing.assert_array_equal(direction_from_angles(angle,90.),[0.,0.,1.])
        self.assertIsNone(angles_from_direction([0.,0.,1.])['azimuth_deg'])
        with self.assertRaises(ValueError):
            rotate_direction(u,np.diag([-1.,1.,1.]))

    def test_normalization_jacobian_and_radial_null(self):
        v=np.array([0.,2.,0.]); result=normalized_direction_covariance(v,np.eye(3)*.04)
        np.testing.assert_allclose(result['covariance'],np.diag([.01,0.,.01]))
        np.testing.assert_array_equal(result['jacobian']@v,[0.,0.,0.])
        eps=1e-5
        numeric=np.column_stack([((v+eps*e)/np.linalg.norm(v+eps*e)-(v-eps*e)/np.linalg.norm(v-eps*e))/(2*eps) for e in np.eye(3)])
        np.testing.assert_allclose(result['jacobian'],numeric,atol=1e-10)

    def test_vmf_bayes_density_ratio_and_solid_angle_integral(self):
        result=vmf_static_update([0.,1.,0.],3.,[1.,0.,0.],4.)
        np.testing.assert_allclose(result['natural_parameter'],[4.,3.,0.])
        self.assertEqual(result['concentration'],5.)
        np.testing.assert_allclose(result['mean_direction'],[.8,.6,0.])
        # Bayes product and posterior have the same relative log density;
        # normalizing constants cancel across two arbitrary test directions.
        a=np.array([0.,0.,1.]); b=np.array([1.,0.,0.])
        lhs=sum(vmf_log_density(a,mu,k)-vmf_log_density(b,mu,k) for mu,k in [([0.,1.,0.],3.),([1.,0.,0.],4.)])
        rhs=vmf_log_density(a,result['mean_direction'],5.)-vmf_log_density(b,result['mean_direction'],5.)
        self.assertAlmostEqual(lhs,rhs)
        # Independent 2pi integral over t=cos(polar angle) via Gauss quadrature.
        nodes,weights=np.polynomial.legendre.leggauss(100)
        for k in (0.,1e-6,3.,5.):
            values=[math.exp(vmf_log_density([math.sqrt(1-t*t),t,0.],[0.,1.,0.],k)) for t in nodes]
            self.assertAlmostEqual(2*math.pi*weights@values,1.,places=12)

    def test_uniform_antipodal_large_kappa_and_invalid_inputs(self):
        self.assertEqual(vmf_resultant_length(0.),0.)
        self.assertAlmostEqual(vmf_resultant_length(1e-12),1e-12/3,places=25)
        self.assertAlmostEqual(vmf_resultant_length(5.),.8000908039820194,places=15)
        self.assertEqual(vmf_resultant_length(1e100),1.)
        with np.errstate(all='raise'):
            with self.assertRaises(ValueError):
                rotate_direction([0.,1.,0.],np.full((3,3),1e308))
        result=vmf_static_update([0.,1.,0.],3.,[0.,-1.,0.],3.)
        self.assertEqual(result['status'],'uniform'); self.assertIsNone(result['mean_direction'])
        self.assertEqual(result['concentration'],0.)
        self.assertAlmostEqual(vmf_log_density([0.,1.,0.],[0.,1.,0.],1e100),math.log(1e100)-math.log(2*math.pi))
        for bad in ([0.,0.,0.],[0.,2.,0.],[False,True,False],[0j,1j,0j]):
            with self.assertRaises(ValueError):
                angles_from_direction(bad)
        with self.assertRaises(ValueError):
            direction_from_angles(0.,91.)
        with self.assertRaises(ValueError):
            vmf_static_update([0.,1.,0.],-1.,[1.,0.,0.],4.)


if __name__=='__main__':
    unittest.main()
