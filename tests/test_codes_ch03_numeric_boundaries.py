"""Independent integer, scale and finite-range boundaries for chapter 3."""
import unittest
import numpy as np
from codes.chapters.ch03.core.calibration import ula_prediction, fit_anchored_ula
from codes.chapters.ch03.core.covariance import spatial_covariance, recursive_covariance
from codes.chapters.ch03.coarray_covariance_exercise import average_ordered_lags, virtual_toeplitz


class Chapter03BoundaryTest(unittest.TestCase):
    def test_angles_rejected_before_real_cast(self):
        for bad in (np.array([30+7j]),['30'],[True]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                ula_prediction(bad,[1,1],[[1]],spacing_m=.04,frequency_hz=1000)
            with self.subTest(initial=bad),self.assertRaises(ValueError):
                fit_anchored_ula(np.ones((2,2,3)),spacing_m=.04,frequency_hz=1000,
                                 measured_relative_phase_rad=0,initial_angles_deg=np.repeat(bad,2))
        np.testing.assert_allclose(ula_prediction([0],[1,2],[[3]],spacing_m=.04,frequency_hz=1000),[[[3,6]]])

    def test_nonrepresentable_prediction_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'prediction'):
            ula_prediction([30],[1e200,1e200],[[1e200]],spacing_m=.04,frequency_hz=1000)
        with self.assertRaisesRegex(ValueError,'phase'):
            ula_prediction([30],[1,1],[[1]],spacing_m=1e308,frequency_hz=1e308)

    def test_lags_use_exact_python_integer_subtraction(self):
        positions=np.array([-2**63,2**63-1],dtype=np.int64)
        actual=average_ordered_lags(np.eye(2),positions)
        self.assertEqual(set(actual),{-(2**64-1),0,2**64-1})
        self.assertEqual(actual[0],1)

    def test_lag_finite_size_scale_and_nonpsd_contract(self):
        for value in (complex(np.inf),complex(np.nan),complex(1,np.inf)):
            with self.assertRaises(ValueError):virtual_toeplitz({0:value},1)
        for size in (True,np.bool_(True),0,1.5):
            with self.assertRaises(ValueError):virtual_toeplitz({0:1},size)
        for scale in (1e-320,1e-200,1.,1e200,1e308):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):average_ordered_lags(scale*np.array([[1,1],[0,1]]),[0,1])
                with self.assertRaises(ValueError):virtual_toeplitz({0:scale,1:scale,-1:0},2)
                np.testing.assert_allclose(virtual_toeplitz({0:scale,1:0,-1:0},2),scale*np.eye(2),atol=0)
        # Indefinite direct augmentation is intentional, not a validation bug.
        np.testing.assert_allclose(np.linalg.eigvalsh(virtual_toeplitz({0:2/3,3:1,-3:1,1:0,-1:0,2:0,-2:0},4)),[-1/3,2/3,2/3,5/3])
        self.assertEqual(average_ordered_lags(np.eye(2)*1e308,[0,1])[0],1e308)

    def test_zero_weight_frame_is_unused_before_square(self):
        x=np.array([[[1e200,1.]]])
        np.testing.assert_allclose(spatial_covariance(x,weights=[0,1]),[[[1]]])
        np.testing.assert_allclose(spatial_covariance(x,weights=[0,1],demean=True),[[[0]]])
        # Ratio of weights underflows if formed first, but sqrt weights and
        # amplitudes have a representable contribution: 1e150*1e-150=1.
        np.testing.assert_allclose(spatial_covariance(np.array([[[1e150,1.]]]),weights=[1e-300,1]),[[[2]]],atol=1e-14)

    def test_representable_average_and_true_constant_variance(self):
        for amplitude in (1e-200,1e200,1e308):
            np.testing.assert_array_equal(spatial_covariance(np.full((1,1,3),amplitude),weights=[1,2,3],demean=True),[[[0]]])
        np.testing.assert_allclose(spatial_covariance(np.full((1,1,4),1e154)),[[[1e308]]],rtol=2e-15)
        np.testing.assert_array_equal(spatial_covariance(np.zeros((2,1,3))),np.zeros((1,2,2)))

    def test_positive_nonrepresentable_diagonal_is_not_physical_zero(self):
        with self.assertRaisesRegex(ValueError,'underflows'):
            spatial_covariance(np.full((1,1,2),1e-200))
        with self.assertRaisesRegex(ValueError,'underflows'):
            spatial_covariance(np.array([[[1e-200,-1e-200]]]),demean=True)
        with self.assertRaisesRegex(ValueError,'underflows'):
            recursive_covariance(np.zeros((1,1,1)),[[1e-200]],forgetting_factor=.5)
        with self.assertRaisesRegex(ValueError,'underflows'):
            recursive_covariance([[[1e-200]]],[[0]],forgetting_factor=1e-200)

    def test_recursive_zero_factor_and_weight_before_square(self):
        np.testing.assert_array_equal(recursive_covariance([[[1e308]]],[[1]],forgetting_factor=0),[[[1]]])
        np.testing.assert_allclose(recursive_covariance([[[0]]],[[1.4e154]],forgetting_factor=.5),[[[.98e308]]],rtol=3e-15)
        np.testing.assert_array_equal(recursive_covariance([[[0]]],[[0]],forgetting_factor=.5),[[[0]]])


if __name__=='__main__':unittest.main()
