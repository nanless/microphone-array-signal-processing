"""Independent small-array and boundary checks for the foundation revisions."""
import math
import unittest

import numpy as np

from codes.array_tutorial.conventions import (
    finite_real_array, finite_real_scalar, hermitian_part,
    validate_frequencies, validate_positions, validate_waveforms,
)
from codes.array_tutorial.covariance import recursive_covariance, spatial_covariance
from codes.array_tutorial.geometry import direction_vector, near_field_steering, plane_wave_delays, plane_wave_steering
from codes.array_tutorial.spectral import istft, stft


class FoundationValidationTest(unittest.TestCase):
    def test_real_helpers_reject_nonreal_types_before_conversion(self):
        for value in (1j, np.array([1+0j]), True, [True], "1", ["2"], np.array([1.], dtype=object)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finite_real_array(value)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                finite_real_scalar(value)
        with self.assertRaises(ValueError):
            finite_real_scalar([1.])
        self.assertEqual(finite_real_scalar(np.int64(3)), 3.)
        np.testing.assert_array_equal(finite_real_array([1, 2.5]), [1., 2.5])

    def test_shared_real_validators_reject_complex_arrays(self):
        for function, value in ((validate_waveforms, [1,2]), (validate_positions, [[0,0],[1,0]]),
                                (validate_frequencies, [1000])):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(np.asarray(value, dtype=complex))

    def test_geometry_rejects_complex_and_invalid_reference(self):
        microphones = np.array([[0.,0.],[.04,0.]])
        for call in (lambda: direction_vector(np.array([1j])),
                     lambda: near_field_steering(microphones, np.array([1+1j,2]), [1000]),
                     lambda: plane_wave_delays(microphones, 0., sound_speed=True),
                     lambda: plane_wave_delays(microphones, 0., reference=.5),
                     lambda: plane_wave_delays(microphones, 0., reference=True)):
            with self.assertRaises(ValueError):
                call()

    def test_geometry_zero_elevation_shape_and_3d_padding(self):
        directions = direction_vector([0., math.pi/2], np.zeros(2))
        self.assertEqual(directions.shape, (2,2))
        np.testing.assert_allclose(directions, [[0,1],[1,0]], atol=1e-15)
        delays = plane_wave_delays([[0,0,0],[.04,0,0]], math.pi/2)
        np.testing.assert_allclose(delays, [0,-.04/343], atol=1e-16)

    def test_geometry_nonrepresentable_distance_or_phase_rejected(self):
        with self.assertRaises(ValueError):
            near_field_steering([[0,0],[.04,0]], [1e200,1e200], [1000])
        with self.assertRaises(ValueError):
            plane_wave_steering([[0,0],[1e200,0]], [1e200], math.pi/2)

    def test_covariance_masks_are_scale_invariant(self):
        x = np.array([[1.,0.,1.],[0.,1.,1.]])[:,None,:]
        expected = [[1.,.5],[.5,.5]]
        for scale in (1e-200, 1., 1e200):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(spatial_covariance(x, weights=np.array([1.,0.,1.])*scale)[0], expected, atol=1e-14)
        with self.assertRaises(ValueError):
            spatial_covariance(x, weights=np.zeros(3))

    def test_weighted_demeaning_uses_normalized_weights(self):
        x = np.array([[1.,3.]])[:,None,:]
        for scale in (1e-200,1.,1e200):
            np.testing.assert_allclose(spatial_covariance(x, weights=np.array([1.,3.])*scale, demean=True), [[[.75]]])

    def test_covariance_rejects_complex_masks_and_nonfinite_result(self):
        with self.assertRaises(ValueError):
            spatial_covariance(np.ones((1,1,2)), weights=np.array([1+1j,1]))
        with self.assertRaises(ValueError):
            spatial_covariance(np.ones((1,1,2))*1e200)
        with self.assertRaises(ValueError):
            recursive_covariance(np.zeros((1,1,1)), np.ones((1,1))*1e200, forgetting_factor=.5)

    def test_hermitian_average_does_not_overflow_finite_diagonal(self):
        np.testing.assert_array_equal(hermitian_part([[1e308]]), [[1e308]])
        with self.assertRaises(ValueError):
            hermitian_part([[np.inf]])

    def test_istft_invalid_floor_cannot_disable_support_check(self):
        spectrum = stft(np.arange(8.), n_fft=4, hop_length=2, center=False)
        for floor in (-1., 0., np.nan, np.inf, True, 1j):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                istft(spectrum, n_fft=4, hop_length=2, center=False, denominator_floor=floor)
        with self.assertRaises(ValueError):
            istft(spectrum, n_fft=4, hop_length=2, center=False)

    def test_complex_window_rejected_and_valid_roundtrip_retained(self):
        with self.assertRaises(ValueError):
            stft(np.arange(8.), n_fft=4, hop_length=2, window=np.ones(4, dtype=complex))
        x = np.arange(11.)-3
        np.testing.assert_allclose(istft(stft(x,n_fft=4,hop_length=2),n_fft=4,hop_length=2,length=11)[0], x, atol=1e-14)

    def test_nonrepresentable_spectral_accumulation_rejected(self):
        with self.assertRaises(ValueError):
            stft(np.full(8,1e308),n_fft=4,hop_length=2,window=np.ones(4))
        with self.assertRaises(ValueError):
            istft(np.ones((1,3,3)),n_fft=4,hop_length=2,window=np.full(4,1e200))

    def test_exercise_covariance_scales_against_hand_answer(self):
        from codes.examples.exercises_spatial import run_exercises
        result = run_exercises()["E02-06"]
        self.assertTrue(result["zero_mask_rejected"])
        self.assertEqual(len(result["cases"]), 3)
        for case in result["cases"]:
            np.testing.assert_allclose(case["matrix_real"], [[1,.5],[.5,.5]], atol=1e-14)

    def test_exercise_gain_symmetry_against_scalar_identity(self):
        from codes.examples.exercises_spatial import run_exercises
        result = run_exercises()["E03-05"]
        p = 2*math.pi*1000*.01/343
        for case in result["cases"]:
            g = case["gain"]
            expected = math.sqrt(1+g*g-2*g*math.cos(p))
            self.assertAlmostEqual(case["front_amplitude"], expected, places=13)
            self.assertAlmostEqual(case["back_amplitude"], expected, places=13)
            self.assertAlmostEqual(case["broadside_amplitude"], abs(1-g), places=13)
        self.assertAlmostEqual(result["delayed_front_amplitude"], 2*math.sin(p), places=13)
        self.assertAlmostEqual(result["delayed_back_amplitude"], 0., places=13)


if __name__ == "__main__":
    unittest.main()
