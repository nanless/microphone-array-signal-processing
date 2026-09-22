"""Regression anchors for finite-noise nulls and real-valued interfaces."""

import copy
import math
import unittest
import warnings

import numpy as np

from codes.array_tutorial.beamforming import dsb_weights, mvdr_weights, wiener_gain
from codes.array_tutorial.doa import gcc_phat, music_spectrum, srp_phat
from codes.array_tutorial.geometry import direction_vector
from codes.array_tutorial.tracking import ConstantVelocityKalman, CircularParticleFilter, systematic_resample, wrap_angle
from codes.examples.exercises_spatial import mvdr_finite_noise_null


class SpatialRoundThreeTest(unittest.TestCase):
    def test_position_azimuth_uses_y_axis_as_zero(self):
        direction = direction_vector(math.atan2(1, 2))
        np.testing.assert_allclose(direction, [1/math.sqrt(5), 2/math.sqrt(5)])
        self.assertAlmostEqual(math.degrees(math.atan2(1, 2)), 26.56505117707799)

    def test_capon_finite_noise_reduces_but_does_not_null_other_direction(self):
        weights = mvdr_weights([[1.25, 1], [1, 1.25]], [1, 1j])
        np.testing.assert_allclose(weights, [.5-.4j, -.4+.5j])
        self.assertAlmostEqual(np.vdot(weights, [1, 1]), .1-.1j)

    def test_finite_inr_null_angles_match_closed_form(self):
        for row in mvdr_finite_noise_null()["cases"]:
            beta = row["inr_linear"]
            expected_phase = math.pi - 2*math.atan(beta/(beta+1))
            expected_angle = math.degrees(math.asin(expected_phase/math.pi))
            self.assertAlmostEqual(row["null_angle_deg"], expected_angle)
            self.assertAlmostEqual(row["interferer_response_real"], 1/(2*(beta+1)))
            self.assertAlmostEqual(row["interferer_response_imag"], 1/(2*(beta+1)))
            self.assertGreater(row["null_angle_deg"], 30)
            self.assertLess(row["null_response_amplitude"], 1e-13)

    def test_music_accepts_ideal_rank_one_singular_covariance(self):
        result = music_spectrum(np.ones((3, 3)), np.ones(3), source_count=1)
        np.testing.assert_allclose(result, [1e15])

    def test_dsb_extreme_scale_preserves_distortionless_constraint(self):
        for scale in (1e-200, 1., 1e200):
            for base in (np.array([1., 1.]), np.array([1+1j, 1-1j])):
                a = scale*base
                with np.errstate(all="raise"):
                    weights = dsb_weights(a)
                self.assertAlmostEqual(np.vdot(weights, a), 1+0j)
                np.testing.assert_allclose(weights*scale, base/np.vdot(base, base).real)
        np.testing.assert_allclose(dsb_weights([[1e200, 1e200], [1e-200, 1e-200]])
                                   * np.array([[1e200], [1e-200]]), .5)
        for bad in ([], [0., 0.], [np.inf], [np.nextafter(0., 1.)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dsb_weights(bad)

    def test_real_interfaces_reject_type_loss_without_warning(self):
        inputs = ([1+0j, 2+1j], [True, False], ["1", "2"], np.array([1, 2], dtype=object))
        for values in inputs:
            array = np.asarray(values)
            calls = (lambda: gcc_phat(array, [1., 2.], 16000),
                     lambda: wiener_gain(array, [1., 1.]),
                     lambda: wrap_angle(array),
                     lambda: CircularParticleFilter(array),
                     lambda: ConstantVelocityKalman(array, np.eye(2), np.eye(2)),
                     lambda: systematic_resample(array, np.random.default_rng(1)),
                     lambda: srp_phat(np.ones((2, 1, 1)), [1000.], [[0., 0.], [.1, 0.]], array))
            for call in calls:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with self.assertRaises(ValueError):
                        call()

    def test_invalid_tracking_scalars_do_not_mutate_state_or_rng(self):
        kf = ConstantVelocityKalman([0., 1.], np.eye(2), np.eye(2))
        pf = CircularParticleFilter([0., 1.])
        rng = np.random.default_rng(12)
        before_rng = copy.deepcopy(rng.bit_generator.state)
        for bad in (1+0j, True, "1", [1.], float("nan")):
            for call in (lambda: kf.predict(bad), lambda: kf.update(bad, 1.),
                         lambda: kf.update(0., bad), lambda: pf.predict(bad, 1., 1., rng),
                         lambda: pf.update(0., bad)):
                with self.assertRaises(ValueError):
                    call()
            np.testing.assert_array_equal(kf.state, [0., 1.])
            np.testing.assert_array_equal(kf.covariance, np.eye(2))
            np.testing.assert_array_equal(pf.particles, [0., 1.])
            np.testing.assert_array_equal(pf.weights, [.5, .5])
            self.assertEqual(rng.bit_generator.state, before_rng)


if __name__ == "__main__":
    unittest.main()
