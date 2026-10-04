"""Independent scalar solutions and rank counterexamples for E03-18."""

import math
import unittest

import numpy as np

from codes.chapters.ch03.core.baseline_calibration import solve_baseline


class BaselineCalibrationTest(unittest.TestCase):
    directions = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    # Derive each observation from its one scalar projection, independently
    # of the fitted matrix: +x, -x, +y, +z.
    delays = np.array([20e-6 - .04/343, 20e-6 + .04/343,
                       20e-6 - .03/343, 20e-6 - .02/343])

    def test_four_direction_hand_solution_and_condition(self):
        result = solve_baseline(self.directions, self.delays)
        np.testing.assert_allclose(result["baseline_m"], [.04, .03, .02], atol=2e-17)
        self.assertAlmostEqual(result["offset_s"], (self.delays[0]+self.delays[1])/2,
                               delta=3e-20)
        self.assertEqual(result["rank"], 4)
        # D^T D has eigenvalues 2, 1, (5 +/- sqrt(17))/2.
        expected_condition = math.sqrt((5+math.sqrt(17))/(5-math.sqrt(17)))
        self.assertAlmostEqual(result["condition_number"], expected_condition, places=13)
        np.testing.assert_allclose(result["predicted_delays_s"], self.delays, atol=2e-19)
        np.testing.assert_allclose(result["residual_delays_s"], 0, atol=2e-19)
        # A direction absent from fitting checks the resulting prediction.
        heldout = 20e-6 - (.6*.04+.8*.03)/343
        predicted = result["offset_s"] - (
            .6*result["baseline_m"][0]+.8*result["baseline_m"][1])/343
        self.assertAlmostEqual(predicted, heldout, delta=2e-19)

    def test_equal_elevation_has_direction_rank_three_but_augmented_rank_three(self):
        r = math.sqrt(3)/2
        directions = np.array([[0, r, .5], [r, 0, .5], [0, -r, .5], [-r, 0, .5]])
        # Explicitly distinct heights and delays yield identical observations.
        tau = np.array([-r*.03/343-.01/343+20e-6,
                        -r*.04/343-.01/343+20e-6,
                         r*.03/343-.01/343+20e-6,
                         r*.04/343-.01/343+20e-6])
        other_tau = tau - .5*.01/343 + .005/343
        np.testing.assert_allclose(other_tau, tau, atol=1e-20)
        self.assertEqual(np.linalg.matrix_rank(directions), 3)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            solve_baseline(directions, tau)

    def test_overdetermined_error_is_not_silently_removed(self):
        directions = np.vstack((self.directions, [.6, .8, 0]))
        heldout = 20e-6 - .048/343
        delays = np.r_[self.delays, heldout+1e-6]
        result = solve_baseline(directions, delays)
        residual = np.array(result["residual_delays_s"])
        self.assertGreater(np.max(np.abs(residual)), 1e-7)
        # The one-dimensional residual space is spanned by (-2,1,-4,0,5).
        # Its squared norm is 46, so projecting e5*1us gives this exact vector.
        np.testing.assert_allclose(residual, np.array([-10, 5, -20, 0, 25])*1e-6/46,
                                   atol=2e-18)
        # A fitted residual is orthogonal to all four design columns.
        np.testing.assert_allclose(np.column_stack((-directions, np.ones(5))).T@residual,
                                   0, atol=2e-18)
        np.testing.assert_allclose(np.array(result["predicted_delays_s"])+residual,
                                   delays, atol=2e-19)

    def test_inputs_are_not_modified(self):
        directions = self.directions.copy()
        delays = self.delays.copy()
        before_u, before_t = directions.copy(), delays.copy()
        solve_baseline(directions, delays)
        np.testing.assert_array_equal(directions, before_u)
        np.testing.assert_array_equal(delays, before_t)

    def test_uniform_extreme_scale_keeps_condition_and_parameter_ratio(self):
        condition = math.sqrt((5+math.sqrt(17))/(5-math.sqrt(17)))
        for scale in (1e-200, 1., 1e200):
            with self.subTest(scale=scale):
                result = solve_baseline(self.directions, self.delays*scale)
                np.testing.assert_allclose(np.array(result["baseline_m"])/scale,
                                           [.04, .03, .02], atol=3e-17)
                self.assertAlmostEqual(result["offset_s"]/scale, 20e-6, delta=5e-20)
                self.assertAlmostEqual(result["condition_number"], condition, places=13)
                self.assertTrue(np.all(np.isfinite(result["predicted_delays_s"])))
        zero = solve_baseline(self.directions, np.zeros(4))
        self.assertEqual(zero["baseline_m"], [0., 0., 0.])
        self.assertEqual(zero["offset_s"], 0.)

    def test_sound_speed_and_delay_scaling_are_separate(self):
        # The length solution is unchanged if c is multiplied and all seconds
        # divided by the same factor, including extreme finite values.
        for speed in (343e-200, 343e200):
            result = solve_baseline(self.directions, self.delays*343/speed, speed)
            np.testing.assert_allclose(result["baseline_m"], [.04, .03, .02], atol=3e-17)
            self.assertAlmostEqual(result["offset_s"]*speed, 343*20e-6, places=15)

    def test_invalid_shapes_types_units_and_nonrepresentable_lengths(self):
        bad_inputs = [
            (self.directions[:3], self.delays[:3], 343.),
            (self.directions[:, :2], self.delays, 343.),
            (self.directions, self.delays[:, None], 343.),
            (self.directions*2, self.delays, 343.),
            (self.directions.astype(complex), self.delays, 343.),
            (self.directions, self.delays.astype(complex), 343.),
            (self.directions, self.delays, True),
            (self.directions, self.delays, 0.),
            (self.directions, [0, 0, 0, math.nan], 343.),
            (np.full((4, 3), 1e308), self.delays, 343.),
            (self.directions, np.full(4, 1e308), 343.),
            (self.directions, np.full(4, 1e-300), 1e-300),
        ]
        for directions, delays, speed in bad_inputs:
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                solve_baseline(directions, delays, speed)


if __name__ == "__main__":
    unittest.main()
