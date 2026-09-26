"""Independent analytic answers and overflow regressions for spatial processing."""

import math
import unittest
import warnings

import numpy as np

from codes.array_tutorial.beamforming import apply_beamformer, mvdr_weights
from codes.array_tutorial.doa import capon_spectrum
from codes.examples.spatial_precision_exercises import (
    forgetting_time, mismatch_bound, run_exercises, tdoa_consistency,
)


class SpatialPrecisionExercisesTest(unittest.TestCase):
    def test_catalog_contract(self):
        result = run_exercises()
        self.assertEqual(set(result), {"E02-07", "E04-10", "E05-06"})
        self.assertTrue(all(isinstance(value, dict) for value in result.values()))

    def test_elapsed_time_not_frame_count(self):
        result = forgetting_time()
        first, second = result["cases"]
        self.assertAlmostEqual(first["time_constant_seconds"], 0.395986531620073)
        self.assertAlmostEqual(second["time_constant_seconds"], 2 * first["time_constant_seconds"])
        for row in result["cases"]:
            self.assertAlmostEqual(row["old_contribution_after_0_4_seconds"], 1 / math.e)
        self.assertAlmostEqual(result["weight_concentration_limit"], 99.0)

    def test_cycle_projection_and_common_reference_covariance(self):
        result = tdoa_consistency()
        self.assertEqual(result["closure_residual_us"], 10.0)
        projected = np.array(result["equal_weight_projection_us"])
        np.testing.assert_allclose(projected, [80 / 3, 50 / 3, -130 / 3])
        self.assertAlmostEqual(float(projected.sum()), 0.0)
        # Every feasible perturbation sums to zero; the correction must be
        # orthogonal to that constraint plane at the least-squares solution.
        correction = projected - [30, 20, -40]
        self.assertAlmostEqual(float(correction @ [1, -1, 0]), 0.0)
        self.assertAlmostEqual(float(correction @ [0, 1, -1]), 0.0)
        np.testing.assert_array_equal(result["shared_reference_covariance_us_squared"],
                                      [[200, 100], [100, 200]])
        self.assertEqual(result["shared_reference_correlation"], 0.5)
        np.testing.assert_array_equal(result["ideal_nearest_grid_error_degrees"], [0, 0.5, 1])

    def test_bound_and_explicit_attaining_perturbation(self):
        result = mismatch_bound()
        for row, squared_norm in zip(result["cases"], (0.5, 5.0)):
            expected = 0.02 * math.sqrt(squared_norm)
            self.assertAlmostEqual(row["response_error_bound"], expected)
            self.assertAlmostEqual(row["attaining_response"], 1 - expected)
            self.assertAlmostEqual(row["white_noise_gain_linear"], 1 / squared_norm)
            self.assertAlmostEqual(np.linalg.norm(row["attaining_error_vector"]), 0.02)


class SpatialOverflowTest(unittest.TestCase):
    def test_finite_inputs_must_not_produce_false_zero_mvdr_or_capon(self):
        for function in (mvdr_weights, capon_spectrum):
            with self.subTest(function=function.__name__), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "floating-point range"):
                    function(np.eye(2), np.array([1e200, 1e200]))

    def test_beamformer_rejects_overflow_for_both_weight_shapes(self):
        for weights in (np.ones(2), np.ones((1, 2))):
            with self.subTest(shape=weights.shape), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "floating-point range"):
                    apply_beamformer(np.full((2, 1, 1), 1e308), weights)

    def test_large_but_representable_normalization_is_preserved(self):
        for scale in (1e-100, 1.0, 1e100):
            with self.subTest(scale=scale):
                vector = scale * np.array([1.0, 1.0j])
                weights = mvdr_weights(np.eye(2), vector)
                np.testing.assert_allclose(weights * scale, [0.5, 0.5j], rtol=1e-14)
                self.assertAlmostEqual(np.vdot(weights, vector).real, 1.0)
                np.testing.assert_allclose(capon_spectrum(np.eye(2), vector) * scale**2,
                                           [0.5], rtol=1e-14)
        np.testing.assert_allclose(apply_beamformer(np.full((2, 1, 1), 1e300),
                                                    np.array([0.5, 0.5])), [[1e300]])


if __name__ == "__main__":
    unittest.main()
