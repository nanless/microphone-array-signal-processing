"""Independent analytic fixtures for enhancement boundary exercises."""

import copy
import unittest
import numpy as np

from codes.array_tutorial.separation import mask_mvdr_2x2
from codes.array_tutorial.tracking import CircularParticleFilter
from codes.examples.exercises_enhancement import run_exercises


class EnhancementRoundTwo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_exercises()

    def test_echo_phase_and_noise_floor(self):
        phase = self.results['E06-04']
        np.testing.assert_allclose(phase['residual_energy_ratio'], [0, 1, 4], atol=1e-14)
        self.assertIsNone(phase['erle_db'][0])
        self.assertAlmostEqual(phase['erle_db'][2], -6.020599913279624)
        floor = self.results['E06-05']
        self.assertEqual(floor['echo_energy'], 1)
        self.assertAlmostEqual(floor['noise_energy'], .01)
        self.assertAlmostEqual(floor['cross_inner_product'], 0)
        self.assertAlmostEqual(floor['measured_input_output_ratio_db'], 20.043213737826427)

    def test_wpe_future_changes_past_filter(self):
        result = self.results['E07-04']
        np.testing.assert_allclose(result['output_a'], [[1, 0, 0, 0]], atol=1e-14)
        # R=9/4, r=5/2 -> g=10/9, independent of the tested solver.
        np.testing.assert_allclose(result['output_b'], [[1, -1/9, -1/9, 8/9]])
        self.assertFalse(result['causal'])

    def test_correlated_wpd(self):
        result = self.results['E07-05']
        np.testing.assert_allclose(result['weights'], [.4, .6, -.2, 0])
        self.assertAlmostEqual(result['constraint'], 1)
        self.assertAlmostEqual(result['objective'], .6)
        self.assertAlmostEqual(result['current_only_objective'], 2/3)

    def test_projection_back_reference_images(self):
        result = self.results['E08-04']
        np.testing.assert_allclose(result['projection_factors'], [.5, -1/6])
        np.testing.assert_allclose(result['raw_outputs'], [[2, 0, -2, 0], [0, -3, 0, 3]], atol=1e-14)
        np.testing.assert_allclose(result['reference_images'], [[1, 0, -1, 0], [0, .5, 0, -.5]], atol=1e-14)
        self.assertFalse(result['blind'])

    def test_mask_floor_not_additive_epsilon(self):
        result = self.results['E08-05']
        np.testing.assert_allclose(result['floored_covariance'], [10, 10, 10, 5])
        self.assertEqual(result['additive_epsilon_covariance_at_floor'], 5)

    def test_noise_discretization_and_gate(self):
        result = self.results['E09-04']
        np.testing.assert_allclose(result['continuous_Q'], [[1/1500, .01], [.01, .2]])
        np.testing.assert_allclose(result['piecewise_constant_Q'], [[.0001, .002], [.002, .04]])
        gate = self.results['E09-05']
        self.assertEqual(gate['raw_squared_mahalanobis'], 32041)
        self.assertEqual(gate['wrapped_squared_mahalanobis'], 1)
        self.assertFalse(gate['raw_accepted'])
        self.assertTrue(gate['wrapped_accepted'])

    def test_mvdr_common_amplitude_and_complex_phase(self):
        # Target [1,j], interference [1,-j]: analytic weight [1/2,j/2].
        x = np.array([[[1, 1], [1j, -1j]]])
        target = np.array([[.9, .1]])
        for scale in [1., 1e-200, 1e-7, 1e7, 1e200]:
            with self.subTest(scale=scale), np.errstate(over='raise', invalid='raise', divide='raise'):
                output, weights = mask_mvdr_2x2(x*scale, target, 1-target)
                np.testing.assert_allclose(weights, [[.5, .5j]], atol=1e-13)
                np.testing.assert_allclose(output/scale, [[1, 0]], atol=1e-13)

    def test_mvdr_subnormal_complex_amplitude(self):
        # Orthogonal target/interference directions [1,j] and [1,-j] give
        # weight [1/2,j/2] and output [scale,0], independent of input level.
        x = np.array([[[1, 1], [1j, -1j]]])
        target = np.array([[.9, .1]])
        for scale in [1e-309, 1e-320]:
            with self.subTest(scale=scale), np.errstate(
                    over='raise', invalid='raise', divide='raise', under='ignore'):
                output, weights = mask_mvdr_2x2(x * scale, target, 1 - target)
                np.testing.assert_allclose(weights, [[.5, .5j]], atol=1e-13)
                # Keep this assertion independent of complex division's
                # overflowing reciprocal at subnormal scales, too.
                np.testing.assert_allclose(output.real / scale, [[1, 0]], atol=1e-13)
                np.testing.assert_array_equal(output.imag, [[0, 0]])

    def test_mvdr_singular_and_zero_statistics(self):
        x = np.ones((1, 2, 2), complex)
        for target, interference, loading in [(np.zeros((1, 2)), np.ones((1, 2)), 1e-6),
                                              (np.ones((1, 2)), np.ones((1, 2)), 0)]:
            output, weights = mask_mvdr_2x2(x, target, interference, diagonal_loading=loading)
            np.testing.assert_array_equal(weights, [[1, 0]])
            np.testing.assert_array_equal(output, x[:, 0])
        output, weights = mask_mvdr_2x2(x*0, np.ones((1, 2)), np.ones((1, 2)))
        np.testing.assert_array_equal(output, [[0, 0]])
        np.testing.assert_array_equal(weights, [[1, 0]])
        with self.assertRaises(ValueError):
            mask_mvdr_2x2(np.ones((1, 2, 0)), np.ones((1, 0)), np.ones((1, 0)))

    def test_invalid_resampling_threshold_keeps_state_and_rng(self):
        for threshold in [np.nan, np.inf, -np.inf, -1, 5, True, np.array([2]), '2']:
            with self.subTest(threshold=threshold):
                particle_filter = CircularParticleFilter([0, 1, 2, 3])
                particle_filter.weights = np.array([1., 0., 0., 0.])
                rng = np.random.default_rng(12)
                original = copy.deepcopy(rng.bit_generator.state)
                with self.assertRaises(ValueError):
                    particle_filter.resample_if_needed(rng, threshold)
                self.assertEqual(rng.bit_generator.state, original)
                np.testing.assert_array_equal(particle_filter.particles, [0, 1, 2, 3])
                np.testing.assert_array_equal(particle_filter.weights, [1, 0, 0, 0])

    def test_resampling_threshold_endpoints(self):
        particle_filter = CircularParticleFilter([0, 1, 2, 3])
        particle_filter.weights = np.array([1., 0., 0., 0.])
        self.assertFalse(particle_filter.resample_if_needed(np.random.default_rng(0), 0))
        self.assertTrue(particle_filter.resample_if_needed(np.random.default_rng(0), 4))
        np.testing.assert_array_equal(particle_filter.particles, [0, 0, 0, 0])
        np.testing.assert_array_equal(particle_filter.weights, [.25, .25, .25, .25])


if __name__ == '__main__':
    unittest.main()
