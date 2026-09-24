"""Independent free-field retarded-time and generated-asset checks."""

import unittest
import numpy as np

from codes.array_tutorial.moving_source import free_field_array, retarded_emission_times, synthetic_source
from codes.examples.moving_source_audio import build_fixture


class TestMovingSource(unittest.TestCase):
    def test_static_limit_and_exact_delay(self):
        times = np.arange(1000) / 16000
        mics = np.array([[-0.05, 0.0], [0.05, 0.0]])
        source = (-0.8, 1.5)
        waves, emission = free_field_array(times, mics, source_start_xy=source,
                                            source_velocity_xy=(0., 0.))
        distances = np.linalg.norm(np.asarray(source)[None, :] - mics, axis=1)
        expected_times = times[None, :] - distances[:, None] / 343.
        np.testing.assert_allclose(emission, expected_times, atol=2e-15)
        np.testing.assert_allclose(waves, synthetic_source(expected_times) / distances[:, None], atol=1e-12)

    def test_motion_has_signed_propagation_truth(self):
        signals, manifest = build_fixture()
        truth = manifest["truth"]
        self.assertEqual(signals["moving_array"].shape, (2, 32000))
        self.assertEqual(signals["static_array"].shape, (2, 32000))
        self.assertGreater(truth["mic1_minus_mic0_travel_time_seconds"][0], 0)
        self.assertLess(truth["mic1_minus_mic0_travel_time_seconds"][-1], 0)
        self.assertLess(manifest["retarded_equation_max_residual_seconds"], 1e-10)
        self.assertFalse(np.array_equal(signals["moving_array"], signals["static_array"]))

    def test_invalid_trajectory(self):
        with self.assertRaises(ValueError):
            retarded_emission_times(np.arange(4.), np.array([[0., 0.], [0.1, 0.]]),
                                    source_start_xy=(0., 1.), source_velocity_xy=(343., 0.))


if __name__ == "__main__":
    unittest.main()
