"""Independent boundary checks for the four-state NCC teaching detector."""

import unittest
import numpy as np

from codes.array_tutorial.double_talk import confusion_counts, ncc_activity_states
from codes.examples.aec_dtd_demo import run_experiment


class TestAutomaticDTD(unittest.TestCase):
    def test_analytic_one_frame_states(self):
        x = np.array([1., -1., 1., -1.])
        orthogonal = np.array([1., 1., -1., -1.])
        reference = np.concatenate([np.zeros(4), x, np.zeros(4), x])
        microphone = np.concatenate([np.zeros(4), 0.5 * x, orthogonal, orthogonal])
        states, ncc = ncc_activity_states(reference, microphone, frame_size=4,
                                          activity_rms=0.1, coherence_threshold=0.8)
        np.testing.assert_array_equal(states, [0, 1, 2, 3])
        np.testing.assert_allclose(ncc, [0, 1, 0, 0])

    def test_confusion_counts_and_path_change(self):
        result = run_experiment()
        counts = np.asarray(result["confusion_rows_true_columns_predicted"])
        self.assertEqual(counts.shape, (4, 4))
        np.testing.assert_array_equal(counts.sum(axis=1), [20, 40, 20, 20])
        self.assertGreater(result["changed_path_far_only_false_double_frames"], 0)
        self.assertLess(result["nlms"]["detector"]["double_echo_residual_rms"],
                        result["nlms"]["none"]["double_echo_residual_rms"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ncc_activity_states(np.ones(4), np.ones(4), frame_size=3,
                                activity_rms=0.1, coherence_threshold=0.8)
        with self.assertRaises(ValueError):
            confusion_counts(np.array([4]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
