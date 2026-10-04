"""Independent finite controls for reference order and retained FIR history."""
import unittest

import numpy as np

from codes.chapters.ch06.core.reference_timing import reference_gain_order, echo_tail_activity
from codes.chapters.ch06.chapter06_experiments import run_experiments


class ReferenceTimingTests(unittest.TestCase):
    def test_complete_gain_order_and_independent_sum(self):
        case = reference_gain_order()
        np.testing.assert_array_equal(case['true_echo'], [1., 1.5, 2.5, 3., 1.])
        np.testing.assert_array_equal(case['wrong_after_gain_prediction'], [1., 1.5, 3., 3., 1.])
        np.testing.assert_array_equal(case['early_fixed_residual'], [0., 0., 1., 1.5, .5])
        np.testing.assert_array_equal(case['wrong_after_gain_residual'], [0., 0., -.5, 0., 0.])
        np.testing.assert_array_equal(case['late_reference_residual'], np.zeros(5))
        # Independently evaluate the lag-zero/lag-one difference, including n=4.
        x, g = case['source'], case['gain']
        difference = [0.] + [.5 * (g[n-1] - g[n]) * x[n-1] for n in range(1, 5)]
        np.testing.assert_array_equal(case['wrong_after_gain_residual'], difference)

    def test_constant_gain_and_memoryless_path_controls(self):
        x = np.array([1., -2., .5, 0.])
        h = np.array([1., .5])
        g = np.full(4, 2.)
        np.testing.assert_array_equal(np.convolve(g*x, h)[:4], g*np.convolve(x, h)[:4])
        varying = np.array([1., 2., 3., 2.])
        np.testing.assert_array_equal(np.convolve(varying*x, [2.]), varying*np.convolve(x, [2.]))

    def test_tail_is_not_a_near_end_component(self):
        case = echo_tail_activity()
        self.assertEqual(case['operational_labels'], ['far_only', 'near_only', 'silence'])
        self.assertEqual(case['frame_sample_intervals'], [[0, 4], [4, 8], [8, 12]])
        np.testing.assert_array_equal(case['microphone_frames'],
                                      [[0., 0., 0., 0.], [.5, -.5, .5, -.5], [0., 0., 0., 0.]])
        np.testing.assert_array_equal(case['near_end_truth'], np.zeros(12))
        np.testing.assert_array_equal(case['known_path_prior_residual'], np.zeros(12))
        np.testing.assert_array_equal(case['absolute_ncc'], np.zeros(3))
        np.testing.assert_array_equal(case['reference_centered_rms'], [1., 0., 0.])
        np.testing.assert_array_equal(case['microphone_centered_rms'], [0., .5, 0.])

    def test_catalog_preserves_old_cases_and_adds_separate_ids(self):
        cases = run_experiments()
        self.assertEqual(list(cases), [f'E06-{i:02d}' for i in range(22, 34)] + ['E06-40', 'E06-41'])
        self.assertEqual(cases['E06-26']['prior_residual'], [0., 0., 3., 5.5, 2., 0.])


if __name__ == '__main__':
    unittest.main()
