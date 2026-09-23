"""Independent numeric and boundary checks for the chapter 6 NLMS interfaces."""

import unittest

import numpy as np

from codes.array_tutorial.aec import NLMSState, erle_db, nlms
from codes.examples.aec_streaming_demo import run_demo


class TestNLMSState(unittest.TestCase):
    def test_known_fir_prediction_crosses_a_block_boundary(self):
        reference = np.array([1., 2., 3., 4.])
        path = np.array([.5, .25])
        # Hand convolution: .5, 1+.25, 1.5+.5, 2+.75.
        microphone = np.array([.5, 1.25, 2., 2.75])
        state = NLMSState(2, step_size=0., initial_weights=path)
        first = state.process(reference[:2], microphone[:2])
        self.assertEqual(state.history.tolist(), [2.])
        second = state.process(reference[2:], microphone[2:])
        np.testing.assert_array_equal(np.r_[first[0], second[0]], np.zeros(4))
        np.testing.assert_array_equal(np.r_[first[1], second[1]], microphone)
        np.testing.assert_array_equal(state.weights, path)
        self.assertEqual(state.history.tolist(), [4.])

    def test_impulse_updates_have_independent_hand_values(self):
        state = NLMSState(3, step_size=.5, epsilon=0.)
        residual, echo_hat = state.process([1., 0., 0.], [.8, -.2, .1])
        np.testing.assert_array_equal(residual, [.8, -.2, .1])
        np.testing.assert_array_equal(echo_hat, [0., 0., 0.])
        np.testing.assert_allclose(state.weights, [.4, -.1, .05], atol=1e-15)

    def test_freeze_suppresses_update_but_not_output_or_history(self):
        state = NLMSState(1, step_size=.5, epsilon=0.)
        first = state.process([1.], [1.], freeze=[False])
        second = state.process([1.], [1.], freeze=[True])
        third = state.process([0.], [0.], freeze=[False])
        np.testing.assert_array_equal(np.r_[first[0], second[0], third[0]], [1., .5, 0.])
        np.testing.assert_array_equal(np.r_[first[1], second[1], third[1]], [0., .5, 0.])
        np.testing.assert_array_equal(state.weights, [.5])

    def test_arbitrary_chunking_matches_offline_result_and_state(self):
        rng = np.random.default_rng(20260923)
        reference = rng.normal(size=23)
        path = np.array([.6, -.2, .05, .01])
        microphone = np.convolve(reference, path, mode="full")[:reference.size]
        microphone[7:14] += .3 * rng.normal(size=7)
        freeze = np.zeros(reference.size, dtype=bool)
        freeze[7:14] = True
        expected_residual, expected_echo, expected_weights = nlms(
            reference, microphone, 4, step_size=.4, epsilon=1e-8, freeze=freeze)
        for sizes in ([23], [1] * 23, [3, 0, 4, 1, 0, 8, 7]):
            with self.subTest(sizes=sizes):
                state = NLMSState(4, step_size=.4, epsilon=1e-8)
                residual_blocks = []
                echo_blocks = []
                offset = 0
                for size in sizes:
                    residual, echo = state.process(
                        reference[offset:offset + size],
                        microphone[offset:offset + size],
                        freeze=freeze[offset:offset + size],
                    )
                    residual_blocks.append(residual)
                    echo_blocks.append(echo)
                    offset += size
                self.assertEqual(offset, reference.size)
                np.testing.assert_array_equal(np.concatenate(residual_blocks), expected_residual)
                np.testing.assert_array_equal(np.concatenate(echo_blocks), expected_echo)
                np.testing.assert_array_equal(state.weights, expected_weights)
                np.testing.assert_array_equal(state.history, reference[-3:])

    def test_empty_block_and_reset_restore_initial_state(self):
        state = NLMSState(2, step_size=.5, initial_weights=[.5, .25],
                          initial_history=[2.])
        residual, echo = state.process([3.], [2.])
        # .5*3 + .25*2 = 2, so there is no update.
        np.testing.assert_array_equal(residual, [0.])
        np.testing.assert_array_equal(echo, [2.])
        before = (state.weights, state.history)
        empty = state.process([], [], freeze=np.array([], dtype=bool))
        self.assertEqual(empty[0].size, 0)
        self.assertEqual(empty[1].size, 0)
        np.testing.assert_array_equal(state.weights, before[0])
        np.testing.assert_array_equal(state.history, before[1])
        state.process([1.], [0.])
        state.reset()
        np.testing.assert_array_equal(state.weights, [.5, .25])
        np.testing.assert_array_equal(state.history, [2.])
        weights_copy = state.weights
        history_copy = state.history
        weights_copy[:] = 9
        history_copy[:] = 9
        np.testing.assert_array_equal(state.weights, [.5, .25])
        np.testing.assert_array_equal(state.history, [2.])

    def test_invalid_input_does_not_change_state(self):
        state = NLMSState(2, initial_weights=[.5, .25], initial_history=[2.])
        for reference, microphone, freeze in (
            ([1.], [1.], [float("nan")]),
            ([1.], [1.], [1]),
            ([1.], [1.], ["False"]),
            ([1.], [1.], [1 + 0j]),
            ([1.], [float("nan")], [False]),
            ([1.], [1., 2.], [False]),
        ):
            with self.subTest(freeze=freeze, microphone=microphone):
                with self.assertRaises(ValueError):
                    state.process(reference, microphone, freeze=freeze)
                np.testing.assert_array_equal(state.weights, [.5, .25])
                np.testing.assert_array_equal(state.history, [2.])
        overflow_state = NLMSState(2, initial_weights=[1e308, 0.])
        with self.assertRaisesRegex(ValueError, "prediction exceeds"):
            overflow_state.process([1e308], [1e308])
        np.testing.assert_array_equal(overflow_state.weights, [1e308, 0.])
        np.testing.assert_array_equal(overflow_state.history, [0.])

    def test_constructor_checks_and_finite_extremes(self):
        for kwargs in (
            {"filter_length": True},
            {"filter_length": 0},
            {"filter_length": 2, "initial_weights": [1.]},
            {"filter_length": 2, "initial_history": [1., 2.]},
            {"filter_length": 2, "initial_history": [1 + 0j]},
            {"filter_length": 2, "step_size": 2.},
            {"filter_length": 2, "epsilon": -1.},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NLMSState(**kwargs)
        with np.errstate(all="raise"):
            state = NLMSState(1)
            residual, echo = state.process([1e308], [1e308])
            np.testing.assert_array_equal(residual, [1e308])
            np.testing.assert_array_equal(echo, [0.])
            np.testing.assert_array_equal(state.weights, [.5])


class TestERLEMaskAndFloor(unittest.TestCase):
    def test_bool_masks_select_only_far_end_samples(self):
        microphone = [1., 1., 4., 4.]
        residual = [.1, .1, 4., 4.]
        self.assertAlmostEqual(
            erle_db(microphone, residual,
                    valid_mask=[True, True, True, True],
                    double_talk_mask=[False, False, True, True]),
            20., places=9)

    def test_masks_reject_numeric_complex_and_text_values(self):
        for bad in ([float("nan")], [1], [1.], [1 + 0j], ["True"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    nlms([1.], [1.], 1, freeze=bad)
                with self.assertRaises(ValueError):
                    erle_db([1.], [.1], valid_mask=bad)
                with self.assertRaises(ValueError):
                    erle_db([1.], [.1], double_talk_mask=bad)

    def test_regularized_erle_has_epsilon_dependent_finite_cap(self):
        for epsilon in (1e-15, 1e-9):
            with self.subTest(epsilon=epsilon):
                # Direct hand evaluation of the documented regularized ratio.
                expected = 10. * np.log10((1. + epsilon) / epsilon)
                self.assertAlmostEqual(erle_db([1.], [0.], epsilon=epsilon), expected,
                                       places=10)


class TestStreamingDemo(unittest.TestCase):
    def test_demo_numbers_come_from_hand_convolution(self):
        result = run_demo()
        self.assertEqual(result["microphone"], [.5, 1.25, 2., 2.75])
        self.assertEqual(result["history_at_split_oldest_first"], [2.])
        self.assertEqual(result["streaming_echo_hat"], result["microphone"])
        self.assertEqual(result["streaming_residual"], [0., 0., 0., 0.])
        self.assertEqual(result["restarted_residual"], [0., 0., .5, 0.])


if __name__ == "__main__":
    unittest.main()
