"""Independent time-convolution checks for the full-crossband Haar AEC model."""

import unittest

import numpy as np

from codes.array_tutorial.aec_subband import (
    HaarCrossbandNLMSState,
    fir_crossband_matrices,
    haar_analyze,
    haar_synthesize,
    two_tap_crossband_matrices,
)
from codes.examples.aec_crossband_demo import run_demo


class TestHaarCrossbandAEC(unittest.TestCase):
    def test_demo_reports_a_frozen_holdout_not_training_error(self):
        report = run_demo()
        check = report["held_out"]
        self.assertEqual(check["train_blocks"], 1500)
        self.assertEqual(check["frozen_test_blocks"], 500)
        self.assertLess(check["crossband_residual_mean_square"], 1e-24)
        self.assertGreater(check["diagonal_residual_mean_square"], .1)

    def test_general_matrices_match_independent_time_convolution(self):
        rng = np.random.default_rng(20260923)
        for length in (1, 2, 3, 4, 7, 12):
            with self.subTest(length=length):
                x = rng.normal(size=60)
                h = rng.normal(size=length)
                matrices = fir_crossband_matrices(h)
                self.assertEqual(matrices.shape, (length // 2 + 1, 2, 2))
                bands = haar_analyze(x)
                predicted = np.zeros_like(bands)
                for block in range(bands.shape[0]):
                    for delay, matrix in enumerate(matrices):
                        if block >= delay:
                            predicted[block] += matrix @ bands[block - delay]
                physical = np.convolve(x, h)[:x.size]
                np.testing.assert_allclose(haar_synthesize(predicted), physical,
                                           rtol=3e-14, atol=3e-14)
                np.testing.assert_allclose(predicted, haar_analyze(physical),
                                           rtol=3e-14, atol=3e-14)
        a0, a1 = two_tap_crossband_matrices([.2, -.8])
        np.testing.assert_allclose(fir_crossband_matrices([.2, -.8]), [a0, a1],
                                   rtol=0, atol=5e-16)

    def test_hand_update_learns_crossband_path(self):
        # u0=sqrt(2), u1=0; the delayed microphone has a nonzero high band.
        state = HaarCrossbandNLMSState(2, step_size=1, epsilon=1)
        x = [1, 1, 0, 0]
        d = [0, 1, 1, 0]
        residual, echo = state.process(x, d)
        np.testing.assert_allclose(echo, [0, 0, 0, 0], atol=1e-15)
        np.testing.assert_allclose(residual, d, atol=1e-15)
        expected = np.zeros((2, 2, 2))
        expected[0, 0] = [1 / 3, 1 / 3]
        expected[1, 0] = [-1 / 3, 1 / 3]
        np.testing.assert_allclose(state.weights, expected, atol=2e-16)
        np.testing.assert_allclose(state.history, [[0], [0]], atol=1e-15)

    def test_adaptation_predicts_unseen_physical_echo(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=8000)
        physical_path = np.array([0.0, 0.7, -0.2, 0.1])
        d = np.convolve(x, physical_path)[:x.size]
        cross = HaarCrossbandNLMSState(3, step_size=.5, epsilon=.05)
        cross.process(x[:6000], d[:6000])
        holdout_error, holdout_echo = cross.process(
            x[6000:], d[6000:], freeze=np.ones(1000, dtype=bool))
        # First blocks contain the true reference history from the training call.
        np.testing.assert_allclose(holdout_echo + holdout_error, d[6000:], atol=1e-14)
        self.assertLess(np.mean(holdout_error ** 2), 1e-5 * np.mean(d[6000:] ** 2))
        self.assertTrue(np.any(np.abs(cross.weights[:, 0, :]) > 1e-2))
        self.assertTrue(np.any(np.abs(cross.weights[:, 1, :]) > 1e-2))

    def test_chunking_freeze_reset_and_zero_input(self):
        x = np.array([1., 2., -1., 3., 4., 0., 2., 1., -2., 1.])
        d = np.convolve(x, [0., 1., -.25])[:x.size]
        freeze = np.array([False, True, False, False, True])
        kw = dict(num_taps=3, step_size=.5, epsilon=.1)
        whole = HaarCrossbandNLMSState(**kw)
        whole_e, whole_y = whole.process(x, d, freeze=freeze)
        chunk = HaarCrossbandNLMSState(**kw)
        outputs = [chunk.process(x[a:b], d[a:b], freeze=freeze[a//2:b//2])
                   for a, b in ((0, 2), (2, 6), (6, 10))]
        np.testing.assert_array_equal(np.concatenate([o[0] for o in outputs]), whole_e)
        np.testing.assert_array_equal(np.concatenate([o[1] for o in outputs]), whole_y)
        np.testing.assert_array_equal(chunk.weights, whole.weights)
        np.testing.assert_array_equal(chunk.history, whole.history)
        # First flush the reference history without adapting. A zero *new*
        # block alone still has nonzero lagged regressors.
        chunk.process(np.zeros(6), np.zeros(6), freeze=[True, True, True])
        before = chunk.weights
        chunk.process([0., 0.], [0., 0.], freeze=[False])
        np.testing.assert_array_equal(chunk.weights, before)
        copied = chunk.weights
        copied[:] = 99
        self.assertFalse(np.all(chunk.weights == 99))
        chunk.reset()
        np.testing.assert_array_equal(chunk.weights, np.zeros((2, 2, 3)))
        np.testing.assert_array_equal(chunk.history, np.zeros((2, 2)))

    def test_invalid_inputs_and_overflow_are_transactional(self):
        for kwargs in ({"num_taps": 0}, {"num_taps": True},
                       {"num_taps": 2, "step_size": 2},
                       {"num_taps": 2, "epsilon": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HaarCrossbandNLMSState(**kwargs)
        for taps in ([], [[1, 2]], [1j], [np.nan]):
            with self.subTest(taps=taps), self.assertRaises(ValueError):
                fir_crossband_matrices(taps)
        state = HaarCrossbandNLMSState(2)
        for x, d, freeze in (([1], [1], None), ([1, 2], [1], None),
                             ([1j, 2], [1, 2], None), ([1, 2], [1, 2], [1]),
                             ([1, 2], [1, 2], [True, False])):
            with self.subTest(x=x, d=d), self.assertRaises(ValueError):
                state.process(x, d, freeze=freeze)
        with self.assertRaises(ValueError):
            state.process([1., 1., 1e308, 1e308], [1., 1., 1e308, 1e308])
        np.testing.assert_array_equal(state.weights, np.zeros((2, 2, 2)))
        np.testing.assert_array_equal(state.history, np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
