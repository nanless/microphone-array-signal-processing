"""Hand-derived IPNLMS and independent time-convolution subband checks."""

import json
import unittest

import numpy as np

from codes.array_tutorial.aec import NLMSState
from codes.array_tutorial.aec_ipnlms import IPNLMSState, ipnlms
from codes.array_tutorial.aec_subband import (
    HaarDiagonalSubbandNLMSState,
    haar_analyze,
    haar_synthesize,
    two_tap_crossband_matrices,
    two_tap_subband_outputs,
)
from codes.examples.aec_ipnlms_subband_demo import run_demo


class TestIPNLMS(unittest.TestCase):
    def test_nonzero_regularizers_have_hand_exact_one_step(self):
        # g=[9/20,3/10], sum=3/4 (not exactly one with gain floor 2).
        # x=[1,1], denominator=3/4+1/4=1, prior e=1/5.
        state = IPNLMSState(2, step_size=1, kappa=0,
                            denominator_floor=0.25, gain_floor=2,
                            initial_weights=[0.8, 0.2], initial_history=[1])
        np.testing.assert_allclose(state.tap_gains(), [0.45, 0.30], atol=1e-15)
        error, echo = state.process([1], [1.2])
        np.testing.assert_allclose(echo, [1], atol=1e-15)
        np.testing.assert_allclose(error, [0.2], atol=1e-15)
        np.testing.assert_allclose(state.weights, [0.89, 0.26], atol=1e-15)

    def test_kappa_minus_one_matches_nlms_with_scaled_floor(self):
        x = np.array([1., 0., 2., -1., 0., 0.5, 1.])
        d = np.convolve(x, [0.8, 0.2])[:x.size]
        frozen = np.array([False, True, False, False, True, False, False])
        ip = IPNLMSState(2, step_size=.4, kappa=-1,
                         denominator_floor=.125, gain_floor=.01)
        nl = NLMSState(2, step_size=.4, epsilon=.25)
        ip_e, ip_y = ip.process(x, d, freeze=frozen)
        nl_e, nl_y = nl.process(x, d, freeze=frozen)
        np.testing.assert_allclose(ip_e, nl_e, rtol=0, atol=2e-15)
        np.testing.assert_allclose(ip_y, nl_y, rtol=0, atol=2e-15)
        np.testing.assert_allclose(ip.weights, nl.weights, rtol=0, atol=2e-15)

    def test_zero_start_pure_proportional_stalls_but_mixed_starts(self):
        pure = IPNLMSState(2, step_size=1, kappa=1,
                           denominator_floor=.25, gain_floor=1)
        mixed = IPNLMSState(2, step_size=1, kappa=0,
                            denominator_floor=.25, gain_floor=1)
        pure.process([1], [1])
        mixed.process([1], [1])
        np.testing.assert_array_equal(pure.weights, [0, 0])
        self.assertGreater(mixed.weights[0], 0)

    def test_chunking_freeze_reset_and_convenience(self):
        x = np.array([1., 2., 0., -1., 0., 3.])
        d = np.convolve(x, [0.7, -0.2, 0.1])[:x.size]
        freeze = np.array([False, False, True, False, False, False])
        kw = dict(step_size=.6, kappa=-.2, denominator_floor=.1, gain_floor=.3)
        whole = IPNLMSState(3, **kw)
        whole_e, whole_y = whole.process(x, d, freeze=freeze)
        chunk = IPNLMSState(3, **kw)
        outputs = [chunk.process(x[a:b], d[a:b], freeze=freeze[a:b])
                   for a, b in [(0, 2), (2, 3), (3, 6)]]
        np.testing.assert_array_equal(np.concatenate([o[0] for o in outputs]), whole_e)
        np.testing.assert_array_equal(np.concatenate([o[1] for o in outputs]), whole_y)
        np.testing.assert_array_equal(chunk.weights, whole.weights)
        np.testing.assert_array_equal(chunk.history, whole.history)
        copied = whole.weights
        copied[:] = 99
        self.assertFalse(np.all(whole.weights == 99))
        whole.reset()
        np.testing.assert_array_equal(whole.weights, [0, 0, 0])
        np.testing.assert_array_equal(whole.history, [0, 0])
        e, y, w = ipnlms(x, d, 3, freeze=freeze, **kw)
        np.testing.assert_array_equal(e, np.concatenate([o[0] for o in outputs]))
        np.testing.assert_array_equal(y, np.concatenate([o[1] for o in outputs]))
        np.testing.assert_array_equal(w, chunk.weights)

    def test_invalid_and_overflow_do_not_change_state(self):
        for kwargs in ({"filter_length": 0}, {"filter_length": True},
                       {"filter_length": 2, "kappa": 2},
                       {"filter_length": 2, "step_size": np.nan},
                       {"filter_length": 2, "gain_floor": 0},
                       {"filter_length": 2, "initial_weights": [1j, 0]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IPNLMSState(**kwargs)
        state = IPNLMSState(2)
        for x, d, mask in [([1+0j], [1], None), ([1], [1, 2], None),
                           ([1], [1], [1]), ([1], [np.nan], None)]:
            with self.subTest(x=x, d=d), self.assertRaises(ValueError):
                state.process(x, d, freeze=mask)
        with self.assertRaises(ValueError):
            state.process([1, 1e308], [1, 1e308])
        np.testing.assert_array_equal(state.weights, [0, 0])
        np.testing.assert_array_equal(state.history, [0])


class TestHaarSubbandAEC(unittest.TestCase):
    def test_perfect_reconstruction_and_crossband_counterexample(self):
        x = np.array([1., 1., 0., 0.])
        y = np.array([0., 1., 1., 0.])  # independent one-sample delay
        np.testing.assert_allclose(haar_synthesize(haar_analyze(x)), x, atol=1e-15)
        np.testing.assert_allclose(haar_analyze(x)[:, 1], [0, 0], atol=1e-15)
        np.testing.assert_allclose(haar_analyze(y)[:, 1],
                                   [-1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-15)
        a0, a1 = two_tap_crossband_matrices([0, 1])
        np.testing.assert_array_equal(a0, [[.5, .5], [-.5, -.5]])
        np.testing.assert_array_equal(a1, [[.5, -.5], [.5, -.5]])
        exact, diagonal, reconstructed = two_tap_subband_outputs(x, [0, 1])
        np.testing.assert_allclose(exact, haar_analyze(y), atol=1e-15)
        np.testing.assert_allclose(reconstructed, y, atol=1e-15)
        np.testing.assert_array_equal(diagonal[:, 1], [0, 0])
        state = HaarDiagonalSubbandNLMSState(2)
        state.process(x, y)
        np.testing.assert_array_equal(state.weights[1], [0, 0])

    def test_exact_crossband_matches_independent_convolution_for_random_taps(self):
        rng = np.random.default_rng(20260923)
        for length in (2, 8, 28):
            x = rng.normal(size=length)
            h = rng.normal(size=2)
            exact, _, reconstructed = two_tap_subband_outputs(x, h)
            expected = np.convolve(x, h)[:length]
            np.testing.assert_allclose(reconstructed, expected, rtol=2e-15, atol=2e-15)
            np.testing.assert_allclose(exact, haar_analyze(expected),
                                       rtol=2e-15, atol=2e-15)

    def test_diagonal_is_exact_for_memoryless_path_and_chunking(self):
        x = np.array([1., 2., -1., 3., 4., 0., 2., 1.])
        y = .7 * x
        exact, diagonal, reconstructed = two_tap_subband_outputs(x, [.7, 0])
        np.testing.assert_allclose(exact, diagonal, atol=1e-15)
        np.testing.assert_allclose(reconstructed, y, atol=1e-15)
        kw = dict(num_taps=2, step_size=.5, epsilon=.1)
        whole = HaarDiagonalSubbandNLMSState(**kw)
        whole_e, whole_y = whole.process(x, y, freeze=[False, True, False, False])
        chunks = HaarDiagonalSubbandNLMSState(**kw)
        a = chunks.process(x[:2], y[:2], freeze=[False])
        b = chunks.process(x[2:], y[2:], freeze=[True, False, False])
        np.testing.assert_array_equal(np.r_[a[0], b[0]], whole_e)
        np.testing.assert_array_equal(np.r_[a[1], b[1]], whole_y)
        np.testing.assert_array_equal(chunks.weights, whole.weights)
        np.testing.assert_array_equal(chunks.history, whole.history)
        chunks.reset()
        np.testing.assert_array_equal(chunks.weights, np.zeros((2, 2)))

    def test_invalid_inputs_and_overflow_are_transactional(self):
        state = HaarDiagonalSubbandNLMSState(2)
        for x, d, freeze in [([1], [1], None), ([1, 2], [1], None),
                             ([1j, 2], [1, 2], None),
                             ([1, 2], [1, 2], [1]),
                             ([1, 2], [1, 2], [True, False])]:
            with self.subTest(x=x, d=d), self.assertRaises(ValueError):
                state.process(x, d, freeze=freeze)
        with self.assertRaises(ValueError):
            state.process([1, 1, 1e308, 1e308], [1, 1, 1e308, 1e308])
        np.testing.assert_array_equal(state.weights, np.zeros((2, 2)))
        np.testing.assert_array_equal(state.history, np.zeros((2, 1)))

    def test_demo_is_finite_and_hand_reproducible(self):
        report = run_demo()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["ipnlms"]["gains"], [0.45, 0.3])
        np.testing.assert_allclose(report["ipnlms"]["posterior_weights"],
                                   [0.89, 0.26], atol=1e-15)
        self.assertEqual(report["haar"]["physical_echo"], [0, 1, 1, 0])
        np.testing.assert_allclose(report["haar"]["reconstructed_echo"],
                                   [0, 1, 1, 0], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
