"""Independent normal-equation, hand-fraction and boundary tests for RLS."""

from fractions import Fraction
import json
import unittest

import numpy as np

from codes.array_tutorial.aec_rls import RLSState, rls
from codes.examples.aec_rls_demo import run_demo


def _batch_oracle(reference, microphone, length, lam, delta, *,
                  freeze=None, initial_weights=None, initial_history=None):
    """Re-solve normal equations, never using recursive P/gain updates."""

    x = np.asarray(reference, dtype=float)
    d = np.asarray(microphone, dtype=float)
    frozen = np.zeros(len(x), dtype=bool) if freeze is None else np.asarray(freeze)
    weights = (np.zeros(length) if initial_weights is None
               else np.asarray(initial_weights, dtype=float).copy())
    history = (np.zeros(length - 1) if initial_history is None
               else np.asarray(initial_history, dtype=float).copy())
    normal = delta * np.eye(length)
    cross = normal @ weights
    predictions = []
    errors = []
    for xn, dn, skip in zip(x, d, frozen):
        regressor = np.r_[xn, history[::-1]]
        prediction = float(weights @ regressor)
        predictions.append(prediction)
        errors.append(float(dn - prediction))
        if not skip:
            normal = lam * normal + np.outer(regressor, regressor)
            cross = lam * cross + regressor * dn
            weights = np.linalg.solve(normal, cross)
        if history.size:
            history = np.r_[history[1:], xn]
    return np.array(errors), np.array(predictions), weights, np.linalg.inv(normal), history


class TestRLSState(unittest.TestCase):
    def test_two_tap_hand_case_and_direct_batch_solution(self):
        state = RLSState(2, forgetting_factor=0.5,
                         initial_regularization=1)
        first_error, first_echo = state.process([1], [1])
        np.testing.assert_allclose(first_echo, [0], atol=1e-14)
        np.testing.assert_allclose(first_error, [1], atol=1e-14)
        np.testing.assert_allclose(state.weights, [float(Fraction(2, 3)), 0], atol=1e-14)
        np.testing.assert_allclose(state.inverse_covariance,
                                   [[2 / 3, 0], [0, 2]], atol=1e-14)
        second_error, second_echo = state.process([1], [2])
        np.testing.assert_allclose(second_echo, [float(Fraction(2, 3))], atol=1e-14)
        np.testing.assert_allclose(second_error, [float(Fraction(4, 3))], atol=1e-14)
        hand_weights = [Fraction(18, 19), Fraction(16, 19)]
        hand_p = [[Fraction(20, 19), Fraction(-16, 19)],
                  [Fraction(-16, 19), Fraction(28, 19)]]
        np.testing.assert_allclose(state.weights, [float(v) for v in hand_weights],
                                   atol=1e-14)
        np.testing.assert_allclose(state.inverse_covariance,
                                   [[float(v) for v in row] for row in hand_p], atol=1e-14)

        normal = np.array([[7 / 4, 1], [1, 5 / 4]])
        cross = np.array([5 / 2, 2])
        np.testing.assert_allclose(np.linalg.solve(normal, cross), state.weights,
                                   atol=1e-14)
        np.testing.assert_allclose(np.linalg.inv(normal), state.inverse_covariance,
                                   atol=1e-14)

    def test_full_stream_matches_independent_batch_solver_and_chunking(self):
        rng = np.random.default_rng(20260923)
        x = rng.normal(size=29)
        d = np.convolve(x, [0.6, -0.3, 0.1], mode="full")[:len(x)]
        d += rng.normal(scale=0.02, size=len(x))
        frozen = np.zeros(len(x), dtype=bool)
        frozen[9:13] = True
        x[20:23] = 0.0
        initial_weights = np.array([0.1, -0.1, 0.05])
        initial_history = np.array([0.3, -0.2])
        kwargs = dict(forgetting_factor=0.97, initial_regularization=0.4,
                      initial_weights=initial_weights, initial_history=initial_history)
        whole = RLSState(3, **kwargs)
        whole_e, whole_y = whole.process(x, d, freeze=frozen)
        oracle_e, oracle_y, oracle_w, oracle_p, oracle_history = _batch_oracle(
            x, d, 3, 0.97, 0.4, freeze=frozen,
            initial_weights=initial_weights, initial_history=initial_history)
        np.testing.assert_allclose(whole_e, oracle_e, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(whole_y, oracle_y, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(whole.weights, oracle_w, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(whole.inverse_covariance, oracle_p,
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(whole.history, oracle_history, atol=0)

        chunks = RLSState(3, **kwargs)
        chunk_outputs = [chunks.process(x[start:stop], d[start:stop],
                                        freeze=frozen[start:stop])
                         for start, stop in [(0, 4), (4, 12), (12, 13), (13, 29)]]
        np.testing.assert_array_equal(np.concatenate([out[0] for out in chunk_outputs]),
                                      whole_e)
        np.testing.assert_array_equal(np.concatenate([out[1] for out in chunk_outputs]),
                                      whole_y)
        np.testing.assert_array_equal(chunks.weights, whole.weights)
        np.testing.assert_array_equal(chunks.inverse_covariance,
                                      whole.inverse_covariance)

    def test_freeze_skips_statistics_but_keeps_prediction_and_history(self):
        state = RLSState(2, forgetting_factor=0.5, initial_regularization=1)
        initial_p = state.inverse_covariance
        e, y = state.process([1, 2], [3, 4], freeze=[True, True])
        np.testing.assert_array_equal(e, [3, 4])
        np.testing.assert_array_equal(y, [0, 0])
        np.testing.assert_array_equal(state.weights, [0, 0])
        np.testing.assert_array_equal(state.inverse_covariance, initial_p)
        np.testing.assert_array_equal(state.history, [2])
        # The first accepted sample sees the *current* x=0 plus last x=2.
        state.process([0], [1])
        oracle = _batch_oracle([0], [1], 2, 0.5, 1, initial_history=[2])
        np.testing.assert_allclose(state.weights, oracle[2], atol=1e-14)

    def test_unfrozen_zero_reference_applies_forgetting(self):
        state = RLSState(2, forgetting_factor=0.5, initial_regularization=1)
        e, y = state.process([0, 0], [2, 3])
        np.testing.assert_array_equal(y, [0, 0])
        np.testing.assert_array_equal(e, [2, 3])
        np.testing.assert_array_equal(state.weights, [0, 0])
        np.testing.assert_array_equal(state.inverse_covariance, 4 * np.eye(2))

    def test_reset_and_property_copies(self):
        state = RLSState(2, initial_weights=[0.2, -0.1], initial_history=[0.4])
        state.process([1], [2])
        leaked_weights = state.weights
        leaked_p = state.inverse_covariance
        leaked_history = state.history
        leaked_weights[:] = 9
        leaked_p[:] = 9
        leaked_history[:] = 9
        self.assertFalse(np.all(state.weights == 9))
        self.assertFalse(np.all(state.inverse_covariance == 9))
        self.assertFalse(np.all(state.history == 9))
        state.reset()
        np.testing.assert_array_equal(state.weights, [0.2, -0.1])
        np.testing.assert_array_equal(state.inverse_covariance, np.eye(2))
        np.testing.assert_array_equal(state.history, [0.4])
        e, y = state.process([], [])
        self.assertEqual(e.size, 0)
        self.assertEqual(y.size, 0)
        np.testing.assert_array_equal(state.history, [0.4])

    def test_invalid_initialization_and_inputs_are_rejected(self):
        bad_parameters = [
            ("filter_length", 0), ("filter_length", True),
            ("forgetting_factor", 0), ("forgetting_factor", 1.1),
            ("forgetting_factor", np.nan), ("initial_regularization", 0),
            ("initial_regularization", 1e-320),
            ("initial_weights", [1j, 0]), ("initial_weights", [1]),
            ("initial_history", [0j]), ("initial_history", [0, 1]),
        ]
        for key, value in bad_parameters:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                RLSState(**{**{"filter_length": 2}, key: value})
        state = RLSState(2)
        bad_inputs = [
            ([1], [1, 2], None), ([1 + 0j], [1], None),
            ([1], [np.nan], None), ([1], [1], [1]),
            ([1], [1], [True, False]), (["1"], [1], None),
            ([[1]], [[1]], None),
        ]
        for x, d, freeze in bad_inputs:
            with self.subTest(x=x, d=d, freeze=freeze), self.assertRaises(ValueError):
                state.process(x, d, freeze=freeze)
        np.testing.assert_array_equal(state.weights, [0, 0])
        np.testing.assert_array_equal(state.inverse_covariance, np.eye(2))

    def test_overflow_in_later_sample_rolls_back_entire_call(self):
        state = RLSState(2)
        with self.assertRaises(ValueError):
            state.process([1, 1e308], [1, 1e308])
        np.testing.assert_array_equal(state.weights, [0, 0])
        np.testing.assert_array_equal(state.inverse_covariance, np.eye(2))
        np.testing.assert_array_equal(state.history, [0])

    def test_convenience_function_and_demo(self):
        e, y, w = rls([1, 1], [1, 2], 2, forgetting_factor=0.5)
        np.testing.assert_allclose(e, [1, 4 / 3], atol=1e-14)
        np.testing.assert_allclose(y, [0, 2 / 3], atol=1e-14)
        np.testing.assert_allclose(w, [18 / 19, 16 / 19], atol=1e-14)
        report = run_demo()
        json.dumps(report, allow_nan=False)
        self.assertAlmostEqual(report["second"]["posterior_weights"][0], 18 / 19)
        self.assertLess(report["max_recursive_vs_batch_weight_difference"], 1e-14)
        self.assertIn("not measured AEC", report["scope"])


if __name__ == "__main__":
    unittest.main()
