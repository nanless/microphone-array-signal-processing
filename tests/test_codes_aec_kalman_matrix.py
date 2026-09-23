"""Independent fraction, analytic, boundary and state-lifetime tests."""

from fractions import Fraction as F
import json
import unittest

import numpy as np

from codes.array_tutorial.aec_kalman_matrix import KalmanAECState
from codes.examples.aec_kalman_matrix_demo import run_demo


def _case(**overrides):
    settings = dict(
        filter_length=2, transition=1.0,
        process_covariance=np.zeros((2, 2)),
        observation_variance=1.0,
        initial_covariance=np.diag([1.0, 4.0]),
        initial_weights=[0.0, 0.0], initial_history=[1.0],
    )
    settings.update(overrides)
    return KalmanAECState(**settings)


class TestKalmanAECState(unittest.TestCase):
    def test_two_tap_joseph_and_cross_covariance_hand_case(self):
        state = _case()
        first = state.step(1.0, 3.0)
        # Independent fractions: P^- X=[1,4], S=1+1+4=6,
        # K=[1/6,4/6].  P+=P^--(P^-X)(P^-X)^T/S.
        np.testing.assert_array_equal(first["taps"], [1., 1.])
        self.assertEqual(first["prior_echo"], 0.)
        self.assertEqual(first["prior_error"], 3.)
        self.assertEqual(first["innovation_variance"], 6.)
        np.testing.assert_allclose(first["gain"], [float(F(1, 6)), float(F(2, 3))], rtol=0, atol=1e-15)
        np.testing.assert_allclose(first["posterior_weights"], [float(F(1, 2)), float(F(2))], rtol=0, atol=1e-15)
        expected_covariance = np.array([[F(5, 6), F(-2, 3)], [F(-2, 3), F(4, 3)]], dtype=float)
        np.testing.assert_allclose(first["posterior_covariance"], expected_covariance, rtol=0, atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(expected_covariance), 2 / 3)

        second = state.step(-1.0, 0.0)
        np.testing.assert_array_equal(second["taps"], [-1., 1.])
        self.assertAlmostEqual(second["prior_echo"], 1.5)
        self.assertAlmostEqual(second["prior_error"], -1.5)
        self.assertAlmostEqual(second["innovation_variance"], float(F(9, 2)))
        np.testing.assert_allclose(second["gain"], [float(F(-1, 3)), float(F(4, 9))], rtol=0, atol=1e-15)
        np.testing.assert_allclose(second["posterior_weights"], [1., float(F(4, 3))], rtol=0, atol=1e-15)
        np.testing.assert_allclose(second["posterior_covariance"],
                                   np.diag([float(F(1, 3)), float(F(4, 9))]), rtol=0, atol=1e-15)
        diagonal_only_s = float(np.diag(expected_covariance).sum() + 1.)
        self.assertAlmostEqual(diagonal_only_s, float(F(19, 6)))
        self.assertNotAlmostEqual(second["innovation_variance"], diagonal_only_s)
        self.assertTrue(np.all(np.linalg.eigvalsh(second["posterior_covariance"]) >= -1e-14))

    def test_zero_reference_prediction_and_freeze_propagate_but_do_not_update(self):
        state = _case(
            transition=.5, process_covariance=np.diag([.1, .2]),
            initial_weights=[2., 3.], initial_history=[0.],
        )
        first = state.step(0., 9.)
        np.testing.assert_array_equal(first["gain"], [0., 0.])
        np.testing.assert_allclose(first["posterior_weights"], [1., 1.5])
        np.testing.assert_allclose(first["posterior_covariance"], np.diag([.35, 1.2]))
        frozen = state.step(2., 100., freeze=True)
        np.testing.assert_array_equal(frozen["gain"], [0., 0.])
        np.testing.assert_allclose(frozen["posterior_weights"], [.5, .75])
        np.testing.assert_allclose(frozen["posterior_covariance"], np.diag([.1875, .5]))
        np.testing.assert_array_equal(state.history, [2.])
        state.reset()
        np.testing.assert_array_equal(state.weights, [2., 3.])
        np.testing.assert_array_equal(state.covariance, np.diag([1., 4.]))
        np.testing.assert_array_equal(state.history, [0.])

    def test_full_process_covariance_keeps_off_diagonal(self):
        process = [[F(1, 2), F(1, 4)], [F(1, 4), F(1, 2)]]
        state = _case(process_covariance=np.asarray(process, dtype=float), initial_history=[0.])
        step = state.step(0., 5.)  # no informative reference, prediction only
        np.testing.assert_allclose(step["posterior_covariance"],
                                   [[1.5, .25], [.25, 4.5]], rtol=0, atol=1e-15)

    def test_batch_split_equivalence_and_analytic_scalar_precision(self):
        settings = dict(
            filter_length=1, transition=1., process_covariance=[[0.]],
            observation_variance=1., initial_covariance=[[1.]],
        )
        whole = KalmanAECState(**settings)
        split = KalmanAECState(**settings)
        render = np.ones(4)
        microphone = np.full(4, 2.)
        result = whole.process_block(render, microphone)
        a = split.process_block(render[:2], microphone[:2])
        b = split.process_block(render[2:], microphone[2:])
        np.testing.assert_allclose(result["prior_echo"], [0., 1., float(F(4, 3)), float(F(3, 2))], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["prior_echo"], np.r_[a["prior_echo"], b["prior_echo"]])
        np.testing.assert_allclose(whole.weights, split.weights)
        np.testing.assert_allclose(whole.covariance, split.covariance)
        self.assertAlmostEqual(float(whole.weights[0]), float(F(8, 5)))
        self.assertAlmostEqual(float(whole.covariance[0, 0]), float(F(1, 5)))

    def test_freeze_mask_and_transactional_block_failure(self):
        state = _case()
        output = state.process_block([1., -1.], [3., 0.], freeze_mask=[False, True])
        np.testing.assert_array_equal(output["gain"][1], [0., 0.])
        np.testing.assert_allclose(state.weights, [float(F(1, 2)), float(F(2))])
        before = (state.weights.copy(), state.covariance.copy(), state.history.copy())
        with self.assertRaises(ValueError):
            state.process_block([1., 1e308], [2., 3.])
        for current, old in zip((state.weights, state.covariance, state.history), before):
            np.testing.assert_array_equal(current, old)

    def test_rejects_shapes_complex_bool_nonfinite_indefinite_and_overflow(self):
        base = dict(filter_length=2, transition=1., process_covariance=np.eye(2),
                    observation_variance=1., initial_covariance=np.eye(2))
        bad = (
            {"filter_length": True}, {"filter_length": 0},
            {"transition": 1j}, {"observation_variance": 0.},
            {"process_covariance": [[1., 2.], [2., 1.]]},
            {"initial_covariance": [[1., 1.], [0., 1.]]},
            {"initial_weights": [1., 2j]}, {"initial_history": [1., 2.]},
        )
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                KalmanAECState(**{**base, **change})
        state = KalmanAECState(**base)
        for x, d, frozen in ((1j, 1., False), (True, 1., False),
                             (float("nan"), 1., False), (1., "2", False),
                             (1., 1., 1)):
            with self.subTest(x=x, d=d, frozen=frozen), self.assertRaises(ValueError):
                state.step(x, d, freeze=frozen)
        for x, d, mask in (([1.], [1., 2.], None), ([1j], [1.], None),
                           ([1.], [1.], [1])):
            with self.subTest(x=x, d=d, mask=mask), self.assertRaises(ValueError):
                state.process_block(x, d, freeze_mask=mask)
        with self.assertRaises(ValueError):
            state.step(1e308, 1e308)

    def test_demo_json_has_hand_scope(self):
        report = run_demo()
        json.dumps(report, allow_nan=False)
        self.assertIn("not FDKF/PBFDKF", report["scope"])
        self.assertAlmostEqual(report["first"]["posterior_covariance"][0][1], -2 / 3)
        self.assertAlmostEqual(report["second"]["innovation_variance"], 9 / 2)
        self.assertAlmostEqual(report["second_innovation_variance_if_cross_covariance_discarded"], 19 / 6)


if __name__ == "__main__":
    unittest.main()
