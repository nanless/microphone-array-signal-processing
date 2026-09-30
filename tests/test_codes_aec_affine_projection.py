"""Independent rational/constraint and streaming-contract checks for APA."""

from fractions import Fraction as F
import unittest

import numpy as np

from codes.chapters.ch06.core.aec import NLMSState
from codes.chapters.ch06.core.aec_affine_projection import APAState
from codes.chapters.ch06.aec_affine_projection_demo import run_demo


class TestAffineProjection(unittest.TestCase):
    def test_two_column_rational_inverse(self):
        # Independent exact 2x2 inverse, with no call to the implementation's
        # solve, Gram construction or any repository expected-value function.
        a, b, c, d = F(6), F(2), F(2), F(2)
        determinant = a*d-b*c
        e0, e1 = F(5, 4), F(1, 2)
        alpha = [(d*e0-b*e1)/determinant, (-c*e0+a*e1)/determinant]
        expected = [2*alpha[0]+alpha[1], alpha[0]]
        state = APAState(2, 2, step_size=1., regularization=1.)
        state.step(1., .5, freeze=True)
        result = state.step(2., 1.25)
        np.testing.assert_allclose(result["posterior_weights"], list(map(float, expected)), atol=1e-15)
        np.testing.assert_allclose(result["window_posterior_errors"], list(map(float, alpha)), atol=1e-15)
        self.assertEqual(result["prior_echo"], 0.)

    def test_exact_full_rank_projection_is_minimum_change(self):
        prior = np.array([.1, .2, .7])
        state = APAState(3, 2, step_size=1., regularization=0.,
                         initial_weights=prior, initial_history=[1., 0.])
        state.step(2., 1.25, freeze=True)
        result = state.step(3., 2.)
        U = result["projection_matrix"]
        np.testing.assert_allclose(U.T @ state.weights, result["desired_window"], atol=2e-15)
        # Orthogonality to a null direction characterizes the min-norm change.
        null = np.cross(U[:, 0], U[:, 1])
        self.assertAlmostEqual(float(null @ (state.weights-prior)), 0., places=14)

    def test_zero_regularization_independent_column_scales(self):
        state = APAState(2, 2, step_size=1., regularization=0.)
        state.step(1e-100, 1e-100, freeze=True)
        result = state.step(0., 0.)
        np.testing.assert_allclose(result["posterior_weights"], [1., 0.], atol=1e-15)

    def test_k1_independent_scalar_nlms_recursion(self):
        x = [1., 2., -.5, 0., 3.]
        d = [.5, 1.25, -.1, .2, .8]
        w = np.array([.1, -.2]); history = .25
        expected_echo = []; expected_error = []
        for xn, dn in zip(x, d):
            u = np.array([xn, history])
            echo = float(w[0]*u[0]+w[1]*u[1]); error = dn-echo
            w = w + .3 * error * u / (u[0]**2+u[1]**2+.7)
            expected_echo.append(echo); expected_error.append(error); history = xn
        state = APAState(2, 1, step_size=.3, regularization=.7,
                         initial_weights=[.1, -.2], initial_history=[.25])
        result = state.process_block(x, d)
        np.testing.assert_allclose(result["prior_echo"], expected_echo, atol=1e-15)
        np.testing.assert_allclose(result["prior_error"], expected_error, atol=1e-15)
        np.testing.assert_allclose(state.weights, w, atol=1e-15)

    def test_k1_matches_existing_nlms_consumer(self):
        x = np.array([.2, -.1, .7, .3, -.2]); d = np.array([.1, -.2, .3, .5, 0.])
        frozen = np.array([False, True, False, False, True])
        apa = APAState(3, 1, step_size=.2, regularization=.001)
        nlms = NLMSState(3, step_size=.2, epsilon=.001)
        actual = apa.process_block(x, d, freeze_mask=frozen)
        error, echo = nlms.process(x, d, freeze=frozen)
        np.testing.assert_allclose(actual["prior_error"], error, atol=1e-15)
        np.testing.assert_allclose(actual["prior_echo"], echo, atol=1e-15)
        np.testing.assert_allclose(apa.weights, nlms.weights, atol=1e-15)

    def test_history_order_startup_and_no_truncation(self):
        state = APAState(3, 4, step_size=0., initial_history=[7., 9.])
        result = state.step(2., 1.)
        np.testing.assert_array_equal(result["projection_matrix"], [[2.], [7.], [9.]])
        self.assertEqual(result["effective_projection_order"], 1)
        block = state.process_block([3., 4., 5., 6., 7.], [0.]*5)
        self.assertEqual(block["prior_error"].size, 5)
        np.testing.assert_array_equal(state.history, [7., 6.])

    def test_old_errors_are_recomputed(self):
        state = APAState(2, 2, step_size=1., regularization=1.)
        first = state.step(1., .5)
        result = state.step(2., 1.25)
        self.assertEqual(first["prior_error"], .5)
        np.testing.assert_allclose(result["window_prior_errors"], [.75, .25], atol=1e-15)
        np.testing.assert_allclose(result["posterior_weights"], [.5, .125], atol=1e-15)

    def test_freeze_and_explicit_projection_clear(self):
        polluted = APAState(2, 2, step_size=1., regularization=1.)
        polluted.step(1., 10., freeze=True)
        result = polluted.step(2., 1.25)
        np.testing.assert_allclose(result["posterior_weights"], [45/16, -35/16], atol=1e-15)
        cleared = APAState(2, 2, step_size=1., regularization=1.)
        cleared.step(1., 10., freeze=True)
        cleared.reset_projection_history()
        np.testing.assert_array_equal(cleared.history, [1.])
        np.testing.assert_array_equal(cleared.weights, [0., 0.])
        result = cleared.step(2., 1.25)
        self.assertEqual(result["effective_projection_order"], 1)
        np.testing.assert_allclose(result["posterior_weights"], [5/12, 5/24], atol=1e-15)

    def test_rank_deficiency_and_positive_regularization(self):
        state = APAState(2, 2, step_size=1., regularization=0., initial_history=[1.])
        state.step(1., 1., freeze=True)
        old = state.weights
        with self.assertRaises(ValueError): state.step(1., 1.)
        np.testing.assert_array_equal(state.weights, old)
        state = APAState(2, 3, step_size=1., regularization=1., initial_history=[1.])
        state.step(1., 1., freeze=True)
        result = state.step(1., 1.)
        np.testing.assert_allclose(result["posterior_weights"], [.4, .4], atol=1e-15)

    def test_regularization_units(self):
        results = []
        for scale, delta in ((1., 1.), (10., 100.)):
            state = APAState(2, 2, step_size=1., regularization=delta, initial_history=[scale])
            state.step(scale, scale, freeze=True)
            results.append(state.step(scale, scale)["posterior_weights"])
        np.testing.assert_allclose(results[0], results[1], atol=1e-15)

    def test_zero_reference_has_zero_increment_even_with_large_observation(self):
        state = APAState(2, 2, regularization=1e-300)
        result = state.step(0., 1e308)
        self.assertEqual(result["prior_error"], 1e308)
        np.testing.assert_array_equal(state.weights, [0., 0.])

    def test_positive_regularization_supports_more_observations_than_taps(self):
        state = APAState(1, 3, step_size=1., regularization=1.)
        state.step(1., 1., freeze=True)
        state.step(1., 1., freeze=True)
        result = state.step(1., 1.)
        self.assertEqual(result["effective_projection_order"], 3)
        self.assertAlmostEqual(state.weights[0], 3/4)

    def test_subnormal_rhs_is_combined_before_final_rounding(self):
        tiny = np.nextafter(0., 1.)
        state = APAState(1, 4, step_size=1., regularization=1.)
        for _ in range(3): state.step(1., tiny, freeze=True)
        state.step(1., tiny)
        # Four identical constraints: exact increment 4*tiny/5. Each small
        # system coefficient tiny/5 would underflow on its own, yet their
        # combined physical increment rounds to a representable positive value.
        self.assertEqual(state.weights[0], float(F.from_float(float(tiny))*F(4, 5)))

    def test_regularized_increment_satisfies_primal_stationarity(self):
        state = APAState(3, 2, step_size=.7, regularization=1.3,
                         initial_weights=[.1, -.2, .3], initial_history=[.4, -.6])
        state.step(.8, -.2, freeze=True)
        old = state.weights
        result = state.step(-.5, .9)
        U = result["projection_matrix"]; e = result["window_prior_errors"]
        increment = (state.weights-old)/.7
        # Primal L-dimensional gradient is independent of the implemented
        # K-dimensional Gram solve and must vanish at its regularized optimum.
        np.testing.assert_allclose(U @ (U.T @ increment-e)+1.3*increment,
                                   np.zeros(3), atol=3e-16)

    def test_prior_selects_solution_without_restoring_rank(self):
        for prior, heldout_error in (([1., 2.], 0.), ([2., 1.], -2.)):
            state = APAState(2, 2, step_size=1., regularization=1.,
                             initial_weights=prior, initial_history=[1.])
            state.step(1., 3., freeze=True)
            result = state.step(1., 3.)
            self.assertEqual(np.linalg.matrix_rank(result["projection_matrix"]), 1)
            np.testing.assert_allclose(state.weights, prior, atol=1e-15)
            self.assertEqual(-1.-float(np.array([1., -1.]) @ state.weights), heldout_error)

    def test_block_split_and_empty_block(self):
        x = [1., 2., -.5, 0., 3.]; d = [.5, 1.25, -.1, .2, .8]
        a = APAState(2, 2); b = APAState(2, 2)
        whole = a.process_block(x, d)
        p = b.process_block(x[:2], d[:2]); q = b.process_block(x[2:], d[2:])
        np.testing.assert_array_equal(whole["prior_error"], np.r_[p["prior_error"], q["prior_error"]])
        np.testing.assert_array_equal(a.weights, b.weights)
        empty = b.process_block([], [])
        self.assertEqual(empty["prior_echo"].size, 0)
        np.testing.assert_array_equal(empty["final_weights"], a.weights)

    def test_atomic_failure_late_in_block(self):
        state = APAState(2, 2, step_size=1., regularization=0.)
        # The first observation has rank 1; the next two identical regressors
        # make a deficient window, after a successful intermediate update.
        with self.assertRaises(ValueError): state.process_block([1., 1., 1.], [1., 1., 1.])
        np.testing.assert_array_equal(state.weights, [0., 0.])
        np.testing.assert_array_equal(state.history, [0.])
        result = state.step(1., 1., freeze=True)
        self.assertEqual(result["effective_projection_order"], 1)

    def test_returned_arrays_and_reset_are_independent(self):
        state = APAState(2, 2, initial_weights=[.2, .3], initial_history=[.4])
        result = state.step(1., .5)
        result["posterior_weights"][:] = 99.
        copy = state.weights; copy[:] = 88.
        self.assertFalse(np.any(state.weights == 99.))
        self.assertFalse(np.any(state.weights == 88.))
        state.reset()
        np.testing.assert_array_equal(state.weights, [.2, .3])
        np.testing.assert_array_equal(state.history, [.4])
        self.assertEqual(state.step(1., .5, freeze=True)["effective_projection_order"], 1)

    def test_invalid_options_and_data_are_rejected(self):
        for options in ({"filter_length": True}, {"projection_order": 0},
                        {"step_size": 2.}, {"regularization": -1.},
                        {"regularization": 1j}, {"initial_history": [True]}):
            args = {"filter_length": 2, "projection_order": 2, **options}
            with self.subTest(options=options), self.assertRaises(ValueError): APAState(**args)
        state = APAState(2, 2)
        for x, d, frozen in (([1., 2.], [1.], None), ([1j], [1.], None),
                              ([True], [1.], None), ([1.], [1.], [1]),
                              ([1.], [float("nan")], None)):
            with self.subTest(x=x, d=d), self.assertRaises(ValueError):
                state.process_block(x, d, freeze_mask=frozen)
        with self.assertRaises(ValueError): state.step(1., 1., freeze=1)

    def test_unrepresentable_scaled_regularization_is_not_zero(self):
        state = APAState(1, 1, regularization=np.nextafter(0., 1.))
        with self.assertRaises(ValueError): state.step(1e308, 1.)
        np.testing.assert_array_equal(state.weights, [0.])

    def test_mixed_rhs_loss_is_rejected_atomically_for_both_solvers(self):
        for delta in (0., 1.):
            with self.subTest(delta=delta):
                state = APAState(2, 2, step_size=1., regularization=delta)
                state.step(1., 1e300, freeze=True)
                snapshots = [state.weights, state.history, state._columns.copy(), state._desired.copy()]
                # U=[[0,1],[1,0]], Gram=I. The exact second coefficient
                # 1e-300/(1+delta) IS representable, but dividing both error
                # components by 1e300 would erase it. Reject the supported
                # solver's intermediate range instead of claiming a zero.
                exact_small = F.from_float(1e-300)/F(1+delta)
                self.assertGreater(float(exact_small), 0.)
                with self.assertRaisesRegex(ValueError, "normalized solve right-hand side"):
                    state.step(0., 1e-300)
                for actual, expected in zip((state.weights, state.history, state._columns, state._desired), snapshots):
                    np.testing.assert_array_equal(actual, expected)

    def test_mixed_rhs_late_failure_rolls_back_entire_block(self):
        state = APAState(2, 2, step_size=1., regularization=1.)
        with self.assertRaisesRegex(ValueError, "normalized solve right-hand side"):
            state.process_block([1., 0.], [1e300, 1e-300], freeze_mask=[True, False])
        np.testing.assert_array_equal(state.weights, [0., 0.])
        np.testing.assert_array_equal(state.history, [0.])
        self.assertEqual(state._columns.shape, (2, 0))
        self.assertEqual(state._desired.size, 0)

    def test_demo_has_all_five_cases_and_finite_output(self):
        result = run_demo()
        self.assertEqual({k for k in result if k.startswith("E06-")},
                         {f"E06-{i}" for i in range(34, 39)})
        self.assertAlmostEqual(result["E06-35"]["incorrect_posterior_weights"][0], 9/16)


if __name__ == "__main__":
    unittest.main()
