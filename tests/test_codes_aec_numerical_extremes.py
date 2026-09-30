"""Exact independent oracles for exceptional finite AEC inputs.

Expected values use Fraction directly on the public input floats, never the
production numerical helpers. Ordinary arithmetic and transactional failures
are checked separately from correctly rounded exceptional results.
"""
from fractions import Fraction as F
import unittest

import numpy as np

from codes.chapters.ch06.core.aec import NLMSState
from codes.chapters.ch06.core.aec_ipnlms import IPNLMSState
from codes.chapters.ch06.core.aec_rls import RLSState
from codes.chapters.ch06.core.aec_kalman_matrix import KalmanAECState
from codes.chapters.ch06.aec_kalman_scalar_demo import scalar_kalman_step


def f(value):
    return F.from_float(float(value))


def scalar_oracle(a, p, q, psi, x, d, w=0):
    """Independent real scalar Kalman closed form, including all intermediate values."""
    a, p, q, psi, x, d, w = map(f, (a, p, q, psi, x, d, w))
    prior = a*a*p+q
    weight = a*w
    error = d-x*weight
    innovation = prior*x*x+psi
    gain = prior*x/innovation
    posterior_weight = weight+gain*error
    return {
        'prior_covariance': float(prior), 'innovation_variance': float(innovation),
        'gain': float(gain), 'posterior_weights': float(posterior_weight),
        'posterior_covariance': float(prior*psi/innovation),
        'prior_echo': float(x*weight), 'prior_error': float(error),
        'posterior_echo': float(x*posterior_weight),
    }


class AECNumericalExtremes(unittest.TestCase):
    def assert_single_step(self, result, expected):
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(float(np.asarray(result[key]).ravel()[0]), value)

    def test_nlms_complementary_scales_retain_both_products(self):
        state = NLMSState(2, step_size=0, initial_weights=[1e308, 1e-308],
                          initial_history=[1e308])
        residual, echo = state.process([1e-308], [2.])
        expected = float(f(1e308)*f(1e-308)*2)
        self.assertEqual(echo[0], expected)
        self.assertEqual(residual[0], 2.-expected)
        np.testing.assert_array_equal(state.history, [1e-308])

    def test_all_real_filters_cancel_unrepresentable_products(self):
        weights = np.array([1e308, -1e308, 1.])
        taps = np.array([1e308, 1e308, 1.])
        expected = float(sum((f(a)*f(b) for a, b in zip(weights, taps)), F(0)))
        self.assertEqual(expected, 1.)
        for cls in (NLMSState, IPNLMSState, RLSState):
            state = cls(3, initial_weights=weights, initial_history=taps[:0:-1])
            with self.subTest(cls=cls.__name__):
                residual, echo = state.process(taps[:1], [1.], freeze=np.array([True]))
                self.assertEqual(echo[0], expected)
                self.assertEqual(residual[0], 0.)
                np.testing.assert_array_equal(state.weights, weights)

    def test_subnormal_sum_rounds_once_in_each_real_filter(self):
        value = 1e-162
        expected = float(4*f(value)*f(value))
        self.assertEqual(expected, np.nextafter(0., 1.))
        for cls in (NLMSState, IPNLMSState, RLSState):
            state = cls(4, initial_weights=[value]*4, initial_history=[value]*3)
            with self.subTest(cls=cls.__name__):
                residual, echo = state.process([value], [expected], freeze=np.array([True]))
                self.assertEqual(echo[0], expected)
                self.assertEqual(residual[0], 0.)

    def test_ordinary_nlms_scaled_prediction_keeps_original_rounding(self):
        weights = np.array([.137, -.251, .339])
        taps = np.array([.17, -.43, .23])
        ws, xs = np.max(np.abs(weights)), np.max(np.abs(taps))
        wm, we = np.frexp(ws)
        xm, xe = np.frexp(xs)
        old_prediction = float(np.ldexp(float((weights/ws) @ (taps/xs))*wm*xm, int(we+xe)))
        state = NLMSState(3, step_size=0, initial_weights=weights,
                          initial_history=taps[:0:-1])
        _, prediction = state.process(taps[:1], [0.])
        self.assertEqual(prediction[0], old_prediction)

    def test_ordinary_ipnlms_rls_prediction_keeps_numpy_rounding(self):
        weights, taps = np.array([.137, -.251, .339]), np.array([.17, -.43, .23])
        for cls in (IPNLMSState, RLSState):
            state = cls(3, initial_weights=weights, initial_history=taps[:0:-1])
            _, prediction = state.process(taps[:1], [0.], freeze=np.array([True]))
            self.assertEqual(prediction[0], float(weights @ taps))

    def test_real_prediction_overflow_failure_is_atomic(self):
        for cls in (NLMSState, IPNLMSState, RLSState):
            state = cls(2, initial_weights=[1e308, 0.], initial_history=[.25])
            before = state.weights, state.history
            inverse = state.inverse_covariance if cls is RLSState else None
            with self.subTest(cls=cls.__name__), self.assertRaises(ValueError):
                state.process([.5, 2.], [0., 0.], freeze=np.array([True, True]))
            np.testing.assert_array_equal(state.weights, before[0])
            np.testing.assert_array_equal(state.history, before[1])
            if inverse is not None:
                np.testing.assert_array_equal(state.inverse_covariance, inverse)

    def test_compound_updates_recover_representable_results_after_underflow(self):
        cases = [
            (NLMSState(1, step_size=.5, epsilon=1e-300), 1e-308, 1e-308,
             f(.5)*f(1e-308)*f(1e-308)/(f(1e-308)**2+f(1e-300))),
            (IPNLMSState(1, step_size=.5, kappa=0, denominator_floor=1e-8),
             1e150, 1e-30, f(.5)*f(.5)*f(1e150)*f(1e-30)/(f(.5)*f(1e150)**2+f(1e-8))),
            (RLSState(1, forgetting_factor=.99, initial_regularization=1e200),
             1e-200, 1e200, f(1/1e200)*f(1e-200)*f(1e200)
             /(f(.99)+f(1/1e200)*f(1e-200)**2)),
        ]
        for state, x, d, expected in cases:
            with self.subTest(state=type(state).__name__):
                expected = float(expected)
                self.assertNotEqual(expected, 0.)
                residual, echo = state.process([x], [d])
                self.assertEqual(echo[0], 0.)
                self.assertEqual(residual[0], d)
                self.assertEqual(state.weights[0], expected)

    def test_compound_update_freeze_still_advances_history(self):
        for cls in (NLMSState, IPNLMSState, RLSState):
            state = cls(2, initial_history=[.25])
            before = state.weights
            covariance = state.inverse_covariance if cls is RLSState else None
            state.process([1e-308], [1e308], freeze=np.array([True]))
            np.testing.assert_array_equal(state.weights, before)
            np.testing.assert_array_equal(state.history, [1e-308])
            if covariance is not None:
                np.testing.assert_array_equal(state.inverse_covariance, covariance)

    def test_compound_update_unrepresentable_final_result_is_atomic(self):
        cases = ((NLMSState(1, step_size=.5, epsilon=0), 1e-308),
                 (IPNLMSState(1, step_size=.5, kappa=0,
                              denominator_floor=np.nextafter(0., 1.)), 1e-308),
                 (RLSState(1, forgetting_factor=.99, initial_regularization=1e-300), 1e-200))
        for state, x in cases:
            weights, history = state.weights, state.history
            inverse = state.inverse_covariance if isinstance(state, RLSState) else None
            with self.subTest(state=type(state).__name__), self.assertRaises(ValueError):
                state.process([0., x], [0., 1e308])
            np.testing.assert_array_equal(state.weights, weights)
            np.testing.assert_array_equal(state.history, history)
            if inverse is not None:
                np.testing.assert_array_equal(state.inverse_covariance, inverse)

    def test_matrix_transition_square_underflow_and_overflow_are_recoverable(self):
        for a, p, psi in ((1e-200, 1e300, 1e-100), (1e200, 1e-300, 1e100)):
            state = KalmanAECState(1, transition=a, process_covariance=[[0.]],
                                  observation_variance=psi, initial_covariance=[[p]])
            self.assert_single_step(state.step(1., 1.), scalar_oracle(a, p, 0, psi, 1., 1.))

    def test_matrix_projected_underflow_retains_gain_and_update(self):
        state = KalmanAECState(1, transition=1, process_covariance=[[0.]],
                              observation_variance=1e-300, initial_covariance=[[1e-200]])
        result = state.step(1e-200, 1e-100)
        self.assert_single_step(result, scalar_oracle(1, 1e-200, 0, 1e-300, 1e-200, 1e-100))
        self.assertNotEqual(result['gain'][0], 0.)
        self.assertNotEqual(result['posterior_weights'][0], 0.)

    def test_matrix_joseph_outer_underflow_and_overflow_are_recoverable(self):
        for p, x, psi, d in ((1e-100, 1e200, 1e300, 1e200),
                             (1e308, 1e-200, 1e-100, 0.)):
            state = KalmanAECState(1, transition=1, process_covariance=[[0.]],
                                  observation_variance=psi, initial_covariance=[[p]])
            self.assert_single_step(state.step(x, d), scalar_oracle(1, p, 0, psi, x, d))

    def test_matrix_full_joseph_matches_independent_rational_matrix_products(self):
        # Two nonzero channels: compare the Joseph expression itself, rather
        # than using the production fallback's rank-one subtraction formula.
        p = np.array([[2e-100, 1e-100], [1e-100, 3e-100]])
        x = [1e200, -1e200]
        psi = 1e300
        pf = [[f(v) for v in row] for row in p]
        xf = list(map(f, x))
        projected = [sum(pf[i][j]*xf[j] for j in range(2)) for i in range(2)]
        s = f(psi)+sum(xf[i]*projected[i] for i in range(2))
        k = [v/s for v in projected]
        b = [[F(int(i == j))-k[i]*xf[j] for j in range(2)] for i in range(2)]
        expected = np.array([[float(sum(b[i][r]*pf[r][t]*b[j][t]
                                      for r in range(2) for t in range(2))+f(psi)*k[i]*k[j])
                              for j in range(2)] for i in range(2)])
        state = KalmanAECState(2, transition=1, process_covariance=np.zeros((2, 2)),
                              observation_variance=psi, initial_covariance=p,
                              initial_history=[x[1]])
        result = state.step(x[0], 1.)
        np.testing.assert_array_equal(result['posterior_covariance'], expected)

    def test_matrix_prediction_uses_exact_cancellation_and_subnormal_sum(self):
        for weights, taps, expected in (([1e308, -1e308, 1.], [1e308, 1e308, 1.], 1.),
                                       ([1e-162]*4, [1e-162]*4, np.nextafter(0., 1.))):
            length = len(weights)
            state = KalmanAECState(length, transition=1, process_covariance=np.zeros((length, length)),
                observation_variance=1, initial_covariance=np.zeros((length, length)),
                initial_weights=weights, initial_history=taps[1:])
            result = state.step(taps[0], expected, freeze=True)
            self.assertEqual(result['prior_echo'], expected)
            self.assertEqual(result['posterior_echo'], expected)
            self.assertEqual(result['prior_error'], 0.)

    def test_matrix_positive_covariance_underflow_rejected_atomically(self):
        tiny = np.nextafter(0., 1.)
        for freeze in (False, True):
            # Measurement update case and time prediction case are distinct.
            state = KalmanAECState(1, transition=.5 if freeze else 1.,
                process_covariance=[[0]], observation_variance=tiny,
                initial_covariance=[[tiny]], initial_weights=[.25])
            before = state.weights.copy(), state.covariance.copy(), state.history.copy()
            with self.assertRaisesRegex(ValueError, 'positive variance'):
                state.step(1., 0., freeze=freeze)
            for actual, original in zip((state.weights, state.covariance, state.history), before):
                np.testing.assert_array_equal(actual, original)

    def test_matrix_block_rolls_back_exceptional_prefix(self):
        state = KalmanAECState(1, transition=1, process_covariance=[[0]],
                              observation_variance=1, initial_covariance=[[0]],
                              initial_weights=[1e308])
        before = state.weights.copy(), state.covariance.copy(), state.history.copy()
        with self.assertRaises(ValueError):
            state.process_block([.5, 2.], [0., 0.], freeze_mask=np.array([True, True]))
        for actual, original in zip((state.weights, state.covariance, state.history), before):
            np.testing.assert_array_equal(actual, original)

    def test_exceptional_freeze_propagates_time_and_history(self):
        state = KalmanAECState(2, transition=.5, process_covariance=np.zeros((2, 2)),
                              observation_variance=1, initial_covariance=np.eye(2)*1e-100,
                              initial_weights=[2., 4.], initial_history=[.25])
        result = state.step(1e-100, 0., freeze=True)
        np.testing.assert_array_equal(result['gain'], [0, 0])
        np.testing.assert_array_equal(state.weights, [1, 2])
        np.testing.assert_array_equal(state.covariance, np.eye(2)*2.5e-101)
        np.testing.assert_array_equal(state.history, [1e-100])

    def test_scalar_complex_square_sum_retains_subnormal_variance(self):
        value = 1.5e-162
        expected = float(2*f(value)*f(value))
        self.assertEqual(expected, np.nextafter(0., 1.))
        result = scalar_kalman_step(previous_weight=0, previous_variance=1,
            transition=complex(value, value), process_variance=0, reference=0,
            observation=0, observation_variance=1)
        self.assertEqual(result['prior_variance'], expected)
        self.assertEqual(result['posterior_variance'], expected)

    def test_scalar_complex_gain_and_update_retain_compound_scales(self):
        p, x, psi, d = 1e-200, 1e-200, 1e-300, 1e-100
        result = scalar_kalman_step(previous_weight=0, previous_variance=p,
            transition=1, process_variance=0, reference=1j*x, observation=d,
            observation_variance=psi)
        denominator = f(psi)+f(p)*f(x)*f(x)
        expected_gain = -float(f(p)*f(x)/denominator)
        self.assertEqual(result['gain'], complex(0, expected_gain))
        self.assertEqual(result['posterior_weight'], complex(0, -float(f(p)*f(x)*f(d)/denominator)))
        self.assertEqual(result['posterior_variance'], float(f(p)*f(psi)/denominator))

    def test_scalar_unrepresentable_positive_posterior_is_not_zero(self):
        tiny = np.nextafter(0., 1.)
        exact = f(tiny)/2
        self.assertGreater(exact, 0)
        self.assertEqual(float(exact), 0.)
        with self.assertRaisesRegex(ValueError, 'positive variance'):
            scalar_kalman_step(previous_weight=0, previous_variance=tiny, transition=1,
                process_variance=0, reference=1, observation=0, observation_variance=tiny)

    def test_true_zero_and_representable_small_variances_remain_legal(self):
        tiny = np.nextafter(0., 1.)
        for p, a, x, expected in ((0., 1., 1., 0.), (1., 0., 1., 0.), (tiny, 1., 0., tiny)):
            result = scalar_kalman_step(previous_weight=0, previous_variance=p, transition=a,
                process_variance=0, reference=x, observation=0, observation_variance=tiny)
            self.assertEqual(result['posterior_variance'], expected)
            state = KalmanAECState(1, transition=a, initial_covariance=[[p]],
                process_covariance=[[0.]], observation_variance=tiny)
            self.assertEqual(state.step(x, 0.)['posterior_covariance'][0, 0], expected)


if __name__ == '__main__':
    unittest.main()
