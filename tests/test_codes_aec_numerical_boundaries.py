"""Independent scale, covariance and input-contract regressions for AEC."""
from decimal import Decimal, localcontext
import unittest
import numpy as np

from codes.chapters.ch06.core.double_talk import ncc_activity_states
from codes.chapters.ch06.core.aec_ipnlms import IPNLMSState
from codes.chapters.ch06.core.aec_kalman_matrix import KalmanAECState
from codes.chapters.ch06.aec_kalman_scalar_demo import scalar_kalman_step


class AECNumericalBoundaries(unittest.TestCase):
    def test_ncc_scale_polarity_and_quiet_mic(self):
        for scale in (1e-300, 1., 1e100, 1e308):
            x = scale * np.array([1., -1., 1., -1.])
            for polarity in (1, -1):
                states, ncc = ncc_activity_states(x, polarity*x, frame_size=4,
                    activity_rms=scale/10, coherence_threshold=.8)
                np.testing.assert_array_equal(states, [1])
                np.testing.assert_allclose(ncc, [1.], rtol=0, atol=1e-15)
            states, _ = ncc_activity_states(x, np.zeros(4), frame_size=4,
                activity_rms=scale/10, coherence_threshold=.8)
            np.testing.assert_array_equal(states, [1])
        # A constant DC frame has zero centered activity even at extreme scale.
        states, ncc = ncc_activity_states(np.full(4, 1e308), np.full(4, 1e308),
            frame_size=4, activity_rms=.1, coherence_threshold=.8)
        np.testing.assert_array_equal(states, [0])
        np.testing.assert_array_equal(ncc, [0])

    def test_ncc_validates_before_casting(self):
        base = dict(reference=np.ones(4), microphone=np.ones(4), frame_size=4,
                    activity_rms=.1, coherence_threshold=.8)
        for key, value in [('reference', np.ones(4, complex)), ('microphone', [True]*4),
                           ('activity_rms', True), ('activity_rms', [1]),
                           ('coherence_threshold', 1j), ('frame_size', True)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ncc_activity_states(**{**base, key: value})

    def test_ipnlms_extreme_gain_uses_exact_proportions_and_floors(self):
        for weights in ([8e307, 2e307], [1.6e308, 4e307]):
            gains = IPNLMSState(2, kappa=0, initial_weights=weights).tap_gains()
            np.testing.assert_allclose(gains, [.65, .35], rtol=2e-15, atol=0)
        # Relative epsilon_g is .2: .25 + .8/(2+.2) and .25 + .2/(2+.2).
        gains = IPNLMSState(2, kappa=0, initial_weights=[8e307, 2e307],
                           gain_floor=2e307).tap_gains()
        np.testing.assert_allclose(gains, [.25+4/11, .25+1/11], rtol=2e-15)
        np.testing.assert_array_equal(IPNLMSState(2, kappa=1).tap_gains(), [0, 0])

    def test_ipnlms_process_uses_stable_gain_and_failure_is_atomic(self):
        state = IPNLMSState(2, step_size=.5, kappa=0, denominator_floor=.75,
                            initial_weights=[8e307, 2e307], initial_history=[-4.])
        e, _ = state.process([1.], [1e307])
        self.assertAlmostEqual(e[0]/1e307, 1.)
        # Exact arithmetic: g=(13/20,7/20), D=7; delta/1e307=(13/280,-1/10).
        np.testing.assert_allclose(state.weights/1e307, [8+13/280, 1.9], rtol=2e-15)
        before = state.weights, state.history
        with self.assertRaises(ValueError):
            state.process([1., 1e308], [1., 1.])
        np.testing.assert_array_equal(state.weights, before[0])
        np.testing.assert_array_equal(state.history, before[1])

    def test_scalar_posterior_variance_matches_high_precision_extremes(self):
        for p, psi in [('1e-200', '1e-200'), ('1e200', '1e-200'),
                       ('1e-200', '1e200'), ('1e200', '1e200')]:
            with localcontext() as context:
                context.prec = 100
                expected = float(Decimal(p)*Decimal(psi)/(Decimal(p)+Decimal(psi)))
            result = scalar_kalman_step(previous_weight=0, previous_variance=float(p),
                transition=1, process_variance=0, reference=1, observation=1,
                observation_variance=float(psi))
            self.assertGreater(result['posterior_variance'], 0)
            self.assertAlmostEqual(result['posterior_variance']/expected, 1., places=14)

    def test_scalar_removable_square_overflow_and_complex_conjugate(self):
        result = scalar_kalman_step(previous_weight=0, previous_variance=1e-200,
            transition=1e200, process_variance=0, reference=1j, observation=1,
            observation_variance=1e200)
        self.assertAlmostEqual(result['prior_variance']/1e200, 1.)
        self.assertAlmostEqual(result['gain'].imag, -.5)
        self.assertAlmostEqual(result['posterior_variance']/1e200, .5)

    def test_covariance_validation_is_relative_and_accepts_singular_psd(self):
        for scale in (1e-200, 1., 1e200):
            for bad in (np.array([[1., 1.], [0., 1.]]),
                        np.array([[1., 2.], [2., 1.]]), np.diag([-1e-11, 1.])):
                with self.assertRaises(ValueError):
                    KalmanAECState(2, transition=1, process_covariance=np.zeros((2,2)),
                        observation_variance=1, initial_covariance=scale*bad)
            state = KalmanAECState(2, transition=1, process_covariance=np.zeros((2,2)),
                observation_variance=1, initial_covariance=scale*np.ones((2,2)))
            np.testing.assert_allclose(state.covariance/scale, np.ones((2,2)), atol=2e-15)

    def test_smallest_positive_variance_is_not_erased_by_symmetrizing(self):
        value = np.nextafter(0., 1.)
        state = KalmanAECState(1, transition=1, process_covariance=[[0.]],
            observation_variance=1, initial_covariance=[[value]])
        self.assertEqual(state.covariance[0, 0], value)

    def test_roundoff_spectrum_is_explicitly_projected_not_propagated(self):
        eps = np.finfo(float).eps
        source = np.array([[1., 1.+eps], [1.+eps, 1.]])
        state = KalmanAECState(2, transition=1, process_covariance=np.zeros((2,2)),
            observation_variance=1, initial_covariance=source)
        # Analytic null direction must no longer have a negative quadratic form.
        direction = np.array([1., -1.])
        self.assertGreaterEqual(float(direction @ state.covariance @ direction), 0.)
        before = state.weights.copy(), state.covariance.copy(), state.history.copy()
        with self.assertRaises(ValueError):
            state.process_block([1., 1e308], [2., 3.])
        for found, old in zip((state.weights, state.covariance, state.history), before):
            np.testing.assert_array_equal(found, old)


if __name__ == '__main__':
    unittest.main()
