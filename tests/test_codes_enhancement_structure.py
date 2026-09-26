"""Independent exact expectations and structural boundaries for three exercises."""

from fractions import Fraction
import unittest

import numpy as np

from codes.examples.enhancement_structure_exercises import (
    cacg_relative_density, cacg_shape_step, imm_mix, wpd_factorization,
)


class WPDFactorizationTests(unittest.TestCase):
    def test_fraction_anchor_and_complete_square(self):
        r = np.array([[2, 0, 1, 0], [0, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 1.]])
        result = wpd_factorization(r, [1, 1])
        np.testing.assert_allclose(result["full"], [2/5, 3/5, -1/5, 0], atol=1e-15)
        self.assertAlmostEqual(np.vdot(result["full"], r @ result["full"]).real, 3/5)
        snapshot = np.array([2., 1, 2, 3])
        self.assertAlmostEqual(np.vdot(result["full"], snapshot).real, 1)
        np.testing.assert_allclose(snapshot[:2]-result["prediction"].conj().T@snapshot[2:], [1, 1])
        # Arbitrary nonoptimal coefficients verify the identity, not only its minimizer.
        w, h = np.array([1+2j, 3-1j]), np.array([2-1j, -3j])
        delta = h + result["prediction"] @ w
        lhs = np.vdot(np.r_[w, h], r @ np.r_[w, h])
        rhs = np.vdot(w, result["schur"] @ w) + np.vdot(delta, r[2:, 2:] @ delta)
        self.assertAlmostEqual(abs(lhs-rhs), 0)

    def test_complex_direct_solution_and_scale(self):
        generator = np.array([[1, 1j, .5], [.3j, 2, 1], [.2, 0, 1j]])
        r = generator @ generator.conj().T + np.eye(3)
        v = np.array([1, 1j])
        direct = np.linalg.solve(r, np.r_[v, 0])
        direct /= np.vdot(np.r_[v, 0], direct)
        actual = wpd_factorization(r, v)
        np.testing.assert_allclose(actual["full"], direct, atol=1e-14)
        np.testing.assert_allclose(wpd_factorization(7*r, v)["full"], direct, atol=1e-14)
        np.testing.assert_allclose(np.vdot(actual["current"], v), 1)

    def test_no_correlation_and_singular_rejected(self):
        np.testing.assert_allclose(wpd_factorization(np.diag([2, 1, 4]), [1, 1])["history"], 0)
        for r in (np.zeros((3, 3)), [[1, 1j], [1j, 2]]):
            with self.assertRaises(ValueError):
                wpd_factorization(r, [1])

    def test_extreme_scale_and_relative_hermitian_check(self):
        result = wpd_factorization(np.eye(3), [1e200, 1e200])
        np.testing.assert_allclose(result["current"] * 1e200, [.5, .5])
        asymmetric = 1e-20*np.array([[2, 1, 0], [0, 2, 0], [0, 0, 2.]])
        with self.assertRaises(ValueError):
            wpd_factorization(asymmetric, [1, 1])


class CACGShapeTests(unittest.TestCase):
    def test_exact_density_and_trace(self):
        raw, shape = cacg_shape_step(np.eye(2), [3, 1], np.eye(2))
        np.testing.assert_allclose(raw, np.diag([1.5, .5]))
        np.testing.assert_allclose(shape, raw)
        raw2, shape2 = cacg_shape_step(np.eye(2), [3, 1], shape)
        np.testing.assert_allclose(raw2, np.diag([9/4, 1/4]))
        np.testing.assert_allclose(shape2, np.diag([1.8, .2]))
        np.testing.assert_allclose(cacg_relative_density(np.eye(2), shape), [3, 1/3])
        for scale in (.01, 2, 100):
            np.testing.assert_allclose(cacg_relative_density(np.eye(2), scale*shape), [3, 1/3])
        # Multiplying the old shape changes the raw step, not its chosen trace gauge.
        np.testing.assert_allclose(cacg_shape_step(np.eye(2), [3, 1], 5*np.eye(2))[1], shape)

    def test_complex_conjugation_and_rank_boundary(self):
        z = np.array([[1, 1j], [1, -1j]]) / np.sqrt(2)
        _, shape = cacg_shape_step(z, [3, 1], np.eye(2))
        np.testing.assert_allclose(shape, [[1, -.5j], [.5j, 1]], atol=1e-15)
        _, singular = cacg_shape_step([[1, 0], [1, 0]], [3, 1], np.eye(2))
        np.testing.assert_allclose(singular, np.diag([2, 0]))
        with self.assertRaises(ValueError):
            cacg_relative_density([[1, 0]], singular)

    def test_invalid_inputs(self):
        for z, weights in (([[0, 0]], [1]), ([[2, 0]], [1]), ([[1, 0]], [0]), ([[1, 0]], [-1])):
            with self.assertRaises(ValueError):
                cacg_shape_step(z, weights, np.eye(2))

    def test_extreme_shapes_weights_and_complex_rejection(self):
        np.testing.assert_allclose(cacg_shape_step(np.eye(2), [1e308, 1e308], np.eye(2))[1], np.eye(2))
        for scale in (1e-300, 1e300):
            np.testing.assert_allclose(cacg_relative_density(np.eye(2), scale*np.diag([1.5, .5])), [3, 1/3])
            np.testing.assert_allclose(cacg_shape_step(np.eye(2), [3, 1], scale*np.eye(2))[1], np.diag([1.5, .5]))
        with self.assertRaises(ValueError):
            cacg_shape_step(np.eye(2), np.array([1+3j, 1]), np.eye(2))


class IMMMixingTests(unittest.TestCase):
    def test_independent_rational_moments(self):
        out = imm_mix([.75, .25], [[.9, .1], [.2, .8]], [[0], [10]], [[[1]], [[4]]])
        np.testing.assert_allclose(out["prior"], [29/40, 11/40])
        np.testing.assert_allclose(out["mixing"], [[27/29, 3/11], [2/29, 8/11]])
        np.testing.assert_allclose(out["means"][:, 0], [20/29, 80/11])
        # E[x²] - E[x]², independently of the implementation's centered sum.
        variance1 = Fraction(27, 29)*1 + Fraction(2, 29)*104 - Fraction(20, 29)**2
        variance2 = Fraction(3, 11)*1 + Fraction(8, 11)*104 - Fraction(80, 11)**2
        np.testing.assert_allclose(out["covariances"][:, 0, 0], [float(variance1), float(variance2)])
        self.assertEqual(variance1, Fraction(6415, 841))
        self.assertEqual(variance2, Fraction(2785, 121))
        posterior = out["prior"] * [.2, .8]
        np.testing.assert_allclose(posterior / posterior.sum(), [29/73, 44/73])

    def test_identity_and_total_variance(self):
        mu = np.array([.75, .25]); means = np.array([[0., 1], [10., 3]])
        cov = np.array([np.eye(2), 4*np.eye(2)])
        identity = imm_mix(mu, np.eye(2), means, cov)
        np.testing.assert_allclose(identity["means"], means)
        np.testing.assert_allclose(identity["covariances"], cov)
        mixed = imm_mix(mu, [[.9, .1], [.2, .8]], means, cov)
        mean = mu @ means
        before = sum(mu[i] * (cov[i] + np.outer(means[i]-mean, means[i]-mean)) for i in range(2))
        after = sum(mixed["prior"][j] * (mixed["covariances"][j] + np.outer(mixed["means"][j]-mean, mixed["means"][j]-mean)) for j in range(2))
        np.testing.assert_allclose(mixed["prior"] @ mixed["means"], mean)
        np.testing.assert_allclose(after, before)

    def test_impossible_mode_and_transpose_rejected(self):
        for transition in ([[1, 0], [1, 0]], [[.9, .2], [.1, .8]]):
            with self.assertRaises(ValueError):
                imm_mix([.75, .25], transition, [[0], [10]], [[[1]], [[4]]])

    def test_nonreal_negative_and_overflow_rejected(self):
        for mean, covariance in ((np.array([[1+2j]]), [[[1]]]), ([[0]], [[[-1e-13]]])):
            with self.assertRaises(ValueError):
                imm_mix([1], [[1]], mean, covariance)
        with self.assertRaises(ValueError):
            imm_mix([.5, .5], [[.5, .5], [.5, .5]], [[1e200], [-1e200]], [[[1]], [[1]]])
        for arguments in (([1+0j], [[1]], [[0]], [[[1]]]),
                          ([1], [[1+0j]], [[0]], [[[1]]]),
                          ([1], [[1]], [[0]], [[[1+0j]]])):
            with self.assertRaises(ValueError):
                imm_mix(*arguments)


if __name__ == "__main__":
    unittest.main()
