"""Independent range, units and subnormal-power oracles for Chapter 5."""
import math
import unittest
import warnings
import numpy as np
from codes.chapters.ch05.core.beamforming import diffuse_coherence, blocking_matrix, wiener_gain
from codes.chapters.ch08.core.separation import masked_spatial_covariance


class BeamformingRangeTest(unittest.TestCase):
    def test_finite_distances_and_frequency_products_avoid_false_nan_or_unity(self):
        expected = math.sin(2*math.pi/343)/(2*math.pi/343)
        for distance, frequency in [(1e200, 1e-200), (1e-308, 1e308)]:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                result = diffuse_coherence([[0, 0], [distance, 0]], [frequency])
            np.testing.assert_allclose(result[0], [[1, expected], [expected, 1]], rtol=1e-15, atol=0)
        result = diffuse_coherence([[-1e308, 0], [1e308, 0]], [1e-308, 0])
        expected = math.sin(4*math.pi/343)/(4*math.pi/343)
        self.assertAlmostEqual(result[0, 0, 1], expected)
        np.testing.assert_array_equal(result[1], np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, 'floating-point range'):
            diffuse_coherence([[0, 0], [1e308, 0]], [1e308])

    def test_ordinary_coherence_retains_original_arithmetic(self):
        positions = np.array([[0., 0.], [.03, .04], [.1, -.02]])
        frequencies = np.array([0., 100., 3000.])
        expected = np.sinc(2*frequencies[:, None, None]*np.linalg.norm(positions[:, None]-positions[None], axis=-1)[None]/343)
        np.testing.assert_array_equal(diffuse_coherence(positions, frequencies), expected)

    def test_blocker_column_units_and_legitimate_zero_dimension(self):
        for scales in ([1e-300, 1e300], [1e-20, 1], [1, 1]):
            c = np.array([[scales[0], 0], [0, scales[1]], [0, 0]], complex)
            before = c.copy()
            b = blocking_matrix(c)
            self.assertEqual(b.shape, (3, 1))
            np.testing.assert_allclose(b@b.conj().T, np.diag([0, 0, 1]), atol=1e-15)
            np.testing.assert_array_equal(c, before)
            self.assertEqual(blocking_matrix(np.diag(scales)).shape, (2, 0))
        c = np.column_stack(([1, 1, 0], [2, 2, 0], [0, 0, 0]))
        b = blocking_matrix(c)
        self.assertEqual(b.shape, (3, 2))
        np.testing.assert_allclose(c.T@b, 0, atol=1e-14)
        np.testing.assert_allclose(b.T@b, np.eye(2), atol=1e-15)
        for bad in (np.empty((0, 1)), np.empty((2, 0))):
            with self.assertRaises(ValueError):
                blocking_matrix(bad)
        with self.assertRaises(np.linalg.LinAlgError):
            blocking_matrix(np.zeros((3, 2)))

    def test_scalar_options_reject_lossy_types_and_gain_has_no_overflow_warning(self):
        for bad in (True, '1', 1+0j, [1.], float('inf'), float('nan')):
            for call in (lambda: diffuse_coherence([[0, 0], [.1, 0]], [1000], sound_speed=bad),
                         lambda: blocking_matrix([1, 1], rtol=bad),
                         lambda: wiener_gain(1, .5, gain_floor=bad),
                         lambda: wiener_gain(1, .5, power_floor=bad)):
                with self.assertRaises(ValueError):
                    call()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(wiener_gain(1e-320, 1., power_floor=1e-320), 0.)
        self.assertEqual(wiener_gain(0, 0), 1.)
        self.assertEqual(wiener_gain(1., 2., gain_floor=.1), .1)


class MaskedDiagonalRangeTest(unittest.TestCase):
    def test_nonzero_diagonal_underflow_is_not_physical_zero(self):
        with self.assertRaisesRegex(ValueError, 'positive.*underflows'):
            masked_spatial_covariance([[[1e-200]]], [[1.]])
        # A tiny channel beside a normal channel still has nonzero truth.
        with self.assertRaisesRegex(ValueError, 'positive.*underflows'):
            masked_spatial_covariance([[[1.], [1e-200]]], [[1.]])
        np.testing.assert_array_equal(masked_spatial_covariance([[[0., 0.]]], [[1., 1.]]), 0)
        np.testing.assert_array_equal(masked_spatial_covariance([[[1e300]]], [[0.]]), 0)
        np.testing.assert_array_equal(masked_spatial_covariance([[[1e308, 1.]]], [[0., 1.]]), 1)

    def test_positive_terms_sum_before_subnormal_rounding(self):
        # x²=2^-1074; two half-weight terms each round to zero alone, but
        # their exact average is the minimum positive float, not zero.
        value = math.ldexp(1., -537)
        result = masked_spatial_covariance([[[value, value]]], [[1., 1.]])
        self.assertEqual(result[0, 0, 0].real, np.nextafter(0., 1.))
        self.assertEqual(result[0, 0, 0].imag, 0.)
        x = np.array([[[1+1j, 2j], [0, 1.]]])
        expected = np.array([[3., 1j], [-1j, .5]])
        np.testing.assert_array_equal(masked_spatial_covariance(x, [[1, 1]])[0], expected)


if __name__ == '__main__':
    unittest.main()
