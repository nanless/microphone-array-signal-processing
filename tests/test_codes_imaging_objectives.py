"""Independent full-matrix objectives, feasible ambiguity and range controls."""
from fractions import Fraction
import unittest
import numpy as np

from codes.chapters.ch14.core.imaging import (
    csm_from_amplitudes, csm_residual, damas_csm_objective_experiment,
    distinct_column_ambiguity_experiment, clean_sc_full_csm,
)


class ImagingObjectiveTests(unittest.TestCase):
    def test_full_matrix_and_scan_have_distinct_minima(self):
        case = damas_csm_objective_experiment()
        np.testing.assert_allclose(case['G'], [[4, 1], [1, 4]], atol=2e-15)
        np.testing.assert_allclose(case['h'], [4, 0], atol=2e-15)
        np.testing.assert_allclose(case['q_GS'], [1, 0], atol=2e-15)
        np.testing.assert_allclose(case['q_CSM'], [1, 0], atol=2e-15)
        np.testing.assert_allclose(case['q_scan'], [16/17, 0], atol=2e-15)
        # Scalar polynomial expansion is independent of dictionary routines.
        for name, u, v in [('GS', Fraction(1), Fraction(0)),
                            ('CSM', Fraction(1), Fraction(0)),
                            ('scan', Fraction(16, 17), Fraction(0))]:
            scan = (u+v/4-1)**2+(u/4+v)**2
            full = 4*u*u+2*u*v+4*v*v-8*u+Fraction(64, 9)
            self.assertAlmostEqual(case['losses'][name]['scan_squared'], float(scan), places=14)
            self.assertAlmostEqual(case['losses'][name]['csm_frobenius_squared'], float(full), places=13)
        self.assertEqual(Fraction(28, 9)+Fraction(4, 289), Fraction(8128, 2601))
        np.testing.assert_allclose(case['csm_gradient_at_solution'], [0, 1], atol=2e-15)
        # Direct element sums retain both conjugate off-diagonals.
        for name, q in [('GS', case['q_GS']), ('scan', case['q_scan'])]:
            model = sum(q[j]*np.outer(case['A'][:, j], case['A'][:, j].conj()) for j in range(2))
            error = sum(abs(model[i, j]-case['R'][i, j])**2 for i in range(2) for j in range(2))
            self.assertAlmostEqual(error, case['losses'][name]['csm_frobenius_squared'], places=13)
        scale = case['objective_common_scale']
        for name, value in case['scaled_csm_objectives'].items():
            self.assertAlmostEqual(value*scale**2, case['losses'][name]['csm_frobenius_squared'], places=13)

    def test_unequal_column_norms_require_positive_row_scaling(self):
        case = damas_csm_objective_experiment()['nonuniform_control']
        np.testing.assert_allclose(case['D'], [[4, 0], [0, 64]], atol=3e-14)
        np.testing.assert_allclose(case['P'], [[1, 1], [1/16, 1]], atol=2e-15)
        np.testing.assert_allclose(case['G'], [[4, 4], [4, 64]], atol=3e-14)
        np.testing.assert_allclose(case['G'], case['D']@case['P'], atol=3e-14)
        np.testing.assert_allclose(case['h'], case['D']@case['b'], atol=2e-14)
        self.assertGreater(abs(case['P'][0, 1]-case['P'][1, 0]), .9)
        # Removing only observation diagonals fails the full-CSM identity.
        full = damas_csm_objective_experiment()
        removed = full['R']-np.diag(np.diag(full['R']))
        changed_b = np.array([(column.conj()@removed@column).real for column in full['W'].T])
        self.assertGreater(np.max(abs(full['h']-full['D']@changed_b)), 2.)

    def test_distinct_templates_have_a_feasible_collective_null_direction(self):
        case = distinct_column_ambiguity_experiment()
        expected_p = np.array([[1, .5, 0, .5], [.5, 1, .5, 0],
                               [0, .5, 1, .5], [.5, 0, .5, 1]])
        np.testing.assert_allclose(case['P'], expected_p, atol=2e-15)
        self.assertEqual(case['dictionary_rank'], 3)
        self.assertEqual(case['psf_rank'], 3)
        np.testing.assert_allclose(case['psf_eigenvalues'], [0, 1, 1, 2], atol=2e-15)
        # Explicit real dictionary, not an oracle imported from the core.
        root2 = np.sqrt(2)
        design = np.array([[1, 1, 1, 1], [1, 1, 1, 1],
                           [root2, 0, -root2, 0], [0, -root2, 0, root2]])
        np.testing.assert_allclose(case['csm_dictionary_real'], design, atol=1e-15)
        for i in range(4):
            for j in range(i):
                self.assertGreater(np.linalg.norm(design[:, i]-design[:, j]), 1.)
        np.testing.assert_array_equal(design@case['null_direction'], np.zeros(4))
        for t in [-.5, -.125, 0, .375, .5]:
            q = np.full(4, .5)+t*np.array([1, -1, 1, -1])
            self.assertTrue(np.all(q >= 0))
            # Both diagonal sums and explicit complex cross sum are fixed.
            z = [1, 1j, -1, -1j]
            cross = sum(q[j]*z[j].conjugate() for j in range(4))
            self.assertEqual(cross, 0)
            self.assertEqual(sum(q), 2)
            np.testing.assert_allclose(expected_p@q, np.ones(4), atol=1e-15)
        np.testing.assert_allclose(case['model_csms'], np.broadcast_to(2*np.eye(2), (3, 2, 2)), atol=3e-16)
        np.testing.assert_allclose(case['b_examples'], np.ones((3, 4)))
        np.testing.assert_array_equal(case['alternative_white_noise']['q'], np.zeros(4))
        self.assertEqual(case['alternative_white_noise']['variance'], 2.)


class ImagingRangeTests(unittest.TestCase):
    def test_relative_error_is_invariant_when_physical_squares_underflow(self):
        for scale in [1., 1e-100, 1e-160, 1e-200]:
            result = csm_residual(scale*np.eye(2), 2*scale*np.eye(2))
            self.assertEqual(result['relative_frobenius'], 1.)
            if scale == 1e-200:
                self.assertEqual(result['frobenius_squared'], 0.)
                self.assertTrue(all(result['squared_underflow'].values()))
            else:
                self.assertGreater(result['reference_frobenius_squared'], 0.)
                self.assertFalse(any(result['squared_underflow'].values()))

    def test_zero_reference_identical_tiny_and_complex_error(self):
        result = csm_residual(np.zeros((2, 2)), 1e-200*np.eye(2))
        self.assertIsNone(result['relative_frobenius'])
        self.assertFalse(result['squared_underflow']['reference_frobenius_squared'])
        self.assertTrue(result['squared_underflow']['frobenius_squared'])
        result = csm_residual(1e-200*np.eye(2), 1e-200*np.eye(2))
        self.assertEqual(result['relative_frobenius'], 0.)
        self.assertFalse(result['squared_underflow']['frobenius_squared'])
        measured = np.array([[2, 1+2j], [1-2j, 3]], complex)
        model = np.array([[3, 2-1j], [2+1j, 1]], complex)
        # Diagonal errors 1,-2; off-diagonal errors 1±3j.
        expected = np.sqrt(25/23)
        for scale in [1, 1e-200]:
            actual = csm_residual(scale*measured, scale*model)
            self.assertAlmostEqual(actual['relative_frobenius'], expected, places=14)

    def test_common_scale_does_not_erase_a_small_nonzero_reference(self):
        result = csm_residual([[1e-300]], [[1e-160]])
        self.assertAlmostEqual(result['relative_frobenius']/1e140, 1., places=14)
        self.assertTrue(result['squared_underflow']['reference_frobenius_squared'])
        with self.assertRaises(ValueError):
            csm_residual([[1e200]], [[2e200]])
        tiny = np.nextafter(0., 1.)
        result = csm_residual([[tiny]], [[2*tiny]])
        self.assertEqual(result['relative_frobenius'], 1.)
        self.assertTrue(result['squared_underflow']['reference_frobenius_squared'])
        complex_tiny = np.array([[tiny, 1j*tiny], [-1j*tiny, tiny]])
        self.assertEqual(csm_residual(complex_tiny, 2*complex_tiny)['relative_frobenius'], 1.)
        with self.assertRaisesRegex(ValueError, 'Hermitian'):
            csm_residual([[0, tiny], [-tiny, 0]], np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            clean_sc_full_csm([[-tiny]], [[1]])

    def test_amplitude_csm_rejects_total_underflow_but_keeps_zero_and_subnormal(self):
        with self.assertRaisesRegex(ValueError, 'underflows'):
            csm_from_amplitudes([[1e-200, 1e-200]])
        with self.assertRaises(ValueError):
            csm_from_amplitudes([[1e200]])
        np.testing.assert_array_equal(csm_from_amplitudes([[0, 0]]), [[0]])
        self.assertGreater(csm_from_amplitudes([[1e-160]])[0, 0].real, 0.)
        np.testing.assert_allclose(csm_from_amplitudes([[.2, .2], [.1j, -.1j]]), [[.02, 0], [0, .005]], atol=1e-18)


if __name__ == '__main__':
    unittest.main()
