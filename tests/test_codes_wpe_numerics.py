"""Independent WPE product-range, design-solve and centered-power checks."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch07.core.dereverberation import (
    offline_wpe, solve_prediction_design, _check_product_range,
)


class WPEDesignNumerics(unittest.TestCase):
    def test_e19_full_rank_design_survives_information_lost_by_gram(self):
        epsilon = 1e-9
        a = np.array([[1., 1.], [0., epsilon]])
        b = np.array([0., -epsilon])
        # Exact determinant epsilon != 0; solve first row g1+g2=0,
        # second row epsilon*g2=-epsilon, hence (1,-1).
        direct, status = solve_prediction_design(a, b, return_diagnostics=True)
        np.testing.assert_allclose(direct, [1., -1.], atol=1e-14, rtol=0)
        self.assertEqual((status.matrix_kind, status.rank), ("augmented_design", 2))
        self.assertTrue(status.used_lstsq)
        self.assertLess(np.linalg.norm(a@direct-b), 1e-14)
        normal, status = solve_prediction_design(a, b, solver="normal", return_diagnostics=True)
        self.assertEqual(status.rank, 1)
        self.assertTrue(status.used_lstsq)
        self.assertLess(np.linalg.norm(normal), 1e-16)

    def test_design_path_does_not_call_normal_solver(self):
        with patch("codes.chapters.ch07.core.dereverberation._normal_solve",
                   side_effect=AssertionError("formed normal equations")):
            g = solve_prediction_design([[1., 1.], [0., 1e-9]], [0., -1e-9])
        np.testing.assert_allclose(g, [1., -1.], rtol=0, atol=1e-14)

    def test_complex_rows_and_relative_loading_match_fraction_e08(self):
        # Rows sqrt(w) q^H; targets sqrt(w) X*. Independent exact solution
        # of [[3,-j],[j,4]]g=[3,4j] is (8/11,9j/11).
        a = np.array([[1, 0], [0, np.sqrt(2)], [1, -1j]], complex)
        b = np.array([0, np.sqrt(2)*.5j, 3], complex)
        for solver in ("normal", "design_lstsq"):
            g, status = solve_prediction_design(a, b, diagonal_loading=.4,
                                                 solver=solver, return_diagnostics=True)
            np.testing.assert_allclose(g, [float(Fraction(8,11)), 1j*float(Fraction(9,11))], atol=2e-15)
            self.assertAlmostEqual(status.absolute_loading, 1)
            self.assertEqual(status.used_lstsq, solver == "design_lstsq")
            self.assertAlmostEqual(float(np.vdot(a@g-b,a@g-b).real), float(Fraction(689,242)))

    def test_multiple_outputs_and_zero_design(self):
        g = solve_prediction_design(np.eye(2), np.array([[1,2j],[3,4]]))
        np.testing.assert_allclose(g, [[1,2j],[3,4]])
        np.testing.assert_array_equal(solve_prediction_design(np.zeros((3,2)), [1,2,3]), [0,0])

    def test_offline_design_and_default_share_complex_prediction_convention(self):
        coefficient = .5+.25j
        x = np.array([[coefficient.conjugate()**n for n in range(12)]])
        for solver in ("normal", "design_lstsq"):
            y, diagnostic = offline_wpe(x, taps=1, delay=1, iterations=1,
                                        diagonal_loading=0, solver=solver, return_diagnostics=True)
            self.assertEqual(y[0,0], 1)
            np.testing.assert_allclose(y[0,1:], 0, atol=2e-16)
            self.assertEqual(diagnostic[0].solver, solver)
            self.assertEqual(diagnostic[0].least_squares_count, 0)

    def test_centered_power_uses_actual_edge_count(self):
        # Powers [1,4,9,16,25]. Radius two gives 14/3,30/4,55/5,
        # 54/4,50/3. Only frames 1..4 contribute to the K1 fit.
        powers = [Fraction(14,3), Fraction(30,4), Fraction(55,5),
                  Fraction(54,4), Fraction(50,3)]
        numerator = sum(Fraction(t*(t+1),1)/powers[t] for t in range(1,5))
        denominator = sum(Fraction(t*t,1)/powers[t] for t in range(1,5))
        g = float(numerator/denominator)
        x = np.array([[1,2,3,4,5]], complex)
        y = offline_wpe(x,taps=1,delay=1,iterations=1,diagonal_loading=0,power_context=2)
        np.testing.assert_allclose(y, [[1,2-g,3-2*g,4-3*g,5-4*g]], atol=2e-15)

    def test_three_underflow_layers_are_explicit_support_failures(self):
        cases = [([1e-308,1e308],1), ([1e308,1e146,0,1e146,0],2),
                 ([1e308,1e108,0],1), ([1e300,1e100,0],1)]
        for row, delay in cases:
            x=np.array([row],complex); saved=x.copy()
            for solver in ("normal", "design_lstsq"):
                with self.subTest(row=row,solver=solver):
                    with self.assertRaisesRegex(ValueError,"supported float64"):
                        offline_wpe(x,taps=1,delay=delay,iterations=1,solver=solver)
                    np.testing.assert_array_equal(x,saved)
        # Prediction can lose a term even if each factor is nonzero.
        with self.assertRaisesRegex(ValueError,"prediction"):
            _check_product_range(np.array([1e-200]),np.array([1e-200]),stage="prediction")
        _check_product_range(np.array([1.]),np.array([1e-308]),stage="prediction")

    def test_options_are_validated_even_for_bypass(self):
        x=np.ones((1,2),complex)
        for option in (-1,True,1.5,1+0j):
            with self.assertRaises(ValueError):
                offline_wpe(x,taps=0,delay=1,power_context=option)
        with self.assertRaises(ValueError):
            offline_wpe(x,taps=0,delay=1,solver="qr")
        for a,b in (([["1"]],[1]), ([[1]],[np.inf]), (np.empty((0,1)), [])):
            with self.assertRaises(ValueError):
                solve_prediction_design(a,b)
        with self.assertRaises(ValueError):
            solve_prediction_design([[1]],[1],diagonal_loading=True)

    def test_startup_is_raw_copy_and_bypass_labels_follow_requested_solver(self):
        x=np.array([[.1+.3j,.7+.2j,1.3+.9j, .6+.8j]])
        y=offline_wpe(x,taps=1,delay=2,iterations=1)
        np.testing.assert_array_equal(y[:,:2],x[:,:2])
        for options, data in ((dict(taps=0,delay=1),x),
                              (dict(taps=3,delay=2),x),
                              (dict(taps=1,delay=1),np.zeros((1,4),complex))):
            y,status=offline_wpe(data,**options,solver="design_lstsq",return_diagnostics=True)
            self.assertEqual(status[0].solver,"design_lstsq")
            self.assertEqual(status[0].matrix_kind,"augmented_design")
            np.testing.assert_array_equal(y,data)

    def test_nonzero_final_output_underflow_is_not_returned_as_zero(self):
        smallest=np.nextafter(0.,1.)
        x=np.array([[smallest,smallest]],complex)
        # g=1/2, so the exact output is half the smallest positive float.
        # It is mathematically nonzero but cannot be returned in this dtype.
        with self.assertRaisesRegex(ValueError,"output restoration"):
            offline_wpe(x,taps=1,delay=1,iterations=1,diagonal_loading=1)
        np.testing.assert_array_equal(x,[[smallest,smallest]])


if __name__ == "__main__":
    unittest.main()
