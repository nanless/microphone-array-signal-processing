"""WPE scalar contracts and independently soluble huge-loading boundary."""
import unittest
from unittest.mock import patch
import numpy as np
from codes.array_tutorial.dereverberation import offline_wpe


class WPENumericalBoundaries(unittest.TestCase):
    def test_real_scalar_contract_even_on_bypass(self):
        x = np.ones((1,2),complex)
        for key in ('diagonal_loading','power_floor'):
            for value in (True,np.bool_(False),1+0j,np.array([1.]),np.array([1.,2.]),np.nan,np.inf):
                for taps in (0,1):
                    with self.subTest(parameter=key,value=value,taps=taps):
                        with self.assertRaises(ValueError):
                            offline_wpe(x,taps=taps,delay=1,**{key:value})
        with self.assertRaises(ValueError):
            offline_wpe(x,taps=0,delay=1,diagonal_loading=-1)
        with self.assertRaises(ValueError):
            offline_wpe(x,taps=0,delay=1,power_floor=0)

    def test_representable_huge_loading_no_false_intermediate_overflow(self):
        original_solve = np.linalg.solve
        captured = []
        def solve(a,b):
            result = original_solve(a,b)
            captured.append((a.copy(),result.copy()))
            return result
        # R=ones(2,2), r=ones(2,2). Exact g columns=ones/(1e308+2).
        # Output rounds back to one; inspect actual solve path to test tiny g.
        with patch('codes.array_tutorial.dereverberation.np.linalg.solve',side_effect=solve):
            result = offline_wpe(np.ones((1,2,2),complex),taps=1,delay=1,
                                 iterations=1,diagonal_loading=1e308)
        matrix,g = captured[0]
        self.assertTrue(np.all(np.isfinite(matrix)))
        np.testing.assert_array_equal(matrix.diagonal(),[1.,1.])
        np.testing.assert_allclose(g.real/1e-308,np.ones((2,2)),rtol=1e-14,atol=0)
        np.testing.assert_array_equal(g.imag,np.zeros((2,2)))
        np.testing.assert_array_equal(result,np.ones((1,2,2)))

    def test_actually_unrepresentable_loading_fails_without_mutation(self):
        x = np.ones((1,1,4),complex)
        saved = x.copy()
        with self.assertRaisesRegex(ValueError,'loading'):
            offline_wpe(x,taps=1,delay=1,iterations=1,diagonal_loading=1e308)
        np.testing.assert_array_equal(x,saved)


if __name__ == '__main__':
    unittest.main()
