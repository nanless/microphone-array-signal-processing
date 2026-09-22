"""Analytic boundary checks for the WPE and mask-MVDR teaching baselines."""

import unittest

import numpy as np

from codes.array_tutorial.dereverberation import offline_wpe
from codes.array_tutorial.separation import mask_mvdr_2x2


class EnhancementRoundFour(unittest.TestCase):
    def test_wpe_extreme_finite_levels_match_one_tap_hand_solution(self):
        # At t=1,3,5,7, (history,current)=(1,2), giving R=4/4, r=4/2.
        # At t=2,4,6, (history,current)=(2,1), giving R=3*4, r=3*2.
        # Thus g=(2+6)/(1+12)=8/13, independently of this solver.
        x = np.array([[1, 2, 1, 2, 1, 2, 1, 2]], dtype=np.complex128)
        expected = np.array([[1, 18 / 13, -3 / 13, 18 / 13,
                              -3 / 13, 18 / 13, -3 / 13, 18 / 13]])
        for scale in (1e-200, 1.0, 1e200):
            with self.subTest(scale=scale), np.errstate(over="raise", invalid="raise"):
                result = offline_wpe(x * scale, taps=1, delay=1, iterations=1,
                                     diagonal_loading=0.0)
            np.testing.assert_allclose(result.real / scale, expected, rtol=2e-13, atol=2e-13)
            np.testing.assert_array_equal(result.imag, np.zeros_like(expected))

    def test_wpe_unrepresentable_relative_floor_is_explicit_error(self):
        # The first channel carries the signal; averaging over two channels
        # makes its normalized peak power 1/2.  Half the smallest positive
        # float64 power floor is not representable, so no valid solve exists
        # under the caller's requested floor.
        x = np.array([[[1, 2, 1, 2], [0, 0, 0, 0]]], dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "power floor"):
            offline_wpe(x, taps=1, delay=1,
                        power_floor=np.nextafter(0.0, 1.0))

    def test_mvdr_mask_rescaling_preserves_closed_form_cancellation(self):
        # Target [1,1] and interference [1,-1] are orthogonal.  The
        # distortionless, minimum-interference two-channel weight is
        # [1/2,1/2], giving [1,1,0,0] on these four snapshots.
        x = np.array([[[1, 1, 1, 1], [1, 1, -1, -1]]], dtype=np.complex128)
        target = np.array([[1, 1, 0, 0]], dtype=float)
        interference = np.array([[0, 0, 1, 1]], dtype=float)
        for target_scale, interference_scale in (
            (1.0, 1.0), (1e-13, 1e-13), (1e-200, 1e200), (1e200, 1e-200)
        ):
            with self.subTest(target_scale=target_scale, interference_scale=interference_scale):
                output, weights = mask_mvdr_2x2(
                    x, target * target_scale, interference * interference_scale)
                np.testing.assert_allclose(weights, [[0.5, 0.5]], atol=1e-12)
                np.testing.assert_allclose(output, [[1, 1, 0, 0]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
