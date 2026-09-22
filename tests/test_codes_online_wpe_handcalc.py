"""Independent rational checks of the stateful nara-wpe 0.0.11 reference.

All frame indices are zero-based.  The expectations below are fixed hand
calculations, not outputs from the upstream implementation or its comparison
helpers.  One frequency, one channel, one tap, delay=1 and alpha=1/2 are used.
This checks the NumPy reference's timing and complex arithmetic, not speech
quality or the general correctness of the upstream online-WPE algorithm.
"""

import unittest

import numpy as np

from codes.examples.compare_online_wpe_reference import NumpyOnlineWPE011


class TestOnlineWPEHandCalculation(unittest.TestCase):
    def setUp(self):
        self.state = NumpyOnlineWPE011(
            taps=1, delay=1, alpha=0.5, frequency_bins=1, channels=1
        )

    def assert_scalar_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

    def step(self, value):
        return self.state.step_frame(np.array([[value]], dtype=np.complex128))

    def test_real_first_update_and_next_prediction(self):
        # At t=0,1,2 the old three-frame buffer's first entry is zero.
        # Thus g stays zero, outputs equal inputs, and P doubles each time.
        for value, inverse_covariance in ((1, 2), (2, 4), (3, 8)):
            with self.subTest(frame=value - 1):
                self.assert_scalar_close(self.step(value), value)
                self.assert_scalar_close(
                    self.state.inverse_covariance, inverse_covariance
                )
                self.assert_scalar_close(self.state.filter_taps, 0)

        # At t=3: old buffer=[1,2,3], regressor=1, P_before=8, g_before=0.
        # Output=4; AFTER inserting 4, power=(2^2+3^2+4^2)/3=29/3.
        # Denominator=(1/2)*(29/3)+8=77/6; gain=8/(77/6)=48/77.
        # P_after=(8-(48/77)*8)/(1/2)=464/77; g_after=(48/77)*4=192/77.
        self.assert_scalar_close(self.step(4), 4)
        self.assert_scalar_close(self.state.inverse_covariance, 464 / 77)
        self.assert_scalar_close(self.state.filter_taps, 192 / 77)
        np.testing.assert_array_equal(self.state.buffer[:, 0, 0], [2, 3, 4])

        # At t=4, prediction uses the PRE-update tap and old regressor 2:
        # residual=5-(192/77)*2=1/77, before learning from this fifth frame.
        self.assert_scalar_close(self.step(5), 1 / 77)

    def test_complex_conjugates_and_next_prediction(self):
        for frame_index, (value, inverse_covariance) in enumerate(
            ((1 + 1j, 2), (2 - 1j, 4), (-1 + 2j, 8))
        ):
            with self.subTest(frame=frame_index):
                self.assert_scalar_close(self.step(value), value)
                self.assert_scalar_close(
                    self.state.inverse_covariance, inverse_covariance
                )
                self.assert_scalar_close(self.state.filter_taps, 0)

        # At t=3: z=1+i, P_before=8, g_before=0, output=3+i.
        # Updated buffer powers are 5,5,10, so power=20/3.
        # Denominator=(1/2)*(20/3)+8*|1+i|^2=58/3.
        # gain=(12/29)*(1+i).
        # P_after=(8-gain*conj(8*(1+i)))/(1/2)=80/29.
        # g_after=gain*conj(3+i)=(48+24i)/29.
        self.assert_scalar_close(self.step(3 + 1j), 3 + 1j)
        self.assert_scalar_close(self.state.inverse_covariance, 80 / 29)
        self.assert_scalar_close(self.state.filter_taps, (48 + 24j) / 29)
        np.testing.assert_array_equal(
            self.state.buffer[:, 0, 0], [2 - 1j, -1 + 2j, 3 + 1j]
        )

        # At t=4: z=2-i and conj(g)*z=(72-96i)/29.
        # New observation 2+3i therefore gives residual=(-14+183i)/29.
        self.assert_scalar_close(self.step(2 + 3j), (-14 + 183j) / 29)


if __name__ == "__main__":
    unittest.main()
