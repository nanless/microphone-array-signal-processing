import numpy as np
import unittest

from scripts.make_aec_figures import block_erle, causal_delay, colored_x


class AecFiguresTest(unittest.TestCase):
    def test_causal_delay_zero_fills_instead_of_wrapping(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        delayed = causal_delay(x, 2)

        np.testing.assert_array_equal(delayed, [0.0, 0.0, 1.0, 2.0])

    def test_causal_delay_handles_zero_and_delay_past_signal(self):
        x = np.array([1, 2, 3])

        np.testing.assert_array_equal(causal_delay(x, 0), x)
        np.testing.assert_array_equal(causal_delay(x, len(x)), np.zeros_like(x))
        with self.assertRaises(ValueError):
            causal_delay(x, -1)

    def test_block_erle_uses_power_ratio_in_decibels(self):
        echo = np.ones(1200)
        residual = np.full(1200, 0.1)

        _, erle = block_erle(echo, residual, blk=400)

        np.testing.assert_allclose(erle, 20.0, atol=1e-12)

    def test_colored_signal_seed_is_repeatable_and_independent(self):
        first = colored_x(128, seed=101)
        _ = colored_x(128, seed=202)
        repeated = colored_x(128, seed=101)

        np.testing.assert_array_equal(first, repeated)


if __name__ == "__main__":
    unittest.main()
