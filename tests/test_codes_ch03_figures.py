"""Independent scalar checks for the two chapter 3 teaching figures."""
import math
import unittest

import numpy as np

from scripts.make_figures import hexagon_frequency_evidence, near_planar_sensitivity


class Chapter03FiguresTest(unittest.TestCase):
    def test_unconstrained_component_gain_and_mirror_separation(self):
        data = near_planar_sensitivity()
        np.testing.assert_allclose(data["absolute_delta_u_z"], [.8575, .08575, .008575], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(data["condition_numbers"], [100, 10, 1])
        for height, separation in zip([.0004, .004, .04], data["mirror_delay_separation_us"]):
            # Distances of the two mirror projections, computed in metres.
            positive = -height * math.sqrt(3) / 2 / 343
            negative = height * math.sqrt(3) / 2 / 343
            self.assertAlmostEqual(separation, (negative - positive) * 1e6, places=12)

    def test_signed_complex_coherence_and_profile_residual_are_distinct(self):
        data = hexagon_frequency_evidence()
        np.testing.assert_allclose(data["relative_ratios"][4000], [1, -1, -1, 1, -1, -1], atol=3e-15)
        np.testing.assert_allclose(data["relative_ratios"][8000], [1] * 6, atol=3e-15)
        self.assertAlmostEqual(sum([1, -1, -1, 1, -1, -1]) / 6, -1 / 3)
        self.assertAlmostEqual(data["wrong_direction_residual"][4000], 8 / 9, places=14)
        self.assertLess(data["wrong_direction_residual"][8000], 1e-28)

    def test_scanned_coherence_from_six_scalar_phasors(self):
        data = hexagon_frequency_evidence()
        ux = math.sqrt(1 - (343 / (8000 * math.sqrt(3) * .0375)) ** 2)
        uy = 343 / (8000 * math.sqrt(3) * .0375)
        for frequency in (4000, 8000):
            for angle in (0, 30, 90, 150, 180):
                dx = math.sin(math.radians(angle)) - ux
                dy = math.cos(math.radians(angle)) - uy
                real = imag = 0.
                for index in range(6):
                    position_x = .0375 * (math.cos(index * math.pi / 3) - 1)
                    position_y = .0375 * math.sin(index * math.pi / 3)
                    phase = 2 * math.pi * frequency * (position_x * dx + position_y * dy) / 343
                    real += math.cos(phase)
                    imag += math.sin(phase)
                self.assertAlmostEqual(data["power_coherence"][frequency][angle * 10],
                                       (real * real + imag * imag) / 36, places=13)


if __name__ == "__main__":
    unittest.main()
