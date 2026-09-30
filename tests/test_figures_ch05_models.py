"""Independent phasor sums and Bayes likelihoods for Chapter 5's new figures."""
import cmath
import math
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np
from scripts import make_figures


class Chapter05FigureModelsTest(unittest.TestCase):
    def capture(self, function):
        captured = []
        with patch.object(make_figures, 'save', side_effect=lambda fig, name: captured.append((fig, name))):
            function()
        self.assertEqual(len(captured), 1)
        self.addCleanup(make_figures.plt.close, captured[0][0])
        return captured[0]

    def test_derivative_response_and_expected_power_have_independent_oracles(self):
        fig, name = self.capture(make_figures.fig_derivative_constraints)
        self.assertEqual(name, 'fig55_derivative_constraints.png')
        complex_ax, error_ax, power_ax = fig.axes
        weights = [(Fraction(4, 7), Fraction(2, 7), Fraction(1, 7)),
                   (Fraction(4, 13), Fraction(5, 13), Fraction(4, 13))]
        for method, w in enumerate(weights):
            line = error_ax.lines[method]
            phase = line.get_xdata()
            expected = np.array([abs(sum(float(t)*cmath.exp(1j*m*p)
                                        for t, m in zip(w, (-1, 0, 1)))-1)**2 for p in phase])
            np.testing.assert_allclose(line.get_ydata(), expected, rtol=2e-6, atol=1e-28)
            complex_line = complex_ax.lines[method]
            response = np.array([sum(float(t)*cmath.exp(1j*m*p)
                                     for t, m in zip(w, (-1, 0, 1))) for p in phase])
            np.testing.assert_allclose(complex_line.get_xdata(), response.real, rtol=0, atol=3e-16)
            np.testing.assert_allclose(complex_line.get_ydata(), response.imag, rtol=0, atol=3e-16)
        expected_noise = [Fraction(4, 7), Fraction(10, 13)]
        # First bar container is noise; second is clean target distortion.
        for k, w in enumerate(weights):
            self.assertAlmostEqual(power_ax.containers[0][k].get_height(), .02**2*float(expected_noise[k])/1e-4, places=12)
            clean = .08**2/2*sum(abs(sum(float(t)*cmath.exp(1j*m*p)
                                             for t, m in zip(w, (-1, 0, 1)))-1)**2 for p in (.1, .3))
            self.assertAlmostEqual(power_ax.containers[1][k].get_height(), clean/1e-4, places=12)

    def test_probability_and_geometric_gain_follow_density_ratio(self):
        fig, name = self.capture(make_figures.fig_omlsa_probability)
        self.assertEqual(name, 'fig56_omlsa_probability.png')
        posterior_ax, gain_ax = fig.axes
        for q, posterior_index, gain_index in ((.5, 0, 0), (.2, 1, 2)):
            line = posterior_ax.lines[posterior_index]
            gamma = line.get_xdata()
            # Independent normalized complex-Gaussian density model: phi_n=1, phi_s=3.
            expected = np.array([(1-q)*math.exp(-g/4)/4 /
                                 (q*math.exp(-g)+(1-q)*math.exp(-g/4)/4) for g in gamma])
            np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-14, atol=0)
            np.testing.assert_allclose(gain_ax.lines[gain_index].get_ydata(),
                                       .8**expected*.1**(1-expected), rtol=1e-14, atol=0)
        expected_p = 1/(1+4*math.exp(-3))
        self.assertAlmostEqual(posterior_ax.collections[0].get_offsets()[0, 1], expected_p, places=14)
        self.assertAlmostEqual(gain_ax.collections[0].get_offsets()[0, 1], .5663821076747687, places=14)


if __name__ == '__main__':
    unittest.main()
