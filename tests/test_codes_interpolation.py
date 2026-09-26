"""Independent sinusoidal/impulse checks for E13-02 and its PCM assets."""
import math
import unittest
import numpy as np

from codes.array_tutorial.audio_samples import interpolation_case, delay_samples, prepare_exports, read_pcm16
from codes.examples.interpolation_exercise import run_exercises


class InterpolationTest(unittest.TestCase):
    def test_two_passes_are_binomial_fir_not_integer_shift(self):
        impulse = np.array([1., 0., 0., 0.])
        first = .5 * (impulse + delay_samples(impulse, 1))
        second = .5 * (first + delay_samples(first, 1))
        np.testing.assert_array_equal(first, [.5, .5, 0, 0])
        np.testing.assert_array_equal(second, [.25, .5, .25, 0])
        self.assertFalse(np.array_equal(second, delay_samples(impulse, 1)))

    def test_entire_float_output_matches_independent_convolution(self):
        case = interpolation_case()
        n = np.arange(32000)
        t = n/16000
        env = np.minimum(np.clip(t/.02, 0, 1), np.clip((2-t)/.02, 0, 1))
        x = .18 * env * (np.sin(2*np.pi*500*t) + np.sin(2*np.pi*6000*t))
        for name, kernel in [('half', [.5, .5]), ('twice', [.25, .5, .25])]:
            actual = case['signals']['interpolation_linear_' + name]
            np.testing.assert_allclose(actual, np.convolve(x, kernel)[:32000], atol=2e-16)
        np.testing.assert_allclose(case['signals']['interpolation_ideal_one'],
                                   np.r_[0., x[:-1]], atol=2e-12)

    def test_pcm_projection_matches_independent_radical_high_frequency(self):
        r = run_exercises()['E13-02']
        expected = math.sqrt(2-math.sqrt(2))/2  # cos(3*pi/8)
        self.assertAlmostEqual(r['analytic_one_pass_amplitude'][1], expected, places=14)
        self.assertAlmostEqual(r['analytic_two_pass_amplitude'][1], (2-math.sqrt(2))/4, places=14)
        for name, power in [('linear_half', 1), ('linear_twice', 2)]:
            np.testing.assert_allclose(r['pcm_amplitude_ratios'][name],
                                      np.array([math.cos(math.pi/32), expected])**power,
                                      atol=2e-4, rtol=0)
        self.assertEqual(r['common_export_gain'], 1.)
        self.assertEqual(r['scoring_interval_samples'], [1600, 30400])

    def test_known_endpoint_delays_and_pcm_quantization(self):
        case = interpolation_case()
        files, groups = prepare_exports({'interpolation': case})
        self.assertEqual(len(files), 4)
        self.assertEqual(groups['interpolation']['common_export_gain'], 1.)
        for name, (blob, _) in files.items():
            rate, pcm = read_pcm16(blob)
            self.assertEqual((rate, pcm.shape), (16000, (1, 32000)))
            self.assertLessEqual(np.max(abs(pcm[0]-case['signals'][name[:-4]])), .5/32768)
            self.assertEqual(pcm[0, 0], 0.)


if __name__ == '__main__':
    unittest.main()
