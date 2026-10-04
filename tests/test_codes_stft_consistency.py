"""Exact overlapping-sample requirements and full-DFT Parseval controls."""
import cmath
import unittest

import numpy as np

from codes.chapters.ch02.core.stft_consistency import consistency_example


class StftConsistencyTest(unittest.TestCase):
    def test_endpoint_valid_frames_conflict_and_wola_averages_each_shared_sample(self):
        result = consistency_example()
        self.assertEqual(result['modified_frames_real'], [[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        self.assertEqual(result['modified_frames_imag'], [[0, 0, 0]]*3)
        self.assertEqual(result['inverse_frames'], [[0, 0, 0, 0], [1, 0, -1, 0], [0, 0, 0, 0]])
        self.assertEqual(result['retained_window_squared_sum'], [2]*4)
        # Independent pointwise LS: min (x-1)^2+x^2 and (x+1)^2+x^2.
        self.assertEqual(result['synthesized_waveform'], [.5, 0, -.5, 0])
        self.assertTrue(result['input_unchanged'])

    def test_reanalysis_by_explicit_complex_exponential_sum(self):
        result = consistency_example()
        frames = [[0, 0, .5, 0], [.5, 0, -.5, 0], [-.5, 0, 0, 0]]
        expected = [[sum(x*cmath.exp(-2j*np.pi*k*q/4) for q, x in enumerate(frame))
                     for k in range(3)] for frame in frames]
        np.testing.assert_allclose(result['reanalysed_frames_real'], np.real(expected), atol=1e-15)
        np.testing.assert_allclose(result['reanalysed_frames_imag'], np.imag(expected), atol=1e-15)
        self.assertEqual(result['second_projection_max_abs_difference'], 0)

    def test_full_dft_weights_and_pythagorean_residual_are_not_unweighted_rfft(self):
        result = consistency_example()
        self.assertEqual(result['full_dft_weights'], [1, 2, 1])
        self.assertEqual(result['full_dft_modified_energy'], 8)
        self.assertEqual(result['full_dft_projected_energy'], 4)
        self.assertEqual(result['full_dft_residual_energy'], 4)
        self.assertEqual(result['relative_full_dft_residual_energy'], .5)
        original = np.array(result['modified_frames_real'])
        projected = np.array(result['reanalysed_frames_real'])
        self.assertNotEqual(float(np.sum((original-projected)**2)), 4.)
        weights = np.array([1, 2, 1])
        self.assertEqual(float(np.sum(weights*(original-projected)*projected)), 0.)


if __name__ == '__main__':
    unittest.main()
