"""Independent phase/conjugation, rational SCM and published PCM controls."""
from pathlib import Path
import unittest
import wave

import numpy as np

from codes.chapters.ch05 import chapter05_experiments as ex


ROOT = Path(__file__).resolve().parents[1]


class PhaseAndMaskModelsTest(unittest.TestCase):
    def test_complex_scaling_preserves_snr_but_not_reference_response(self):
        a = np.array([1., 1j])
        w = np.array([.5, .25+.5j])
        rn = np.diag([2., 3.])
        c = 2+3j
        g = np.vdot(w, a)
        scaled = c*w
        self.assertAlmostEqual(np.vdot(scaled, a), c.conjugate()*g)
        snr = abs(g)**2 / np.vdot(w, rn@w).real
        scaled_snr = abs(np.vdot(scaled, a))**2 / np.vdot(scaled, rn@scaled).real
        self.assertAlmostEqual(snr, scaled_snr)
        self.assertAlmostEqual(np.vdot(w/g.conjugate(), a), 1.)
        self.assertGreater(abs(np.vdot(w/g, a)-1), .1)

    def test_two_tone_waveforms_have_independent_integer_cycle_scores(self):
        t = np.arange(2400, 29600)/16000
        low, high, quadrature = np.cos(2*np.pi*500*t), np.cos(2*np.pi*1500*t), np.sin(2*np.pi*1500*t)
        reference = .1*(low+high)
        outputs = [reference, .1*(low-high), .1*(low+quadrature)]
        results = ex.frequency_phase_ambiguity(include_published=False)['cases']
        for row, output, mse, nmse in zip(results, outputs, [0., .02, .01], [0., 2., 1.]):
            self.assertAlmostEqual(np.mean(output**2), .01)
            self.assertAlmostEqual(np.mean((output-reference)**2), mse)
            self.assertAlmostEqual(row['steady_error_mean_square'], mse)
            self.assertAlmostEqual(row['steady_nmse'], nmse)
        # A +j WEIGHT becomes -j OUTPUT, and Re[-j exp(jwt)] is +sin.
        np.testing.assert_allclose(np.real(-1j*np.exp(2j*np.pi*1500*t)), quadrature, atol=1e-14)

    def test_conjugate_frequency_pair_is_real_and_has_positive_sine(self):
        spectrum = np.zeros(32, complex)
        spectrum[3], spectrum[-3] = -1j, 1j
        signal = np.fft.ifft(spectrum)
        np.testing.assert_allclose(signal.imag, 0., atol=1e-16)
        np.testing.assert_allclose(signal.real, np.sin(2*np.pi*3*np.arange(32)/32)/16, atol=1e-15)
        spectrum[-3] = -1j  # the wrong equal phase leaves an imaginary signal
        self.assertGreater(np.max(abs(np.fft.ifft(spectrum).imag)), .05)

    def test_published_pcm_has_exact_integer_scores_not_equal_quantized_power(self):
        directory = ROOT/'codes/chapters/ch05/phase_audio'
        def read(name):
            with wave.open(str(directory/f'phase_{name}.wav'), 'rb') as stream:
                self.assertEqual((stream.getframerate(), stream.getnchannels(), stream.getnframes(), stream.getsampwidth()),
                                 (16000, 1, 32000, 2))
                return np.frombuffer(stream.readframes(32000), '<i2').astype(np.int64)[2400:29600]
        reference = read('reference')
        # Exact answers are independently derived integer sums, no manifest oracle.
        self.assertEqual(int(reference@reference), 292026771800)
        energies = []
        for name, energy, error in [('reference', 292026771800, 0),
                                    ('flip', 292035584600, 584062356400),
                                    ('quadrature', 292051763500, 292037208100)]:
            samples = read(name)
            energies.append(int(samples@samples))
            self.assertEqual(energies[-1], energy)
            self.assertEqual(int((samples-reference)@(samples-reference)), error)
        self.assertEqual(len(set(energies)), 3)
        self.assertEqual(27200*32768**2, 29205777612800)

    def test_channel_preprocessing_changes_direction_and_rank(self):
        report = ex.channel_mask_prefilter_model()
        fixed, alternate = report['cases']
        np.testing.assert_array_equal(report['original_scm'], np.ones((2, 2)))
        np.testing.assert_array_equal(fixed['frame_average_scm'], [[1., .5], [.5, .25]])
        np.testing.assert_array_equal(alternate['frame_average_scm'], [[5/8, .5], [.5, 5/8]])
        np.testing.assert_allclose(fixed['eigenvalues'], [0., 5/4], atol=1e-15)
        np.testing.assert_allclose(alternate['eigenvalues'], [1/8, 9/8])
        self.assertEqual((fixed['rank'], alternate['rank']), (1, 2))
        self.assertAlmostEqual(np.linalg.det(alternate['frame_average_scm']), 9/64)
        np.testing.assert_allclose(report['pseudo_rtf_mvdr_weights'], [4/5, 2/5])
        self.assertAlmostEqual(report['response_on_original_target'], 6/5)
        self.assertAlmostEqual(report['response_on_prefiltered_target'], 1.)

    def test_scalar_denominators_change_scale_without_channel_rank_change(self):
        for row in ex.channel_mask_prefilter_model()['cases']:
            over_frames, over_mass = row['scalar_frame_average_scm'], row['scalar_mass_normalized_scm']
            np.testing.assert_array_equal(over_frames, np.full((2, 2), 5/8))
            np.testing.assert_array_equal(over_mass, np.ones((2, 2)))
            self.assertEqual(np.linalg.matrix_rank(over_frames), 1)
            self.assertEqual(np.linalg.matrix_rank(over_mass), 1)
            self.assertAlmostEqual(np.trace(over_frames), 5/4)
            self.assertAlmostEqual(np.trace(over_mass), 2.)


if __name__ == '__main__':
    unittest.main()
