import tempfile
import unittest
from pathlib import Path

import numpy as np

from codes.array_tutorial.audio_samples import (delay_samples, pcm16_bytes, read_pcm16,
                                               build_cases, prepare_exports)
from codes.examples.generate_audio_samples import generate


class AudioSamplesTest(unittest.TestCase):
    def test_delay_is_causal_not_circular(self):
        np.testing.assert_array_equal(delay_samples(np.array([1., 2., 3.]), 1), [0, 1, 2])
        np.testing.assert_array_equal(delay_samples(np.ones(3), 4), np.zeros(3))
        for invalid in (-1, .5, True):
            with self.assertRaises(ValueError):
                delay_samples(np.ones(3), invalid)
        with self.assertRaises(ValueError):
            delay_samples([1j], 1)

    def test_pcm_channels_endpoints_and_quantization(self):
        x = np.array([[-.5, 0, .5], [.2, -.2, .25]])
        rate, result = read_pcm16(pcm16_bytes(x))
        self.assertEqual(rate, 16000)
        self.assertEqual(result.shape, (2, 3))
        self.assertLessEqual(np.max(np.abs(result-x)), .5/32768 + 1e-15)
        np.testing.assert_array_equal(result[0], x[0])

    def test_invalid_pcm_is_rejected_not_clipped(self):
        for x in ([1.], [np.nan], [np.inf], [], [1j]):
            with self.assertRaises(ValueError):
                pcm16_bytes(x)
        for rate in (0, -1, .5, True):
            with self.assertRaises(ValueError):
                pcm16_bytes([0.], rate)

    def test_case_models_and_shared_gain(self):
        cases = build_cases()
        sep = cases['separation']['signals']
        np.testing.assert_allclose(sep['separation_recovered1'], sep['separation_source1'], atol=1e-15)
        np.testing.assert_allclose(sep['separation_mixture'][0], sep['separation_source1'] + .5*sep['separation_source2'])
        pan = cases['tracking']['signals']['tracking_pan']
        np.testing.assert_allclose(np.sum(pan**2, axis=0), sep['separation_source1']**2, atol=1e-15)
        files, groups = prepare_exports(cases)
        self.assertEqual(len(files), 44)
        for blob, info in files.values():
            self.assertEqual(info['common_export_gain'], groups[info['group']]['common_export_gain'])
            self.assertLess(info['peak'], .801)
            self.assertLessEqual(info['quantization_max_abs_error'], .5/32768 + 1e-15)

    def test_manifest_check_detects_modified_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(generate(root)['files'], 44)
            self.assertTrue(generate(root, check=True)['checked'])
            (root/'spatial_reference.wav').write_bytes(b'not a WAV')
            with self.assertRaisesRegex(ValueError, 'audio content differs'):
                generate(root, check=True)
            self.assertEqual((root/'spatial_reference.wav').read_bytes(), b'not a WAV')

    def test_common_gain_attenuates_all_signals_without_boost(self):
        files, groups = prepare_exports({'test': {'signals': {'reference': np.array([1., -1.]),
                                                            'output': np.array([2., -2.])},
                                                  'parameters': {}, 'limits': 'fixture'}})
        self.assertEqual(groups['test']['common_export_gain'], .4)
        for name, expected in [('reference.wav', .4), ('output.wav', .8)]:
            _, x = read_pcm16(files[name][0])
            self.assertAlmostEqual(x[0, 0], expected, delta=.5/32768)

    def test_correlation_and_polarity_models_independent_identities(self):
        cases = build_cases()
        signal = cases['correlation']['signals']
        np.testing.assert_array_equal(signal['correlation_common'], signal['correlation_single'])
        noise = signal['correlation_single'] - signal['correlation_reference']
        mean_noise = signal['correlation_independent'] - signal['correlation_reference']
        # Finite-sample check, deliberately not exact population variance equality.
        self.assertTrue(.47 < np.mean(mean_noise**2) / np.mean(noise**2) < .53)
        polarity = cases['polarity']['signals']
        np.testing.assert_allclose(polarity['polarity_uncorrected'], .05*polarity['polarity_reference'], atol=1e-16)
        np.testing.assert_allclose(polarity['polarity_corrected'], .95*polarity['polarity_reference'], atol=1e-16)

    def test_conditioning_matches_closed_form_inverse_not_another_solve(self):
        signal = build_cases()['conditioning']['signals']
        reference = signal['conditioning_reference']
        self.assertEqual(reference.shape, (2, 32000))
        noises = []
        for label, off_diagonal in [('well', .5), ('ill', .99)]:
            observed = signal[f'conditioning_{label}_input']
            self.assertEqual(observed.shape, (2, 32000))
            a = off_diagonal
            noise1 = observed[0] - reference[0] - a*reference[1]
            noise2 = observed[1] - a*reference[0] - reference[1]
            noises.append(np.vstack((noise1, noise2)))
            expected = np.vstack((noise1-a*noise2, noise2-a*noise1))/(1-a*a)
            actual = signal[f'conditioning_{label}_output'] - reference
            np.testing.assert_allclose(actual, expected, atol=2e-14)
        np.testing.assert_allclose(noises[0], noises[1], atol=2e-16, rtol=0)

    def test_all_export_metadata_is_measured_from_pcm_and_original_float(self):
        cases = build_cases()
        files, groups = prepare_exports(cases)
        for name, (blob, record) in files.items():
            _, pcm = read_pcm16(blob)
            raw = np.atleast_2d(cases[record['group']]['signals'][name[:-4]])
            raw = raw * groups[record['group']]['common_export_gain']
            self.assertAlmostEqual(record['rms'], float(np.sqrt(np.sum(pcm*pcm)/pcm.size)), places=15)
            self.assertEqual(record['quantization_max_abs_error'], float(np.max(np.abs(pcm-raw))))
        params = groups['conditioning']['parameters']
        for matrix, expected in zip([params['well_matrix'], params['ill_matrix']], [3., 199.]):
            a = matrix[0][1]
            self.assertAlmostEqual((1+a)/(1-a), expected, places=10)
        self.assertEqual(params['condition_numbers_2norm'], [3., 199.])

    def test_exported_polarity_gain_is_not_peak_normalized(self):
        files, _ = prepare_exports()
        decoded = {name: read_pcm16(files[f'polarity_{name}.wav'][0])[1]
                   for name in ('reference', 'corrected', 'uncorrected')}
        # Difference of two independently rounded PCM signals is bounded by
        # half an LSB for each term, with its associated analytic gain.
        for name, gain in [('corrected', .95), ('uncorrected', .05)]:
            self.assertLessEqual(np.max(np.abs(decoded[name]-gain*decoded['reference'])),
                                 (1+gain)*.5/32768 + 1e-15)

    def test_nonlinear_projection_matches_independent_trigonometric_identity(self):
        case = build_cases()['nonlinear']
        signals = case['signals']
        phase = 2*np.pi*500*np.arange(32000)/16000
        fundamental, third = np.sin(phase), np.sin(3*phase)
        # Independent phase multiplication order accumulates float64 roundoff
        # over 1000 periods; 1e-12 remains far below one PCM quantization step.
        np.testing.assert_allclose(signals['nonlinear_echo'], .496*fundamental-.032*third, atol=1e-12, rtol=0)
        np.testing.assert_allclose(signals['nonlinear_estimate'], .496*fundamental, atol=1e-12, rtol=0)
        np.testing.assert_allclose(signals['nonlinear_residual'], -.032*third, atol=1e-12, rtol=0)
        self.assertAlmostEqual(case['parameters']['fit_gain'], 1.24, places=13)
        self.assertAlmostEqual(np.mean(signals['nonlinear_echo']**2), .12352, places=13)
        self.assertAlmostEqual(np.mean(signals['nonlinear_residual']**2), .000512, places=13)

    def test_nonlinear_pcm_retains_common_gain_and_harmonic_amplitudes(self):
        files, groups = prepare_exports()
        self.assertEqual(groups['nonlinear']['common_export_gain'], 1.)
        for name, expected in [('reference', [.4, 0]), ('echo', [.496, .032]),
                               ('estimate', [.496, 0]), ('residual', [0, .032])]:
            _, decoded = read_pcm16(files[f'nonlinear_{name}.wav'][0])
            self.assertEqual(decoded.shape, (1, 32000))
            amplitude = 2*np.abs(np.fft.rfft(decoded[0])[[1000, 3000]])/32000
            # DFT amplitude error <= twice the pointwise rounding bound.
            np.testing.assert_allclose(amplitude, expected, atol=1/32768, rtol=0)

    def test_four_mic_fractional_array_geometry_and_alignment(self):
        case = build_cases()['fractional_array']
        signals = case['signals']
        params = case['parameters']
        expected_adjacent = .04 * .5 / 343
        np.testing.assert_allclose(-np.diff(params['relative_arrival_seconds']), expected_adjacent,
                                   rtol=1e-13, atol=0)
        self.assertAlmostEqual(params['arrival_difference_adjacent_samples'],
                               16000 * expected_adjacent, places=12)
        self.assertEqual(signals['fractional_array'].shape, (4, 32000))
        # The independent model says correct geometry restores coherence; this
        # test evaluates the actual generated arrays without per-file gain fit.
        reference = signals['fractional_reference'][320:-320]
        mse_unaligned = np.mean((signals['fractional_unaligned'][320:-320]-reference)**2)
        mse_aligned = np.mean((signals['fractional_aligned'][320:-320]-reference)**2)
        self.assertLess(mse_aligned, mse_unaligned / 10)
        files, groups = prepare_exports({'fractional_array': case})
        self.assertEqual(len(files), 4)
        self.assertEqual(groups['fractional_array']['parameters']['azimuth_deg'], 30.)
        _, decoded = read_pcm16(files['fractional_array.wav'][0])
        self.assertEqual(decoded.shape, (4, 32000))


if __name__ == '__main__':
    unittest.main()
