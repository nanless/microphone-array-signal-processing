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
        self.assertEqual(len(files), 23)
        for blob, info in files.values():
            self.assertEqual(info['common_export_gain'], groups[info['group']]['common_export_gain'])
            self.assertLess(info['peak'], .801)
            self.assertLessEqual(info['quantization_max_abs_error'], .5/32768 + 1e-15)

    def test_manifest_check_detects_modified_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(generate(root)['files'], 23)
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


if __name__ == '__main__':
    unittest.main()
