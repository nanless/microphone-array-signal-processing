"""Analytic clock identities, PCM bounds and read-only export verification."""
import unittest
import numpy as np
from codes.array_tutorial.audio_samples import clock_drift_case, prepare_exports, read_pcm16

class ClockAudioTest(unittest.TestCase):
    def test_clock_sampling_and_interior_sum_identity(self):
        case = clock_drift_case()
        signals = case['signals']
        self.assertEqual(signals['clock_array'].shape, (2, 128000))
        # n=53339 gives a physical time difference close to 1/(2*1500).
        n = np.arange(321, 127680)
        t = n / 16000
        t2 = n / 16001.6
        expected = sum(.18 * np.sin(np.pi*f*(t+t2))*np.cos(np.pi*f*(t-t2))
                       for f in (500, 1500))
        np.testing.assert_allclose(signals['clock_index_mean'][n], expected, atol=5e-12, rtol=0)
        np.testing.assert_array_equal(signals['clock_oracle_mean'], signals['clock_reference'])
        self.assertAlmostEqual((4-4/1.0001)*16000, 6.39936006399, places=9)
        first_null_time = 1.0001 / (2*1500*1e-4)
        self.assertAlmostEqual(first_null_time, 3.3336666666666663)
        self.assertLess(abs(np.cos(np.pi*1500*first_null_time*1e-4/1.0001)), 1e-14)

    def test_pcm_common_gain_and_no_fake_resampler(self):
        case = clock_drift_case()
        files, groups = prepare_exports({'clock_drift': case})
        self.assertEqual(len(files), 4)
        gain = groups['clock_drift']['common_export_gain']
        self.assertEqual(gain, 1)
        pcm = {}
        for name,(blob, info) in files.items():
            rate, x = read_pcm16(blob)
            self.assertEqual(rate, 16000)
            self.assertEqual(x.shape[-1], 128000)
            self.assertLessEqual(info['quantization_max_abs_error'], .5/32768 + 1e-15)
            pcm[name] = x
        np.testing.assert_array_equal(pcm['clock_oracle_mean.wav'], pcm['clock_reference.wav'])
        error = pcm['clock_index_mean.wav'] - pcm['clock_array.wav'].mean(axis=0)
        self.assertLessEqual(np.max(abs(error)), 1/32768 + 1e-15)
        self.assertIn('not measured compensation', case['limits'])

if __name__ == '__main__':
    unittest.main()
