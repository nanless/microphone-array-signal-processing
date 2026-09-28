"""Reference transport failure verified by delayed missing samples, not NLMS."""
import unittest
import numpy as np
from codes.array_tutorial.audio_samples import aec_dropout_case, prepare_exports, read_pcm16


class AECDropoutAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = aec_dropout_case()

    def test_known_path_rejects_only_unavailable_reference_not_target(self):
        signals = self.case['signals']
        target = signals['aec_dropout_target']
        good = signals['aec_dropout_complete_reference_residual']
        bad = signals['aec_dropout_missing_reference_residual']
        np.testing.assert_allclose(good, target, atol=1e-16, rtol=0)
        # The missing interval is away from the 320-sample file fades. Build
        # only its source samples independently and add each delayed path.
        random = np.random.default_rng(20261028).uniform(-.18,.18,32000)
        missing = np.zeros(32000)
        n = np.arange(12000,16000)
        missing[n] = random[n] + .08*np.sin(2*np.pi*310*n/16000)
        expected = .7*missing
        expected[80:] -= .3*missing[:-80]
        expected[240:] += .15*missing[:-240]
        np.testing.assert_allclose(bad-target, expected, atol=2e-14, rtol=0)
        np.testing.assert_allclose(bad[:12000], target[:12000], atol=1e-16, rtol=0)
        np.testing.assert_allclose(bad[16240:], target[16240:], atol=1e-16, rtol=0)
        self.assertGreater(abs(bad[16239]-target[16239]), 1e-5)
        self.assertEqual(self.case['parameters']['first_fully_recovered_output_sample'],16240)

    def test_pcm_scores_use_pcm_target_and_one_gain(self):
        files, groups = prepare_exports({'aec_dropout': self.case})
        self.assertEqual(len(files),4)
        self.assertEqual(groups['aec_dropout']['common_export_gain'],1.)
        decoded = {}
        for name,(blob,record) in files.items():
            rate, pcm = read_pcm16(blob)
            self.assertEqual((rate,pcm.shape),(16000,(1,32000)))
            self.assertLessEqual(record['quantization_max_abs_error'],1/65536+1e-15)
            decoded[name[:-4]] = pcm[0]
        target = decoded['aec_dropout_target']
        params = self.case['parameters']
        for label in ('complete_reference_residual','missing_reference_residual'):
            output = decoded['aec_dropout_'+label]
            for region,(start,end) in params['score_windows_samples'].items():
                delta = output[start:end]-target[start:end]
                expected = np.sum(delta**2)/np.sum(target[start:end]**2)
                self.assertAlmostEqual(expected,params['pcm_to_pcm_reference_scores'][label][region]['relative_squared_reference_error'],places=14)
        np.testing.assert_array_equal(decoded['aec_dropout_complete_reference_residual'],target)
        np.testing.assert_array_equal(decoded['aec_dropout_missing_reference_residual'][16240:],target[16240:])


if __name__ == '__main__':
    unittest.main()
