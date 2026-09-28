"""Known inverse, geometric impulse train and independent one-tap WLS audio checks."""
import unittest
import numpy as np
from codes.chapters.ch00.core.audio_samples import wpe_predictable_case,prepare_exports,read_pcm16
from codes.chapters.ch02.core.spectral import stft,istft


class WPEPredictableAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = wpe_predictable_case()

    def test_feedback_against_explicit_geometric_impulse_train(self):
        s = self.case['signals']['wpe_predictable_target']
        # Build the causal impulse response independently of the feedback loop.
        expected = np.zeros_like(s)
        for k in range(63):
            offset = 512*k
            expected[offset:] += .65**k*s[:len(s)-offset]
        np.testing.assert_allclose(self.case['signals']['wpe_predictable_reverberant'],expected,atol=3e-16,rtol=0)
        np.testing.assert_allclose(self.case['signals']['wpe_predictable_oracle_inverse'],s,atol=3e-17,rtol=0)
        np.testing.assert_array_equal(s[:4000],0)
        np.testing.assert_array_equal(s[16000:],0)
        self.assertLessEqual(np.max(abs(s)),.12)

    def test_one_tap_wpe_against_unscaled_scalar_quotients(self):
        signals = self.case['signals']
        X = stft(signals['wpe_predictable_reverberant'],n_fft=512,hop_length=128)[0]
        expected = X.copy()
        floor = 1e-5*np.max(abs(X)**2,axis=1)
        past = X[:,:-4]
        for _ in range(3):
            power = np.maximum(abs(expected)**2,floor[:,None])
            # One-tap direct scalar arithmetic: no covariance solve, input
            # normalization, or common weight rescaling from the implementation.
            inverse = 1/power[:,4:]
            numerator = np.sum(inverse*past*X[:,4:].conj(),axis=1)
            denominator = 1.000001*np.sum(inverse*abs(past)**2,axis=1)
            g = numerator/denominator
            expected[:,4:] = X[:,4:]-g.conj()[:,None]*past
        reconstructed = istft(expected[None],n_fft=512,hop_length=128,length=32000)[0]
        np.testing.assert_allclose(signals['wpe_predictable_output'],reconstructed,atol=1e-14,rtol=1e-12)

    def test_pcm_reference_one_gain_and_real_decoded_scores(self):
        files,groups = prepare_exports({'wpe_predictable':self.case})
        self.assertEqual(groups['wpe_predictable']['common_export_gain'],1)
        self.assertEqual(len(files),4)
        pcm = {}
        for filename,(data,info) in files.items():
            fs,decoded = read_pcm16(data)
            self.assertEqual((fs,decoded.shape),(16000,(1,32000)))
            self.assertLessEqual(info['quantization_max_abs_error'],1/65536+1e-15)
            pcm[filename[:-4]] = decoded[0]
        truth = pcm['wpe_predictable_target'][6400:14400]
        scores = self.case['parameters']['pcm_to_pcm_reference_scores']
        for name,value in pcm.items():
            active = value[6400:14400]
            self.assertAlmostEqual(scores['files'][name]['steady_projection_gain'],np.dot(active,truth)/np.dot(truth,truth),places=14)
            self.assertAlmostEqual(scores['files'][name]['steady_relative_squared_reference_error'],np.sum((active-truth)**2)/np.sum(truth**2),places=14)
            self.assertAlmostEqual(scores['files'][name]['tail_mean_square'],np.mean(value[16000:]**2),places=17)
        np.testing.assert_array_equal(pcm['wpe_predictable_oracle_inverse'],pcm['wpe_predictable_target'])
        # Independent scalar WLS above establishes the actual output. These
        # inequalities express the intended teaching boundary, not a ranking.
        self.assertLess(scores['output_to_input_tail_power_ratio_db'],-11)
        gain = scores['files']['wpe_predictable_output']['steady_projection_gain']
        self.assertTrue(.52 < gain < .54)
        self.assertEqual(self.case['parameters']['stft']['frames'],251)


if __name__ == '__main__':
    unittest.main()
