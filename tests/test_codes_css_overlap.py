"""Independent overlap association and encoded-audio checks."""
import unittest
import numpy as np
from codes.array_tutorial.css import match_two_source_overlap
from codes.array_tutorial.audio_samples import css_overlap_case, pcm16_bytes, read_pcm16


class CSSOverlapTests(unittest.TestCase):
    def test_exact_mapping_both_directions_and_no_mutation(self):
        a = np.array([[1., -1, 0, 0], [0, 0, 1, -1]])
        original = a.copy()
        for order in ([0, 1], [1, 0]):
            r = match_two_source_overlap(a, a[order])
            self.assertEqual(r['current_indices_for_previous'], order)
        np.testing.assert_array_equal(a, original)

    def test_scale_phase_and_policy_are_distinct(self):
        a = np.array([[1., -1, 0, 0], [0, 0, 1, -1]])
        for scale in (1e-310, 1e300):
            r = match_two_source_overlap(a*scale, -a[::-1]*scale, minimum_rms=0)
            self.assertEqual(r['current_indices_for_previous'], [1, 0])
        self.assertEqual(match_two_source_overlap(a*1e-10, a*1e-10)['reason'], 'low_energy_overlap')

    def test_weak_overlap_and_ties(self):
        old = np.array([[1., -1, 0, 0], [0, 0, 1, -1]])
        weak = np.array([[1., 1, -1, -1], [1., 1, -1, -1]])
        self.assertIsNone(match_two_source_overlap(old, weak)['current_indices_for_previous'])
        r = match_two_source_overlap(np.ones((2, 4)), np.ones((2, 4)))
        self.assertEqual(r['reason'], 'low_energy_overlap')

    def test_input_contract(self):
        a = np.eye(2)
        for bad in (np.zeros((1, 4)), np.ones((2, 1)), np.ones((2, 2))*np.nan, np.eye(2, dtype=complex)):
            with self.assertRaises(ValueError): match_two_source_overlap(bad, bad)
        for kwargs in ({'minimum_rms': True}, {'minimum_margin': [0]}, {'minimum_correlation': 2}):
            with self.assertRaises(ValueError): match_two_source_overlap(a, a, **kwargs)

    def test_waveforms_against_given_signal_algebra(self):
        case = css_overlap_case()
        s = case['signals']['css_overlap_reference']
        expected = np.stack((s[0]+s[1]/10, s[1]+s[0]/10))
        np.testing.assert_allclose(case['signals']['css_overlap_aligned'], expected, atol=1e-16)
        np.testing.assert_allclose(case['signals']['css_overlap_naive'][:, 19200:], expected[::-1, 19200:])
        midpoint = 16000
        w = (midpoint-12800)/6399
        np.testing.assert_allclose(case['signals']['css_overlap_naive'][:, midpoint], (1-w)*expected[:, midpoint]+w*expected[::-1, midpoint])

    def test_pcm_scores_use_pcm_references(self):
        c = css_overlap_case()
        decoded = {}
        for key, array in c['signals'].items():
            fs, decoded[key] = read_pcm16(pcm16_bytes(array))
            self.assertEqual(fs, 16000)
            self.assertEqual(decoded[key].shape, array.shape)
            self.assertLess(np.max(np.abs(decoded[key]-array)), 1/32768)
        ref = decoded['css_overlap_reference']
        for key, values in c['parameters']['pcm_si_sdr_db'].items():
            for i, score in enumerate(values):
                s = ref[i]-ref[i].mean()
                y = decoded[key][0 if key.endswith('mixture') else i]
                y = y-y.mean()
                # Independent normalized squared correlation identity.
                rho2 = (y@s)**2 / ((y@y)*(s@s))
                self.assertAlmostEqual(score, 10*np.log10(rho2/(1-rho2)), places=10)
