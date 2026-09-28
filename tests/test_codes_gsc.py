"""Independent finite-step and actual-PCM checks for the scalar GSC fixture."""
import unittest
import numpy as np

from codes.chapters.ch05.core.gsc import ScalarGSCNLMS
from codes.chapters.ch00.core.audio_samples import gsc_gate_case, prepare_exports, read_pcm16


class ScalarGSCTest(unittest.TestCase):
    def test_complex_step_uses_previous_weight_and_conjugated_error(self):
        state = ScalarGSCNLMS(step_size=.5, epsilon=1.)
        np.testing.assert_array_equal(state.process([1+1j], [1+2j]), [1+1j])
        self.assertAlmostEqual(state.coefficient, .25+1j/12)
        np.testing.assert_allclose(state.process([1+1j], [1+2j], update=False), [7/12*(1+1j)])
        self.assertAlmostEqual(state.coefficient, .25+1j/12)
        self.assertEqual(state.samples_processed, 2)

    def test_blocks_are_continuous_and_reset_is_explicit(self):
        d = np.array([1, 2j, 0, 3-1j, -2])
        u = np.array([1j, 2, 0, 1+1j, 2])
        gate = np.array([True, False, True, True, False])
        whole, split = ScalarGSCNLMS(), ScalarGSCNLMS()
        a = whole.process(d, u, update=gate)
        b = np.concatenate([split.process(d[:2], u[:2], update=gate[:2]),
                            split.process(d[2:], u[2:], update=gate[2:])])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(whole.coefficient, split.coefficient)
        self.assertEqual(split.samples_processed, 5)
        split.reset()
        self.assertEqual((split.coefficient, split.samples_processed), (0j, 0))
        np.testing.assert_array_equal(split.process(d, u, update=gate), a)

    def test_zero_reference_and_empty_block_preserve_state(self):
        state = ScalarGSCNLMS()
        np.testing.assert_array_equal(state.process([1, 2j], [0, 0]), [1, 2j])
        self.assertEqual(state.coefficient, 0)
        self.assertEqual(state.process([], []).size, 0)
        self.assertEqual(state.samples_processed, 2)

    def test_invalid_block_is_atomic_including_arithmetic_failure(self):
        state = ScalarGSCNLMS()
        state.process([1], [1])
        before = state.coefficient, state.samples_processed
        for d, u, gate in [([1, np.nan], [1, 1], True), ([1], [1, 2], True),
                           ([1], [1], 1), ([1, 1e308], [1, 1e308], True)]:
            with self.subTest(d=d), self.assertRaises(ValueError):
                state.process(d, u, update=gate)
            self.assertEqual((state.coefficient, state.samples_processed), before)
        for kwargs in [{'step_size': 0}, {'step_size': 2}, {'epsilon': 0},
                       {'step_size': True}, {'epsilon': float('nan')}]:
            with self.assertRaises(ValueError):
                ScalarGSCNLMS(**kwargs)


class GSCAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = gsc_gate_case()
        cls.files, cls.groups = prepare_exports({'gsc_gate': cls.case})

    def test_mixture_and_known_gate_have_distinct_physical_claims(self):
        sig = self.case['signals']
        s, x = sig['gsc_reference'], sig['gsc_array']
        n = np.arange(16000, 30000)
        t = n/16000
        expected = .16*np.sin(2*np.pi*220*t)+.1*np.sin(2*np.pi*330*t)+.07*np.sin(2*np.pi*660*t)
        np.testing.assert_allclose(s[n], expected, atol=1e-14)
        np.testing.assert_allclose(x[1, n], .96*s[n], atol=1e-15)
        np.testing.assert_allclose(x[1, :8000], .5*x[0, :8000], atol=0)
        np.testing.assert_array_equal(x[:, 8000:9600], 0)
        np.testing.assert_allclose(sig['gsc_gate_frozen'][n], .92*s[n], atol=2e-15)
        self.assertLess(np.max(np.abs(sig['gsc_always_adapt'][n])), 1e-14)
        for name, last in [('always_adapt', 49), ('gate_frozen', 3)]:
            checkpoints = self.case['parameters']['coefficient_checkpoints'][name]
            self.assertAlmostEqual(checkpoints[0]['coefficient'], 3, places=12)
            self.assertAlmostEqual(checkpoints[-1]['coefficient'], last, places=12)

    def test_pcm_reference_scores_and_quantization_are_actual_file_bytes(self):
        decoded = {}
        self.assertEqual(self.groups['gsc_gate']['common_export_gain'], 1)
        self.assertEqual(len(self.files), 4)
        for name, (blob, info) in self.files.items():
            rate, x = read_pcm16(blob)
            self.assertEqual(rate, 16000)
            self.assertEqual(x.shape, (2 if name == 'gsc_array.wav' else 1, 32000))
            self.assertLessEqual(info['quantization_max_abs_error'], 1/65536+1e-15)
            decoded[name] = x
        r = decoded['gsc_reference.wav'][0, 16000:30000]
        for key, expected_gain, expected_error in [('always_adapt', 0., 1.),
                  ('gate_frozen', .9200033979526868, .0799966476118432)]:
            y = decoded['gsc_'+key+'.wav'][0, 16000:30000]
            gain = float(np.dot(r, y)/np.dot(r, r))
            error = float(np.linalg.norm(y-r)/np.linalg.norm(r))
            self.assertAlmostEqual(gain, expected_gain, places=12)
            self.assertAlmostEqual(error, expected_error, places=12)
            recorded = self.case['parameters']['pcm_to_pcm_reference_scores'][key]
            self.assertAlmostEqual(gain, recorded['reference_projection_gain'])
            self.assertAlmostEqual(error, recorded['normalized_reference_error'])


if __name__ == '__main__':
    unittest.main()
