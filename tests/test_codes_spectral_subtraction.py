"""Independent arithmetic and asset invariants for the E10-13 baseline."""

import hashlib
import unittest

import numpy as np

from codes.array_tutorial.audio_samples import build_cases, prepare_exports, read_pcm16
from codes.array_tutorial.noise_suppression import power_spectral_subtraction
from codes.examples.spectral_subtraction_demo import run_demo


class SpectralSubtractionTest(unittest.TestCase):
    def test_hand_power_and_phase_without_using_the_implementation_as_oracle(self):
        y = np.array([[2j, 3j, 1j]], dtype=complex)
        enhanced, noise = power_spectral_subtraction(y, np.array([0]), floor_ratio=.04)
        np.testing.assert_array_equal(noise, [4.])
        np.testing.assert_allclose(enhanced[0, 1], 1j * np.sqrt(5), rtol=0, atol=1e-15)
        np.testing.assert_allclose(enhanced[0, 2], .2j, rtol=0, atol=1e-15)
        zero_floor, _ = power_spectral_subtraction(y, np.array([0]), floor_ratio=0.)
        self.assertEqual(zero_floor[0, 2], 0j)

    def test_zero_energy_and_scale_invariance(self):
        y = np.array([[0j, 2 + 0j, 3j], [0j, 0j, 0j]])
        result, noise = power_spectral_subtraction(y, np.array([0, 1]), floor_ratio=.04)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(result[1], np.zeros(3))
        np.testing.assert_allclose(noise, [2., 0.], rtol=0, atol=0)
        scaled, _ = power_spectral_subtraction(y * 1e-7, np.array([0, 1]), floor_ratio=.04)
        np.testing.assert_allclose(scaled / 1e-7, result, rtol=1e-14, atol=1e-14)

    def test_invalid_inputs_fail_before_coercion(self):
        y = np.array([[1j, 2j]])
        invalid = [np.array([[np.nan, 0]]), np.array([[np.inf, 0]]),
                   np.array([[1e308 + 1e308j, 0]]), np.zeros((0, 2)),
                   np.zeros((1, 0)), np.array(['text'])]
        for value in invalid:
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                power_spectral_subtraction(value, np.array([0]))
        for indices in ([], [0, 0], [-1], [2], [0.5], [True], [[0]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                power_spectral_subtraction(y, np.asarray(indices))
        for kwargs in ({'floor_ratio': -1}, {'floor_ratio': 1.1},
                       {'floor_ratio': np.nan}, {'floor_ratio': 1j},
                       {'oversubtraction': -1}, {'oversubtraction': np.inf},
                       {'oversubtraction': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                power_spectral_subtraction(y, np.array([0]), **kwargs)

    def test_new_audio_group_and_original_55_wav_byte_hashes(self):
        cases = build_cases()
        files, groups = prepare_exports(cases)
        self.assertEqual(len(files), 60)
        self.assertEqual(len(groups), 14)
        old = [(name, hashlib.sha256(blob).hexdigest()) for name, (blob, _) in files.items()
               if not name.startswith('spectral_')]
        aggregate = hashlib.sha256(''.join(name + digest for name, digest in sorted(old)).encode()).hexdigest()
        self.assertEqual(aggregate, 'debc8d44e275a14cc6cec77f9eb7bd27d7f98dc42652cabc8c28d533f6935540')
        case = cases['spectral_subtraction']
        signals = case['signals']
        np.testing.assert_allclose(signals['spectral_noisy'],
                                   signals['spectral_clean'] + signals['spectral_noise'], rtol=0, atol=0)
        self.assertTrue(np.all(signals['spectral_clean'][:6400] == 0))
        self.assertEqual(case['parameters']['noise_only_stft_frame_indices'][0], 2)
        for name in signals:
            self.assertIn(name + '.wav', files)
            _, decoded = read_pcm16(files[name + '.wav'][0])
            self.assertEqual(decoded.shape, (1, 32000))
            self.assertLessEqual(np.max(np.abs(decoded - signals[name][None] *
                                              groups['spectral_subtraction']['common_export_gain'])),
                                 .5 / 32768 + 1e-15)

    def test_demo_reports_separate_math_and_audio_observations(self):
        result = run_demo()
        self.assertEqual(result['exercise_id'], 'E10-13')
        np.testing.assert_allclose(result['hand']['output_powers'], [5., .04], atol=1e-14)
        self.assertEqual(result['hand']['zero_floor_low_bin_amplitude'], 0.)
        self.assertEqual(result['audio']['noise_only_frame_count'], 47)
        self.assertEqual({name: round(value, 4) for name, value in
                          result['audio']['noise_only_rms'].items()},
                         {'spectral_noisy': .0697, 'spectral_floor04': .0404,
                          'spectral_floor00': .0381})


if __name__ == '__main__':
    unittest.main()
