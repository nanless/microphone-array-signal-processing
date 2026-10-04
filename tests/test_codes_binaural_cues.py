"""Independent exact-input and published-asset checks for E01-08.

Analytic four-tone values and rational PCM input establish expectations;
generated manifests are not used to invent the expected waveforms or gains.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from codes.chapters.ch01.core.binaural_cues import build_cues, measure_cues
from codes.chapters.ch01.examples.generate_binaural_cues import (
    DEFAULT_DIRECTORY, check_assets, generate_assets,
)


class BinauralCueTest(unittest.TestCase):
    def test_source_samples_gains_causal_shift_and_complete_tail(self):
        cases = build_cues()
        # Evaluate a separate scalar trigonometric expression at representative
        # fade/interior/tail points; no call to the source generator for expected values.
        points = (0, 1, 160, 319, 320, 1601, 30399, 31680, 31840, 31999)
        for index in points:
            time = index / 16000
            fade = min(1, time / .02, (2 - time) / .02)
            expected = .08 * sum(math.sin(2 * math.pi * frequency * time)
                                 for frequency in (500, 800, 1300, 2400)) * fade
            self.assertAlmostEqual(cases['reference'][0, index], expected, places=14)
        configurations = {'reference': (0, 1, 1), 'itd_only': (8, 1, 1),
                          'ild_only': (0, .5, 1), 'consistent': (8, .5, 1),
                          'conflicting': (8, 1, .5)}
        for name, (delay, left_gain, right_gain) in configurations.items():
            signal = cases[name]
            self.assertEqual(signal.shape, (2, 32008))
            np.testing.assert_array_equal(signal[0, :delay], 0)
            np.testing.assert_array_equal(signal[0, delay:delay + 32000], cases['reference'][0, :32000] * left_gain)
            np.testing.assert_array_equal(signal[1, :32000], cases['reference'][0, :32000] * right_gain)
            np.testing.assert_array_equal(signal[0, delay + 32000:], 0)
            np.testing.assert_array_equal(signal[1, 32000:], 0)
        self.assertNotEqual(cases['itd_only'][0, -1], 0)

    def test_known_delays_and_power_match_analytic_orthogonality(self):
        for name, waveform in build_cues().items():
            lag = 8 if name in ('itd_only', 'consistent', 'conflicting') else 0
            sign = 1 if name in ('ild_only', 'consistent') else -1 if name == 'conflicting' else 0
            result = measure_cues(waveform, (lag, 0))
            self.assertEqual(result['estimated_itd_samples'], lag)
            self.assertEqual(result['truth_itd_samples'], lag)
            self.assertEqual(result['correlation_window_samples'], 28800)
            # Each orthogonal sinusoid contributes A^2/2 over 1.8 seconds.
            expected_power = 4 * .08**2 / 2
            self.assertAlmostEqual(result['left_mean_square'], expected_power / (4 if sign == 1 else 1), places=14)
            self.assertAlmostEqual(result['right_mean_square'], expected_power / (4 if sign == -1 else 1), places=14)
            self.assertAlmostEqual(result['ild_right_minus_left_db'], sign * 20 * math.log10(2), places=12)
            self.assertEqual(result['power_receive_windows'], [[1600 + lag, 30400 + lag], [1600, 30400]])
            self.assertEqual(result['correlation_lags_samples'], list(range(-16, 17)))
            self.assertEqual(len(result['correlation_energy_denominators']), 33)

    def test_actual_committed_pcm_values_with_independent_integer_energy(self):
        manifest = check_assets()
        for name, lag, gains in (('reference', 0, (1, 1)), ('itd_only', 8, (1, 1)),
                                 ('ild_only', 0, (.5, 1)), ('consistent', 8, (.5, 1)),
                                 ('conflicting', 8, (1, .5))):
            raw = (DEFAULT_DIRECTORY / (name + '.wav')).read_bytes()
            with wave.open(io.BytesIO(raw), 'rb') as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()),
                                 (16000, 2, 2, 32008))
                pcm = np.frombuffer(wav.readframes(32008), dtype='<i2').reshape(-1, 2).astype(np.int64)
            left = pcm[1600 + lag:30400 + lag, 0]
            right = pcm[1600:30400, 1]
            # Integer square sums are exact here (below 2^53); derive powers and
            # ILD independently of the implementation's float normalization.
            left_sum, right_sum = int(np.sum(left**2)), int(np.sum(right**2))
            score = manifest['samples'][name]['pcm_measurements']
            self.assertEqual(score['left_mean_square'], left_sum / (28800 * 32768**2))
            self.assertEqual(score['right_mean_square'], right_sum / (28800 * 32768**2))
            self.assertAlmostEqual(score['ild_right_minus_left_db'], 10 * math.log10(right_sum / left_sum), places=13)
            # A separate squared normalized-signal difference chooses the lag.
            right_float = right.astype(float)
            right_float /= math.sqrt(sum(value * value for value in right_float))
            errors = []
            for candidate in range(-16, 17):
                candidate_left = pcm[1600 + candidate:30400 + candidate, 0].astype(float)
                candidate_left /= math.sqrt(sum(value * value for value in candidate_left))
                errors.append(float(np.sum((candidate_left - right_float)**2)))
            self.assertEqual(int(np.argmin(errors)) - 16, lag)
            self.assertEqual(score['estimated_itd_samples'], lag)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['files'][name + '.wav']['sha256'])

    def test_pcm_rounding_is_independently_calculated_and_no_channel_swap(self):
        generate_source = build_cues()['conflicting']
        with wave.open(str(DEFAULT_DIRECTORY / 'conflicting.wav'), 'rb') as wav:
            pcm = np.frombuffer(wav.readframes(32008), dtype='<i2').reshape(-1, 2)
        for frame in (0, 8, 17, 1608, 30399, 32007):
            for channel in (0, 1):
                self.assertEqual(int(pcm[frame, channel]), round(float(generate_source[channel, frame]) * 32768))
        error = np.abs(pcm.T / 32768 - generate_source)
        self.assertLessEqual(float(np.max(error)), .5 / 32768)

    def test_check_is_read_only_on_success_and_every_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'fixtures'
            generate_assets(directory)

            def snapshot():
                return {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in directory.iterdir() if path.is_file()}

            before = snapshot()
            check_assets(directory)
            self.assertEqual(snapshot(), before)
            payload = (directory / 'itd_only.wav').read_bytes()
            (directory / 'itd_only.wav').write_bytes(payload[:-2] + bytes([payload[-2] ^ 1, payload[-1]]))
            changed = snapshot()
            with self.assertRaisesRegex(ValueError, 'Stale or modified WAV'):
                check_assets(directory)
            self.assertEqual(snapshot(), changed)
            (directory / 'itd_only.wav').write_bytes(payload)
            (directory / 'extra.wav').write_bytes(b'unrelated')
            changed = snapshot()
            with self.assertRaisesRegex(ValueError, 'extra'):
                check_assets(directory)
            self.assertEqual(snapshot(), changed)
            (directory / 'extra.wav').unlink()
            (directory / 'itd_only.wav').unlink()
            changed = snapshot()
            with self.assertRaisesRegex(ValueError, 'missing'):
                check_assets(directory)
            self.assertEqual(snapshot(), changed)
            absent = Path(temporary) / 'absent'
            with self.assertRaises(ValueError):
                check_assets(absent)
            self.assertFalse(absent.exists())

    def test_source_measurement_and_format_manifest_changes_fail_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = generate_assets(directory)
            cases = [('source', lambda m: m['source_sha256'].__setitem__(next(iter(m['source_sha256'])), '0' * 64)),
                     ('measurement', lambda m: m['samples']['consistent']['pcm_measurements'].__setitem__('estimated_itd_samples', -8)),
                     ('format', lambda m: m['files']['reference.wav'].__setitem__('channels', 1)),
                     ('gain', lambda m: m.__setitem__('common_export_gain', .9)),
                     ('environment', lambda m: m['environment'].__setitem__('numpy', 'wrong'))]
            for label, change in cases:
                with self.subTest(label=label):
                    modified = json.loads(json.dumps(manifest))
                    change(modified)
                    path = directory / 'MANIFEST.json'
                    path.write_text(json.dumps(modified), encoding='utf-8')
                    before = (path.read_bytes(), path.stat().st_mtime_ns)
                    with self.assertRaisesRegex(ValueError, 'Manifest differs'):
                        check_assets(directory)
                    self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_cli_check_failure_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / 'missing'
            result = subprocess.run([sys.executable, '-m', 'codes.chapters.ch01.examples.generate_binaural_cues',
                                     '--check', '--output', str(missing)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('asset directory is missing', result.stderr)
            self.assertFalse(missing.exists())

    def test_analysis_rejects_silence_complex_and_wrong_shape(self):
        for invalid in (np.zeros((2, 32008)), np.ones((2, 32008), dtype=complex),
                        np.ones((2, 32000)), np.full((2, 32008), np.nan),
                        np.full((2, 32008), np.inf)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    measure_cues(invalid, (0, 0))
        with self.assertRaises(ValueError):
            measure_cues(build_cues()['reference'], (0, 8))

    def test_integer_amplitudes_are_squared_without_dtype_wrap(self):
        # Python integer arithmetic supplies the expected squares, independent
        # of both NumPy's original dtype and the measurement implementation.
        for dtype, amplitude in ((np.int16, 300), (np.int16, 32767),
                                 (np.int32, 100000)):
            with self.subTest(dtype=dtype, amplitude=amplitude):
                values = np.full((2, 32008), amplitude, dtype=dtype)
                before = values.copy()
                result = measure_cues(values, (0, 0))
                self.assertEqual(result['left_mean_square'], amplitude * amplitude)
                self.assertEqual(result['right_mean_square'], amplitude * amplitude)
                self.assertEqual(result['left_rms'], amplitude)
                self.assertEqual(result['ild_right_minus_left_db'], 0.0)
                self.assertAlmostEqual(result['winning_correlation'], 1.0)
                np.testing.assert_array_equal(values, before)


if __name__ == '__main__':
    unittest.main()
