"""Independent source-clock sums, actual wave/struct PCM and asset preflight."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np

from codes.chapters.ch07.core import delay_audio as core
from codes.chapters.ch07.examples import generate_delay_audio as generator


def raw_channels(blob):
    with wave.open(io.BytesIO(blob), 'rb') as stream:
        channels = stream.getnchannels()
        if (stream.getframerate(), channels, stream.getnframes(), stream.getsampwidth(),
                stream.getcomptype()) != (16000, channels, 33024, 2, 'NONE'):
            raise AssertionError('actual PCM16 header')
        values = struct.unpack('<'+'h'*(33024*channels), stream.readframes(33024))
        return [values[c::channels] for c in range(channels)]


def snapshot(directory):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


class DelayAudioContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = core.generate_experiment()
        cls.blobs, cls.manifest = generator.prepare_assets()
        cls.raw = {name: raw_channels(cls.blobs[filename]) for name, filename in core.FILE_NAMES.items()}

    def write_fixture(self, output):
        output.mkdir()
        for filename, blob in self.blobs.items():
            (output/filename).write_bytes(blob)
        (output/'MANIFEST.json').write_text(json.dumps(self.manifest, allow_nan=False))

    def test_independent_scalar_source_delays_and_support(self):
        # Scalar trigonometry and direct source indices, without the shared
        # shift/LS helpers, establish the entire declared finite waveform.
        source = []
        for n in range(32256):
            local = n % 1536
            pulse_n = local if local < 128 else local-512
            if 0 <= pulse_n < 128:
                source.append(.1*math.cos(2*math.pi*500*pulse_n/16000)
                              *math.sin(math.pi*pulse_n/127)**2)
            else:
                source.append(0.)
        source += [0.]*768
        def delayed(n):
            return [source[j-n] if j >= n else 0. for j in range(33024)]
        expected = {'reference': [delayed(384)], 'array': [delayed(384), source],
                    'common_history': [delayed(384)], 'aligned_history': [delayed(768)],
                    'common_residual': [[0.]*33024], 'aligned_residual': [delayed(384)]}
        for name, value in expected.items():
            # Scalar/vector trigonometric argument rounding differs by up to
            # 2.19e-16 here, far below the 1/65536 PCM half-step bound.
            np.testing.assert_allclose(self.experiment['signals'][name], value, atol=3e-16, rtol=0)
        self.assertEqual(self.manifest['parameters']['physical_propagation_tail_samples'], 384)
        self.assertEqual(self.manifest['parameters']['maximum_history_padding_samples'], 768)
        self.assertEqual(self.manifest['parameters']['scoring_source_periods'], 19)
        self.assertTrue(np.all(self.experiment['signals']['reference'][0, 32256+384:] == 0))
        self.assertTrue(np.all(self.experiment['signals']['aligned_history'][0, 32256+768:] == 0))

    def test_single_regressor_identity_and_disjoint_support_in_float_and_pcm(self):
        p = self.manifest
        ref = self.experiment['signals']['reference'][0, 1536:30720]
        common = self.experiment['signals']['common_history'][0, 1536:30720]
        aligned = self.experiment['signals']['aligned_history'][0, 1536:30720]
        self.assertEqual(float(common@ref)/float(common@common), 1.)
        self.assertEqual(float(aligned@ref), 0.)
        self.assertFalse(np.any((aligned != 0) & (ref != 0)))
        for kind in ('float_regression', 'pcm_regression'):
            self.assertEqual(p[kind]['common']['coefficient'], 1.)
            self.assertEqual(p[kind]['aligned']['coefficient'], 0.)
            self.assertEqual(p[kind]['aligned']['cross'], 0.)
        self.assertEqual(set(self.raw['common_residual'][0]), {0})
        self.assertEqual(self.blobs[core.FILE_NAMES['reference']], self.blobs[core.FILE_NAMES['aligned_residual']])
        self.assertEqual(self.blobs[core.FILE_NAMES['reference']], self.blobs[core.FILE_NAMES['common_history']])
        # A stationary one-frequency control remains predictable after the
        # same corrected source gap; alignment alone is not target protection.
        tone = .1*np.cos(2*np.pi*500*np.arange(33024)/16000)
        delayed = np.r_[np.zeros(384), tone[:-384]]
        gain = float(tone[1536:30720]@delayed[1536:30720]) / float(delayed[1536:30720]@delayed[1536:30720])
        self.assertAlmostEqual(gain, 1., places=13)

    def test_analytic_finite_pulse_sum_and_actual_integer_denominators(self):
        pulse_energy = math.fsum((.1*math.cos(2*math.pi*n/32)*math.sin(math.pi*n/127)**2)**2
                                 for n in range(128))
        expected_power = 2*pulse_energy/1536
        self.assertAlmostEqual(expected_power, .0003100559925140138, places=18)
        reference = self.raw['reference'][0][1536:30720]
        ref_energy = sum(v*v for v in reference)
        self.assertEqual(ref_energy, 9715561194)
        denominator = 29184*32768**2
        for name, channels in self.raw.items():
            row = self.manifest['samples'][name]['pcm_integer_measurements']
            energies = [sum(v*v for v in x[1536:30720]) for x in channels]
            errors = [sum((v-r)**2 for v,r in zip(x[1536:30720],reference)) for x in channels]
            self.assertEqual(row['integer_squared_sum_E_per_channel'], energies)
            self.assertEqual(row['integer_denominator_D_per_channel'], denominator)
            self.assertEqual(row['integer_denominator_D_all_channels'], denominator*len(channels))
            self.assertEqual(row['mean_square_all_channels'], sum(energies)/(len(channels)*denominator))
            self.assertEqual(row['integer_reference_error_squared_sum_per_channel'], errors)
            self.assertEqual(row['integer_reference_squared_sum_per_channel'], ref_energy)
            self.assertEqual(row['reference_NMSE_per_channel'], [v/ref_energy for v in errors])
            self.assertEqual(len(channels), 2 if name == 'array' else 1)
        self.assertEqual(self.manifest['samples']['common_residual']['pcm_integer_measurements']['reference_NMSE_per_channel'], [1.])
        self.assertEqual(self.manifest['samples']['aligned_residual']['pcm_integer_measurements']['reference_NMSE_per_channel'], [0.])
        for path, digest in self.manifest['source_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((generator.ROOT/path).read_bytes()).hexdigest())
        self.assertEqual(len(generator.SOURCE_PATHS), 5)

    def test_all_fixed_parameters_and_nested_types_fail_before_synthesis_or_write(self):
        changes = []
        for key in generator.REQUIRED_PARAMETERS:
            params = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            params[key] = None
            changes.append((key, params))
        for key, value in [('source_samples', 32256.0), ('sample_rate_hz', True),
                           ('amplitude', 0), ('common_export_gain', True)]:
            params = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            params[key] = value
            changes.append((key+' type', params))
        params = copy.deepcopy(generator.REQUIRED_PARAMETERS)
        params['observation_delays_samples']['reference'] = True
        changes.append(('nested bool', params))
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary)/'absent'
            existing = Path(temporary)/'existing'
            self.write_fixture(existing)
            before = snapshot(existing)
            for label, params in changes:
                for output in (absent, existing):
                    with self.subTest(label=label), patch.object(generator, 'parameters', return_value=params), \
                            patch.object(generator, 'generate_experiment') as synth, self.assertRaises(ValueError):
                        generator.generate_assets(output)
                    synth.assert_not_called()
                self.assertFalse(absent.exists())
                self.assertEqual(snapshot(existing), before)

    def test_generation_and_full_check_remain_readonly(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'generated'
            self.assertEqual(generator.generate_assets(output), self.manifest)
            before = snapshot(output)
            self.assertEqual(generator.check_assets(output), self.manifest)
            self.assertEqual(snapshot(output), before)
            member = output/core.FILE_NAMES['array']
            data = bytearray(member.read_bytes())
            data[5000] ^= 1
            member.write_bytes(data)
            before = snapshot(output)
            with self.assertRaises(ValueError):
                generator.check_assets(output)
            self.assertEqual(snapshot(output), before)

    def test_measurement_support_and_malformed_synthesis_fail_without_writing(self):
        reference = self.experiment['signals']['reference']
        for x in (np.ones((1, 33023)), np.full((1, 33024), True),
                  np.full((1, 33024), np.inf), np.full((1, 33024), 1e308)):
            with self.assertRaises(ValueError):
                core.measure_signal(x, reference)
        with self.assertRaises(ValueError):
            core.measure_signal(reference, np.zeros_like(reference))
        malformed = copy.deepcopy(self.experiment)
        malformed['signals']['array'] = np.zeros((1, 33024))
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary)/'absent'
            with patch.object(generator, 'generate_experiment', return_value=malformed), self.assertRaises(ValueError):
                generator.generate_assets(absent)
            self.assertFalse(absent.exists())

    def test_actual_pcm_scoring_catches_forged_record_even_with_replay_patched(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'assets'
            self.write_fixture(output)
            changed = copy.deepcopy(self.manifest)
            changed['samples']['array']['pcm_integer_measurements']['integer_squared_sum_E_per_channel'][1] += 1
            (output/'MANIFEST.json').write_text(json.dumps(changed))
            before = snapshot(output)
            with patch.object(generator, 'prepare_assets', return_value=(self.blobs, changed)), self.assertRaises(ValueError):
                generator.check_assets(output)
            self.assertEqual(snapshot(output), before)

    def test_extra_members_linked_parent_hardlink_and_strict_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent/'assets'
            self.write_fixture(output)
            extra = output/'extra'
            extra.write_text('preserve')
            for operation in (generator.generate_assets, generator.check_assets):
                with patch.object(generator, 'generate_experiment') as synth, self.assertRaises(ValueError):
                    operation(output)
                synth.assert_not_called()
            extra.unlink()
            linked = parent/'linked'
            linked.symlink_to(output, target_is_directory=True)
            with self.assertRaises(ValueError):
                generator.generate_assets(linked/'new')
            member = output/core.FILE_NAMES['array']
            member.unlink()
            member.symlink_to(output/core.FILE_NAMES['reference'])
            with self.assertRaises(ValueError):
                generator.check_assets(output)
            member.unlink()
            member.hardlink_to(output/core.FILE_NAMES['reference'])
            with self.assertRaises(ValueError):
                generator.generate_assets(output)
            member.unlink()
            member.write_bytes(self.blobs[core.FILE_NAMES['array']])
            for document in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                (output/'MANIFEST.json').write_text(document)
                with self.assertRaises(ValueError):
                    generator.check_assets(output)


if __name__ == '__main__':
    unittest.main()
