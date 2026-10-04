"""Independent convolution, actual wave/struct integers and writer boundaries."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np

from codes.chapters.ch06.core import reference_audio as core
from codes.chapters.ch06.examples import generate_reference_audio as generator


def integers(blob):
    with wave.open(io.BytesIO(blob), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getnframes(),
                stream.getsampwidth(), stream.getcomptype()) != (16000, 1, 33600, 2, 'NONE'):
            raise AssertionError('actual WAV format')
        return struct.unpack('<33600h', stream.readframes(33600))


class ReferenceAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals = core.generate_signals()
        with patch.object(generator, 'generate_signals', return_value=cls.signals):
            cls.blobs, cls.manifest = generator.prepare_assets()
        cls.raw = {k: integers(cls.blobs[v]) for k, v in core.FILE_NAMES.items()}

    def test_independent_full_causal_convolution_and_gain_order(self):
        n = np.arange(32000)
        source = .08*np.cos(2*np.pi*500*n/16000)+.06*np.sin(2*np.pi*730*n/16000)
        envelope = np.ones(32000)
        ramp = np.sin(np.linspace(0, np.pi/2, 320))**2
        envelope[:320], envelope[-320:] = ramp, ramp[::-1]
        early = np.pad(source*envelope, (0, 1600))
        gain = np.where(np.arange(33600) < 16000, 1., 2.)
        path = np.zeros(1601)
        path[0], path[1600] = .7, .35
        true = np.convolve((early*gain)[:32000], path)
        early_prediction = np.convolve(early[:32000], path)
        expected = {'early': early, 'late': early*gain, 'echo': true,
                    'early_residual': true-early_prediction,
                    'wrong_gain_residual': true-gain*early_prediction,
                    'late_residual': np.zeros(33600)}
        for name, signal in expected.items():
            np.testing.assert_allclose(self.signals[name][0], signal, atol=4e-13, rtol=0)
        self.assertGreater(np.max(abs(expected['wrong_gain_residual'][16000:17600])), .04)
        self.assertGreater(np.max(abs(true[32000:])), .09)
        self.assertLess(np.max(abs(self.signals['late_residual'])), 1e-15)
        self.assertEqual(set(self.raw['late_residual']), {0})

    def test_closed_form_constant_envelope_windows(self):
        factors = {'early': [1, 1], 'late': [2, 2], 'echo': [1.75, 2.1],
                   'early_residual': [.7, 1.05], 'wrong_gain_residual': [-.35, 0],
                   'late_residual': [0, 0]}
        for name, pair in factors.items():
            for label, factor in zip(('step', 'stable'), pair):
                start, stop = core.WINDOWS[label]
                actual = float(np.mean(self.signals[name][0, start:stop]**2))
                self.assertAlmostEqual(actual, .005*factor**2, places=15)
        self.assertEqual(core.WINDOWS['stable'], (19200, 28800))

    def test_actual_pcm_integer_energy_sources_and_pure_tail_label(self):
        self.assertEqual(len(generator.MEMBERS), 7)
        for name, raw in self.raw.items():
            self.assertLess(max(abs(v) for v in raw), 32768)
            for label, (start, stop) in core.WINDOWS.items():
                energy = sum(v*v for v in raw[start:stop])
                denominator = (stop-start)*32768**2
                row = self.manifest['samples'][name]['pcm_integer_measurements'][label]
                self.assertEqual(row['integer_squared_sum_E_per_channel'], [energy])
                self.assertEqual(row['integer_squared_sum_E_all_channels'], energy)
                self.assertEqual(row['integer_denominator_D_all_channels'], denominator)
                self.assertEqual(row['mean_square_all_channels'], energy/denominator)
        for path, digest in self.manifest['source_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((generator.ROOT/path).read_bytes()).hexdigest())
        report = self.manifest['pcm_tail_activity']
        self.assertEqual(report['tail_labels'], ['near_only']*10)
        self.assertEqual(report['tail_near_end_truth_present'], [False]*10)
        self.assertEqual(report['tail_ncc_defined'], [False]*10)
        self.assertEqual(report['available_after_samples'][-1], 33600)

    def test_every_fixed_parameter_rejected_before_synthesis_or_writing(self):
        mutants = []
        for key, value in generator.REQUIRED_PARAMETERS.items():
            changed = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            changed[key] = None
            mutants.append((key, changed))
        for key in ('sample_rate_hz', 'common_export_gain', 'nlms_step_size', 'ncc_frame_samples'):
            changed = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            changed[key] = True
            mutants.append((key+' bool', changed))
        changed = copy.deepcopy(generator.REQUIRED_PARAMETERS)
        changed['frequencies_hz'][0] = 500
        mutants.append(('nested exact type', changed))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'not_created'
            for label, changed in mutants:
                with self.subTest(label=label), patch.object(generator, 'parameters', return_value=changed), patch.object(generator, 'generate_signals') as synth:
                    with self.assertRaises(ValueError):
                        generator.generate_assets(output)
                    synth.assert_not_called()
                    self.assertFalse(output.exists())

    def make_assets(self, output):
        output.mkdir()
        for filename, blob in self.blobs.items():
            (output/filename).write_bytes(blob)
        (output/'MANIFEST.json').write_text(json.dumps(self.manifest, allow_nan=False))

    def test_check_actual_pcm_and_manifest_replay_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'audio'
            self.make_assets(output)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            with patch.object(generator, 'generate_signals', return_value=self.signals):
                generator.check_assets(output)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})
            pcm = output/core.FILE_NAMES['echo']
            changed = bytearray(pcm.read_bytes())
            changed[-20] ^= 1
            pcm.write_bytes(changed)
            with patch.object(generator, 'generate_signals', return_value=self.signals), self.assertRaises(ValueError):
                generator.check_assets(output)

    def test_extra_linked_members_parent_and_strict_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/'audio'
            self.make_assets(output)
            extra = output/'extra'
            extra.write_text('preserve')
            for operation in (generator.check_assets, generator.generate_assets):
                with patch.object(generator, 'generate_signals') as synth, self.assertRaises(ValueError):
                    operation(output)
                synth.assert_not_called()
            extra.unlink()
            member = output/core.FILE_NAMES['early']
            member.unlink()
            member.symlink_to(output/core.FILE_NAMES['late'])
            with self.assertRaises(ValueError):
                generator.check_assets(output)
            member.unlink()
            member.write_bytes(self.blobs[core.FILE_NAMES['early']])
            linked = root/'linked'
            linked.symlink_to(output, target_is_directory=True)
            with self.assertRaises(ValueError):
                generator.generate_assets(linked/'nested')
            for document in ('{"a":1,"a":2}', '{"a":NaN}'):
                (output/'MANIFEST.json').write_text(document)
                with self.assertRaises(ValueError):
                    generator.check_assets(output)


if __name__ == '__main__':
    unittest.main()
