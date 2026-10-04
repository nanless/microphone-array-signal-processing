"""Independent finite-waveform/integer checks and strict CSS asset contracts."""
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
from codes.chapters.ch08.core import css_audio as core
from codes.chapters.ch08.examples import generate_css_audio as demo


class CSSAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = core.generate_experiment()
        cls.blobs, cls.manifest = demo.prepare_assets()

    def fixture(self, out):
        out.mkdir()
        for name, blob in self.blobs.items():
            (out/name).write_bytes(blob)
        (out/'MANIFEST.json').write_text(json.dumps(self.manifest))

    def snapshot(self, out):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}

    def test_closed_form_crossfade_and_observed_overlap_fit(self):
        signal = self.experiment['signals']
        np.testing.assert_array_equal(self.experiment['overlap_application_gain'], [-.5, -.5])
        overlap = self.experiment['observed_overlap']
        # Independent unscaled dot formula uses only two observed overlaps.
        for p, c in zip(overlap['previous'], overlap['current']):
            self.assertEqual(float(c@p/(c@c)), -.5)
        for name, slope, after in (('naive', -3, -2), ('polarity', 1, 2), ('corrected', 0, 1)):
            for ch, f in enumerate((500, 1500)):
                # No stitch, LS, or measurement helper in the expected waveform.
                for n in (2400, 12800, 14000, 15999, 17000, 19199, 19200, 25000, 29599):
                    factor = 1 if n < 12800 else 1+slope*(n-12800)/6399 if n < 19200 else after
                    expected = .1*math.sin(2*math.pi*f*(n/16000))*factor
                    self.assertAlmostEqual(signal[name][ch, n], expected, delta=5e-13)
            self.assertTrue(np.all(signal[name][:, [0, -1]] == 0))
        for name, nmse in (('naive', 9), ('polarity', 1), ('corrected', 0)):
            actual = self.manifest['samples'][name]['float_measurements']['post_overlap']
            np.testing.assert_allclose(actual['reference_NMSE_per_channel'], [nmse]*2, atol=2e-14)
            np.testing.assert_allclose(self.manifest['samples'][name]['analytic']['post_overlap']['reference_NMSE_per_channel'],
                                       [nmse]*2, atol=2e-14)
        self.assertEqual(self.blobs['css_reference.wav'], self.blobs['css_corrected.wav'])

    def test_standard_library_pcm_two_windows_all_integer_numerators(self):
        pcm = {}
        for name, filename in core.FILE_NAMES.items():
            with wave.open(io.BytesIO(self.blobs[filename])) as reader:
                self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth()),
                                 (16000, 2, 32000, 2))
                values = struct.unpack('<64000h', reader.readframes(32000))
            pcm[name] = [values[0::2], values[1::2]]
        expected_error = {'primary': {'naive': [605638471357, 605638513911],
                                      'polarity': [67298151177, 67298221568]},
                          'post_overlap': {'naive': [502527654850]*2, 'polarity': [55841196450]*2}}
        for window, (start, stop) in core.SCORING_INTERVALS.items():
            for name in core.FILE_NAMES:
                record = self.manifest['samples'][name]['pcm_integer_measurements'][window]
                energies, errors, reference, cross = [], [], [], []
                for ch in range(2):
                    x, r = pcm[name][ch][start:stop], pcm['reference'][ch][start:stop]
                    energies.append(sum(v*v for v in x)); errors.append(sum((a-b)**2 for a,b in zip(x,r)))
                    reference.append(sum(v*v for v in r)); cross.append(sum(a*b for a,b in zip(x,r)))
                self.assertEqual(record['integer_squared_sum_E_per_channel'], energies)
                self.assertEqual(record['integer_reference_error_squared_sum_per_channel'], errors)
                self.assertEqual(record['integer_reference_squared_sum_per_channel'], reference)
                self.assertEqual(record['integer_output_reference_cross_sum_per_channel'], cross)
                self.assertEqual(record['integer_denominator_D_per_channel'], (stop-start)*32768**2)
                self.assertEqual(record['integer_squared_sum_E_all_channels'], sum(energies))
                self.assertEqual(record['integer_denominator_D_all_channels'], 2*(stop-start)*32768**2)
                if name in expected_error[window]:
                    self.assertEqual(errors, expected_error[window][name])
                else:
                    self.assertEqual(errors, [0,0])
                self.assertEqual(reference, [146027417700]*2 if window == 'primary' else [55834012650]*2)
                self.assertEqual(record['reference_NMSE_per_channel'], [v/r for v,r in zip(errors,reference)])
                self.assertEqual(self.manifest['files'][core.FILE_NAMES[name]]['sha256'],
                                 hashlib.sha256(self.blobs[core.FILE_NAMES[name]]).hexdigest())

    def test_real_generation_and_check_are_exact_and_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'assets'
            demo.generate_assets(out)
            before = self.snapshot(out)
            demo.check_assets(out)
            self.assertEqual(before, self.snapshot(out))
            self.assertEqual(set(before), demo.MEMBERS)

    def test_all_fixed_fields_and_limits_rejected_before_directory_creation(self):
        # Mutate every leaf, including nested types, not only a few convenient fields.
        def leaves(value, path=()):
            if isinstance(value, dict):
                for k,v in value.items(): yield from leaves(v,path+(k,))
            elif isinstance(value, list):
                for k,v in enumerate(value): yield from leaves(v,path+(k,))
            else: yield path,value
        for path, value in leaves(core.parameters()):
            bad = copy.deepcopy(core.parameters()); target=bad
            for key in path[:-1]: target=target[key]
            target[path[-1]] = (not value if type(value) is bool else True if type(value) in (int,float)
                                else value+' altered')
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'assets'
                with patch.object(demo, 'parameters', return_value=bad), self.assertRaises(ValueError):
                    demo.generate_assets(out)
                self.assertFalse(out.exists())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'assets'
            with patch.object(demo, 'LIMITS', 'false scope'), self.assertRaises(ValueError): demo.generate_assets(out)
            self.assertFalse(out.exists())

    def test_unsafe_members_parents_and_hardlinks_preflight_before_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'assets'; self.fixture(out)
            original=self.snapshot(out)
            (out/'extra').write_text('x')
            with patch.object(demo, 'prepare_assets', side_effect=AssertionError('must preflight')):
                with self.assertRaises(ValueError): demo.generate_assets(out)
            (out/'extra').unlink()
            (root/'linked').symlink_to(out,target_is_directory=True)
            with self.assertRaises(ValueError): demo.check_assets(root/'linked')
            with self.assertRaises(ValueError): demo.generate_assets(root/'linked'/'new')
            member=out/'css_naive.wav'; member.unlink(); member.symlink_to(out/'css_reference.wav')
            with self.assertRaises(ValueError): demo.generate_assets(out)
            member.unlink(); member.write_bytes(original[member.name][0])
            import os
            os.link(member,root/'linked.wav')
            with self.assertRaises(ValueError): demo.check_assets(out)

    def test_manifest_false_fields_strict_json_and_changed_pcm_fail_without_repair(self):
        edits=[lambda m: m['source_sha256'].update({demo.SOURCE_PATHS[0]:'0'*64}),
               lambda m: m.update(common_export_gain=True),
               lambda m: m['samples']['naive']['pcm_integer_measurements']['primary'].update(integer_denominator_D_per_channel=True),
               lambda m: m['samples']['naive']['float_measurements']['primary'].update(mean_square_all_channels=.5),
               lambda m: m['samples']['reference']['analytic']['primary'].update(channels=True)]
        for edit in edits:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'assets'; self.fixture(out); m=copy.deepcopy(self.manifest); edit(m)
                (out/'MANIFEST.json').write_text(json.dumps(m)); before=self.snapshot(out)
                with self.assertRaises(ValueError): demo.check_assets(out)
                self.assertEqual(before,self.snapshot(out))
        for text in ('{"schema_version":1,"schema_version":1}', '{"value":1e999}', '{"value":NaN}'):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'assets'; self.fixture(out); (out/'MANIFEST.json').write_text(text)
                with self.assertRaises(ValueError): demo.check_assets(out)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'assets'; self.fixture(out); path=out/'css_naive.wav'
            blob=bytearray(path.read_bytes()); blob[10000]^=1; path.write_bytes(blob); before=self.snapshot(out)
            with self.assertRaises(ValueError): demo.check_assets(out)
            self.assertEqual(before,self.snapshot(out))


if __name__ == '__main__': unittest.main()
