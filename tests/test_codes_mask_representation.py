"""Independent tone algebra, real FFT/PCM and safe asset checks for E08-28."""
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

from codes.chapters.ch08.core import mask_representation as core
from codes.chapters.ch08.examples import mask_representation_demo as demo


class MaskRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.arrays = core.run_experiment()
        cls.manifest, cls.contents = demo.prepare_assets()

    def _fixture(self, directory):
        for name, blob in self.contents.items():
            (directory/name).write_bytes(blob)

    def _manifest(self, directory):
        return json.loads((directory/'MANIFEST.json').read_text())

    def _save(self, directory, manifest):
        (directory/'MANIFEST.json').write_text(json.dumps(manifest, allow_nan=True))

    def test_analytic_tone_responses_and_phase_error_not_amplitude_only(self):
        start, stop = core.SCORING_INTERVAL
        t = np.arange(start, stop)/16000
        c0, c1, s1 = np.cos(2*np.pi*500*t), np.cos(2*np.pi*1250*t), np.sin(2*np.pi*1250*t)
        # Closed-form real sinusoids do not use an FFT or model helper.
        expected = {'target': .1*c0-.1*s1, 'other': -.08*c0+.1*c1,
                    'mixture': .02*c0+.1*c1-.1*s1,
                    'bounded_real': .02*c0+.05*c1-.05*s1,
                    'unbounded_real': .1*c0+.05*c1-.05*s1,
                    'complex_oracle': .1*c0-.1*s1}
        mse = {'target': 0., 'other': .0262, 'mixture': .0082,
               'bounded_real': .0057, 'unbounded_real': .0025, 'complex_oracle': 0.}
        for key in expected:
            np.testing.assert_allclose(self.arrays[key][0, start:stop], expected[key], atol=3e-13, rtol=0)
            self.assertAlmostEqual(self.report['float_measurements'][key]['total_reference_mse'], mse[key], places=13)
            self.assertEqual(self.arrays[key].shape, (1, 32000))
            self.assertEqual(self.arrays[key][0, 0], 0.)
            self.assertEqual(self.arrays[key][0, -1], 0.)
        self.assertLess(self.report['float_measurements']['complex_oracle']['total_reference_mse'], 1e-24)
        self.assertGreater(self.report['float_measurements']['complex_oracle']['total_reference_mse'], 0.)
        p = self.report['float_measurements']['unbounded_real']['phase_ls']['phasors_real_imag']
        np.testing.assert_allclose(p, [[.1, 0], [.05, .05]], atol=2e-14)
        self.assertFalse(self.report['float_measurements']['unbounded_real']['phase_ls']['gain_or_time_compensation'])

    def test_actual_wave_struct_integer_scores_and_independent_cos_sin_projection(self):
        pcm = {}
        for key, filename in core.FILE_NAMES.items():
            with wave.open(io.BytesIO(self.contents[filename])) as reader:
                self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth()),
                                 (16000, 1, 32000, 2))
                payload = reader.readframes(32000)
            pcm[key] = struct.unpack('<32000h', payload)
        start, stop = 2400, 29600
        reference = pcm['target'][start:stop]
        denominator = sum(v*v for v in reference)
        self.assertEqual(denominator, 292062085900)
        expected_sse = {'mixture': 239495337850, 'bounded_real': 166479798100,
                        'unbounded_real': 73012418550, 'complex_oracle': 0}
        for key, sse in expected_sse.items():
            actual = sum((a-b)**2 for a, b in zip(pcm[key][start:stop], reference))
            self.assertEqual(actual, sse)
            record = self.manifest['samples'][key]['pcm_measurements']
            self.assertEqual(record['integer_error_squared_sum'], actual)
            self.assertEqual(record['integer_reference_squared_sum'], denominator)
            self.assertEqual(record['integer_mse_denominator'], 27200*32768**2)
            self.assertEqual(record['total_reference_mse'], actual/(27200*32768**2))
            self.assertEqual(record['relative_squared_reference_error'], actual/denominator)
            # Orthogonal integer-cycle projections independently use dot sums,
            # whereas the measurement uses joint least squares.
            y = np.array(pcm[key][start:stop])/32768
            t = np.arange(start, stop)/16000
            coefficients = [2*np.sum(y*fn(2*np.pi*f*t))/27200
                            for f in (500, 1250) for fn in (np.cos, np.sin)]
            np.testing.assert_allclose(record['phase_ls']['coefficients'], coefficients, atol=4e-15, rtol=0)
        self.assertEqual(pcm['complex_oracle'], pcm['target'])

    def test_real_generated_directory_and_check_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'assets'
            # Actual generation, not a mocked prepare/replay.
            demo.generate_assets(out)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
            demo.check_assets(out, replay=False)
            demo.generate_assets(out, check=True)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()})

    def test_actual_pcm_score_tamper_rejected_even_without_source_replay(self):
        for field, value in (('integer_error_squared_sum', 1), ('integer_mse_denominator', 27200),
                             ('total_reference_mse', float('nan')), ('relative_squared_reference_error', True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self._fixture(out)
                manifest = self._manifest(out)
                manifest['samples']['bounded_real']['pcm_measurements'][field] = value
                self._save(out, manifest)
                before = (out/'MANIFEST.json').read_bytes()
                with self.assertRaises(ValueError):
                    demo.check_assets(out, replay=False)
                self.assertEqual((out/'MANIFEST.json').read_bytes(), before)

    def test_model_float_source_sha_and_member_mismatches_rejected(self):
        edits = [lambda m: m['source_sha256'].update({'unexpected.py': '0'*64}),
                 lambda m: m['source_sha256'].update({demo.SOURCE_PATHS[0]: '0'*64}),
                 lambda m: m.update({'common_export_gain': True}),
                 lambda m: m['samples']['mixture']['float_measurements'].update({'total_reference_mse': .1})]
        for edit in edits:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self._fixture(out)
                manifest = self._manifest(out)
                edit(manifest)
                self._save(out, manifest)
                with self.assertRaises(ValueError):
                    demo.check_assets(out)
        for filename in ('EXTRA.txt', 'mask_target.wav'):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self._fixture(out)
                if filename.startswith('EXTRA'):
                    (out/filename).write_text('extra')
                else:
                    (out/filename).unlink()
                with self.assertRaises(ValueError):
                    demo.check_assets(out)

    def test_valid_format_zero_reference_and_wrong_format_updated_sha_fail_without_repair(self):
        for rate in (16000, 8000):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self._fixture(out)
                stream = io.BytesIO()
                with wave.open(stream, 'wb') as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(rate)
                    writer.writeframes(bytes(64000))
                blob = stream.getvalue()
                (out/'mask_target.wav').write_bytes(blob)
                manifest = self._manifest(out)
                manifest['files']['mask_target.wav']['sha256'] = hashlib.sha256(blob).hexdigest()
                self._save(out, manifest)
                before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
                with self.assertRaisesRegex(ValueError, 'reference power|format mismatch'):
                    demo.check_assets(out, replay=False)
                self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()})

    def test_changed_pcm_payload_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._fixture(out)
            path = out/'mask_mixture.wav'
            blob = bytearray(path.read_bytes())
            blob[44+2*2500] ^= 1
            path.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                demo.check_assets(out)

    def test_symlink_extra_and_wrong_directory_guard_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp)/'assets'
            out.mkdir()
            outside = root/'outside'
            outside.write_bytes(b'unchanged')
            (out/'mask_target.wav').symlink_to(outside)
            with patch.object(demo, 'prepare_assets', side_effect=AssertionError('no model on unsafe destination')):
                with self.assertRaises(ValueError):
                    demo.generate_assets(out)
            self.assertEqual(outside.read_bytes(), b'unchanged')
            (out/'mask_target.wav').unlink()
            (out/'extra').write_bytes(b'extra')
            with self.assertRaises(ValueError):
                demo.generate_assets(out)
            link = root/'linked'
            link.symlink_to(out, target_is_directory=True)
            with self.assertRaises(ValueError):
                demo.generate_assets(link)
            with self.assertRaises(ValueError):
                demo.generate_assets(outside)

    def test_measurement_public_type_shape_and_pcm_range(self):
        reference = self.arrays['target']
        for bad in (reference.astype(complex), reference.astype(str), np.ones((1, 32000), dtype=bool),
                    reference[:, :-1], np.full((1, 32000), np.nan)):
            with self.assertRaises(ValueError):
                core.measure_signal(bad, reference)
        for value in (1., -1.-1/32768, .5/32768):
            with self.assertRaises(ValueError):
                core.measure_signal(np.full((1, 32000), value), reference, pcm=True)
        with self.assertRaises(ValueError):
            core.measure_signal(reference, np.zeros_like(reference))
        with self.assertRaises(ValueError):
            core.measure_signal(reference, reference, pcm='yes')


if __name__ == '__main__':
    unittest.main()
