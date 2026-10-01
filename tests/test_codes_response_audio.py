"""Independent analytic, actual wave/struct and no-write asset checks."""
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave
import numpy as np
from codes.chapters.appendix_b.core import response_audio as core
from codes.chapters.appendix_b.examples import generate_response_audio as assets


def integers(data):
    with wave.open(io.BytesIO(data), 'rb') as reader:
        assert (reader.getnchannels(), reader.getframerate(), reader.getnframes(), reader.getsampwidth()) == (1, 16000, 32002, 2)
        return struct.unpack('<32002h', reader.readframes(32002))


class ResponseAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.buffers, cls.metadata = assets.expected_assets()

    def copy_assets(self, root):
        for name, data in self.buffers.items():
            (root/name).write_bytes(data)

    def test_response_and_power_use_independent_closed_trigonometry(self):
        fixture = core.build_fixture()
        report = core.analytic_results()
        root2 = math.sqrt(2)
        expected = {
            'a': [(root2/4, -(root2+2)/4), (-.5, -.5)],
            'b': [(root2/4, (2-root2)/4), (.5, -.5)]}
        for label, responses in expected.items():
            row = report['reflections']['response_reflection_'+label+'.wav']
            np.testing.assert_allclose(row['complex_response_real_imag'], responses, atol=3e-16)
            power = .005*sum(real**2+imag**2 for real, imag in responses)
            self.assertAlmostEqual(row['output_power'], power, places=15)
            self.assertAlmostEqual(report['candidates']['response_full_'+label+'.wav']['nmse'], power/.01, places=14)
        source = fixture['signals']['response_source']
        self.assertEqual((source[0], source[31999], source[32000], source[32001]), (0., 0., 0., 0.))
        for label, sign in (('a', 1), ('b', -1)):
            reflection = fixture['signals']['response_reflection_'+label]
            # Another direct, causal sample route; no np.convolve in this oracle.
            expected = [(.5*source[n-1] if n >= 1 else 0)+ (sign*.5*source[n-2] if n >= 2 else 0) for n in range(32002)]
            np.testing.assert_allclose(reflection, expected, atol=0, rtol=0)
            np.testing.assert_allclose(fixture['signals']['response_full_'+label], source+expected, atol=0, rtol=0)

    def test_actual_pcm_integer_scores_and_no_compensating_fit(self):
        lo, hi = 1600, 30400
        ref = integers(self.buffers['response_source.wav'])[lo:hi]
        d = sum(v*v for v in ref)
        self.assertEqual(d, 309262788000)
        expected_errors = {'response_full_a.wav': 209299622400, 'response_full_b.wav': 99939578400}
        for filename, expected in expected_errors.items():
            output = integers(self.buffers[filename])[lo:hi]
            e = sum((x-y)**2 for x, y in zip(output, ref))
            self.assertEqual(e, expected)
            row = self.metadata['pcm_analysis']['candidates'][filename]
            self.assertEqual(row['integer_error_squared_sum'], e)
            self.assertEqual(row['integer_reference_squared_sum'], d)
            self.assertEqual(row['mse'], e/(28800*32768**2))
            self.assertEqual(row['nmse'], e/d)
            self.assertFalse(row['gain_fitting']); self.assertFalse(row['delay_fitting'])
        for filename in expected_errors:
            actual = self.metadata['pcm_analysis']['signals'][filename]['complex_tone_amplitudes_real_imag']
            closed = self.metadata['analytic']['candidates'][filename]['complex_response_real_imag']
            np.testing.assert_allclose(actual, np.array(closed)*.1, atol=3e-5)

    def test_real_generation_and_check_are_strictly_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'audio'
            assets.generate(directory)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            actual = assets.check_assets(directory)
            assets.generate(directory, check=True)
            self.assertEqual(actual['pcm_analysis']['integer_reference_squared_sum'], 309262788000)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()})

    def test_missing_extra_symlink_and_parent_traversal_do_not_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); directory = root/'audio'; directory.mkdir(); self.copy_assets(directory)
            (directory/'extra').mkdir()
            before = {p.name for p in directory.iterdir()}
            for operation in (assets.generate, assets.check_assets):
                with self.assertRaises(ValueError): operation(directory)
            self.assertEqual(before, {p.name for p in directory.iterdir()})
            (directory/'extra').rmdir()
            outside = root/'outside.wav'; outside.write_bytes(b'outside-preserved')
            member = directory/'response_full_a.wav'; member.unlink(); member.symlink_to(outside)
            with self.assertRaises(ValueError): assets.generate(directory)
            self.assertEqual(outside.read_bytes(), b'outside-preserved')
            linked = root/'linked'; linked.symlink_to(directory, target_is_directory=True)
            with self.assertRaises(ValueError): assets.generate(linked/'..'/'new')
            self.assertFalse((root/'new').exists())

    def test_metadata_type_nonfinite_duplicate_and_source_digest_reject(self):
        bads = [self.buffers['MANIFEST.json'].replace(b'"schema_version": 1', b'"schema_version": true'),
                self.buffers['MANIFEST.json'].replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'),
                self.buffers['MANIFEST.json'].replace(b'"common_export_gain": 1.0', b'"common_export_gain": NaN'),
                self.buffers['MANIFEST.json'].replace(b'"common_export_gain": 1.0', b'"common_export_gain": 1e999')]
        changed = json.loads(self.buffers['MANIFEST.json'])
        changed['source_sha256'][assets.SOURCE_PATHS[0]] = '0'*64
        bads.append(json.dumps(changed).encode())
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); self.copy_assets(directory)
            for data in bads:
                (directory/'MANIFEST.json').write_bytes(data)
                before = (directory/'MANIFEST.json').read_bytes()
                with self.assertRaises(ValueError): assets.check_assets(directory)
                self.assertEqual(before, (directory/'MANIFEST.json').read_bytes())
            with self.assertRaises(ValueError): assets.generate(directory, check='yes')

    def test_format_valid_zero_reference_and_modified_pcm_reject(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); self.copy_assets(directory)
            target = directory/'response_source.wav'
            blob = bytearray(target.read_bytes()); blob[44:] = b'\0'*(len(blob)-44); target.write_bytes(blob)
            metadata = json.loads(self.buffers['MANIFEST.json'])
            metadata['files'][target.name]['sha256'] = hashlib.sha256(blob).hexdigest()
            (directory/'MANIFEST.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'reference power'):
                assets.check_assets(directory)
            self.copy_assets(directory)
            target = directory/'response_full_b.wav'; blob = bytearray(target.read_bytes()); blob[-2] ^= 1; target.write_bytes(blob)
            before = target.read_bytes()
            with self.assertRaises(ValueError): assets.check_assets(directory)
            self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
