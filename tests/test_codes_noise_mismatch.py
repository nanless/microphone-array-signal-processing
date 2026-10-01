"""Independent PCM, window-power and filesystem checks for E10-33."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
from codes.chapters.ch10.core.noise_mismatch import build_fixture, analyze_fixture, analyze_pcm
from codes.chapters.ch10.examples import generate_noise_mismatch as generator


def integer_codes(data):
    with wave.open(io.BytesIO(data), 'rb') as reader:
        assert reader.getparams()[:4] == (1, 2, 16000, 32000)
        return struct.unpack('<32000h', reader.readframes(32000))


class NoiseMismatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = build_fixture()
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)/'actual'
        cls.report = generator.generate(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_same_input_fixed_gain_and_complete_window_variance(self):
        f = self.fixture
        np.testing.assert_array_equal(f['signals']['noise_mixture'],
                                      f['signals']['noise_reference']+f['signals']['noise_component'])
        self.assertEqual(f['parameters']['common_export_gain'], 1.)
        self.assertEqual(f['parameters']['pure_noise_frame_indices'], list(range(2, 49)))
        self.assertEqual(f['parameters']['polluted_frame_indices'], list(range(65, 99)))
        self.assertEqual(f['signals']['noise_reference'][0, 6400], 0.)
        self.assertEqual(f['signals']['noise_reference'][0, -1], 0.)
        # Periodic N512 Hann squared sum is 3N/8=192. The sample at its
        # midpoint belongs to the new regime: half sums are 95.5 and 96.5.
        power = f['spectral']['known_variance_power']
        self.assertAlmostEqual(power[100], .03**2*192, places=14)
        self.assertAlmostEqual(power[200], .12**2*192, places=13)
        self.assertAlmostEqual(power[150], .03**2*95.5+.12**2*96.5, places=13)
        # Independent direct DFT of a window known to lie in the pure prefix.
        samples = f['signals']['noise_component'][0, 0:512]
        w = [.5-.5*math.cos(2*math.pi*n/512) for n in range(512)]
        z = sum(samples[n]*w[n]*complex(math.cos(-2*math.pi*16*n/512),
                                        math.sin(-2*math.pi*16*n/512)) for n in range(512))
        self.assertAlmostEqual(abs(z-self.fixture['spectral']['mixture'][16, 2]), 0., places=12)

    def test_float_components_and_mse_from_independent_sample_differences(self):
        report = analyze_fixture(self.fixture)
        for window, (lo, hi) in {'before_step': (9600, 16000), 'after_step': (22400, 28800)}.items():
            reference = self.fixture['signals']['noise_reference'][0, lo:hi]
            expected_power = math.fsum(float(x)**2 for x in reference)/(hi-lo)
            self.assertAlmostEqual(expected_power, .0104, places=13)
            for name in ['noise_fixed', 'noise_polluted', 'noise_known_variance']:
                actual = self.fixture['signals'][name][0, lo:hi]
                expected = math.fsum((float(a)-float(b))**2 for a,b in zip(actual, reference))/(hi-lo)
                row = report['score_windows'][window]['scores'][name+'.wav']
                self.assertAlmostEqual(row['mse'], expected, places=16)
                self.assertAlmostEqual(row['component_sum_mse'], expected, places=16)
                self.assertNotEqual(row['twice_cross_term'], 0.)
        bad = copy.deepcopy(self.fixture)
        bad['components']['noise_fixed']['noise'][0, 10000] = np.nan
        with self.assertRaises(ValueError):
            analyze_fixture(bad)
        bad = copy.deepcopy(self.fixture)
        bad['components']['noise_fixed']['noise'].fill(0)
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            analyze_fixture(bad)

    def test_actual_pcm_integer_sums_and_no_quantization_noise_decomposition(self):
        codes = {name: integer_codes((self.directory/name).read_bytes()) for name in generator.MEMBERS if name.endswith('.wav')}
        for window, (lo, hi) in {'before_step': (9600, 16000), 'after_step': (22400, 28800)}.items():
            reference = codes['noise_reference.wav'][lo:hi]
            expected_D = sum(v*v for v in reference)
            self.assertEqual(expected_D, 71466999200)
            record = self.report['pcm_analysis']['score_windows'][window]
            self.assertEqual(record['reference_squared_sum_pcm_integer'], expected_D)
            self.assertEqual(record['samples'], 6400)
            for name, row in record['scores'].items():
                expected_E = sum((x-r)**2 for x,r in zip(codes[name][lo:hi], reference))
                self.assertEqual(row['error_squared_sum_pcm_integer'], expected_E)
                self.assertEqual(row['nmse'], expected_E/expected_D)
                self.assertEqual(row['mse'], expected_E/(6400*32768**2))
                self.assertNotIn('residual_noise_mse', row)
        self.assertGreater(self.report['pcm_analysis']['score_windows']['after_step']['scores']['noise_known_variance.wav']['nmse'], .1)

    def test_check_is_readonly_and_cli_checks_real_assets(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        self.assertEqual(generator.check_assets(self.directory), self.report)
        self.assertEqual(generator.generate(self.directory, check=True), self.report)
        completed = subprocess.run([sys.executable, '-B', '-m',
                                    'codes.chapters.ch10.examples.generate_noise_mismatch',
                                    '--output', str(self.directory), '--check'], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        self.assertEqual(before, after)

    def test_strict_manifest_and_actual_pcm_fail_without_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'copy'
            mutations = [lambda r: r.update(schema_version=True),
                         lambda r: r.update(schema_version=999),
                         lambda r: r['source_sha256'].update({generator.SOURCE_PATHS[0]: '0'*64}),
                         lambda r: r['pcm_analysis']['score_windows']['before_step'].update(reference_squared_sum_pcm_integer=True)]
            for mutation in mutations:
                shutil.copytree(self.directory, directory)
                manifest = copy.deepcopy(self.report); mutation(manifest)
                (directory/'MANIFEST.json').write_text(json.dumps(manifest))
                before = {p.name: p.read_bytes() for p in directory.iterdir()}
                with self.assertRaises(ValueError): generator.check_assets(directory)
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
                shutil.rmtree(directory)
            for text in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}',
                         '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}', '[]'):
                shutil.copytree(self.directory, directory)
                (directory/'MANIFEST.json').write_text(text)
                with self.assertRaises(ValueError): generator.check_assets(directory)
                shutil.rmtree(directory)
            shutil.copytree(self.directory, directory)
            target = directory/'noise_fixed.wav'; original=target.read_bytes()
            changed=bytearray(original);changed[44+2*10000] ^= 1; target.write_bytes(changed)
            manifest=copy.deepcopy(self.report)
            manifest['files']['noise_fixed.wav']['sha256']=hashlib.sha256(changed).hexdigest()
            (directory/'MANIFEST.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): generator.check_assets(directory)
            self.assertEqual(target.read_bytes(), changed)

    def test_format_zero_reference_and_member_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'copy';shutil.copytree(self.directory,directory)
            b=io.BytesIO()
            with wave.open(b,'wb') as out:
                out.setparams((1,2,16000,32000,'NONE','not compressed'));out.writeframes(bytes(64000))
            (directory/'noise_reference.wav').write_bytes(b.getvalue())
            manifest=copy.deepcopy(self.report)
            manifest['files']['noise_reference.wav']['sha256']=hashlib.sha256(b.getvalue()).hexdigest()
            (directory/'MANIFEST.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'PCM reference'): generator.check_assets(directory)
            b=io.BytesIO()
            with wave.open(b,'wb') as out:
                out.setparams((2,2,16000,32000,'NONE','not compressed'));out.writeframes(bytes(128000))
            (directory/'noise_reference.wav').write_bytes(b.getvalue())
            with self.assertRaisesRegex(ValueError, 'WAV must'): generator.check_assets(directory)
            shutil.rmtree(directory);shutil.copytree(self.directory,directory)
            (directory/'extra').mkdir()
            with self.assertRaises(ValueError): generator.generate(directory)
            (directory/'extra').rmdir(); (directory/'noise_fixed.wav').unlink()
            with self.assertRaises(ValueError): generator.generate(directory)
            (directory/'noise_fixed.wav').write_bytes((self.directory/'noise_fixed.wav').read_bytes())
            external=Path(temp)/'external';external.write_bytes(b'do not touch')
            (directory/'noise_fixed.wav').unlink(); (directory/'noise_fixed.wav').symlink_to(external)
            with self.assertRaises(ValueError): generator.generate(directory)
            self.assertEqual(external.read_bytes(),b'do not touch')
            linked=Path(temp)/'linked';linked.symlink_to(directory, target_is_directory=True)
            with self.assertRaises(ValueError): generator.generate(linked/'child')
            with self.assertRaises(ValueError): generator.generate(linked/'..'/'escape')
            self.assertFalse((Path(temp)/'escape').exists())
            with self.assertRaises(ValueError): generator.generate(Path(temp)/'missing',check='yes')
            self.assertFalse((Path(temp)/'missing').exists())

    def test_missing_directory_does_not_get_created_by_check(self):
        with tempfile.TemporaryDirectory() as temp:
            missing=Path(temp)/'uncreated'/'nested'
            with self.assertRaises(ValueError): generator.check_assets(missing)
            self.assertFalse(missing.parent.exists())


if __name__ == '__main__':
    unittest.main()
