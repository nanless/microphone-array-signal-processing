"""Independent trigonometric/PCM oracles and strict read-only asset contracts."""
import cmath
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import wave
import numpy as np

from codes.chapters.ch05.core import derivative_audio as core
from codes.chapters.ch05.examples import generate_derivative_audio as generator
from codes.chapters.ch05.chapter05_experiments import gsc_audio_results


class DerivativeAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals = core.generate_signals()
        cls.blobs, cls.manifest = generator.prepare_assets()

    def write_fixture(self, directory):
        output = Path(directory)
        for name, blob in self.blobs.items():
            (output/name).write_bytes(blob)
        (output/'MANIFEST.json').write_text(json.dumps(self.manifest, allow_nan=False))
        return output

    def test_geometry_delays_full_tail_and_weights_are_independent_hand_values(self):
        p = self.manifest['parameters']
        self.assertEqual(p['seed'], 20260522)
        np.testing.assert_allclose(p['single_weights'], [4/7, 2/7, 1/7], atol=1e-15)
        np.testing.assert_allclose(p['constrained_weights'], [4/13, 5/13, 4/13], atol=1e-15)
        delta = .1/(2*math.pi*1000)*16000
        expected_angle = math.degrees(math.asin(.1*343/(2*math.pi*1000*.04)))
        self.assertAlmostEqual(p['azimuth_deg'], expected_angle)
        clean = core.generate_components()['clean_array']
        # At several arbitrary absolute samples, independently evaluate source
        # envelope and real propagation (not the shared steering helper).
        for m, offset in enumerate((delta, 0., -delta)):
            for n in (0, 1, 4, 322, 4800, 31999, 32000, 32001):
                t = (n-1-offset)/16000
                envelope = min(max(0., min(1., t/.02)), max(0., min(1., (2-t)/.02)))
                expected = .08*envelope*(math.cos(2*math.pi*1000*t)+math.cos(2*math.pi*3000*t))
                self.assertAlmostEqual(clean[m, n], expected, places=12)
        self.assertNotEqual(clean[0, -1], 0)
        self.assertEqual(clean[-1, -1], 0)
        self.assertGreaterEqual(32002, math.ceil(32000+1+delta))
        self.assertEqual(self.signals['array'].shape, (3, 32002))

    def test_clean_responses_use_another_real_projection_route(self):
        truth = core.generate_components()
        start, stop = 2400, 29600
        time = np.arange(start, stop)/16000
        for method, weights in [('single', [4/7, 2/7, 1/7]), ('constrained', [4/13, 5/13, 4/13])]:
            y = np.asarray(weights)@truth['clean_array']
            for f, phi in ((1000, .1), (3000, .3)):
                # Integer cycles make these real projection columns orthogonal;
                # this expectation does not use the production LS fit.
                z = 2/(stop-start)*(np.dot(y[start:stop], np.cos(2*np.pi*f*time))
                                    -1j*np.dot(y[start:stop], np.sin(2*np.pi*f*time)))
                reference = .08*cmath.exp(-2j*math.pi*f/16000)
                expected = ((2+5*math.cos(phi)-3j*math.sin(phi))/7 if method == 'single' else
                            (5+8*math.cos(phi))/13)
                # Continuous phase evaluations reach ~35000 radians; the
                # independent evaluation orders agree within 1e-12 absolute.
                self.assertLess(abs(z/reference-expected), 1e-12)
            report = self.manifest['float_decomposition'][method]
            self.assertAlmostEqual(report['total_reference_mean_square'], report['decomposition_sum'], places=15)
        theory = self.manifest['parameters']['expected_population_steady_window']
        self.assertAlmostEqual(theory['single']['white_noise_mean_square'], .02**2*4/7)
        self.assertAlmostEqual(theory['constrained']['white_noise_mean_square'], .02**2*10/13)
        self.assertAlmostEqual(theory['single']['steady_clean_distortion_mean_square'], 6.048553938454008e-5)
        self.assertAlmostEqual(theory['constrained']['steady_clean_distortion_mean_square'], 2.4476478932936304e-6)
        self.assertLess(theory['single']['total_reference_mean_square'], theory['constrained']['total_reference_mean_square'])

    def test_actual_pcm_format_and_integer_error_sums_do_not_use_shared_decoder(self):
        decoded = {}
        for key, filename in core.FILE_NAMES.items():
            blob = self.blobs[filename]
            with wave.open(io.BytesIO(blob), 'rb') as source:
                channels = 3 if key == 'array' else 1
                self.assertEqual((source.getframerate(), source.getnchannels(), source.getnframes(), source.getsampwidth(), source.getcomptype()),
                                 (16000, channels, 32002, 2, 'NONE'))
                decoded[key] = np.frombuffer(source.readframes(32002), '<i2').reshape(-1, channels).T.astype(np.int64)
            self.assertEqual(hashlib.sha256(blob).hexdigest(), self.manifest['files'][filename]['sha256'])
            self.assertLessEqual(self.manifest['samples'][key]['quantization_max_abs_error'], 1/65536+1e-15)
        r = decoded['reference'][0, 2400:29600]
        denominator = sum(int(v)**2 for v in r)
        for method in ('single', 'constrained'):
            y = decoded[method][0, 2400:29600]
            error = sum((int(a)-int(b))**2 for a, b in zip(y, r))
            report = self.manifest['samples'][method]['pcm_measurements']
            self.assertEqual(report['integer_reference_squared_sum'], denominator)
            self.assertEqual(report['integer_error_squared_sum_per_channel'], [error])
            self.assertEqual(report['sample_denominator_per_channel'], 27200)
            self.assertEqual(report['total_reference_mse_per_channel'], [error/(32768**2*27200)])
            self.assertEqual(report['pcm_decode_divisor'], 32768)

    def test_check_reads_pcm_recomputes_and_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.write_fixture(directory)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            with (mock.patch.object(Path, 'write_bytes', side_effect=AssertionError('check writes')),
                  mock.patch.object(Path, 'write_text', side_effect=AssertionError('check writes'))):
                self.assertEqual(generator.check_assets(output), self.manifest)
                self.assertEqual(generator.main(['--check', '--output-dir', str(output)]), 0)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})

    def test_strict_check_rejects_missing_extra_corrupt_source_and_scores(self):
        changes = ('missing', 'extra', 'wav', 'source', 'score', 'forged_sha')
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                output = self.write_fixture(directory)
                manifest = json.loads((output/'MANIFEST.json').read_text())
                filename = core.FILE_NAMES['single']
                if change == 'missing':
                    (output/filename).unlink()
                elif change == 'extra':
                    (output/'extra.wav').write_bytes(b'junk')
                elif change in ('wav', 'forged_sha'):
                    blob = bytearray((output/filename).read_bytes()); blob[-1] ^= 1
                    (output/filename).write_bytes(blob)
                    if change == 'forged_sha':
                        manifest['files'][filename]['sha256'] = hashlib.sha256(blob).hexdigest()
                elif change == 'source':
                    manifest['source_sha256'][generator.SOURCE_PATHS[0]] = '0'*64
                else:
                    manifest['samples']['single']['pcm_measurements']['integer_error_squared_sum_per_channel'][0] += 1
                (output/'MANIFEST.json').write_text(json.dumps(manifest))
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                with self.assertRaises(ValueError):
                    generator.check_assets(output)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_public_measurement_shapes_and_pcm_flag_are_strict(self):
        for x in (np.zeros((2, 32002)), np.zeros((1, 32000)), np.full((1, 32002), np.nan)):
            with self.assertRaises(ValueError):
                core.measure_signal(x, self.signals['reference'])
        with self.assertRaises(ValueError):
            core.measure_signal(self.signals['single'], self.signals['reference'], pcm=True)


class PublishedGSCReadbackTest(unittest.TestCase):
    def test_valid_format_and_updated_sha_cannot_make_zero_reference_scorable(self):
        from codes.chapters.ch05.chapter05_experiments import ROOT
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for p in (ROOT/'codes/chapters/ch05/audio').glob('gsc_*.wav'):
                shutil.copyfile(p, output/p.name)
            reference_path = output/'gsc_reference.wav'
            with wave.open(str(reference_path), 'wb') as writer:
                writer.setparams((1, 2, 16000, 32000, 'NONE', 'not compressed'))
                writer.writeframes(bytes(64000))
            manifest = json.loads((ROOT/'codes/chapters/ch00/audio/MANIFEST.json').read_text())
            for row in manifest['files']:
                if row['file'] == reference_path.name:
                    row['sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
                    row['peak'] = row['rms'] = 0.
            manifest_path = output/'MANIFEST.json'
            manifest_path.write_text(json.dumps(manifest))
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, 'reference needs positive PCM energy'):
                gsc_audio_results(audio_dir=output, manifest_path=manifest_path)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})

    def test_actual_files_have_independent_integer_golden_scores(self):
        report = gsc_audio_results()['published_audio']
        always = report['actual_pcm_measurements']['always_adapt']
        frozen = report['actual_pcm_measurements']['gate_frozen']
        self.assertEqual(always['integer_reference_squared_sum'], 304641828594)
        self.assertEqual(always['integer_error_squared_sum'], 304641828594)
        self.assertEqual(frozen['integer_reference_cross_sum'], 280271517465)
        self.assertEqual(frozen['integer_error_squared_sum'], 1949544302)
        self.assertAlmostEqual(frozen['reference_projection_gain'], .9200033979526868)
        self.assertAlmostEqual(frozen['normalized_reference_error'], .0799966476118432)

    def test_published_bad_copy_is_not_regenerated(self):
        from codes.chapters.ch05.chapter05_experiments import ROOT
        for change in ('wav', 'source', 'format'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                source_dir = ROOT/'codes/chapters/ch05/audio'
                for p in source_dir.glob('gsc_*.wav'):
                    shutil.copyfile(p, output/p.name)
                manifest_path = output/'MANIFEST.json'
                manifest = json.loads((ROOT/'codes/chapters/ch00/audio/MANIFEST.json').read_text())
                if change == 'wav':
                    p = output/'gsc_array.wav'
                    blob = bytearray(p.read_bytes()); blob[-1] ^= 1; p.write_bytes(blob)
                elif change == 'source':
                    first = next(iter(manifest['generator_inputs']))
                    manifest['generator_inputs'][first] = '0'*64
                else:
                    p = output/'gsc_reference.wav'
                    with wave.open(str(p), 'wb') as w:
                        w.setparams((1, 2, 8000, 32000, 'NONE', 'not compressed'))
                        w.writeframes(bytes(64000))
                    for row in manifest['files']:
                        if row['file'] == p.name:
                            row['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                with self.assertRaises(ValueError):
                    gsc_audio_results(audio_dir=output, manifest_path=manifest_path)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
