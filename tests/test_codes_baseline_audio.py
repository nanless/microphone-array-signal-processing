"""Independent continuous-source/phasor checks and hostile IO for E03-18."""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.chapters.ch03.core import baseline_audio as audio
from codes.chapters.ch03.examples import generate_baseline_audio as generator


class BaselineAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs, cls.manifest = generator.prepare_assets()

    def write_fixture(self, directory):
        directory = Path(directory)
        for name, blob in self.blobs.items():
            (directory/name).write_bytes(blob)
        (directory/'MANIFEST.json').write_text(json.dumps(self.manifest))
        return directory

    def read_integer(self, blob):
        with wave.open(io.BytesIO(blob), 'rb') as stream:
            n, channels = stream.getnframes(), stream.getnchannels()
            values = np.frombuffer(stream.readframes(n), '<i2').reshape(n, channels).T.copy()
        return values

    def test_fixed_member_set_format_gain_and_current_sources(self):
        self.assertEqual(set(self.blobs), {'baseline_source.wav', 'baseline_train_px.wav',
                         'baseline_train_nx.wav', 'baseline_train_py.wav',
                         'baseline_train_pz.wav', 'baseline_heldout.wav'})
        self.assertEqual(len(generator.MEMBERS), 7)
        self.assertEqual(self.manifest['common_export_gain'], 1.)
        self.assertEqual(set(self.manifest['source_sha256']), set(generator.SOURCE_PATHS))
        self.assertEqual(len(generator.SOURCE_PATHS), 6)
        for name, digest in self.manifest['source_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((generator.ROOT/name).read_bytes()).hexdigest())
        for filename, blob in self.blobs.items():
            with wave.open(io.BytesIO(blob), 'rb') as stream:
                self.assertEqual((stream.getframerate(), stream.getnframes(), stream.getsampwidth(),
                                  stream.getcomptype()), (16000, 32000, 2, 'NONE'))
                self.assertEqual(stream.getnchannels(), 1 if 'source' in filename else 2)
            self.assertEqual(self.manifest['files'][filename]['sha256'], hashlib.sha256(blob).hexdigest())

    def test_complete_pcm_independent_continuous_evaluation(self):
        directions = {'train_px': (1, 0, 0), 'train_nx': (-1, 0, 0),
                      'train_py': (0, 1, 0), 'train_pz': (0, 0, 1), 'heldout': (.6, .8, 0)}
        t = np.arange(32000)/16000
        for name, filename in audio.FILE_NAMES.items():
            pcm = self.read_integer(self.blobs[filename])/32768
            tau = 0 if name == 'source' else -(directions[name][0]*.04+
                        directions[name][1]*.03+directions[name][2]*.02)/343+20e-6
            for m in range(len(pcm)):
                shifted = t-.002-(tau if m else 0)
                envelope = np.minimum(np.clip((shifted-.1)/.02, 0, 1), np.clip((1.9-shifted)/.02, 0, 1))
                expected = .2*envelope*np.sin(2*np.pi*500*shifted)
                np.testing.assert_array_equal(pcm[m]*32768, np.rint(expected*32768))
                self.assertLessEqual(float(np.max(abs(pcm[m]-expected))), .5/32768+1e-15)
                self.assertTrue(np.all(pcm[m, :1400] == 0))
                self.assertTrue(np.all(pcm[m, 31000:] == 0))
            self.assertLess(float(np.max(abs(pcm))), .201)
        mono = self.read_integer(self.blobs['baseline_source.wav'])[0]
        for name, blob in self.blobs.items():
            np.testing.assert_array_equal(self.read_integer(blob)[0], mono)

    def test_integer_energy_denominators_from_actual_pcm(self):
        # 27200 samples = 850 exact 500 Hz periods, each channel counted once.
        for name, filename in audio.FILE_NAMES.items():
            pcm = self.read_integer(self.blobs[filename]).astype(np.int64)[:, 2400:29600]
            expected = [int(sum(int(v)**2 for v in row)) for row in pcm]
            scores = self.manifest['samples'][name]['pcm_integer_measurements']
            self.assertEqual(scores['integer_squared_sum_E_per_channel'], expected)
            self.assertEqual(scores['integer_denominator_D_per_channel'], 27200*32768**2)
            self.assertEqual(scores['integer_squared_sum_E_all_channels'], sum(expected))
            self.assertEqual(scores['integer_denominator_D_all_channels'], len(pcm)*27200*32768**2)
            self.assertEqual(scores['mean_square_all_channels'], sum(expected)/(len(pcm)*27200*32768**2))
            np.testing.assert_allclose(self.manifest['samples'][name]['analytic']['mean_square_per_channel'],
                                       [.02]*len(pcm), rtol=0, atol=1e-16)

    def test_pcm_delay_from_independent_integer_period_dft(self):
        t = np.arange(2400, 29600)/16000
        kernel = np.exp(-2j*np.pi*500*t)
        for name, filename in audio.FILE_NAMES.items():
            pcm = self.read_integer(self.blobs[filename])/32768
            phasors = 2/27200*(pcm[:, 2400:29600]@kernel)
            recorded = self.manifest['samples'][name]['pcm_measurements']
            np.testing.assert_allclose(recorded['phasor_real_imag'],
                                       np.column_stack((phasors.real, phasors.imag)), atol=3e-16, rtol=0)
            if name != 'source':
                tau = -math.atan2((phasors[1]*phasors[0].conjugate()).imag,
                                 (phasors[1]*phasors[0].conjugate()).real)/(2*math.pi*500)
                self.assertAlmostEqual(recorded['delay_s'], tau, delta=2e-17)

    def test_four_training_closed_form_and_excluded_prediction(self):
        for domain in ('analytic', 'float', 'pcm'):
            result = self.manifest['calibration'][domain]
            if domain == 'analytic':
                delays = [-.04/343+20e-6, .04/343+20e-6,
                          -.03/343+20e-6, -.02/343+20e-6]
            else:
                key = domain+'_measurements'
                delays = [self.manifest['samples'][n][key]['delay_s'] for n in
                          ('train_px', 'train_nx', 'train_py', 'train_pz')]
            offset = (delays[0]+delays[1])/2
            b = [343*(delays[1]-delays[0])/2, 343*(offset-delays[2]), 343*(offset-delays[3])]
            np.testing.assert_allclose(result['solver']['baseline_m'], b, rtol=0, atol=5e-17)
            self.assertAlmostEqual(result['solver']['offset_s'], offset, delta=1e-19)
            self.assertEqual(result['solver']['rank'], 4)
            self.assertAlmostEqual(result['heldout']['predicted_delay_s'],
                                   -(.6*b[0]+.8*b[1])/343+offset, delta=1e-19)
            self.assertEqual(result['training_names'], ['train_px', 'train_nx', 'train_py', 'train_pz'])
        np.testing.assert_allclose(self.manifest['calibration']['analytic']['solver']['baseline_m'],
                                   [.04, .03, .02], atol=2e-17, rtol=0)
        np.testing.assert_allclose(self.manifest['calibration']['analytic']['fixed_zero_offset_control']['baseline_m'],
                                   [.04, .02314, .01314], atol=2e-17, rtol=0)
        self.assertAlmostEqual(self.manifest['calibration']['analytic']['fixed_zero_offset_control']
                               ['heldout_prediction_minus_truth_s'], -4e-6, delta=1e-19)
        # Quantization bias is distinct from overdetermination: four equations fit four unknowns.
        pcm = self.manifest['calibration']['pcm']
        self.assertGreater(abs(pcm['baseline_error_m'][0]), 2e-6)
        self.assertLess(abs(pcm['heldout']['prediction_minus_truth_s']), 3e-9)
        self.assertNotEqual(pcm['heldout']['prediction_minus_truth_s'],
                            pcm['heldout']['prediction_minus_measurement_s'])

    def test_measurement_input_types_zero_and_overflow(self):
        for invalid in (np.zeros((2, 32000)), np.zeros((2, 100)),
                        np.ones((2, 32000), dtype=bool), np.ones((2, 32000), dtype=complex),
                        np.full((2, 32000), '0.2'), np.full((2, 32000), np.nan),
                        np.full((2, 32000), 1e308)):
            with self.subTest(dtype=invalid.dtype, shape=invalid.shape), self.assertRaises(ValueError):
                audio.measure_signal(invalid)

    def test_single_tone_half_period_boundary_is_rejected(self):
        time = np.arange(32000)/16000
        tone = np.sin(2*np.pi*500*time)
        # Antiphase cannot choose between +1 ms and -1 ms at 500 Hz.
        with self.assertRaisesRegex(ValueError, 'phase boundary'):
            audio.measure_signal(np.stack((tone, -tone)))

    def test_check_strictly_readonly_and_successful_temp_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'assets'
            generated = generator.generate_assets(path)
            self.assertEqual(generated, self.manifest)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
            with patch.object(Path, 'write_bytes', side_effect=AssertionError('check wrote bytes')), \
                 patch.object(Path, 'write_text', side_effect=AssertionError('check wrote text')), \
                 patch.object(Path, 'mkdir', side_effect=AssertionError('check created directory')):
                self.assertEqual(generator.check_assets(path), self.manifest)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})

    def test_false_source_declarations_fail_before_generation_or_directory_creation(self):
        # The waveforms alone cannot prove that separately declared metadata
        # is true. Check actual source declarations at the write boundary.
        for key, bad in (('source_amplitude', True), ('frequency_hz', '500'),
                         ('fixed_channel_offset_s', 21e-6),
                         ('scoring_cycles', 850.0), ('baseline_m', [.04, .03, .03]),
                         ('channel_order', ['microphone_1', 'reference_microphone_0']),
                         ('waveform_model', 'x0=F(t-base), x1=F(t-base+tau)'),
                         ('direction_convention', 'unit vector from source toward array')):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                parameters = audio.baseline_parameters()
                parameters[key] = bad
                target = Path(temporary)/'not-created'
                with patch.object(generator, 'baseline_parameters', return_value=parameters), \
                     patch.object(generator, 'generate_signals', side_effect=AssertionError('late guard')):
                    with self.assertRaisesRegex(ValueError, 'true parameters'):
                        generator.generate_assets(target)
                self.assertFalse(target.exists())

    def test_manifest_types_sources_scores_and_forged_sha_fail(self):
        for kind in ('missing', 'extra', 'source', 'bool', 'numeric_string', 'float_for_int',
                     'score', 'integer', 'wav', 'forged_sha'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                path = self.write_fixture(temporary)
                m = copy.deepcopy(self.manifest)
                if kind == 'missing':
                    (path/'baseline_train_px.wav').unlink()
                elif kind == 'extra':
                    (path/'extra.wav').write_bytes(b'extra')
                elif kind == 'source':
                    m['source_sha256'][generator.SOURCE_PATHS[0]] = '0'*64
                elif kind == 'bool':
                    m['common_export_gain'] = True
                elif kind == 'numeric_string':
                    m['parameters']['fixed_channel_offset_s'] = '0.00002'
                elif kind == 'float_for_int':
                    m['samples_per_channel'] = 32000.
                elif kind == 'score':
                    m['samples']['train_px']['pcm_measurements']['delay_s'] = 0.
                elif kind == 'integer':
                    m['samples']['train_px']['pcm_integer_measurements']['integer_denominator_D_all_channels'] //= 2
                else:
                    data = bytearray((path/'baseline_train_px.wav').read_bytes())
                    data[12000] ^= 1
                    (path/'baseline_train_px.wav').write_bytes(data)
                    if kind == 'forged_sha':
                        m['files']['baseline_train_px.wav']['sha256'] = hashlib.sha256(data).hexdigest()
                (path/'MANIFEST.json').write_text(json.dumps(m))
                before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
                with self.assertRaises(ValueError):
                    generator.check_assets(path)
                self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})

    def test_json_duplicate_nonfinite_and_wrong_topology_fail(self):
        for data in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}', '{"x":1e400}', '[]'):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as temporary:
                path = self.write_fixture(temporary)
                (path/'MANIFEST.json').write_text(data)
                with self.assertRaises(ValueError):
                    generator.check_assets(path)

    def test_symlink_hardlink_special_member_and_parent_fail_preflight(self):
        for kind in ('symlink', 'hardlink', 'directory', 'fifo', 'parent'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root/'assets'
                path.mkdir()
                self.write_fixture(path)
                if kind == 'parent':
                    (root/'linked').symlink_to(path, target_is_directory=True)
                    target = root/'linked'
                else:
                    target = path
                    leaf = path/'baseline_train_px.wav'
                    payload = leaf.read_bytes()
                    leaf.unlink()
                    if kind == 'symlink':
                        (root/'outside').write_bytes(payload)
                        leaf.symlink_to(root/'outside')
                    elif kind == 'hardlink':
                        (root/'outside').write_bytes(payload)
                        os.link(root/'outside', leaf)
                    elif kind == 'directory':
                        leaf.mkdir()
                    else:
                        os.mkfifo(leaf)
                # Both checking and generation must reject before source replay or writes.
                with patch.object(generator, 'prepare_assets', side_effect=AssertionError('replay preceded preflight')):
                    for operation in (generator.check_assets, generator.generate_assets):
                        with self.assertRaises(ValueError):
                            operation(target)

    def test_missing_and_lexical_parent_traversal_fail_without_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary)/'missing'
            with self.assertRaises(ValueError):
                generator.check_assets(missing)
            self.assertFalse(missing.exists())
            with self.assertRaises(ValueError):
                generator.generate_assets(Path(temporary)/'x'/'..'/'assets')


if __name__ == '__main__':
    unittest.main()
