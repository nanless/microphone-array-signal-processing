"""Independent fixed-channel control, actual PCM integers and prewrite contracts."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
from codes.chapters.ch10.core import channel_audio as core
from codes.chapters.ch10.examples import generate_channel_audio as generator


def read_codes(path):
    with wave.open(str(path), 'rb') as reader:
        channels = reader.getnchannels()
        assert (reader.getframerate(), reader.getnframes(), reader.getsampwidth(), reader.getcomptype()) == (16000, 32000, 2, 'NONE')
        return np.array(struct.unpack('<'+'h'*(32000*channels), reader.readframes(32000)), dtype=np.int64).reshape(-1, channels).T


def snapshot(directory):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


class ChannelAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)/'actual'
        cls.manifest = generator.generate_assets(cls.directory)
        cls.fixture = core.generate_experiment()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_literal_model_weights_target_retention_and_noise_covariance(self):
        fixture = self.fixture
        lo, hi = 2400, 29600
        bases = fixture['noise_bases'][:, lo:hi]
        np.testing.assert_allclose(bases@bases.T/27200, .005*np.eye(3), atol=2e-15, rtol=0)
        matrix = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
        np.testing.assert_allclose(fixture['noise_covariance'], .005*matrix@matrix.T, atol=1e-18)
        np.testing.assert_allclose(fixture['full_weights'], [1, -.5, .5], atol=2e-15)
        np.testing.assert_allclose(fixture['selected_weights'], [.5, .5], atol=2e-15)
        signals = fixture['signals']; reference = signals['reference']
        np.testing.assert_array_equal(signals['faulty_array'][0], np.zeros(32000))
        np.testing.assert_array_equal(signals['faulty_array'][1:], signals['healthy_array'][1:])
        # Independent rational model, without reusing the beamformer or selector.
        expected = {'healthy_output': reference+.5*fixture['noise_bases'][0:1]+.5*fixture['noise_bases'][2:3],
                    'stale_output': -.5*fixture['noise_bases'][0:1]+.5*fixture['noise_bases'][2:3],
                    'recomputed_output': reference+.5*fixture['noise_bases'][0:1]+fixture['noise_bases'][1:2]+.5*fixture['noise_bases'][2:3]}
        for name, values in expected.items():
            np.testing.assert_allclose(signals[name], values, atol=2e-16, rtol=0)
            row = self.manifest['samples'][name]['float_measurements']
            mse = float(np.mean((values[:, lo:hi]-reference[:, lo:hi])**2))
            self.assertAlmostEqual(row['reference_error_mean_square_per_channel'][0], mse, places=16)
            self.assertAlmostEqual(mse, core.analytic_measurements()[name]['reference_error_mean_square'], places=14)
        self.assertEqual(sum([-.5, .5]), 0)
        self.assertEqual(sum([.5, .5]), 1)
        self.assertNotEqual(self.manifest['samples']['stale_output']['pcm_integer_measurements']['reference_NMSE_per_channel'],
                            self.manifest['samples']['recomputed_output']['pcm_integer_measurements']['reference_NMSE_per_channel'])

    def test_actual_pcm_integer_sums_all_channels_and_known_frequency_diagnostics(self):
        codes = {name: read_codes(self.directory/file) for name, file in core.FILE_NAMES.items()}
        ref = codes['reference'][0, 2400:29600]
        Dref = sum(int(x)**2 for x in ref); Dpower = 27200*32768**2
        self.assertEqual(Dref, 146027417700); self.assertEqual(Dpower, 29205777612800)
        expected_errors = {'healthy_output': 73014637900, 'stale_output': 219018750300,
                           'recomputed_output': 219050854800}
        for name, values in codes.items():
            row = self.manifest['samples'][name]['pcm_integer_measurements']
            power = [sum(int(v)**2 for v in channel[2400:29600]) for channel in values]
            errors = [sum((int(v)-int(r))**2 for v, r in zip(channel[2400:29600], ref)) for channel in values]
            cross = [sum(int(v)*int(r) for v, r in zip(channel[2400:29600], ref)) for channel in values]
            self.assertEqual(row['integer_squared_sum_E_per_channel'], power)
            self.assertEqual(row['integer_reference_error_squared_sum_per_channel'], errors)
            self.assertEqual(row['integer_output_reference_cross_sum_per_channel'], cross)
            self.assertEqual(row['integer_reference_squared_sum_per_channel'], [Dref]*len(values))
            self.assertEqual(row['integer_denominator_D_all_channels'], Dpower*len(values))
            self.assertEqual(row['integer_squared_sum_E_all_channels'], sum(power))
            self.assertEqual(row['mean_square_all_channels'], sum(power)/(Dpower*len(values)))
            self.assertEqual(row['reference_NMSE_per_channel'], [x/Dref for x in errors])
            if name in expected_errors:
                self.assertEqual(errors, [expected_errors[name]])
            phase = np.exp(-2j*np.pi*500*np.arange(2400, 29600)/16000)
            phasor = 2*np.mean(values[:, 2400:29600]/32768*phase, axis=1)
            measured = self.manifest['samples'][name]['pcm_measurements']
            np.testing.assert_allclose(measured['target_frequency_phasor_real_per_channel'], phasor.real, atol=5e-14, rtol=0)
            np.testing.assert_allclose(measured['target_frequency_phasor_imag_per_channel'], phasor.imag, atol=5e-14, rtol=0)
        self.assertIn('no gain compensation', self.manifest['samples']['stale_output']['pcm_integer_measurements']['projection_scope'])

    def test_exact_source_closure_and_readonly_check_cli(self):
        expected = {'codes/chapters/ch10/core/channel_audio.py', 'codes/chapters/ch10/examples/generate_channel_audio.py',
                    'codes/chapters/ch10/core/channel_selection.py', 'codes/chapters/ch05/core/beamforming.py',
                    'codes/chapters/ch04/core/covariance.py', 'codes/chapters/ch02/core/conventions.py',
                    'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py'}
        self.assertEqual(set(generator.SOURCE_PATHS), expected)
        self.assertEqual(self.manifest['source_sha256'], {name: hashlib.sha256((generator.ROOT/name).read_bytes()).hexdigest() for name in expected})
        before = snapshot(self.directory)
        self.assertEqual(generator.check_assets(self.directory), self.manifest)
        process = subprocess.run([sys.executable, '-B', '-m', 'codes.chapters.ch10.examples.generate_channel_audio',
                                  '--output', str(self.directory), '--check'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(before, snapshot(self.directory))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'missing'/'nested'
            with self.assertRaises(ValueError): generator.check_assets(output)
            self.assertFalse(output.parent.exists())

    def test_parameter_true_types_fixed_values_and_scope_fail_before_writes(self):
        mutations = [lambda p: p.update(sample_rate_hz=True), lambda p: p.update(samples_per_channel=32000.),
                     lambda p: p.update(common_export_gain=1), lambda p: p.update(tail_samples=False),
                     lambda p: p.update(target_frequency_hz=1000.), lambda p: p.update(scoring_interval_samples=[2400, 29599]),
                     lambda p: p.update(noise_scope='random independent device noise'),
                     lambda p: p['selected_channels'].__setitem__(0, True), lambda p: p.update(extra='claim')]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'uncreated'
            for mutate in mutations:
                with self.subTest(mutate=mutate):
                    p = core.parameters(); mutate(p)
                    with patch.object(generator, 'parameters', return_value=p), patch.object(generator, 'generate_experiment') as builder:
                        with self.assertRaises(ValueError): generator.generate_assets(output)
                        builder.assert_not_called()
                    self.assertFalse(output.exists())
            for attribute, value in [('LIMITS', 'full blind fault recovery'), ('SAMPLE_RATE', True),
                                     ('SCORING_INTERVAL', (2400., 29600)), ('MEMBERS', ('other.wav',))]:
                with self.subTest(attribute=attribute), patch.object(generator, attribute, value):
                    with self.assertRaises(ValueError): generator.generate_assets(output)
                    self.assertFalse(output.exists())
            a = core.analytic_measurements(); a['stale_output']['target_gain'] = False
            with patch.object(generator, 'analytic_measurements', return_value=a):
                with self.assertRaises(ValueError): generator.generate_assets(output)
                self.assertFalse(output.exists())

    def test_fixture_shape_finite_dtype_components_and_weights_fail_before_writes(self):
        mutations = [lambda x: x['signals'].update(reference=np.zeros((32000,))),
                     lambda x: x['signals'].update(reference=x['signals']['reference'].astype(np.float32)),
                     lambda x: x['signals']['reference'].__setitem__((0, 9000), np.nan),
                     lambda x: x['signals']['faulty_array'].__setitem__((0, 9000), .1),
                     lambda x: x['signals'].update(unexpected=x['signals']['reference']),
                     lambda x: x['signals']['stale_output'].__setitem__((0, 9000), .1),
                     lambda x: x.update(full_weights=np.array([1., 0., 0.])),
                     lambda x: x.update(noise_covariance=np.eye(3)),
                     lambda x: x.update(noise_bases=x['noise_bases']*2)]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'uncreated'
            for mutate in mutations:
                fixture = copy.deepcopy(self.fixture); mutate(fixture)
                with self.subTest(mutate=mutate), patch.object(generator, 'generate_experiment', return_value=fixture):
                    with self.assertRaises(ValueError): generator.generate_assets(output)
                    self.assertFalse(output.exists())
            # Rejection also preserves a valid existing output byte-for-byte and mtime.
            before = snapshot(self.directory)
            with patch.object(generator, 'parameters', return_value={}):
                with self.assertRaises(ValueError): generator.generate_assets(self.directory)
            self.assertEqual(before, snapshot(self.directory))

    def test_strict_json_actual_pcm_full_replay_and_member_links(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'copy'
            for text in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}', '{"x":1e999}', '[]'):
                shutil.copytree(self.directory, directory); (directory/'MANIFEST.json').write_text(text)
                before = snapshot(directory)
                with self.assertRaises(ValueError): generator.check_assets(directory)
                self.assertEqual(before, snapshot(directory)); shutil.rmtree(directory)
            shutil.copytree(self.directory, directory)
            file = directory/core.FILE_NAMES['stale_output']; changed = bytearray(file.read_bytes()); changed[44+2*9000] ^= 1
            file.write_bytes(changed); manifest = copy.deepcopy(self.manifest)
            manifest['files'][file.name]['sha256'] = hashlib.sha256(changed).hexdigest()
            (directory/'MANIFEST.json').write_text(json.dumps(manifest))
            before = snapshot(directory)
            with self.assertRaises(ValueError): generator.check_assets(directory)
            self.assertEqual(before, snapshot(directory)); shutil.rmtree(directory)
            shutil.copytree(self.directory, directory); (directory/'extra').write_text('keep')
            before = snapshot(directory)
            with self.assertRaises(ValueError): generator.generate_assets(directory)
            self.assertEqual(before, snapshot(directory)); shutil.rmtree(directory)
            linked = Path(temp)/'linked'; linked.symlink_to(self.directory, target_is_directory=True)
            with self.assertRaises(ValueError): generator.generate_assets(linked/'child')
            external = Path(temp)/'external'; external.write_bytes(b'keep')
            shutil.copytree(self.directory, directory); file = directory/core.FILE_NAMES['stale_output']
            file.unlink(); file.symlink_to(external)
            with self.assertRaises(ValueError): generator.generate_assets(directory)
            self.assertEqual(external.read_bytes(), b'keep')
            with patch.object(generator, 'ROOT', Path(temp)/'missing-source-root'):
                with self.assertRaises(ValueError): generator.generate_assets(Path(temp)/'source-fail')
            self.assertFalse((Path(temp)/'source-fail').exists())

    def test_generic_measurement_rejects_invalid_or_nonfinite_arithmetic(self):
        reference = self.fixture['signals']['reference']
        for bad in (np.ones((32000,)), np.ones((2, 32000)), np.ones((1, 32000), dtype=complex), np.full((1, 32000), np.nan)):
            with self.assertRaises(ValueError): core.measure_signal(bad, reference)
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(ValueError): core.measure_signal(np.full((1, 32000), 1e200), reference)
        with self.assertRaises(ValueError): core.measure_signal(reference, np.zeros_like(reference))


if __name__ == '__main__':
    unittest.main()
