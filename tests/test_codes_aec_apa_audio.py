"""Actual PCM and causal-state contracts of the E06-39 audio fixture.

One real experiment/prepare call is shared by the class. Expected PCM sums
come from standard-library wave/struct, not the production decoder/scorer.
"""
import contextlib
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

from codes.chapters.ch06.core import apa_audio as core
from codes.chapters.ch06.examples import generate_apa_audio as generator


NAMES = {
    'reference': 'apa_reference.wav', 'true_echo': 'apa_true_echo.wav',
    'microphone': 'apa_microphone.wav', 'nlms_residual': 'apa_nlms_residual.wav',
    'apa2_residual': 'apa_apa2_residual.wav', 'apa4_residual': 'apa_apa4_residual.wav',
}


def decode(blob):
    with wave.open(io.BytesIO(blob), 'rb') as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                reader.getsampwidth(), reader.getcomptype()) == (16000, 1, 32013, 2, 'NONE')
        payload = reader.readframes(32013)
    assert len(payload) == 64026
    return struct.unpack('<32013h', payload)


def wav_from_integers(values):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack('<32013h', *values))
    return buffer.getvalue()


def snapshot(directory):
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}


class APAActualAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Capture the real run made by prepare_assets, without substituting a
        # fake experiment or running its 192k adaptive observations twice.
        actual_run = generator.run_experiment
        captured = {}

        def capture():
            signals, experiment = actual_run()
            captured.update(signals=signals, experiment=experiment)
            return signals, experiment

        with patch.object(generator, 'run_experiment', side_effect=capture):
            cls.blobs, cls.manifest = generator.prepare_assets()
        cls.signals = captured['signals']
        cls.experiment = captured['experiment']
        cls.integers = {key: decode(cls.blobs[name]) for key, name in NAMES.items()}
        cls.decoded = {key: np.array(values, dtype=float)[None, :]/32768
                       for key, values in cls.integers.items()}

    def make_assets(self, directory):
        for name, blob in self.blobs.items():
            (directory/name).write_bytes(blob)
        self.write_manifest(directory, self.manifest)

    @staticmethod
    def write_manifest(directory, manifest):
        (directory/'MANIFEST.json').write_text(json.dumps(manifest, allow_nan=False)+'\n')

    def test_real_prepare_formats_common_gain_and_source_binding(self):
        self.assertEqual(set(self.blobs), set(NAMES.values()))
        self.assertEqual(core.FILE_NAMES, NAMES)
        self.assertEqual(self.manifest['sample_rate_hz'], 16000)
        self.assertEqual(self.manifest['samples_per_channel'], 32013)
        self.assertEqual(self.manifest['common_export_gain'], 1.)
        self.assertEqual(self.manifest['pcm_decode_divisor'], 32768)
        expected_sources = {
            'codes/chapters/ch06/core/apa_audio.py',
            'codes/chapters/ch06/examples/generate_apa_audio.py',
            'codes/chapters/ch06/core/aec.py', 'codes/chapters/ch06/core/aec_numeric.py',
            'codes/chapters/ch06/core/aec_affine_projection.py',
            'codes/chapters/ch02/core/conventions.py',
            'codes/chapters/ch00/core/audio_samples.py',
            'codes/chapters/ch00/io_contracts.py',
        }
        self.assertEqual(set(self.manifest['source_sha256']), expected_sources)
        for path in expected_sources:
            self.assertEqual(self.manifest['source_sha256'][path],
                             hashlib.sha256((generator.ROOT/path).read_bytes()).hexdigest())
        for name, blob in self.blobs.items():
            record = self.manifest['files'][name]
            self.assertEqual(record['sha256'], hashlib.sha256(blob).hexdigest())
            self.assertEqual((record['sample_rate_hz'], record['channels'], record['samples_per_channel']),
                             (16000, 1, 32013))
            decode(blob)

    def test_pcm_integer_holdout_sums_and_ratios_independent_of_scorer(self):
        scores = self.manifest['pcm_measurements']
        self.assertEqual(scores['holdout_interval_samples'], [24000, 32000])
        sums = {key: sum(v*v for v in values[24000:32000])
                for key, values in self.integers.items()}
        for key, total in sums.items():
            record = scores['powers'][key]
            self.assertEqual(record['integer_squared_sum'], total)
            self.assertEqual(record['sample_denominator'], 8000)
            self.assertEqual(record['mean_square'], total/(32768**2*8000))
        for key in ('nlms_residual', 'apa2_residual', 'apa4_residual'):
            record = scores['ratios'][key]
            self.assertEqual(record['integer_numerator'], sums['microphone'])
            self.assertEqual(record['integer_denominator'], sums[key])
            self.assertFalse(record['zero_output'])
            expected = 10*np.log10(sums['microphone']/sums[key])
            self.assertAlmostEqual(record['microphone_to_residual_total_power_ratio_db'], expected, places=14)

    def test_quantization_and_full_thirteen_sample_echo_tail(self):
        reference = self.signals['reference'][0]
        echo = self.signals['true_echo'][0]
        np.testing.assert_array_equal(reference[32000:], np.zeros(13))
        self.assertEqual(np.count_nonzero(echo[32000:]), 13)
        # Independent causal sparse convolution includes every nonzero tail.
        expected = np.zeros(32013)
        for delay, amplitude in ((0, .6), (3, -.2), (7, .1), (13, .05)):
            expected[delay:delay+32000] += amplitude*reference[:32000]
        np.testing.assert_allclose(echo, expected, rtol=2e-15, atol=3e-17)
        self.assertNotEqual(self.integers['true_echo'][-1], 0)
        for key, signal in self.signals.items():
            # Shared encoder and actual decoder use the same /32768 lattice;
            # quantization is round-to-even, without per-file normalization.
            expected_pcm = np.rint(signal[0]*32768).astype(np.int64)
            np.testing.assert_array_equal(self.integers[key], expected_pcm)
            self.assertLessEqual(float(np.max(np.abs(signal-self.decoded[key]))), 1/65536)

    def test_ar_draw_order_noise_and_same_source_components(self):
        rng = np.random.Generator(np.random.PCG64(20261001))
        excitation = rng.standard_normal(32000)*.015
        noise = rng.standard_normal(32013)*.003
        reference = self.signals['reference'][0]
        for index in (0, 1, 7, 15999, 31999):
            # Closed-form AR sum is an independent route from the recurrence.
            expected = float(excitation[:index+1] @ (.98**np.arange(index, -1, -1)))
            self.assertAlmostEqual(reference[index], expected, places=14)
        np.testing.assert_allclose(self.signals['microphone'][0]-self.signals['true_echo'][0],
                                   noise, rtol=1e-13, atol=2e-17)
        self.assertEqual(self.experiment['parameters']['random_draw_order'],
                         '32000 AR excitation draws, then 32013 independent observation-noise draws')

    def test_prior_output_startup_and_frozen_holdout(self):
        reference = self.signals['reference'][0]
        observation = self.signals['microphone'][0]
        x0, x1, d0 = reference[0], reference[1], observation[0]
        first_weight = .2*d0*x0/(x0*x0+.001)
        for key, order in (('nlms_residual', '1'), ('apa2_residual', '2'), ('apa4_residual', '4')):
            residual = self.signals[key][0]
            self.assertEqual(residual[0], d0)
            self.assertAlmostEqual(residual[1], observation[1]-first_weight*x1, places=15)
            run = self.experiment['conditions']['noisy'][order]
            weights = np.array(run['training_final_weights'])
            # Independent frozen FIR output; no further microphone-dependent
            # adaptation is allowed anywhere in holdout or the complete tail.
            prediction = np.zeros(32013)
            for delay, coefficient in enumerate(weights):
                prediction[delay:] += coefficient*reference[:32013-delay]
            np.testing.assert_allclose(residual[24000:],
                observation[24000:]-prediction[24000:], rtol=2e-13, atol=5e-17)
            frozen = [row for row in run['trace'] if row['frozen']]
            self.assertEqual(frozen[0]['state_after_samples'], 24160)
            self.assertEqual(frozen[-1]['state_after_samples'], 32013)
            for row in frozen:
                self.assertEqual(row['weights'], weights.tolist())

    def test_float_component_decomposition_uses_same_eight_thousand_samples(self):
        window = slice(24000, 32000)
        d = self.signals['microphone'][0]
        echo = self.signals['true_echo'][0]
        noise = (d-echo)[window]
        for key, order in (('nlms_residual', '1'), ('apa2_residual', '2'), ('apa4_residual', '4')):
            residual = self.signals[key][0][window]
            prediction = d[window]-residual
            clean_error = echo[window]-prediction
            expected = {'clean_echo_prediction_error_mse': np.mean(clean_error**2),
                'observation_noise_mse': np.mean(noise**2),
                'twice_clean_error_noise_cross_mean': 2*np.mean(clean_error*noise),
                'total_prior_residual_mse': np.mean(residual**2)}
            actual = self.experiment['conditions']['noisy'][order]['holdout_float']
            self.assertEqual(actual['sample_denominator'], 8000)
            for metric, value in expected.items():
                self.assertAlmostEqual(actual[metric]/value, 1., places=12)
            self.assertAlmostEqual(actual['decomposition_sum']/actual['total_prior_residual_mse'], 1., places=14)
            noiseless = self.experiment['conditions']['noiseless'][order]['holdout_float']
            self.assertEqual(noiseless['observation_noise_mse'], 0.)
            self.assertEqual(noiseless['twice_clean_error_noise_cross_mean'], 0.)

    def test_readonly_actual_check_preserves_bytes_mtime_and_input_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_assets(directory)
            before = snapshot(directory)
            self.assertEqual(generator.check_assets(directory, replay=False), self.manifest)
            self.assertEqual(snapshot(directory), before)
            copies = {key: value.copy() for key, value in self.decoded.items()}
            self.assertEqual(core.measure_pcm(self.decoded), self.manifest['pcm_measurements'])
            for key in copies:
                np.testing.assert_array_equal(self.decoded[key], copies[key])
            self.assertEqual(snapshot(directory), before)

    def test_cli_check_runs_real_replay_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_assets(directory)
            before = snapshot(directory)
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(generator.main(['--check', '--output-dir', temporary]), 0)
            self.assertEqual(json.loads(captured.getvalue())['status'], 'checked')
            self.assertEqual(snapshot(directory), before)

    def test_tampered_manifest_scores_and_source_sha_fail_without_repair(self):
        for target in ('scores', 'source'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.make_assets(directory)
                manifest = copy.deepcopy(self.manifest)
                if target == 'scores':
                    manifest['pcm_measurements']['powers']['microphone']['integer_squared_sum'] += 1
                else:
                    first = next(iter(manifest['source_sha256']))
                    manifest['source_sha256'][first] = '0'*64
                self.write_manifest(directory, manifest)
                before = snapshot(directory)
                with self.assertRaises(ValueError):
                    generator.check_assets(directory, replay=False)
                self.assertEqual(snapshot(directory), before)

    def test_current_source_bytes_are_checked_not_only_manifest_key_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'root'
            directory = Path(temporary)/'assets'
            directory.mkdir()
            self.make_assets(directory)
            for path in generator.SOURCE_PATHS:
                target = root/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((generator.ROOT/path).read_bytes())
            (root/generator.SOURCE_PATHS[0]).write_bytes(b'changed actual source\n')
            before = snapshot(directory)
            with patch.object(generator, 'ROOT', root), self.assertRaisesRegex(ValueError, 'source'):
                generator.check_assets(directory, replay=False)
            self.assertEqual(snapshot(directory), before)

    def test_valid_pcm_tamper_requires_sha_and_rescored_actual_payload(self):
        for update_sha in (False, True):
            with self.subTest(update_sha=update_sha), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.make_assets(directory)
                values = list(self.integers['apa2_residual'])
                values[24000] += 17
                changed = wav_from_integers(values)
                name = NAMES['apa2_residual']
                (directory/name).write_bytes(changed)
                if update_sha:
                    manifest = copy.deepcopy(self.manifest)
                    manifest['files'][name]['sha256'] = hashlib.sha256(changed).hexdigest()
                    self.write_manifest(directory, manifest)
                before = snapshot(directory)
                with self.assertRaises(ValueError):
                    generator.check_assets(directory, replay=False)
                self.assertEqual(snapshot(directory), before)

    def test_valid_zero_microphone_pcm_has_undefined_input_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_assets(directory)
            blob = wav_from_integers([0]*32013)
            name = NAMES['microphone']
            (directory/name).write_bytes(blob)
            manifest = copy.deepcopy(self.manifest)
            manifest['files'][name]['sha256'] = hashlib.sha256(blob).hexdigest()
            self.write_manifest(directory, manifest)
            before = snapshot(directory)
            with self.assertRaisesRegex(ValueError, 'positive microphone energy'):
                generator.check_assets(directory, replay=False)
            self.assertEqual(snapshot(directory), before)

    def test_measure_pcm_rejects_positive_32768_off_grid_nonreal_and_nonfinite(self):
        for value in (1., -32769/32768, .1, np.nan, np.inf, 1j, True, '0'):
            decoded = {key: samples.copy() for key, samples in self.decoded.items()}
            dtype = complex if value == 1j else (bool if value is True else str if value == '0' else float)
            invalid = np.zeros((1, 32013), dtype=dtype)
            invalid[0, 24000] = value
            decoded['reference'] = invalid
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.measure_pcm(decoded)
        decoded = {key: samples.copy() for key, samples in self.decoded.items()}
        decoded['reference'][0, 24000] = -1.
        core.measure_pcm(decoded)  # -32768 is a valid signed PCM16 endpoint.

    def test_zero_residual_is_explicit_without_inventing_finite_db(self):
        decoded = {key: samples.copy() for key, samples in self.decoded.items()}
        decoded['apa2_residual'][:] = 0
        record = core.measure_pcm(decoded)['ratios']['apa2_residual']
        self.assertEqual(record['integer_denominator'], 0)
        self.assertTrue(record['zero_output'])
        self.assertIsNone(record['microphone_to_residual_total_power_ratio_db'])

    def test_exact_member_set_missing_extra_directory_and_symlink_are_rejected(self):
        for case in ('missing', 'extra', 'directory', 'symlink'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)/'assets'
                directory.mkdir()
                self.make_assets(directory)
                target = directory/NAMES['reference']
                external = Path(temporary)/'external.wav'
                external.write_bytes(b'keep external bytes')
                if case == 'missing':
                    target.unlink()
                elif case == 'extra':
                    (directory/'extra.wav').write_bytes(b'extra')
                elif case == 'directory':
                    target.unlink()
                    target.mkdir()
                else:
                    target.unlink()
                    target.symlink_to(external)
                external_before = external.stat().st_mtime_ns, external.read_bytes()
                with self.assertRaises(ValueError):
                    generator.check_assets(directory, replay=False)
                self.assertEqual((external.stat().st_mtime_ns, external.read_bytes()), external_before)

    def test_fixed_true_parameters_reject_before_simulation_and_write(self):
        cases = []
        for key in generator.REQUIRED_PARAMETERS:
            changed = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            changed[key] = None
            cases.append((key, changed))
        for key in ('seed', 'common_export_gain', 'sample_rate_hz', 'step_size'):
            changed = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            changed[key] = True
            cases.append((key+' bool', changed))
        for key, index in [('projection_orders', 0), ('true_path_current_first', 1)]:
            changed = copy.deepcopy(generator.REQUIRED_PARAMETERS)
            changed[key][index] = True
            cases.append((key+' nested bool', changed))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'not_created'
            for label, changed in cases:
                with self.subTest(label=label), patch.object(generator, 'parameters', return_value=changed), patch.object(generator, 'run_experiment') as run:
                    with self.assertRaises(ValueError):
                        generator.main(['--output-dir', str(output)])
                    run.assert_not_called()
                    self.assertFalse(output.exists())

    def test_returned_false_parameters_rejected_before_encoding(self):
        for key, value in [('ar_coefficient', .5), ('seed', True),
                           ('projection_orders', [True, 2, 4]), ('common_export_gain', True),
                           ('freeze', 'weights and histories frozen'),
                           ('random_draw_order', 'different order'),
                           ('alignment', 'fitted delay'), ('output', 'posterior error')]:
            changed = copy.deepcopy(self.experiment)
            changed['parameters'][key] = value
            with self.subTest(key=key), patch.object(generator, 'run_experiment', return_value=(self.signals, changed)), patch.object(generator, 'pcm16_bytes') as encode:
                with self.assertRaises(ValueError):
                    generator.prepare_assets()
                encode.assert_not_called()

    def test_generation_member_and_output_symlink_guards_precede_expensive_run(self):
        for member in (False, True):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temporary:
                external = Path(temporary)/'external'
                external.mkdir()
                protected = external/'protected.wav'
                protected.write_bytes(b'external must survive')
                output = Path(temporary)/'assets'
                if member:
                    output.mkdir()
                    (output/NAMES['reference']).symlink_to(protected)
                else:
                    output.symlink_to(external, target_is_directory=True)
                before = protected.stat().st_mtime_ns, protected.read_bytes()
                with patch.object(generator, 'prepare_assets', side_effect=AssertionError('guard must run first')):
                    with self.assertRaises(ValueError):
                        generator.main(['--output-dir', str(output)])
                self.assertEqual((protected.stat().st_mtime_ns, protected.read_bytes()), before)


if __name__ == '__main__':
    unittest.main()
