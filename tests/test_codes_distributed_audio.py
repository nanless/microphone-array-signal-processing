"""Known-tone independent scores, genuine PCM and strict read-only replay."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch13.core.distributed_audio import (
    FILE_NAMES, OUTPUT_REFERENCES, SAMPLE_RATE, SAMPLES, SCORING_SAMPLES,
    POWER, run_experiment, measure_signal, generate_components, continuous_components,
)
from codes.chapters.ch13.examples.generate_distributed_audio import (
    prepare_assets, check_assets, generate_assets, source_digests, SOURCE_PATHS, ROOT,
)


class DistributedAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.signals = run_experiment()
        cls.manifest, cls.contents = prepare_assets()

    def _populate(self, directory):
        directory = Path(directory)
        directory.mkdir()
        for name, blob in self.contents.items():
            (directory/name).write_bytes(blob)
        return directory

    def _manifest_change(self, directory, change):
        manifest = json.loads((directory/'MANIFEST.json').read_text())
        change(manifest)
        (directory/'MANIFEST.json').write_text(json.dumps(manifest, allow_nan=False))

    def test_independent_file_set_and_shapes(self):
        expected = {'reference_node1', 'reference_node2', 'array_white', 'array_correlated',
                    'local_node1', 'central_white', 'compressed_white', 'central_correlated',
                    'compressed_correlated', 'stale_correlated', 'central_node2_correlated',
                    'remote_scalar_white', 'transport_pcm16_white', 'clock_misaligned_white',
                    'clock_linear_corrected_white', 'packet_zerofill_white', 'packet_local_fallback_white'}
        self.assertEqual(set(FILE_NAMES), expected)
        self.assertEqual(len(self.contents), 18)
        for key, array in self.signals.items():
            self.assertEqual(array.shape, (4 if key.startswith('array_') else 1, 32000))
            self.assertLess(np.max(abs(array)), 32767/32768)

    def test_finite_orthogonal_window_not_random_independence(self):
        self.assertAlmostEqual(self.report['finite_steady_target_power'], .0068, places=14)
        np.testing.assert_allclose(self.report['finite_steady_basis_covariance'], .0068*np.eye(4), atol=3e-15)
        np.testing.assert_allclose(self.report['finite_steady_source_noise_cross'], 0., atol=3e-15)
        self.assertEqual(SCORING_SAMPLES, 28800)

    def test_independent_analytic_costs_and_actual_decomposition(self):
        expected = {'local_node1': 4/9, 'central_white': 2/13, 'compressed_white': 2/13,
                    'central_correlated': 8/69, 'compressed_correlated': 8/69,
                    'stale_correlated': 2/13, 'central_node2_correlated': 8/69}
        for key, nmse in expected.items():
            analytic = self.report['analytic_expected'][key]['steady']
            actual = self.report['float_components'][key]['steady']
            self.assertAlmostEqual(analytic['normalized_mse'], nmse, places=13)
            self.assertAlmostEqual(actual['normalized_mse'], nmse, places=12)
            self.assertLess(abs(actual['decomposition_residual']), 2e-17)
        node2 = self.report['analytic_expected']['central_node2_correlated']['steady']
        self.assertAlmostEqual(node2['reference_target_power'], 4*.0068)
        self.assertAlmostEqual(node2['total_mse'], 4*.0068*8/69)

    def test_deliberate_static_compression_pcm_byte_equality(self):
        for scene in ('white', 'correlated'):
            self.assertEqual(self.contents['central_'+scene+'.wav'], self.contents['compressed_'+scene+'.wav'])
        self.assertNotEqual(self.contents['central_correlated.wav'], self.contents['stale_correlated.wav'])

    def test_true_integer_pcm_scores_and_distinct_reference_denominators(self):
        expected = {'central_white': (32352145920, 210281178240),
                    'central_correlated': (24381142920, 210281178240),
                    'stale_correlated': (32353470360, 210281178240),
                    'central_node2_correlated': (97517652480, 841117066920)}
        # Raw RIFF payload route avoids the production decoder and scorer.
        def integers(key):
            with wave.open(io.BytesIO(self.contents[key+'.wav'])) as wav:
                return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.int64)
        for key, (e, d) in expected.items():
            ref = integers(OUTPUT_REFERENCES[key]); y = integers(key)
            independent_e = sum(int(v)**2 for v in y[1600:30400]-ref[1600:30400])
            independent_d = sum(int(v)**2 for v in ref[1600:30400])
            self.assertEqual((independent_e, independent_d), (e, d))
            score = self.manifest['samples'][key]['pcm_measurements']['windows']['steady']
            self.assertEqual(score['integer_error_squared_sum_E'], e)
            self.assertEqual(score['integer_reference_squared_sum_D'], d)
            self.assertEqual(score['integer_nmse_E_over_D'], e/d)
        self.assertNotEqual(expected['central_node2_correlated'][1], 4*expected['central_white'][1])

    def test_transport_really_quantizes_scalar_before_receiver(self):
        data = generate_components(); x = data['signals']['array_white']
        z = 2*x[2]-.5*x[3]
        quantized = np.rint(z*32768)/32768  # Independent scalar nearest-even calculation.
        expected = (2*x[0]+x[1]+2*quantized)/13
        np.testing.assert_array_equal(data['signals']['transport_pcm16_white'][0], expected)
        difference = max(abs(expected-data['signals']['central_white'][0]))
        self.assertLessEqual(difference, (2/13)*(.5/32768)+1e-16)
        self.assertGreater(difference, 2e-6)
        self.assertGreater(data['transport']['remote_quantization_max_abs_error'], 1e-5)

    def test_real_clock_correction_support_state_and_model_change(self):
        control = self.report['clock']
        self.assertEqual(control['device_input_samples'], 32004)
        self.assertEqual(control['reference_output_samples'], 32000)
        self.assertEqual(control['invalid_reference_indices'], [])
        self.assertTrue(control['whole_vs_chunked_exact_equal'])
        self.assertAlmostEqual(control['same_index_drift_samples_last'], 31999*.0001/1.0001)
        self.assertGreater(control['uncorrected_peer_reconstruction_nmse'], .7)
        self.assertLess(control['corrected_peer_reconstruction_nmse'], .003)
        raw = self.report['float_components']['clock_misaligned_white']['steady']
        corrected = self.report['float_components']['clock_linear_corrected_white']['steady']
        original = self.report['float_components']['central_white']['steady']
        self.assertAlmostEqual(raw['normalized_mse'], .350880476532835, places=10)
        self.assertAlmostEqual(corrected['normalized_mse'], .135724013151017, places=10)
        self.assertLess(corrected['noise_power'], original['noise_power'])
        self.assertIn('not blind SRO', control['limitations'])

    def test_drop_window_and_complete_window_have_separate_scores(self):
        expected = {'packet_zerofill_white': 461/676, 'packet_local_fallback_white': 4/9}
        for key, during in expected.items():
            actual = self.report['float_components'][key]
            self.assertAlmostEqual(actual['packet']['normalized_mse'], during, places=12)
            self.assertAlmostEqual(actual['steady']['normalized_mse'], (35*(2/13)+during)/36, places=12)
            pcm = self.manifest['samples'][key]['pcm_measurements']['windows']
            self.assertEqual(pcm['packet']['integer_sample_denominator'], 800)
            self.assertEqual(pcm['steady']['integer_sample_denominator'], 28800)

    def test_measurement_rejects_false_pcm_or_wrong_reference_shape(self):
        for array, kwargs in [(self.signals['central_white'], {'pcm': True, 'reference': self.signals['reference_node1']}),
                              (np.zeros((1, 31999)), {'reference': self.signals['reference_node1']}),
                              (self.signals['central_white'], {}),
                              (self.signals['central_white'], {'reference': np.zeros((2, 32000))})]:
            with self.assertRaises(ValueError):
                measure_signal(array, 'central_white', **kwargs)
        with self.assertRaises(ValueError):
            measure_signal(np.ones((1, 32000), complex), 'reference_node1')
        for times in ([0j], [True], [], [-1.], [2.1], [1e308]):
            with self.assertRaises(ValueError):
                continuous_components(times)

    def test_every_executed_dependency_has_current_real_sha(self):
        expected = {'codes/chapters/ch13/core/distributed.py', 'codes/chapters/ch13/core/distributed_audio.py',
                    'codes/chapters/ch13/examples/generate_distributed_audio.py', 'codes/chapters/ch02/core/conventions.py',
                    'codes/chapters/ch04/core/covariance.py', 'codes/chapters/ch10/sro_closed_loop_demo.py',
                    'codes/chapters/ch10/core/engineering.py', 'codes/chapters/ch00/core/audio_samples.py',
                    'codes/chapters/ch00/io_contracts.py'}
        self.assertEqual(set(SOURCE_PATHS), expected)
        self.assertEqual(source_digests(), {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in expected})

    def test_full_check_is_read_only_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._populate(Path(temporary)/'assets')
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            with (patch.object(Path, 'write_bytes', side_effect=AssertionError('check attempted write')),
                  patch.object(Path, 'mkdir', side_effect=AssertionError('check attempted mkdir'))):
                checked = check_assets(directory)
            self.assertEqual(checked, self.manifest)
            after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            self.assertEqual(before, after)

    def test_missing_check_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'missing'
            with self.assertRaises(ValueError):
                generate_assets(directory, check=True)
            self.assertFalse(directory.exists())

    def test_actual_generator_to_isolated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'generated'
            manifest = generate_assets(directory)
            self.assertEqual(set(p.name for p in directory.iterdir()), set(self.contents))
            self.assertEqual(manifest, self.manifest)
            self.assertEqual(generate_assets(directory, check=True), manifest)

    def test_internal_fixture_drift_cannot_overwrite_existing_members(self):
        # The output folder can already contain ordinary user files with the
        # expected names. Model validation must precede their first overwrite.
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._populate(Path(temporary)/'assets')
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            parameters = json.loads(json.dumps(self.manifest['parameters']))
            parameters['target_frequencies_hz'][0] = 701
            with (patch('codes.chapters.ch13.examples.generate_distributed_audio.parameters', return_value=parameters),
                  patch.object(Path, 'write_bytes', side_effect=AssertionError('preflight attempted overwrite')) as writer,
                  patch.object(Path, 'mkdir', side_effect=AssertionError('preflight attempted mkdir')) as mkdir):
                with self.assertRaises(ValueError):
                    generate_assets(directory)
                writer.assert_not_called(); mkdir.assert_not_called()
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()})

    def test_extra_missing_symlink_and_linked_parent_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); directory = self._populate(base/'assets')
            (directory/'extra').write_text('extra')
            with self.assertRaises(ValueError):
                check_assets(directory)
            with patch('codes.chapters.ch13.examples.generate_distributed_audio.prepare_assets', side_effect=AssertionError('preflight missing')):
                with self.assertRaises(ValueError):
                    generate_assets(directory)
            (directory/'extra').unlink(); (directory/'central_white.wav').unlink()
            with self.assertRaises(ValueError):
                check_assets(directory)
            (directory/'central_white.wav').symlink_to(directory/'compressed_white.wav')
            with self.assertRaises(ValueError):
                check_assets(directory)
            (base/'linked').symlink_to(directory, target_is_directory=True)
            with self.assertRaises(ValueError):
                check_assets(base/'linked')

    def test_hard_linked_member_rejected(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); directory = self._populate(base/'assets')
            os.link(directory/'central_white.wav', base/'duplicate.wav')
            with self.assertRaises(ValueError):
                check_assets(directory)

    def test_strict_json_rejects_duplicate_nonfinite_and_type_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._populate(Path(temporary)/'assets')
            for data in (b'{"schema_version":1,"schema_version":1}', b'{"value":NaN}', b'{"value":1e999}'):
                (directory/'MANIFEST.json').write_bytes(data)
                with self.assertRaises(ValueError):
                    check_assets(directory)
            (directory/'MANIFEST.json').write_bytes(self.contents['MANIFEST.json'])
            self._manifest_change(directory, lambda m: m['files']['central_white.wav'].update(channels=True))
            with self.assertRaises(ValueError):
                check_assets(directory)

    def test_replay_rejects_forged_float_and_actual_pcm_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._populate(Path(temporary)/'assets')
            self._manifest_change(directory, lambda m: m['samples']['central_white']['float_components']['steady'].update(normalized_mse=.5))
            with self.assertRaises(ValueError):
                check_assets(directory)
            (directory/'MANIFEST.json').write_bytes(self.contents['MANIFEST.json'])
            self._manifest_change(directory, lambda m: m['samples']['central_white']['pcm_measurements']['windows']['steady'].update(integer_error_squared_sum_E=1))
            with self.assertRaises(ValueError):
                check_assets(directory)

    def test_wrong_source_sha_and_wav_format_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._populate(Path(temporary)/'assets')
            self._manifest_change(directory, lambda m: m['source_sha256'].update({SOURCE_PATHS[0]: '0'*64}))
            with self.assertRaises(ValueError):
                check_assets(directory)
            (directory/'MANIFEST.json').write_bytes(self.contents['MANIFEST.json'])
            (directory/'central_white.wav').write_bytes(pcm16_bytes(np.zeros((2, SAMPLES)), SAMPLE_RATE))
            with self.assertRaises(ValueError):
                check_assets(directory)


if __name__ == '__main__':
    unittest.main()
