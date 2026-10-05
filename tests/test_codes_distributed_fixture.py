"""Fixed trusted-model preflight: drift must fail before mkdir or WAV writes.

These controls replace internal model returns. They are not an external attack
claim and do not assert filesystem race safety. Expectations use the fixed
published sample contract and independent integer payloads, never fitted gain.
"""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.chapters.ch13.examples import generate_distributed_audio as generator


class DistributedFixturePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.signals = generator.run_experiment()
        cls.parameters = generator.parameters()

    def assert_prewrite_rejected(self, *, report=None, signals=None, parameters=None, limits=None):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'not-created'
            with (patch.object(generator, 'run_experiment', return_value=(
                    self.report if report is None else report,
                    self.signals if signals is None else signals)),
                  patch.object(generator, 'parameters', return_value=(
                    self.parameters if parameters is None else parameters)),
                  patch.object(generator, 'LIMITS', generator.LIMITS if limits is None else limits),
                  patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir before rejection')) as mkdir,
                  patch.object(Path, 'write_bytes', side_effect=AssertionError('write before rejection')) as writer):
                with self.assertRaises(ValueError):
                    generator.generate_assets(target)
                mkdir.assert_not_called(); writer.assert_not_called()
            self.assertFalse(target.exists())

    def test_five_observed_drift_classes_now_fail_before_writes(self):
        parameters = copy.deepcopy(self.parameters); parameters['sample_rate_hz'] = 16000.0
        self.assert_prewrite_rejected(parameters=parameters)
        parameters = copy.deepcopy(self.parameters); parameters['target_frequencies_hz'][0] = 701
        self.assert_prewrite_rejected(parameters=parameters)
        report = copy.deepcopy(self.report)
        report['float_measurements']['central_white']['windows']['steady']['samples'] = 28799
        self.assert_prewrite_rejected(report=report)
        report = copy.deepcopy(self.report)
        report['float_components']['central_white']['steady']['target_distortion_power'] = -1.
        self.assert_prewrite_rejected(report=report)
        self.assert_prewrite_rejected(limits='Full blind DANSE has been executed on a real network.')

    def test_nested_bool_integer_float_and_structure_drift(self):
        changes = [
            ('sample_rate_hz', True), ('samples_per_channel', 32000.),
            ('common_export_gain', True), ('common_export_gain', 1),
            ('seed', 1502026.), ('complete_convolution_tail_samples', False),
            ('target_frequencies_hz', [700., 1300]), ('a', [1, .5, 2., -.5]),
            ('target_peak_amplitudes', [.1, .061]), ('reference_target_factors', [1., 1.]),
            ('channel_order', 'x4,x3,x2,x1'), ('covariance_units', 'Pa squared'),
            ('orthogonality_scope', 'random statistical independence'),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                p = copy.deepcopy(self.parameters); p[key] = value
                self.assert_prewrite_rejected(parameters=p)
        p = copy.deepcopy(self.parameters); p['clock']['rate_is_known'] = 1
        self.assert_prewrite_rejected(parameters=p)
        p = copy.deepcopy(self.parameters); p['packet']['known_flag'] = 1
        self.assert_prewrite_rejected(parameters=p)
        p = copy.deepcopy(self.parameters); p['scoring_windows_samples']['packet'] = [16000, 16801]
        self.assert_prewrite_rejected(parameters=p)
        p = copy.deepcopy(self.parameters); p['extra'] = 'undeclared field'
        self.assert_prewrite_rejected(parameters=p)
        p = copy.deepcopy(self.parameters); del p['phase_rule']
        self.assert_prewrite_rejected(parameters=p)

    def test_report_scope_type_counts_and_decomposition_drift(self):
        for field, value in (
            ('samples', 28800.), ('samples', True), ('target_distortion_power', -1e-30),
            ('noise_power', -1e-30), ('normalized_mse', float('nan')),
            ('target_noise_cross_power', .001), ('decomposition_residual', .001),
        ):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(self.report)
                report['float_components']['central_white']['steady'][field] = value
                self.assert_prewrite_rejected(report=report)
        for field, value in (('reference_output_samples', 31999), ('whole_vs_chunked_exact_equal', 1),
                             ('invalid_reference_indices', [31999])):
            report = copy.deepcopy(self.report); report['clock'][field] = value
            self.assert_prewrite_rejected(report=report)
        report = copy.deepcopy(self.report); report['parameters']['common_export_gain'] = .8
        self.assert_prewrite_rejected(report=report)
        report = copy.deepcopy(self.report); report['limits'] = 'real speech MOS verified'
        self.assert_prewrite_rejected(report=report)

    def test_signal_set_real_shape_finite_and_literal_samples(self):
        for replacement in (
            np.zeros((2, 32000)), np.zeros((1, 31999)), np.zeros((1, 32000), dtype=bool),
            self.signals['central_white'].astype(complex),
            np.full((1, 32000), float('inf')), self.signals['central_white']+.001,
        ):
            signals = dict(self.signals); signals['central_white'] = replacement
            self.assert_prewrite_rejected(signals=signals)
        signals = dict(self.signals); del signals['central_white']
        self.assert_prewrite_rejected(signals=signals)

    def test_wrong_pcm_scorer_is_independently_rejected_before_write(self):
        original = generator.measure_signal
        def wrong(samples, key, **kwargs):
            score = original(samples, key, **kwargs)
            if kwargs.get('pcm') and key == 'central_white':
                score['windows']['steady']['integer_reference_squared_sum_D'] += 1
            return score
        with patch.object(generator, 'measure_signal', side_effect=wrong):
            self.assert_prewrite_rejected()

    def test_wrong_encoded_format_or_decoder_cannot_be_published(self):
        original_encode = generator.pcm16_bytes
        original_decode = generator.read_pcm16
        def wrong_rate(samples, rate):
            return original_encode(samples, 8000)
        with patch.object(generator, 'pcm16_bytes', side_effect=wrong_rate):
            self.assert_prewrite_rejected()
        def false_decoded_rate(blob):
            _, samples = original_decode(blob)
            return 16000., samples
        with patch.object(generator, 'read_pcm16', side_effect=false_decoded_rate):
            self.assert_prewrite_rejected()
        def false_samples(blob):
            rate, samples = original_decode(blob)
            return rate, samples+.0001
        with patch.object(generator, 'read_pcm16', side_effect=false_samples):
            self.assert_prewrite_rejected()

    def test_known_negative_cross_is_accepted_not_clipped(self):
        # This fixture's clock error correlates with noise; negative cross is
        # required and cannot be banned as though it were a variance.
        cross = self.report['float_components']['clock_misaligned_white']['steady']['target_noise_cross_power']
        self.assertLess(cross, 0.)
        manifest, _ = generator.prepare_assets()
        actual = manifest['samples']['clock_misaligned_white']['float_components']['steady']['target_noise_cross_power']
        self.assertEqual(actual, cross)

    def test_generated_all_pcm_payloads_preserve_published_fixture(self):
        # Formal manifests currently retain their old source identity until
        # ROOT regenerates them. Compare immutable real WAVs, not their labels.
        manifest, contents = generator.prepare_assets()
        self.assertEqual(len(contents), 18)
        for filename in manifest['files']:
            with self.subTest(filename=filename):
                with wave.open(io.BytesIO(contents[filename])) as fresh, \
                     wave.open(str(generator.DEFAULT_OUTPUT/filename)) as old:
                    self.assertEqual(fresh.getparams(), old.getparams())
                    self.assertEqual(fresh.readframes(32000), old.readframes(32000))
                self.assertEqual(contents[filename], (generator.DEFAULT_OUTPUT/filename).read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'assets'
            generated = generator.generate_assets(target)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()}
            with (patch.object(Path, 'write_bytes', side_effect=AssertionError('check wrote')),
                  patch.object(Path, 'mkdir', side_effect=AssertionError('check mkdir'))):
                self.assertEqual(generator.check_assets(target), generated)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()})


if __name__ == '__main__':
    unittest.main()
