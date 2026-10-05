"""Independent fixed-model oracles and first-write timing for weighted audio.

Substituted trusted model functions simulate developer drift, not an external
code attack. All writable publications are temporary; formal files are read only.
"""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from codes.chapters.appendix_a.core import weighted_audio as core
from codes.chapters.appendix_a.examples import generate_weighted_audio as generator


class WeightedFixtureTests(unittest.TestCase):
    def reject_before_io(self, attribute, value):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            destination = Path(directory)/'not-created'
            with patch.object(generator, attribute, return_value=value), \
                 patch.object(Path, 'mkdir', side_effect=AssertionError('premature mkdir')) as mkdir, \
                 patch.object(Path, 'write_bytes', side_effect=AssertionError('premature write')) as write:
                with self.assertRaises(ValueError):
                    generator.generate(destination)
            mkdir.assert_not_called()
            write.assert_not_called()
            self.assertFalse(destination.exists())

    def test_full_waveform_matches_independent_literal_components(self):
        f = core.build_fixture()
        # Direct samplewise scalar oracle at every envelope boundary and
        # nonzero samples; it does not invoke the validator's literal helper.
        for n in (0, 1, 639, 640, 1600, 9007, 30399, 30400, 31359, 31360, 31998, 31999):
            envelope = min(1, n/640, (31999-n)/640)
            target = .2*math.cos(math.tau*700*n/16000)*envelope
            noise = [.03*math.cos(math.tau*3500*n/16000)*envelope,
                     .06*math.cos(math.tau*4000*n/16000)*envelope]
            self.assertAlmostEqual(f['signals']['weighted_target'][n], target, delta=3e-13)
            for stem, weights in [('weighted_ols', [.5,.5]), ('weighted_gls', [.8,.2]),
                                  ('weighted_reversed', [.2,.8])]:
                value = weights[0]*noise[0]+weights[1]*noise[1]
                self.assertAlmostEqual(f['components'][stem]['noise'][n], value, delta=3e-13)
                self.assertAlmostEqual(f['signals'][stem][n], target+value, delta=3e-13)
        core.validate_fixed_fixture(f)
        generator.expected_assets()

    def test_parameters_values_structure_and_exact_types_rejected_before_io(self):
        changes = [('sample_rate_hz', True), ('sample_rate_hz', 16000.),
                   ('target_frequency_hz', 701), ('target_frequency_hz', 700.),
                   ('target_amplitude', 0), ('common_export_gain', 1),
                   ('scored_samples', 1), ('scored_samples', 28800.),
                   ('source_score', (1600,30400)), ('source_score', [1600.,30400]),
                   ('delay_samples', False), ('noise_frequencies_hz', [3500.,4000]),
                   ('noise_amplitudes', [.03,.061]),
                   ('weights', {'weighted_ols':[.5,.5], 'weighted_gls':[.2,.8],
                                'weighted_reversed':[.2,.8]}),
                   ('envelope', 'no fade'), ('limits', 'blind measured speech result')]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                f = copy.deepcopy(core.build_fixture()); f['parameters'][key] = value
                self.reject_before_io('build_fixture', f)
        for kind in ('extra_parameter', 'missing_parameter', 'extra_fixture_field'):
            with self.subTest(kind=kind):
                f = copy.deepcopy(core.build_fixture())
                if kind == 'extra_parameter': f['parameters']['extra'] = 1
                elif kind == 'missing_parameter': del f['parameters']['envelope']
                else: f['extra'] = 1
                self.reject_before_io('build_fixture', f)

    def test_signal_and_components_drift_rejected_before_io(self):
        for kind in ('polarity', 'shape', 'complex', 'nan', 'inf', 'bool', 'int',
                     'list', 'channel_order', 'component', 'extra_component', 'extra_signal', 'endpoint'):
            with self.subTest(kind=kind):
                f = copy.deepcopy(core.build_fixture()); x = f['signals']['weighted_gls']
                if kind == 'polarity': f['signals']['weighted_gls'] = -x
                elif kind == 'shape': f['signals']['weighted_gls'] = x[:-1]
                elif kind == 'complex': f['signals']['weighted_gls'] = x.astype(complex)
                elif kind in ('nan', 'inf'): x[500] = float(kind)
                elif kind == 'bool': f['signals']['weighted_gls'] = x.astype(bool)
                elif kind == 'int': f['signals']['weighted_gls'] = x.astype(int)
                elif kind == 'list': f['signals']['weighted_gls'] = x.tolist()
                elif kind == 'channel_order': f['signals']['weighted_array'] = f['signals']['weighted_array'][::-1]
                elif kind == 'component': f['components']['weighted_gls']['noise'] *= -1
                elif kind == 'endpoint': x[0] = 1e-14
                elif kind == 'extra_component': f['components']['weighted_gls']['extra'] = x
                else: f['signals']['extra'] = x
                self.reject_before_io('build_fixture', f)

    def test_analytic_and_float_score_drift_rejected_before_io(self):
        for attribute in ('analytic_results', 'analyze_fixture'):
            baseline = getattr(core, attribute)()
            for kind in ('negative_power', 'false_nmse', 'bool_power', 'extra', 'cross', 'count'):
                with self.subTest(attribute=attribute, kind=kind):
                    report = copy.deepcopy(baseline)
                    row = report['candidates']['weighted_gls.wav']
                    if kind == 'negative_power': row['target_distortion_power'] = -1e-30
                    elif kind == 'false_nmse': row['nmse'] = .1
                    elif kind == 'bool_power': row['target_distortion_power'] = False
                    elif kind == 'extra': row['unknown'] = 1
                    elif kind == 'cross': row['cross_term'] = -.001
                    elif attribute == 'analyze_fixture': report['scored_samples'] = 28800.
                    else: report['target_power'] = 1
                    self.reject_before_io(attribute, report)

    def test_signed_cross_roundoff_is_legal_and_is_retained(self):
        baseline = core.analyze_fixture()
        baseline['candidates']['weighted_gls.wav']['cross_term'] = -1e-18
        with patch.object(generator, 'analyze_fixture', return_value=baseline):
            _, metadata = generator.expected_assets()
        self.assertEqual(metadata['floating_point']['candidates']['weighted_gls.wav']['cross_term'], -1e-18)

    def test_export_constants_cannot_smuggle_equal_bool_or_numeric_types(self):
        for name, value in [('GAIN', True), ('GAIN', 1), ('GAIN', .9),
                            ('SAMPLES', 32000.), ('SAMPLE_RATE', 16000.),
                            ('STEMS', list(generator.STEMS))]:
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
                target = Path(directory)/'new'
                with patch.object(generator, name, value), \
                     patch.object(Path, 'mkdir', side_effect=AssertionError('premature mkdir')) as mkdir, \
                     patch.object(Path, 'write_bytes', side_effect=AssertionError('premature write')) as write:
                    with self.assertRaises(ValueError): generator.generate(target)
                mkdir.assert_not_called(); write.assert_not_called()

    def test_pcm_actual_encoding_and_score_drift_rejected_before_io(self):
        _, metadata = generator.expected_assets()
        for kind in ('E', 'D', 'bool', 'gainfit'):
            report = copy.deepcopy(metadata['pcm_analysis']); row = report['candidates']['weighted_gls.wav']
            if kind == 'E': row['integer_error_squared_sum'] += 1
            elif kind == 'D': row['integer_reference_squared_sum'] += 1
            elif kind == 'bool': row['scored_samples'] = True
            else: row['gain_fitting'] = True
            self.reject_before_io('analyze_pcm', report)
        encode = generator.pcm16_bytes
        def altered(signal, rate):
            data = bytearray(encode(signal, rate)); data[4000] ^= 1
            return bytes(data)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            destination = Path(directory)/'not-created'
            with patch.object(generator, 'pcm16_bytes', side_effect=altered), \
                 patch.object(Path, 'mkdir', side_effect=AssertionError('premature mkdir')) as mkdir, \
                 patch.object(Path, 'write_bytes', side_effect=AssertionError('premature write')) as write:
                with self.assertRaisesRegex(ValueError, 'independent literal quantization'):
                    generator.generate(destination)
            mkdir.assert_not_called(); write.assert_not_called()

    def test_new_temporary_publication_same_five_payloads_and_readonly_check(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            target = Path(directory)/'publication'; generator.generate(target)
            for filename in generator.MEMBERS:
                if filename.endswith('.wav'):
                    self.assertEqual((target/filename).read_bytes(), (generator.OUTPUT/filename).read_bytes())
            before = {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in target.iterdir()}
            with patch.object(Path, 'mkdir', side_effect=AssertionError('check mkdir')) as mkdir, \
                 patch.object(Path, 'write_bytes', side_effect=AssertionError('check write')) as write, \
                 patch.object(Path, 'write_text', side_effect=AssertionError('check write')):
                generator.generate(target, check=True)
            mkdir.assert_not_called(); write.assert_not_called()
            self.assertEqual(before, {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in target.iterdir()})
            self.assertEqual(generator.SOURCE_PATHS, (
                'codes/chapters/appendix_a/core/weighted_audio.py',
                'codes/chapters/appendix_a/examples/generate_weighted_audio.py',
                'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py'))
