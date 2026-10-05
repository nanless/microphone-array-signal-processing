"""Trusted internal fixture/report drift must fail before any filesystem write."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from codes.chapters.appendix_b.core import response_audio as core
from codes.chapters.appendix_b.examples import generate_response_audio as assets


class ResponseFixedPreflightTests(unittest.TestCase):
    def reject_before_write(self, producer, replacement):
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td)/'not-created'
            with patch.object(assets, producer, replacement), \
                    patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir reached')) as mkdir, \
                    patch.object(Path, 'write_bytes', side_effect=AssertionError('write reached')) as write, \
                    patch.object(Path, 'write_text', side_effect=AssertionError('text reached')) as text:
                with self.assertRaises(ValueError):
                    assets.generate(destination)
                self.assertEqual((mkdir.call_count, write.call_count, text.call_count), (0, 0, 0))
            self.assertFalse(destination.exists())

    def fixture_drift(self, mutate):
        original = core.build_fixture
        def replacement():
            value = original()
            mutate(value)
            return value
        self.reject_before_write('build_fixture', replacement)

    def test_five_original_fixture_drifts_have_zero_writes(self):
        mutations = [
            lambda x: x['parameters'].__setitem__('sample_rate_hz', 16000.),
            lambda x: x['parameters'].__setitem__('frequencies_hz', [1000, 4000]),
            lambda x: x['parameters'].__setitem__('scored_samples', 1),
            lambda x: x['parameters'].__setitem__('scope', 'blind real room performance'),
            lambda x: x['signals']['response_full_a'].__setitem__(1234, x['signals']['response_full_a'][1234]+.001),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index): self.fixture_drift(mutation)

    def test_complete_parameter_types_fields_and_near_valid_values(self):
        original = core.build_fixture()['parameters']
        # Every top-level parameter is bound, including descriptive model scope.
        for name in original:
            with self.subTest(missing=name):
                self.fixture_drift(lambda x, name=name: x['parameters'].pop(name))
        mutations = [
            lambda x: x['parameters'].__setitem__('extra', 'not declared'),
            lambda x: x['parameters'].__setitem__('common_export_gain', 1),
            lambda x: x['parameters'].__setitem__('common_export_gain', np.nextafter(1., 0.)),
            lambda x: x['parameters'].__setitem__('delay_fitting', 0),
            lambda x: x['parameters'].__setitem__('source_score', [1600., 30400]),
            lambda x: x['parameters'].__setitem__('amplitudes', [.1, float('nan')]),
            lambda x: x['parameters']['reflection_rirs'].__setitem__('b', [0., .5, .5]),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index): self.fixture_drift(mutation)

    def test_float_shape_dtype_finite_and_tiny_sample_drift(self):
        mutations = [
            lambda x: x['signals'].__setitem__('response_source', x['signals']['response_source'].tolist()),
            lambda x: x['signals'].__setitem__('response_source', x['signals']['response_source'].astype(np.float32)),
            lambda x: x['signals'].__setitem__('response_source', x['signals']['response_source'][:-1]),
            lambda x: x['signals']['response_source'].__setitem__(400, float('nan')),
            lambda x: x['signals']['response_source'].__setitem__(400, x['signals']['response_source'][400]+1e-10),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index): self.fixture_drift(mutation)

    def report_drift(self, producer, path, value):
        original = getattr(assets, producer)
        def replacement(*args):
            result = copy.deepcopy(original(*args))
            cursor = result
            for key in path[:-1]: cursor = cursor[key]
            cursor[path[-1]] = value
            return result
        self.reject_before_write(producer, replacement)

    def test_three_original_report_drifts_and_typed_measurements(self):
        controls = [
            ('analytic_results', ['candidates', 'response_full_a.wav', 'total_error_power'], -1.),
            ('analyze_fixture', ['candidates', 'response_full_a.wav', 'nmse'], -1.),
            ('analyze_pcm', ['integer_reference_squared_sum'], 1),
            ('analyze_fixture', ['scored_samples'], 28800.),
            ('analyze_pcm', ['scored_samples'], True),
            ('analytic_results', ['source_power'], float('inf')),
            ('analyze_fixture', ['candidates', 'response_full_a.wav', 'component_identity_max_error'], -1e-30),
        ]
        for producer, path, value in controls:
            with self.subTest(producer=producer, path=path): self.report_drift(producer, path, value)

    def test_changed_encoder_payload_rejected_without_using_analyzer_as_oracle(self):
        original = assets.pcm16_bytes
        def replacement(*args):
            data = bytearray(original(*args)); data[-2] ^= 1
            return bytes(data)
        self.reject_before_write('pcm16_bytes', replacement)

    def test_signed_cross_is_valid_and_pure_replay_does_not_mkdir(self):
        with patch.object(Path, 'mkdir', side_effect=AssertionError('unexpected mkdir')), \
                patch.object(Path, 'write_bytes', side_effect=AssertionError('unexpected write')):
            buffers, metadata = assets.expected_assets()
        self.assertLess(metadata['analytic']['candidates']['response_full_a.wav']['full_energy_cross_term'], 0)
        self.assertGreater(metadata['analytic']['candidates']['response_full_b.wav']['full_energy_cross_term'], 0)
        self.assertEqual(metadata['pcm_analysis']['integer_reference_squared_sum'], 309262788000)
        self.assertEqual(len(buffers), 6)


if __name__ == '__main__':
    unittest.main()
