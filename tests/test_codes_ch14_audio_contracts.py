"""Imaging publication contract drift controls, with only temporary writes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.chapters.ch12.examples import generate_imaging_audio as audio


class ImagingAudioPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root/'new-assets'

    def refuse_without_writes(self, context):
        with context, self.assertRaises((ValueError, TypeError)):
            audio.generate_assets(self.output)
        self.assertFalse(self.output.exists(), 'preflight must precede mkdir/member writes')

    def test_fixed_parameters_reject_each_type_or_value_drift_before_writes(self):
        genuine = audio.parameters()
        for name, value in genuine.items():
            if type(value) is int:
                replacement = True
            elif type(value) is float:
                replacement = True
            elif isinstance(value, list):
                replacement = value[:-1]
            else:
                replacement = value+' false claim'
            drift = copy.deepcopy(genuine)
            drift[name] = replacement
            with self.subTest(field=name):
                self.refuse_without_writes(patch.object(audio, 'parameters', return_value=drift))

    def test_original_five_accepted_drifts_now_refuse_before_writes(self):
        genuine = audio.parameters()
        frequency = copy.deepcopy(genuine); frequency['frequency_hz'] = 1000
        boolean_gain = copy.deepcopy(genuine); boolean_gain['common_export_gain'] = True
        scoring = copy.deepcopy(genuine); scoring['scoring_samples_per_channel'] = 1
        for name, context in (
            ('frequency', patch.object(audio, 'parameters', return_value=frequency)),
            ('boolean gain', patch.object(audio, 'parameters', return_value=boolean_gain)),
            ('scoring', patch.object(audio, 'parameters', return_value=scoring)),
            ('limits', patch.object(audio, 'LIMITS', 'Actual measured industrial array performance')),
        ):
            with self.subTest(case=name):
                self.refuse_without_writes(context)
        report, arrays = audio.run_experiment()
        report['float_measurements']['source_1']['csm_real_imag'][0][0][0] = .04
        self.refuse_without_writes(patch.object(audio, 'run_experiment', return_value=(report, arrays)))

    def test_model_parameters_are_exact_even_when_top_parameters_are_genuine(self):
        for name, value in [('common_export_gain', 1.+1e-15), ('sample_rate_hz', 24000.),
                            ('phase_code', [True, -1]*10)]:
            report, arrays = audio.run_experiment()
            report['parameters'][name] = value
            with self.subTest(field=name):
                self.refuse_without_writes(patch.object(audio, 'run_experiment', return_value=(report, arrays)))

    def test_shape_dtype_nonfinite_waveform_tail_and_members_drift_refuse(self):
        for name in ('missing', 'extra', 'shape', 'complex', 'float32', 'nan', 'waveform', 'tail'):
            report, arrays = audio.run_experiment()
            if name == 'missing': del arrays['source_2_coherent']
            elif name == 'extra': arrays['extra'] = arrays['source_1'].copy()
            elif name == 'shape': arrays['source_1'] = arrays['source_1'][:, :-1]
            elif name == 'complex': arrays['source_1'] = arrays['source_1'].astype(complex)
            elif name == 'float32': arrays['source_1'] = arrays['source_1'].astype(np.float32)
            elif name == 'nan': arrays['source_1'][0, 10] = np.nan
            elif name == 'waveform': arrays['source_1'][0, 1000] += .001
            elif name == 'tail': arrays['array_coherent'][1, -3] = 0.1
            with self.subTest(case=name):
                self.refuse_without_writes(patch.object(audio, 'run_experiment', return_value=(report, arrays)))

    def test_independent_float_pair_scores_and_measurement_types_refuse(self):
        for name in ('coherence', 'integer count', 'bool float', 'false units', 'nan score'):
            report, arrays = audio.run_experiment()
            if name == 'coherence': report['float_source_pairs']['coherent']['magnitude_squared_coherence'] = 0.
            elif name == 'integer count': report['float_measurements']['source_1']['samples_per_snapshot'] = 1908.
            elif name == 'bool float': report['float_measurements']['source_1']['peak_all_samples'] = True
            elif name == 'false units': report['float_measurements']['source_1']['csm_units'] = 'Pa^2'
            else: report['float_measurements']['source_1']['csm_real_imag'][0][0][0] = float('nan')
            with self.subTest(case=name):
                self.refuse_without_writes(patch.object(audio, 'run_experiment', return_value=(report, arrays)))

    def test_pcm_codec_and_scorer_drift_refuse_before_writes(self):
        genuine_encode = audio.pcm16_bytes
        def shifted_encode(values, rate):
            changed = values.copy(); changed[0, 5000] += 1/32768
            return genuine_encode(changed, rate)
        self.refuse_without_writes(patch.object(audio, 'pcm16_bytes', side_effect=shifted_encode))
        genuine_score = audio.measure_signal
        def false_score(*args, **kwargs):
            result = genuine_score(*args, **kwargs)
            result['integer_scored_squared_sum_by_channel'][0] += 1
            return result
        self.refuse_without_writes(patch.object(audio, 'measure_signal', side_effect=false_score))
        self.refuse_without_writes(patch.object(audio, 'measure_source_pairs', return_value={}))

    def test_real_raw_pcm_integer_denominator_phase_and_tail(self):
        manifest = audio.generate_assets(self.output)
        expected_energy = {'source_1': [819589833360], 'source_2_phase_code': [204876619800],
                           'source_2_coherent': [204876619800],
                           'array_phase_code': [1024274797740, 1024370623860],
                           'array_coherent': [1843672975680, 614629853040]}
        for key, name in audio.FILE_NAMES.items():
            with self.subTest(file=name), wave.open(str(self.output/name)) as reader:
                self.assertEqual((reader.getframerate(), reader.getnframes(), reader.getsampwidth()), (24000, 48004, 2))
                channels = reader.getnchannels()
                raw = struct.unpack('<'+'h'*48004*channels, reader.readframes(48004))
                energy = [sum(raw[n*channels+c]**2 for j in range(20)
                              for n in range(2400*j+252, 2400*j+2160)) for c in range(channels)]
                self.assertEqual(energy, expected_energy[key])
                scores = manifest['samples'][key]['pcm_measurements']
                self.assertEqual(scores['integer_scored_squared_sum_by_channel'], energy)
                self.assertEqual(scores['integer_power_sample_denominator']*scores['integer_power_scale_denominator'], 40973988003840)
                if key.startswith('array_'):
                    self.assertNotEqual(raw[-7], 0)  # microphone 2, first retained tail sample
        # The published waveform is unchanged by the strengthened contract.
        for name in audio.FILE_NAMES.values():
            self.assertEqual((self.output/name).read_bytes(), (audio.DEFAULT_OUTPUT/name).read_bytes())

    def test_check_is_pure_read_only_and_drift_is_not_repaired(self):
        audio.generate_assets(self.output)
        def snapshot():
            return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                    for p in self.output.iterdir()}
        before = snapshot()
        with patch.object(Path, 'write_bytes', side_effect=AssertionError('check wrote bytes')), \
                patch.object(Path, 'write_text', side_effect=AssertionError('check wrote text')):
            audio.generate_assets(self.output, check=True)
        self.assertEqual(snapshot(), before)
        path = self.output/'MANIFEST.json'
        manifest = json.loads(path.read_text()); manifest['parameters']['common_export_gain'] = True
        path.write_text(json.dumps(manifest))
        before = snapshot()
        with self.assertRaises(ValueError): audio.check_assets(self.output)
        self.assertEqual(snapshot(), before)

    def test_illegal_json_extra_members_links_and_parent_chains_refuse(self):
        audio.generate_assets(self.output)
        manifest_path = self.output/'MANIFEST.json'
        genuine = manifest_path.read_bytes()
        for invalid in (b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}', b'{"x":1e999}'):
            manifest_path.write_bytes(invalid)
            with self.subTest(json=invalid), self.assertRaises(ValueError): audio.check_assets(self.output)
        manifest_path.write_bytes(genuine)
        extra = self.output/'extra'; extra.write_text('preserve')
        with self.assertRaises(ValueError): audio.generate_assets(self.output)
        self.assertEqual(extra.read_text(), 'preserve'); extra.unlink()
        member = self.output/'source_1.wav'; backup = self.root/'backup.wav'; backup.write_bytes(member.read_bytes())
        member.unlink(); member.symlink_to(backup)
        with self.assertRaises(ValueError): audio.check_assets(self.output)
        member.unlink(); os.link(backup, member)
        with self.assertRaises(ValueError): audio.check_assets(self.output)
        parent = self.root/'linked-parent'; parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): audio.generate_assets(parent/'different-assets')
        self.assertFalse((self.root/'different-assets').exists())


if __name__ == '__main__':
    unittest.main()
