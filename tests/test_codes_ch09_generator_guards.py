"""Fixed trusted-model drift must fail before creating/replacing assets."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from codes.chapters.ch09.examples import moving_source_audio as moving
from codes.chapters.ch09.examples import chapter09_tracking_audio as tracking


class GeneratorPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moving_model = moving.build_fixture()
        cls.tracking_model = tracking.build_fixture()

    def rejected_before_write(self, module, mutation, *, existing=False):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)/'assets'
            if existing:
                module.generate(output)
                saved = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            fixture = copy.deepcopy(self.moving_model if module is moving else self.tracking_model)
            mutation(*fixture)
            with patch.object(module, 'build_fixture', return_value=fixture):
                with self.assertRaises(ValueError):
                    module.generate(output)
            if existing:
                self.assertEqual(saved, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})
            else:
                self.assertFalse(output.exists())

    def test_moving_fixed_metadata_true_types_truth_gain_and_shape(self):
        mutations = [lambda s,m:m.update(model='false model'),
            lambda s,m:m.update(sample_rate_hz=8000), lambda s,m:m.update(sample_rate_hz=True),
            lambda s,m:m.update(duration_seconds=2), lambda s,m:m.update(common_export_gain=float('nan')),
            lambda s,m:m.update(common_export_gain=1.),
            lambda s,m:m['microphones_xy_m'][0].__setitem__(0, 0),
            lambda s,m:m['truth']['angle_degrees_from_positive_y'].__setitem__(0, 123.),
            lambda s,m:m.update(retarded_equation_max_residual_seconds=True),
            lambda s,m:s.__setitem__('moving_array', s['moving_array'][:1,:-1]),
            lambda s,m:s.__setitem__('moving_array', s['moving_array'].astype(str)),
            lambda s,m:s['moving_array'].__setitem__((0,0),float('inf')),
            lambda s,m:s.__setitem__('unexpected',s['source'])]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i): self.rejected_before_write(moving, mutate)
        self.rejected_before_write(moving, mutations[7], existing=True)

    def test_tracking_model_pcm_and_float_score_drift(self):
        mutations = [lambda b,m:m.update(model='false model'),
            lambda b,m:m.update(sample_rate_hz=True), lambda b,m:m.update(seed=9002),
            lambda b,m:m.update(common_export_gain=float('nan')),
            lambda b,m:m['analysis_config'].__setitem__('window_samples', True),
            lambda b,m:m['analysis_config'].__setitem__('lookahead_from_state_ms',16),
            lambda b,m:m['phases_rad'].__setitem__(0,0.),
            lambda b,m:b.__setitem__('array_noisy.wav', b'not PCM'),
            lambda b,m:b.__setitem__('source.wav', bytearray(b['source.wav'])),
            lambda b,m:b.__setitem__('unexpected.wav', b['source.wav']),
            lambda b,m:m['pcm_analysis']['scores'].__setitem__('raw_valid_rmse_deg',123.),
            lambda b,m:m['float_analysis']['scores'].__setitem__('raw_valid_rmse_deg',123.),
            lambda b,m:m['pcm_analysis']['scores'].__setitem__('valid_observation_count',True),
            lambda b,m:m['pcm_analysis']['frames']['last_valid_measurement_time_s'].__setitem__(105,1.06596875)]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i): self.rejected_before_write(tracking, mutate)
        self.rejected_before_write(tracking, mutations[10], existing=True)

    def test_valid_generators_preserve_old_wav_bytes_and_check_is_read_only(self):
        for module in (moving,tracking):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as name:
                out=Path(name)/'assets'; report=module.generate(out)
                original=module.OUT if module is moving else module.OUTPUT
                for filename in report['files']:
                    self.assertEqual((out/filename).read_bytes(), (original/filename).read_bytes())
                for relative,digest in report['source_sha256'].items():
                    self.assertEqual(digest,hashlib.sha256((module.ROOT/relative).read_bytes()).hexdigest())
                before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
                module.generate(out,check=True)
                self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()})
                manifest=out/'MANIFEST.json'
                manifest.write_text('{"files":{},"files":{}}')
                saved=manifest.read_bytes()
                with self.assertRaises(ValueError):module.generate(out,check=True)
                self.assertEqual(saved,manifest.read_bytes())

    def test_lexical_parent_and_singly_linked_member_guards_precede_builder(self):
        for module in (moving,tracking):
            with self.subTest(module=module.__name__),tempfile.TemporaryDirectory() as name:
                base=Path(name);outside=base/'outside';outside.mkdir();linked=base/'linked';linked.symlink_to(outside)
                with patch.object(module,'build_fixture',side_effect=AssertionError('must not build')):
                    for output in (linked/'assets',base/'x'/'..'/'assets'):
                        with self.assertRaises(ValueError):module.generate(output)
                self.assertEqual(list(outside.iterdir()),[])
                out=base/'assets';module.generate(out)
                extra=out/'extra';extra.mkdir()
                with patch.object(module,'build_fixture',side_effect=AssertionError('must not build')):
                    with self.assertRaises(ValueError):module.generate(out)
