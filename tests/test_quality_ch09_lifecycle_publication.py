"""Publish figure77 only with exact fixed-PCM epochs and lifecycle records.

Every fixture is an ordinary temporary copy; no generator/FSM is the oracle
and no published asset is modified by this test.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import quality_check as quality


ROOT = Path(__file__).resolve().parents[1]
AUDIO = 'codes/chapters/ch09/tracking_audio'
REPORT = 'codes/chapters/ch09/reports/figure77_tracking_lifecycle.json'
FIGURE = 'figures/fig77_tracking_lifecycle.png'
SOURCES = {
    'scripts/make_tracking_figures.py', 'codes/chapters/ch09/core/lifecycle.py',
    'codes/chapters/ch09/examples/tracking_lifecycle_demo.py',
    'codes/chapters/ch09/examples/chapter09_tracking_audio.py',
    'codes/chapters/ch09/core/tracking_audio.py',
    'codes/chapters/ch09/core/moving_source.py', 'codes/chapters/ch09/core/tracking.py',
    'codes/chapters/ch04/core/doa.py', 'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py',
}


class LifecyclePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = SOURCES | {REPORT, FIGURE} | {AUDIO+'/'+name for name in
                    ('source.wav', 'array_noisy.wav', 'MANIFEST.json')}
        cls.blobs = {name: (ROOT/name).read_bytes() for name in names}
        cls.original = json.loads(cls.blobs[REPORT])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, blob in self.blobs.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        self.report = copy.deepcopy(self.original)
        context = patch.object(quality, 'ROOT', self.root)
        context.start(); self.addCleanup(context.stop)

    def save(self):
        (self.root/REPORT).write_text(json.dumps(self.report, allow_nan=False))

    def issues(self):
        errors = []
        quality.check_tracking_lifecycle_figure(errors)
        return errors

    def test_both_legitimate_protocols_exact_equality_and_read_only(self):
        # Fixed 512-point windows start at 160*i; ticks are half samples.
        for name, expiry in (('lifecycle', 100), ('state_age_comparison', 102)):
            rows = self.report[name]['rows']
            self.assertEqual(len(rows), 197)
            for i, row in enumerate(rows):
                valid = not 82 <= i <= 105
                phase = ('tentative' if i < 2 or 106 <= i < 108 else
                         'confirmed' if i < expiry or i >= 108 else 'absent')
                identity = None if expiry <= i < 106 else 1 if i < 106 else 2
                last = 320*i+511 if valid else 26431 if i < expiry else None
                now = 320*i+(1024 if name == 'lifecycle' else 511)
                expected = {
                    'frame': i, 'observation_valid': valid,
                    'state_tick': 320*i+511, 'available_tick': 320*i+1024,
                    'state_time_s': (320*i+511)/32000,
                    'publication_time_s': (320*i+1024)/32000,
                    'phase': phase, 'track_id': identity,
                    'consecutive_valid_frames': i+1 if i < 82 else 0 if i < 106 else i-105,
                    'last_valid_measurement_tick': last,
                    'last_valid_measurement_time_s': None if last is None else last/32000,
                    'age_ticks': None if last is None else now-last,
                    'age_s': None if last is None else (now-last)/32000,
                    'publish_confirmed': phase == 'confirmed',
                    'events': {0:['candidate'], 2:['confirmed'], expiry:['expired'],
                               106:['candidate'], 108:['confirmed']}.get(i, []),
                }
                self.assertEqual(row, expected)
            self.assertEqual(self.report[name]['counts']['confirmed_publications'],
                             187 if expiry == 100 else 189)
        state = self.report['state_age_comparison']['rows']
        self.assertEqual(state[101]['age_ticks'], 6400)
        self.assertTrue(state[101]['publish_confirmed'])
        self.assertEqual(self.report['state_age_comparison']['events'][2]['age_ticks'], 6720)
        before = {p.relative_to(self.root).as_posix():
                  (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(self.issues(), [])
        after = {p.relative_to(self.root).as_posix():
                 (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_thirteen_original_unchecked_row_drifts_rejected(self):
        cases = (
            ('lifecycle',99,'phase','absent'), ('lifecycle',108,'track_id',1),
            ('lifecycle',0,'track_id',True),
            ('lifecycle',2,'consecutive_valid_frames',3.0),
            ('lifecycle',99,'last_valid_measurement_tick',32191),
            ('lifecycle',99,'last_valid_measurement_time_s',1.0),
            ('lifecycle',99,'age_ticks',1), ('lifecycle',99,'age_s',1.0),
            ('lifecycle',99,'state_time_s',9.0),
            ('lifecycle',99,'publication_time_s',9.0),
            ('lifecycle',100,'events',[]),
            ('state_age_comparison',102,'phase','confirmed'),
            ('state_age_comparison',108,'track_id',2.0),
        )
        for name,i,key,value in cases:
            with self.subTest(protocol=name,frame=i,field=key):
                self.report = copy.deepcopy(self.original)
                self.report[name]['rows'][i][key] = value
                self.save(); self.assertTrue(self.issues())

    def test_exact_field_sets_and_true_numeric_types(self):
        changes = (
            lambda r: r['lifecycle']['rows'][2].pop('age_s'),
            lambda r: r['state_age_comparison']['rows'][101].update(extra=1),
            lambda r: r['lifecycle']['rows'][0].update(frame=False),
            lambda r: r['lifecycle']['rows'][0].update(state_tick=511.0),
            lambda r: r['lifecycle']['rows'][0].update(observation_valid=1),
            lambda r: r['lifecycle']['events'][0].update(track_id=True),
            lambda r: r['state_age_comparison']['events'][2].update(age_ticks=6720.0),
            lambda r: r['lifecycle']['events'][1].update(extra=1),
            lambda r: r['lifecycle']['parameters'].update(maximum_age_ticks=6400.0),
            lambda r: r['lifecycle']['parameters'].update(event_publication_epoch='state centre'),
            lambda r: r['state_age_comparison']['parameters'].update(extra=1),
            lambda r: r['lifecycle']['counts'].update(confirmed_publications=187.0),
            lambda r: r.update(schema_version=True),
        )
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                self.report = copy.deepcopy(self.original)
                change(self.report); self.save(); self.assertTrue(self.issues())

    def test_state_protocol_cannot_be_relabelled_available(self):
        self.report['lifecycle'] = copy.deepcopy(self.report['state_age_comparison'])
        self.report['lifecycle']['parameters']['age_clock'] = 'available'
        self.report['parameters'] = copy.deepcopy(self.report['lifecycle']['parameters'])
        self.save(); self.assertTrue(self.issues())

    def test_false_execution_identity_and_latency_scopes_rejected(self):
        false_claim = ('Full LOCATA multi-target Bayesian Bernoulli/LMB filter '
                       'with permanent speaker identity and zero-latency publication.')
        for section in ('lifecycle', 'state_age_comparison'):
            with self.subTest(protocol=section):
                self.report = copy.deepcopy(self.original)
                self.report[section]['scope'] = false_claim
                self.save(); self.assertTrue(self.issues())
        self.report = copy.deepcopy(self.original)
        self.report['limits'] = false_claim
        self.save(); self.assertTrue(self.issues())

    def test_illegal_report_json(self):
        for text in ('[]', '{"schema_version":1,"schema_version":1}',
                     '{"x":NaN}', '{"x":1e999}', '{'):
            with self.subTest(text=text):
                (self.root/REPORT).write_text(text)
                self.assertTrue(self.issues())

    def test_extra_input_and_changed_pcm_rejected(self):
        audio = self.root/AUDIO
        extra = audio/'extra.bin'; extra.write_bytes(b'extra')
        self.assertTrue(self.issues()); extra.unlink()
        pcm = audio/'array_noisy.wav'; original = pcm.read_bytes()
        pcm.write_bytes(original[:-2]+b'\x00\x00')
        self.assertTrue(self.issues())

    def test_linked_or_hardlinked_input_report_and_png_rejected(self):
        for name in (AUDIO+'/array_noisy.wav', REPORT, FIGURE):
            path = self.root/name
            original = path.read_bytes()
            target = self.root/('outside-'+path.name); target.write_bytes(original)
            for hard in (False, True):
                with self.subTest(name=name,hardlink=hard):
                    path.unlink()
                    if hard: os.link(target, path)
                    else: path.symlink_to(target)
                    self.assertTrue(self.issues())
                    path.unlink(); path.write_bytes(original)
            target.unlink()
        parent = self.root/'codes/chapters/ch09'
        real = self.root/'real-ch09'; parent.rename(real); parent.symlink_to(real)
        self.assertTrue(self.issues())
