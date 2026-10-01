"""Tracking publication checks actual bytes, arithmetic and usable controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_site, quality_check as quality

ROOT = Path(__file__).resolve().parents[1]


class TrackingPublicationTests(unittest.TestCase):
    directory = 'tracking_audio'
    checker = staticmethod(quality.check_tracking_audio)
    stage = staticmethod(build_site.stage_tracking_audio)
    quality_root_name = 'TRACKING_AUDIO_ROOT'

    @classmethod
    def setUpClass(cls):
        from codes.chapters.ch09.examples.chapter09_tracking_audio import generate
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'assets'; generate(source)
            cls.assets = {p.name: p.read_bytes() for p in source.iterdir()}
        cls.original = json.loads(cls.assets['MANIFEST.json'])
        cls.sources = {name: (ROOT/name).read_bytes() for name in cls.original['source_sha256']}

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root/'codes/chapters/ch09'/self.directory
        self.site = self.root/'site'; self.published = self.site/self.directory
        self.source.mkdir(parents=True); self.published.mkdir(parents=True)
        (self.site/'research').mkdir()
        for name, blob in self.sources.items():
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        for name, blob in self.assets.items():
            (self.source/name).write_bytes(blob); (self.published/name).write_bytes(blob)
        self.manifest = copy.deepcopy(self.original)
        self.pages = ((self.site/'09_source-tracking.html', ''),
                      (self.site/'research/05_exercises_and_audio.html', '../'))
        for page, prefix in self.pages:
            page.write_text(self.html(prefix))
        for name, value in (('ROOT', self.root), ('SITE', self.site), (self.quality_root_name, self.source)):
            patcher = patch.object(quality, name, value); patcher.start(); self.addCleanup(patcher.stop)

    def html(self, prefix):
        base = prefix+self.directory+'/'
        return ''.join(f'<audio controls preload="none" aria-label="sample {name}" src="{base}{name}"></audio>'
                       for name in sorted(self.original['files']))+f'<a href="{base}MANIFEST.json">Manifest</a>'

    def save(self):
        for folder in (self.source, self.published):
            (folder/'MANIFEST.json').write_text(json.dumps(self.manifest))

    def issues(self):
        errors = []; self.checker(errors); return errors

    def rejected(self):
        self.assertTrue(self.issues())
        destination = self.root/'stage'
        with self.assertRaises(ValueError):
            self.stage(self.source, destination)
        self.assertFalse(destination.exists())

    def test_valid_actual_assets_are_read_only_and_publish_identically(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(), [])
        self.assertEqual(self.stage(self.source, self.root/'stage'), set(self.assets))
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_nonfinite_overflow_duplicate_and_wrong_root_json_are_rejected(self):
        original = self.assets['MANIFEST.json'].decode()
        for invalid in ('[]', '{"files":{},"files":{}}',
                        original.replace('16000', 'NaN', 1),
                        original.replace('16000', '1e999', 1)):
            with self.subTest(invalid=invalid[:40]):
                for folder in (self.source, self.published):
                    (folder/'MANIFEST.json').write_text(invalid)
                self.rejected()

    def test_empty_or_stale_sources_are_rejected(self):
        self.manifest['source_sha256'] = {}; self.save(); self.rejected()

    def test_extra_directory_manifest_link_and_missing_member_are_rejected(self):
        extra = self.source/'extra'; extra.mkdir(); self.rejected(); extra.rmdir()
        member = self.source/'MANIFEST.json'; blob = member.read_bytes()
        member.unlink(); member.symlink_to(self.published/member.name); self.rejected()
        member.unlink(); member.write_bytes(blob)
        next(self.source.glob('*.wav')).unlink(); self.rejected()

    def test_parent_links_cannot_be_read_through_or_overwrite_destination(self):
        linked = self.root/'linked'; linked.symlink_to(self.source.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.stage(linked/self.directory, self.root/'stage')
        self.assertFalse((self.root/'stage').exists())
        outside = self.root/'outside'; outside.mkdir()
        linked_destination = self.root/'linked_destination'; linked_destination.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.stage(self.source, linked_destination/'stage')
        self.assertEqual(list(outside.iterdir()), [])
        (outside/'nested').mkdir()
        with self.assertRaises(ValueError):
            self.stage(self.source, linked_destination/'nested'/'stage')
        self.assertEqual(list((outside/'nested').iterdir()), [])

    def test_published_extra_directory_and_changed_bytes_fail_quality(self):
        (self.published/'extra').mkdir(); self.assertTrue(self.issues()); (self.published/'extra').rmdir()
        member = next(self.published.glob('*.wav'))
        data = member.read_bytes(); member.write_bytes(data[:-1]+bytes([data[-1]^1]))
        self.assertTrue(self.issues())

    def test_hidden_duplicate_unlabelled_or_autoplay_controls_fail_on_both_pages(self):
        for page, prefix in self.pages:
            valid = self.html(prefix)
            for invalid in (valid.replace('<audio ', '<audio hidden ', 1),
                            '<section inert>'+valid+'</section>', '<template>'+valid+'</template>',
                            '<dialog>'+valid+'</dialog>', '<section style="display:none">'+valid+'</section>',
                            '<details><summary>Audio</summary>'+valid+'</details>',
                            '<details><div><summary>'+valid+'</summary></div></details>',
                            '<details><summary>Audio</summary><summary>'+valid+'</summary></details>',
                            valid.replace('controls ', '', 1), valid.replace('<audio ', '<audio autoplay ', 1),
                            valid.replace('aria-label="sample', 'aria-label="" data-unused="sample', 1),
                            valid+valid[:valid.index('</audio>')+8],
                            valid.replace('<a ', '<a hidden ', 1)):
                with self.subTest(page=page.name, invalid=invalid[:40]):
                    page.write_text(invalid); self.assertTrue(self.issues())
            page.write_text(valid)

    def test_open_details_and_visible_summary_links_are_accessible(self):
        for page, prefix in self.pages:
            page.write_text('<details open><summary>Audio</summary>'+self.html(prefix)+'</details>')
        self.assertEqual(self.issues(), [])
        parser = quality.VisibleMediaParser()
        parser.feed('<details><summary><a href="visible">Open</a></summary>'
                    '<a href="hidden">Hidden</a></details>')
        self.assertEqual(parser.links, {'visible'})

    def test_false_phase_clock_score_boolean_count_and_pcm_rms_fail(self):
        cases = (('scores', 'valid_observation_count', True),
                 ('scores', 'filtered_missing_rmse_deg', 0),
                 ('frames', 'available_time_s', 0),
                 ('frames', 'state_phase', 'posterior'),
                 ('frames', 'rms_before_export', 0),
                 ('frames', 'last_valid_measurement_time_s', 99))
        for section, key, value in cases:
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(self.original)
                target = self.manifest['pcm_analysis'][section]
                if section == 'frames': target[key][0] = value
                else: target[key] = value
                self.save(); self.rejected()


class MovingPublicationTests(TrackingPublicationTests):
    directory = 'moving_audio'
    checker = staticmethod(quality.check_moving_audio)
    stage = staticmethod(build_site.stage_moving_audio)
    quality_root_name = 'MOVING_AUDIO_ROOT'

    @classmethod
    def setUpClass(cls):
        from codes.chapters.ch09.examples.moving_source_audio import generate
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'assets'; generate(source)
            cls.assets = {p.name: p.read_bytes() for p in source.iterdir()}
        cls.original = json.loads(cls.assets['MANIFEST.json'])
        cls.sources = {name: (ROOT/name).read_bytes() for name in cls.original['source_sha256']}

    def test_false_phase_clock_score_boolean_count_and_pcm_rms_fail(self):
        self.manifest['truth']['angle_degrees_from_positive_y'][0] = 0
        self.save(); self.rejected()


class IndependentFrameArithmeticTests(unittest.TestCase):
    def test_tampered_pcm_statistics_fail_without_replay(self):
        source = ROOT/'codes/chapters/ch09/tracking_audio'
        original = json.loads((source/'MANIFEST.json').read_text())
        quality._check_tracking_frame_arithmetic(original, source/'array_noisy.wav')
        for section, field, value in (
            ('scores', 'valid_observation_count', True),
            ('scores', 'raw_valid_rmse_deg', 0),
            ('frames', 'rms_before_export', 0),
            ('frames', 'state_time_s', float('nan')),
            ('frames', 'truth_angle_deg', 0)):
            with self.subTest(field=field):
                altered = copy.deepcopy(original)
                if section == 'frames': altered['pcm_analysis'][section][field][0] = value
                else: altered['pcm_analysis'][section][field] = value
                with self.assertRaises(ValueError):
                    quality._check_tracking_frame_arithmetic(altered, source/'array_noisy.wav')
