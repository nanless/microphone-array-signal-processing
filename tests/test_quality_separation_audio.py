"""Publication rejects unsafe members, false PCM scores and invisible players."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_site, quality_check as quality

ROOT = Path(__file__).resolve().parents[1]


class MaskPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from codes.chapters.ch08.examples.mask_representation_demo import prepare_assets
        cls.original, cls.assets = prepare_assets()
        cls.sources = {name: (ROOT/name).read_bytes() for name in cls.original['source_sha256']}

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root/'codes/chapters/ch08/mask_audio'
        self.site = self.root/'site'
        self.published = self.site/'mask_audio'
        self.source.mkdir(parents=True); self.published.mkdir(parents=True)
        (self.site/'research').mkdir()
        for name, blob in self.sources.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        for name, blob in self.assets.items():
            (self.source/name).write_bytes(blob); (self.published/name).write_bytes(blob)
        self.manifest = copy.deepcopy(self.original)
        self.pages = ((self.site/'08_speech-separation.html', ''),
                      (self.site/'research/05_exercises_and_audio.html', '../'))
        for page, prefix in self.pages:
            page.write_text(self.html(prefix))
        for name, value in (('ROOT', self.root), ('SITE', self.site), ('MASK_AUDIO_ROOT', self.source)):
            patcher = patch.object(quality, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def html(self, prefix):
        return ''.join(f'<audio controls preload="none" aria-label="sample {name}" '
                       f'src="{prefix}mask_audio/{name}"></audio>' for name in sorted(self.original['files'])) + \
            f'<a href="{prefix}mask_audio/MANIFEST.json">Manifest</a>'

    def save(self):
        for folder in (self.source, self.published):
            (folder/'MANIFEST.json').write_text(json.dumps(self.manifest))

    def issues(self):
        errors = []; quality.check_mask_audio(errors)
        return errors

    def rejected(self):
        self.assertTrue(self.issues())
        with self.assertRaises(ValueError):
            build_site.stage_mask_audio(self.source, self.root/'stage')

    def test_valid_real_pcm_is_read_only_and_stages_all_members(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(), [])
        self.assertEqual(build_site.stage_mask_audio(self.source, self.root/'stage'), set(self.assets))
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_false_pcm_and_boolean_integer_are_rejected(self):
        scores = self.manifest['samples']['bounded_real']['pcm_measurements']
        for key, value in (('total_reference_mse', 0), ('integer_reference_squared_sum', True)):
            with self.subTest(key=key):
                original = scores[key]; scores[key] = value; self.save(); self.rejected()
                scores[key] = original

    def test_float_phase_or_model_tampering_is_rejected(self):
        self.manifest['samples']['bounded_real']['float_measurements']['phase_ls']['coefficients'][0] = 1
        self.save(); self.rejected()

    def test_pcm_phase_tampering_is_rejected(self):
        self.manifest['samples']['bounded_real']['pcm_measurements']['phase_ls']['coefficients'][0] = 1
        self.save(); self.rejected()

    def test_json_overflow_cannot_look_finite(self):
        self.manifest['samples']['bounded_real']['float_measurements']['peak'] = float('inf')
        text = json.dumps(self.manifest).replace('Infinity', '1e999')
        for folder in (self.source, self.published):
            (folder/'MANIFEST.json').write_text(text)
        self.rejected()

    def test_stale_source_is_rejected(self):
        self.manifest['source_sha256'][next(iter(self.sources))] = '0'*64
        self.save(); self.rejected()

    def test_missing_extra_nested_and_symlink_members_are_rejected(self):
        member = self.source/'mask_bounded_real.wav'
        blob = member.read_bytes()
        member.unlink(); self.rejected(); member.write_bytes(blob)
        extra = self.source/'extra.txt'; extra.write_text('extra'); self.rejected(); extra.unlink()
        extra.mkdir(); self.rejected(); extra.rmdir()
        member.unlink(); member.symlink_to(self.published/member.name); self.rejected()

    def test_changed_published_bytes_are_rejected(self):
        member = self.published/'mask_bounded_real.wav'
        blob = member.read_bytes(); member.write_bytes(blob[:-1]+bytes([blob[-1]^1]))
        self.assertTrue(self.issues())

    def test_hidden_inert_template_duplicate_or_missing_players_are_rejected(self):
        for page, prefix in self.pages:
            valid = self.html(prefix)
            for invalid in (valid.replace('<audio ', '<audio hidden ', 1),
                            '<template>'+valid+'</template>', '<noscript>'+valid+'</noscript>',
                            '<dialog>'+valid+'</dialog>',
                            '<section inert>'+valid+'</section>', '<section style="display:none">'+valid+'</section>',
                            valid.replace('<audio ', '<audio autoplay ', 1),
                            valid+valid[:valid.index('</audio>')+8],
                            valid.replace('controls ', '', 1)):
                with self.subTest(page=page.name, invalid=invalid[:30]):
                    page.write_text(invalid); self.assertTrue(self.issues())
            page.write_text(valid)

    def test_hidden_neighbor_does_not_hide_valid_controls(self):
        page, prefix = self.pages[0]
        page.write_text('<template><audio src="bad.wav"></audio></template>'+self.html(prefix))
        self.assertEqual(self.issues(), [])

    def test_open_dialog_keeps_controls_usable(self):
        for page, prefix in self.pages:
            page.write_text('<dialog open>'+self.html(prefix)+'</dialog>')
        self.assertEqual(self.issues(), [])


class GSSPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from codes.chapters.ch08.examples.gss_teaching_demo import generate
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'assets'; generate(source)
            cls.assets = {p.name: p.read_bytes() for p in source.iterdir()}
        cls.original = json.loads(cls.assets['MANIFEST.json'])
        cls.sources = {name: (ROOT/name).read_bytes() for name in cls.original['generator_inputs']}

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root/'codes/chapters/ch08/gss_audio'
        self.site = self.root/'site'; self.published = self.site/'gss_audio'
        self.source.mkdir(parents=True); self.published.mkdir(parents=True); (self.site/'research').mkdir()
        for name, blob in self.sources.items():
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        for name, blob in self.assets.items():
            (self.source/name).write_bytes(blob); (self.published/name).write_bytes(blob)
        self.pages = ((self.site/'08_speech-separation.html', ''),
                      (self.site/'research/05_exercises_and_audio.html', '../'))
        for page, prefix in self.pages:
            page.write_text(self.html(prefix))
        for name, value in (('ROOT', self.root), ('SITE', self.site), ('GSS_AUDIO_ROOT', self.source)):
            patcher = patch.object(quality, name, value); patcher.start(); self.addCleanup(patcher.stop)

    def html(self, prefix):
        return ''.join(f'<audio controls preload="none" aria-label="sample {name}" '
                       f'src="{prefix}gss_audio/{name}"></audio>' for name in sorted(self.assets) if name.endswith('.wav')) + \
            ''.join(f'<a href="{prefix}gss_audio/{name}">{name}</a>' for name in ('MANIFEST.json', 'STATE.npz'))

    def issues(self):
        errors = []; quality.check_gss_audio(errors); return errors

    def test_valid_pcm_state_replay_and_independent_score_pass(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(), [])
        build_site.stage_gss_audio(self.source, self.root/'stage')
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_extra_members_symlinks_and_false_scores_fail(self):
        member = self.source/'extra.txt'; member.write_text('extra')
        self.assertTrue(self.issues())
        with self.assertRaises(ValueError): build_site.stage_gss_audio(self.source, self.root/'stage')
        member.unlink(); member.mkdir(); self.assertTrue(self.issues()); member.rmdir()
        member = self.source/'enhanced_correct.wav'; member.unlink(); member.symlink_to(self.published/member.name)
        self.assertTrue(self.issues()); member.unlink(); member.write_bytes(self.assets[member.name])
        manifest = copy.deepcopy(self.original); manifest['pcm_si_sdr_db']['enhanced_correct'] = 0
        for folder in (self.source, self.published):
            (folder/'MANIFEST.json').write_text(json.dumps(manifest))
        self.assertTrue(self.issues())

    def test_hidden_players_and_invisible_manifest_fail(self):
        for page, prefix in self.pages:
            valid = self.html(prefix)
            page.write_text('<section inert>'+valid+'</section>'); self.assertTrue(self.issues())
            page.write_text(valid.replace('<a ', '<a hidden ', 1)); self.assertTrue(self.issues())
            page.write_text(valid)

    def test_published_symlink_is_rejected(self):
        member = self.published/'STATE.npz'; member.unlink(); member.symlink_to(self.source/member.name)
        self.assertTrue(self.issues())


class SeparationPublishSafetyTests(unittest.TestCase):
    def test_linked_media_parent_cannot_write_or_remove_external_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); site = root/'site'; outside = root/'other'
            site.mkdir(); outside.mkdir()
            staged = root/'staged.wav'; staged.write_bytes(b'PUBLISHED')
            target = outside/'mask_target.wav'; target.write_bytes(b'PREEXISTING')
            stale = outside/'stale.wav'; stale.write_bytes(b'KEEP')
            (site/'mask_audio').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError): build_site._validate_site_output(site)
            with self.assertRaises(ValueError):
                build_site.publish_files([(staged, site/'mask_audio/mask_target.wav')],
                                         [site/'mask_audio/stale.wav'], boundary=site)
            self.assertEqual(target.read_bytes(), b'PREEXISTING')
            self.assertEqual(stale.read_bytes(), b'KEEP')
            self.assertEqual(staged.read_bytes(), b'PUBLISHED')

    def test_linked_site_root_and_nested_ancestor_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); outside = root/'other'; outside.mkdir()
            (outside/'nested').mkdir()
            site = root/'site'; site.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError): build_site._validate_site_output(site)
            staged = root/'staged'; staged.write_bytes(b'PUBLISHED')
            with self.assertRaises(ValueError):
                build_site.publish_files([(staged, site/'nested/target')], boundary=site)
            self.assertFalse((outside/'nested/target').exists())


if __name__ == '__main__':
    unittest.main()
