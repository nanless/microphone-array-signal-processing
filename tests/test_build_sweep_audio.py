"""Publication boundaries for independent E02-19 excitation/response assets."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch02.examples import generate_sweep_audio as generator
from scripts import build_site, quality_check


class SweepPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        generator.generate_assets(self.source)

    def site(self):
        destination = self.root/'site'
        (destination/'research').mkdir(parents=True)
        build_site.stage_sweep_audio(self.source, destination/'sweep_audio')
        for page, prefix in ((destination/'02_basics-signal-model.html', ''),
                             (destination/'research/05_exercises_and_audio.html', '../')):
            page.write_text('\n'.join(
                f'<audio controls preload="none" aria-label="{name}" src="{prefix}sweep_audio/{name}"></audio>'
                for name in generator.WAV_LENGTHS) +
                f'<a href="{prefix}sweep_audio/MANIFEST.json">Manifest</a>')
        return destination

    def test_stage_copies_verified_assets_and_quality_checks_actual_media(self):
        site = self.site()
        for name in generator.MEMBERS:
            self.assertEqual((self.source/name).read_bytes(), (site/'sweep_audio'/name).read_bytes())
        errors = []
        with patch.object(quality_check, 'SWEEP_AUDIO_ROOT', self.source), patch.object(quality_check, 'SITE', site):
            quality_check.check_sweep_audio(errors)
        self.assertEqual(errors, [])

    def test_source_replay_rejects_changed_sample_even_if_manifest_hash_is_changed(self):
        file = self.source/'sweep_complete.wav'
        file.write_bytes(file.read_bytes()[:-2]+b'\x01\x00')
        import hashlib
        manifest = json.loads((self.source/'MANIFEST.json').read_text())
        manifest['files'][file.name]['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        (self.source/'MANIFEST.json').write_text(json.dumps(manifest))
        destination = self.root/'rejected'
        with self.assertRaises(ValueError):
            build_site.stage_sweep_audio(self.source, destination)
        self.assertFalse(destination.exists())

    def test_preflight_rejects_extra_members_and_linked_destinations(self):
        (self.source/'extra').write_text('keep')
        with self.assertRaises(ValueError):
            build_site.stage_sweep_audio(self.source, self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())
        (self.source/'extra').unlink()
        linked = self.root/'linked'
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            build_site.stage_sweep_audio(self.source, linked)

    def test_quality_rejects_modified_copy_and_missing_visible_player(self):
        site = self.site()
        with patch.object(quality_check, 'SWEEP_AUDIO_ROOT', self.source), patch.object(quality_check, 'SITE', site):
            original = (site/'sweep_audio/sweep_cut.wav').read_bytes()
            (site/'sweep_audio/sweep_cut.wav').write_bytes(original[:-2]+b'\x01\x00')
            errors = []
            quality_check.check_sweep_audio(errors)
            self.assertTrue(errors)
            (site/'sweep_audio/sweep_cut.wav').write_bytes(original)
            page = site/'02_basics-signal-model.html'
            page.write_text(page.read_text().replace('src="sweep_audio/sweep_cut.wav"', 'src="sweep_audio/wrong.wav"'))
            errors = []
            quality_check.check_sweep_audio(errors)
            self.assertTrue(errors)

    def test_links_publish_correct_relative_players_in_chapter_and_research(self):
        for current, prefix, target in [(build_site.SRC/'02_basics-signal-model.md', '', '../codes/chapters/ch02/sweep_audio'),
                                        (build_site.RESEARCH_ROOT/'05_exercises_and_audio.md', '../', '../../ch02/sweep_audio')]:
            source = f'<a href="{target}/sweep_source.wav">Source</a>\n<a href="{target}/MANIFEST.json">Manifest</a>'
            html = build_site.rewrite_site_links(source, current)
            self.assertIn(f'src="{prefix}sweep_audio/sweep_source.wav"', html)
            self.assertIn(f'href="{prefix}sweep_audio/MANIFEST.json"', html)

    def test_every_sweep_member_contributes_to_both_site_source_digests(self):
        # Only this family is redirected. Every other source path remains the
        # real read-only repository path; changes occur in a temporary copy.
        # Both digest implementations encode paths relative to ROOT. A
        # temporary replacement ROOT would alter other inputs, so this small
        # disposable asset copy lives inside ROOT and is removed afterwards.
        with tempfile.TemporaryDirectory(prefix='.sweep-digest-', dir=build_site.ROOT) as directory:
            temporary = Path(directory)/'sweep_audio'
            shutil.copytree(self.source, temporary)
            with patch.object(build_site, 'SWEEP_AUDIO_ROOT', temporary), patch.object(quality_check, 'SWEEP_AUDIO_ROOT', temporary):
                baseline = build_site.source_digest()
                self.assertEqual(baseline, quality_check.site_source_digest())
                for name in sorted(generator.MEMBERS):
                    path = temporary/name
                    original = path.read_bytes()
                    path.write_bytes(original+b'changed')
                    self.assertNotEqual(build_site.source_digest(), baseline)
                    self.assertEqual(build_site.source_digest(), quality_check.site_source_digest())
                    path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
