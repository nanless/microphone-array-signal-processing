"""Publication boundaries for independent E04-24 coherent-reflection assets."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch04.examples import generate_reflection_audio as generator
from scripts import build_site, quality_check


class ReflectionPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        generator.generate_assets(self.source)

    def site(self):
        destination = self.root/'site'
        (destination/'research').mkdir(parents=True)
        build_site.stage_reflection_audio(self.source, destination/'reflection_audio')
        for page, prefix in ((destination/'04_doa-estimation.html', ''),
                             (destination/'research/05_exercises_and_audio.html', '../')):
            page.write_text('\n'.join(
                f'<audio controls preload="none" aria-label="{name}" src="{prefix}reflection_audio/{name}"></audio>'
                for name in quality_check.REFLECTION_AUDIO_CHANNELS) +
                f'<a href="{prefix}reflection_audio/MANIFEST.json">Manifest</a>')
        return destination

    def test_stage_copies_verified_assets_and_quality_checks_actual_media(self):
        site = self.site()
        for name in generator.MEMBERS:
            self.assertEqual((self.source/name).read_bytes(), (site/'reflection_audio'/name).read_bytes())
        errors = []
        with patch.object(quality_check, 'REFLECTION_AUDIO_ROOT', self.source), patch.object(quality_check, 'SITE', site):
            quality_check.check_reflection_audio(errors)
        self.assertEqual(errors, [])

    def test_source_replay_rejects_changed_sample_even_if_manifest_hash_is_changed(self):
        file = self.source/'reflection_direct.wav'
        file.write_bytes(file.read_bytes()[:-2]+b'\x01\x00')
        import hashlib
        manifest = json.loads((self.source/'MANIFEST.json').read_text())
        manifest['files'][file.name]['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        (self.source/'MANIFEST.json').write_text(json.dumps(manifest))
        destination = self.root/'rejected'
        with self.assertRaises(ValueError):
            build_site.stage_reflection_audio(self.source, destination)
        self.assertFalse(destination.exists())

    def test_preflight_rejects_extra_members_and_linked_destinations(self):
        (self.source/'extra').write_text('keep')
        with self.assertRaises(ValueError):
            build_site.stage_reflection_audio(self.source, self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())
        (self.source/'extra').unlink()
        linked = self.root/'linked'
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            build_site.stage_reflection_audio(self.source, linked)

    def test_quality_rejects_modified_copy_and_missing_visible_player(self):
        site = self.site()
        with patch.object(quality_check, 'REFLECTION_AUDIO_ROOT', self.source), patch.object(quality_check, 'SITE', site):
            original = (site/'reflection_audio/reflection_mixed.wav').read_bytes()
            (site/'reflection_audio/reflection_mixed.wav').write_bytes(original[:-2]+b'\x01\x00')
            errors = []
            quality_check.check_reflection_audio(errors)
            self.assertTrue(errors)
            (site/'reflection_audio/reflection_mixed.wav').write_bytes(original)
            page = site/'04_doa-estimation.html'
            page.write_text(page.read_text().replace('src="reflection_audio/reflection_mixed.wav"', 'src="reflection_audio/wrong.wav"'))
            errors = []
            quality_check.check_reflection_audio(errors)
            self.assertTrue(errors)

    def test_links_publish_correct_relative_players_in_chapter_and_research(self):
        for current, prefix, target in [(build_site.SRC/'04_doa-estimation.md', '', '../codes/chapters/ch04/reflection_audio'),
                                        (build_site.RESEARCH_ROOT/'05_exercises_and_audio.md', '../', '../../ch04/reflection_audio')]:
            source = f'<a href="{target}/reflection_reference.wav">Source</a>\n<a href="{target}/MANIFEST.json">Manifest</a>'
            html = build_site.rewrite_site_links(source, current)
            self.assertIn(f'src="{prefix}reflection_audio/reflection_reference.wav"', html)
            self.assertIn(f'href="{prefix}reflection_audio/MANIFEST.json"', html)

    def test_quality_independently_rejects_forged_pcm_integer_power(self):
        site = self.site()
        manifest = json.loads((self.source/'MANIFEST.json').read_text())
        row = next(row for row in manifest['samples'].values()
                   if row['file'] == 'reflection_mixed.wav')
        row['pcm_integer_measurements']['integer_squared_sum_E_per_channel'][0] += 1
        # The publisher's independent integer loop must catch a bad score even
        # when the separate replay validator is replaced by this controlled fixture.
        errors = []
        with patch.object(quality_check, 'REFLECTION_AUDIO_ROOT', self.source), \
             patch.object(quality_check, 'SITE', site), \
             patch.object(generator, 'check_assets', return_value=manifest):
            quality_check.check_reflection_audio(errors)
        self.assertTrue(any('independent integer scoring' in error for error in errors))

    def test_every_reflection_member_contributes_to_both_site_source_digests(self):
        # Only this family is redirected. Every other source path remains the
        # real read-only repository path; changes occur in a temporary copy.
        # Both digest implementations encode paths relative to ROOT. A
        # temporary replacement ROOT would alter other inputs, so this small
        # disposable asset copy lives inside ROOT and is removed afterwards.
        with tempfile.TemporaryDirectory(prefix='.reflection-digest-', dir=build_site.ROOT) as directory:
            temporary = Path(directory)/'reflection_audio'
            shutil.copytree(self.source, temporary)
            with patch.object(build_site, 'REFLECTION_AUDIO_ROOT', temporary), patch.object(quality_check, 'REFLECTION_AUDIO_ROOT', temporary):
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
