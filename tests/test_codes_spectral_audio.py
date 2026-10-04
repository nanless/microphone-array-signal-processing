"""Exact PCM/provenance contracts for four independent E01-10 source controls."""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

from codes.chapters.ch01.examples.generate_spectral_cues import (
    MEMBERS, check_assets, generate_assets,
)
from scripts import build_site, quality_check


class SpectralAudioTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.manifest = generate_assets(self.source)

    def snapshot(self):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.source.iterdir() if p.is_file()}

    def test_independent_pcm_shape_complete_tail_and_integer_energies(self):
        expected_energy = {'flat': (386556004800, 386556004800),
                           'tilted': (492118106400, 126383569200)}
        for name in sorted(MEMBERS - {'MANIFEST.json'}):
            raw = (self.source/name).read_bytes()
            channels = 1 if name.endswith('_source.wav') else 2
            frames = 32000 if channels == 1 else 32001
            with wave.open(io.BytesIO(raw), 'rb') as stream:
                self.assertEqual((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                                  stream.getnframes(), stream.getcomptype()), (16000, channels, 2, frames, 'NONE'))
                data = stream.readframes(frames)
            values = [int.from_bytes(data[i:i+2], 'little', signed=True) for i in range(0, len(data), 2)]
            self.assertEqual(hashlib.sha256(raw).hexdigest(), self.manifest['files'][name]['sha256'])
            if channels == 2:
                self.assertNotEqual(values[-2], 0)
                self.assertEqual(values[-2], -values[-1])
                # Fixed integer expectations came from an independent scalar
                # sine calculation and ties-to-even quantization, before generation.
                energy = tuple(sum(values[2*n+c]**2 for n in range(1600, 30400)) for c in range(2))
                scene = name.split('_')[0]
                self.assertEqual(energy, expected_energy[scene])
                score = self.manifest['samples'][scene]['pcm_integer_measurements']
                self.assertEqual(score['integer_denominator_D'], 28800*32768**2)
                self.assertEqual(score['left_squared_sum_E'], energy[0])
                self.assertEqual(score['right_squared_sum_E'], energy[1])
                self.assertAlmostEqual(score['ild_right_minus_left_db'], 10*math.log10(energy[1]/energy[0]))

    def test_success_and_failures_never_repair_assets(self):
        before = self.snapshot()
        check_assets(self.source)
        self.assertEqual(before, self.snapshot())
        target = self.source/'flat_stereo.wav'
        payload = target.read_bytes()
        target.write_bytes(payload[:-2] + bytes([payload[-2] ^ 1, payload[-1]]))
        changed = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'Stale or modified'):
            check_assets(self.source)
        self.assertEqual(changed, self.snapshot())
        target.write_bytes(payload)
        (self.source/'extra.wav').write_bytes(b'not part of this experiment')
        changed = self.snapshot()
        with self.assertRaises(ValueError):
            check_assets(self.source)
        self.assertEqual(changed, self.snapshot())

    def test_metadata_types_sources_scores_and_duplicate_json_are_rejected(self):
        path = self.source/'MANIFEST.json'
        original = path.read_bytes()
        changes = [lambda m: m.__setitem__('common_export_gain', True),
                   lambda m: m['source_sha256'].__setitem__(next(iter(m['source_sha256'])), '0'*64),
                   lambda m: m['samples']['flat']['pcm_integer_measurements'].__setitem__('left_squared_sum_E', 1),
                   lambda m: m['files']['flat_source.wav'].__setitem__('samples_per_channel', 32001)]
        for change in changes:
            data = json.loads(original)
            change(data)
            path.write_text(json.dumps(data), encoding='utf-8')
            before = self.snapshot()
            with self.assertRaises(ValueError):
                check_assets(self.source)
            self.assertEqual(before, self.snapshot())
        path.write_bytes(b'{"sample_rate_hz":16000,"sample_rate_hz":32000}')
        with self.assertRaises(ValueError):
            check_assets(self.source)

    def test_linked_member_and_linked_parent_are_rejected_without_touching_target(self):
        target = self.root/'external.wav'
        target.write_bytes((self.source/'flat_source.wav').read_bytes())
        original = target.read_bytes()
        (self.source/'flat_source.wav').unlink()
        (self.source/'flat_source.wav').symlink_to(target)
        with self.assertRaises(ValueError):
            check_assets(self.source)
        self.assertEqual(target.read_bytes(), original)
        alias = self.root/'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            generate_assets(alias)
        self.assertEqual(target.read_bytes(), original)

    def test_missing_cli_check_never_creates_directory(self):
        absent = self.root/'absent'
        result = subprocess.run([sys.executable, '-m',
                                 'codes.chapters.ch01.examples.generate_spectral_cues',
                                 '--check', '--output', str(absent)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(absent.exists())

    def test_publication_replays_and_copies_exact_bytes_before_visible_quality_check(self):
        published = self.root/'site'
        published.mkdir()
        destination = published/'spectral_audio'
        self.assertEqual(build_site.stage_spectral_audio(self.source, destination), MEMBERS)
        for name in MEMBERS:
            self.assertEqual((self.source/name).read_bytes(), (destination/name).read_bytes())
        (published/'research').mkdir()
        for page, prefix in ((published/'01_problem-definition.html', ''),
                             (published/'research/05_exercises_and_audio.html', '../')):
            controls = ''.join(f'<audio controls preload="none" aria-label="known FIR" src="{prefix}spectral_audio/{name}"></audio>'
                               for name in MEMBERS if name.endswith('.wav'))
            page.write_text(controls + f'<a href="{prefix}spectral_audio/MANIFEST.json">Manifest</a>', encoding='utf-8')
        with patch.object(quality_check, 'SITE', published), patch.object(quality_check, 'SPECTRAL_AUDIO_ROOT', self.source):
            errors = []
            quality_check.check_spectral_audio(errors)
            self.assertEqual(errors, [])
            (destination/'flat_stereo.wav').write_bytes(b'modified published PCM')
            quality_check.check_spectral_audio(errors)
            self.assertTrue(errors)

    def test_bad_source_does_not_create_publication_destination(self):
        path = self.source/'tilted_source.wav'
        path.write_bytes(path.read_bytes()[:-2])
        destination = self.root/'published'
        with self.assertRaises(ValueError):
            build_site.stage_spectral_audio(self.source, destination)
        self.assertFalse(destination.exists())

    def test_chapter_and_research_links_make_local_players_but_unknown_assets_do_not(self):
        html = '<a href="../codes/chapters/ch01/spectral_audio/flat_stereo.wav">Known response</a>'
        rendered = build_site.rewrite_site_links(html, build_site.SRC/'01_problem-definition.md')
        self.assertIn('src="spectral_audio/flat_stereo.wav"', rendered)
        self.assertNotIn('autoplay', rendered)
        unknown = '<a href="spectral_audio/not_a_fixture.wav">Unknown</a>'
        # Link rewriting recognizes only the fixed members; the general player
        # formatter is deliberately not used to validate an asset's existence.
        unknown = build_site.rewrite_site_links(unknown, build_site.SRC/'01_problem-definition.md')
        self.assertNotIn('src=', unknown)


if __name__ == '__main__':
    unittest.main()
