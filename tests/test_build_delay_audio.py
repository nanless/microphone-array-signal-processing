"""Publish real fixed controls, with independent corruption and IO negatives."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_site, quality_check


class DelayPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        shutil.copytree(build_site.DELAY_AUDIO_ROOT, self.source)

    def test_complete_real_set_is_copied_without_source_writes(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.source.iterdir()}
        target = self.root / "published"
        self.assertEqual(build_site.stage_delay_audio(self.source, target), set(before))
        self.assertEqual(len(before), 7)
        for name, (data, mtime) in before.items():
            self.assertEqual((target / name).read_bytes(), data)
            self.assertEqual((self.source / name).read_bytes(), data)
            self.assertEqual((self.source / name).stat().st_mtime_ns, mtime)

    def test_changed_pcm_cannot_be_hidden_by_updating_its_digest(self):
        wav = self.source / "delay_common_residual.wav"
        data = bytearray(wav.read_bytes())
        data[-1] ^= 1
        wav.write_bytes(data)
        manifest = self.source / "MANIFEST.json"
        obj = json.loads(manifest.read_text())
        obj["files"][wav.name]["sha256"] = hashlib.sha256(data).hexdigest()
        manifest.write_text(json.dumps(obj))
        target = self.root / "published"
        with self.assertRaises(ValueError):
            build_site.stage_delay_audio(self.source, target)
        self.assertFalse(target.exists())

    def test_extra_member_and_linked_destination_are_rejected_before_copy(self):
        (self.source / "extra.txt").write_text("unexpected")
        target = self.root / "published"
        with self.assertRaises(ValueError):
            build_site.stage_delay_audio(self.source, target)
        self.assertFalse(target.exists())
        (self.source / "extra.txt").unlink()
        external = self.root / "external"
        external.mkdir()
        sentinel = external / "sentinel"
        sentinel.write_bytes(b"keep")
        before = sentinel.stat().st_mtime_ns
        alias = self.root / "alias"
        alias.symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            build_site.stage_delay_audio(self.source, alias / "published")
        self.assertEqual({p.name for p in external.iterdir()}, {"sentinel"})
        self.assertEqual(sentinel.stat().st_mtime_ns, before)

    def make_site(self):
        site = self.root / "site"
        (site / "research").mkdir(parents=True)
        build_site.stage_delay_audio(self.source, site / "delay_audio")
        for page, prefix in ((site / "07_wpe-dereverberation.html", ""),
                             (site / "research/05_exercises_and_audio.html", "../")):
            page.write_text("\n".join(
                f'<audio controls preload="none" aria-label="control" '
                f'src="{prefix}delay_audio/{name}"></audio>'
                for name in build_site.DELAY_AUDIO_WAVS)
                + f'\n<a href="{prefix}delay_audio/MANIFEST.json">Manifest</a>')
        return site

    def test_quality_reads_integer_pcm_and_rejects_inert_players(self):
        site = self.make_site()
        with patch.object(quality_check, "DELAY_AUDIO_ROOT", self.source), \
                patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_delay_audio(errors)
            self.assertEqual(errors, [])
            page = site / "research/05_exercises_and_audio.html"
            page.write_text("<template>" + page.read_text() + "</template>")
            quality_check.check_delay_audio(errors)
            self.assertTrue(errors)

    def test_truncated_decay_curve_is_rejected_before_png_identity(self):
        report = build_site.ROOT / "codes/chapters/ch07/reports/figure75_prediction_delays.json"
        obj = json.loads(report.read_bytes())
        obj["power_decay_control"]["relative_power"].pop()
        altered = json.dumps(obj).encode()
        original_read = Path.read_bytes

        def read(path):
            return altered if path == report else original_read(path)

        with patch.object(Path, "read_bytes", read):
            errors = []
            quality_check.check_delay_figure(errors)
        self.assertTrue(any("shorter" in error for error in errors), errors)

    def test_links_become_local_players_in_chapter_and_research(self):
        for source, href, expected in (
            (build_site.SRC / "07_wpe-dereverberation.md",
             "../codes/chapters/ch07/delay_audio/delay_reference.wav",
             "delay_audio/delay_reference.wav"),
            (build_site.ROOT / "codes/chapters/ch00/research/05_exercises_and_audio.md",
             "../../ch07/delay_audio/delay_reference.wav",
             "../delay_audio/delay_reference.wav"),
        ):
            with self.subTest(source=source):
                rendered = build_site.rewrite_site_links(
                    f'<p><a href="{href}">Reference</a></p>', source)
                self.assertIn(f'src="{expected}"', rendered)
                self.assertIn('preload="none"', rendered)
                self.assertIn('aria-label="Reference"', rendered)
                self.assertNotIn("autoplay", rendered)


if __name__ == "__main__":
    unittest.main()
