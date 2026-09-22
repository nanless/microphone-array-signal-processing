"""Read-only real-recording publication checks with independent release inventory."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_site, quality_check

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "demand_nriver_16ch_10s.wav", "demand_nriver_ch01_10s.wav",
    "demand_nriver_mean02_10s.wav", "demand_nriver_mean16_10s.wav",
}


class RealAudioPublishingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "codes/real_audio"
        self.site = self.root / "site"
        shutil.copytree(ROOT / "codes/real_audio", self.source)
        shutil.copytree(self.source, self.site / "real_audio")
        for name in ("codes/array_tutorial/real_recordings.py", "codes/examples/prepare_real_recordings.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.page = self.site / "research/05_exercises_and_audio.html"
        self.page.parent.mkdir(parents=True)
        self.page.write_text("\n".join(
            f'<audio controls preload="none" aria-label="录音" src="../real_audio/{n}"></audio>'
            for n in sorted(NAMES) if "16ch" not in n), encoding="utf-8")

    def errors(self):
        with patch.object(quality_check, "ROOT", self.root), patch.object(quality_check, "SITE", self.site):
            errors = []
            quality_check.check_real_audio(errors)
            return errors

    def mutate_record(self, field, value):
        path = self.source / "MANIFEST.json"
        data = json.loads(path.read_text())
        data["files"][0][field] = value
        path.write_text(json.dumps(data), encoding="utf-8")
        shutil.copy2(path, self.site / "real_audio/MANIFEST.json")

    def test_release_inventory_and_readonly_check(self):
        self.assertEqual(build_site.REAL_AUDIO_WAVS, NAMES)
        self.assertEqual(set(quality_check.EXPECTED_REAL_AUDIO_CHANNELS), NAMES)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.errors(), [])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_staging_keeps_license_and_bytes(self):
        destination = self.root / "staged"
        files = build_site.stage_real_audio(self.source, destination)
        self.assertEqual(files, NAMES | {"MANIFEST.json", "README.md", "LICENSE.txt", "ATTRIBUTION.txt"})
        for name in files:
            self.assertEqual((self.source / name).read_bytes(), (destination / name).read_bytes())

    def test_staging_rejects_path_injection(self):
        self.mutate_record("file", "../escape.wav")
        with self.assertRaises(ValueError):
            build_site.stage_real_audio(self.source, self.root / "staged")

    def test_missing_attribution_blocks_release(self):
        (self.source / "ATTRIBUTION.txt").unlink()
        self.assertTrue(self.errors())
        with self.assertRaises(ValueError):
            build_site.stage_real_audio(self.source, self.root / "staged")

    def test_changed_audio_blocks_staging(self):
        self.mutate_record("sha256", "0" * 64)
        self.assertTrue(self.errors())
        with self.assertRaises(ValueError):
            build_site.stage_real_audio(self.source, self.root / "staged")

    def test_bad_rms_is_rejected(self):
        self.mutate_record("rms", [float("nan")])
        self.assertTrue(self.errors())

    def test_changed_generator_is_rejected(self):
        path = self.root / "codes/array_tutorial/real_recordings.py"
        path.write_text(path.read_text() + "\n# fixture change\n")
        self.assertTrue(any("生成源过期" in error for error in self.errors()))

    def test_bad_peak_is_rejected(self):
        self.mutate_record("peak", float("nan"))
        self.assertTrue(self.errors())

    def test_bad_gain_is_rejected(self):
        self.mutate_record("common_export_gain", .5)
        self.assertTrue(self.errors())

    def test_bad_duration_is_rejected(self):
        self.mutate_record("duration_s", 11)
        self.assertTrue(self.errors())

    def test_unknown_player_is_rejected(self):
        with self.page.open("a") as stream:
            stream.write('<audio src="https://example.org/unlicensed.wav"></audio>')
        self.assertTrue(self.errors())

    def test_autoplay_is_rejected(self):
        self.page.write_text(self.page.read_text().replace("controls", "autoplay controls", 1))
        self.assertTrue(self.errors())

    def test_rewritten_links_keep_real_audio_and_license_local(self):
        with patch.object(build_site, "ROOT", self.root), patch.object(build_site, "SRC", self.root / "chapters"):
            output = build_site.rewrite_site_links(
                '<a href="../real_audio/demand_nriver_ch01_10s.wav">录音</a>'
                '<a href="../real_audio/ATTRIBUTION.txt">署名</a>',
                self.root / "codes/research/05_exercises_and_audio.md")
        self.assertIn('src="../real_audio/demand_nriver_ch01_10s.wav"', output)
        self.assertIn('href="../real_audio/ATTRIBUTION.txt"', output)
        self.assertNotIn("autoplay", output)

    def test_multichannel_input_stays_a_download_link(self):
        with patch.object(build_site, "ROOT", self.root), patch.object(build_site, "SRC", self.root / "chapters"):
            output = build_site.rewrite_site_links(
                '<a href="../real_audio/demand_nriver_16ch_10s.wav">16 通道输入</a>',
                self.root / "codes/research/05_exercises_and_audio.md")
        self.assertIn('href="../real_audio/demand_nriver_16ch_10s.wav"', output)
        self.assertNotIn("<audio", output)


if __name__ == "__main__":
    unittest.main()
