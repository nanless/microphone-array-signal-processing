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
        self.source = self.root / "codes/chapters/ch02/real_audio"
        self.site = self.root / "site"
        shutil.copytree(ROOT / "codes/chapters/ch02/real_audio", self.source)
        self.site.mkdir()
        build_site.stage_real_audio(self.source, self.site / "real_audio")
        for name in ("codes/chapters/ch02/core/real_recordings.py", "codes/chapters/ch02/examples/prepare_real_recordings.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.page = self.site / "research/05_exercises_and_audio.html"
        self.page.parent.mkdir(parents=True)
        self.page.write_text("\n".join(
            f'<audio controls preload="none" aria-label="录音" src="../real_audio/{n}"></audio>'
            for n in sorted(NAMES) if "16ch" not in n), encoding="utf-8")

    def errors(self):
        with patch.object(quality_check, "ROOT", self.root), \
                patch.object(quality_check, "REAL_AUDIO_ROOT", self.source), \
                patch.object(quality_check, "SITE", self.site):
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
        for name in files - {"README.md"}:
            self.assertEqual((self.source / name).read_bytes(), (destination / name).read_bytes())
        original = (self.source / "README.md").read_text(encoding="utf-8")
        published = (destination / "README.md").read_text(encoding="utf-8")
        for old, new in build_site.REAL_AUDIO_SOURCE_LINKS.items():
            self.assertIn(f"]({old})", original)
            self.assertNotIn(f"]({old})", published)
            self.assertIn(f"]({new})", published)
        self.assertIn("](ATTRIBUTION.txt)", published)
        self.assertIn("](demand_nriver_ch01_10s.wav)", published)

    def test_missing_or_similar_source_link_cannot_be_published(self):
        readme = self.source / "README.md"
        original = readme.read_text(encoding="utf-8")
        readme.write_text(original.replace("](../core/real_recordings.py)",
                                           "](../core/real_recording.py)"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "缺少唯一的源码链接"):
            build_site.stage_real_audio(self.source, self.root / "missing-link")
        readme.write_text(original + "\n[近似文件](../core/real_recordings.py.bak)\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "失效的本地链接"):
            build_site.stage_real_audio(self.source, self.root / "similar-link")

    def test_quality_gate_rejects_wrong_code_url_and_other_broken_local_link(self):
        published = self.site / "real_audio/README.md"
        original = published.read_text(encoding="utf-8")
        published.write_text(original.replace("/core/real_recordings.py)",
                                              "/core/real_recordings.py.bak)"), encoding="utf-8")
        self.assertTrue(any("发布说明与定向转换后的源文件不符" in issue for issue in self.errors()))
        published.write_text(original + "\n[缺失的音频](missing.wav)\n", encoding="utf-8")
        self.assertTrue(any("失效的本地链接：missing.wav" in issue for issue in self.errors()))
        published.write_text(original + "\n[不安全链接](javascript:alert(1))\n", encoding="utf-8")
        self.assertTrue(any("不安全链接" in issue for issue in self.errors()))

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
        path = self.root / "codes/chapters/ch02/core/real_recordings.py"
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
        output = build_site.rewrite_site_links(
            '<a href="../../ch02/real_audio/demand_nriver_ch01_10s.wav">录音</a>'
            '<a href="../../ch02/real_audio/ATTRIBUTION.txt">署名</a>',
            ROOT / "codes/chapters/ch00/research/05_exercises_and_audio.md")
        self.assertIn('src="../real_audio/demand_nriver_ch01_10s.wav"', output)
        self.assertIn('href="../real_audio/ATTRIBUTION.txt"', output)
        self.assertNotIn("autoplay", output)

    def test_multichannel_input_stays_a_download_link(self):
        output = build_site.rewrite_site_links(
            '<a href="../../ch02/real_audio/demand_nriver_16ch_10s.wav">16 通道输入</a>',
            ROOT / "codes/chapters/ch00/research/05_exercises_and_audio.md")
        self.assertIn('href="../real_audio/demand_nriver_16ch_10s.wav"', output)
        self.assertNotIn("<audio", output)


if __name__ == "__main__":
    unittest.main()
