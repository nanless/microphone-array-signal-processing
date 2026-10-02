"""Independent WAV fixtures exercise publication and rejection of cue assets."""
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import build_site, quality_check


class BinauralPublicationTest(unittest.TestCase):
    NAMES = {"reference.wav", "itd_only.wav", "ild_only.wav",
             "consistent.wav", "conflicting.wav"}
    SOURCES = ("codes/chapters/ch01/core/binaural_cues.py",
               "codes/chapters/ch01/examples/generate_binaural_cues.py",
               "codes/chapters/ch00/core/audio_samples.py",
               'codes/chapters/ch00/io_contracts.py')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x01\x00\x02\x00" * 32008)
        encoded = data.getvalue()
        for name in self.NAMES:
            (self.source / name).write_bytes(encoded)
        self.manifest = {
            "sample_rate_hz": 16000, "common_export_gain": 1,
            "files": {name: {"channels": 2, "samples_per_channel": 32008,
                             "sha256": hashlib.sha256(encoded).hexdigest()}
                      for name in self.NAMES},
            "source_sha256": {name: hashlib.sha256((build_site.ROOT / name).read_bytes()).hexdigest()
                              for name in self.SOURCES},
        }
        self.save_manifest()

    def save_manifest(self):
        (self.source / "MANIFEST.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_files_are_copied_byte_for_byte(self):
        destination = self.root / "published"
        expected = self.NAMES | {"MANIFEST.json"}
        self.assertEqual(build_site.stage_binaural_audio(self.source, destination), expected)
        for name in expected:
            self.assertEqual((destination / name).read_bytes(), (self.source / name).read_bytes())

    def test_truncated_pcm_is_rejected_even_with_updated_digest(self):
        path = self.source / "reference.wav"
        path.write_bytes(path.read_bytes()[:-16])
        self.manifest["files"]["reference.wav"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "截断"):
            build_site.stage_binaural_audio(self.source, self.root / "published")
        site = self.root / "site"
        (site / "research").mkdir(parents=True)
        shutil.copytree(self.source, site / "binaural_audio")
        with patch.object(quality_check, "BINAURAL_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_binaural_audio(errors)
        self.assertTrue(any("截断" in error for error in errors), errors)

    def test_checksum_detects_a_changed_sample(self):
        path = self.source / "reference.wav"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "摘要"):
            build_site.stage_binaural_audio(self.source, self.root / "published")

    def test_format_is_checked_even_if_checksum_was_updated(self):
        path = self.source / "reference.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0" * (32008 * 4))
        self.manifest["files"][path.name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "PCM格式"):
            build_site.stage_binaural_audio(self.source, self.root / "published")

    def test_missing_extra_and_symlink_files_are_rejected(self):
        for case in ("missing", "extra", "symlink"):
            with self.subTest(case=case):
                source = self.root / case
                shutil.copytree(self.source, source)
                if case == "extra":
                    (source / "other.wav").write_bytes(b"not a WAV")
                else:
                    (source / "reference.wav").unlink()
                    if case == "symlink":
                        (source / "reference.wav").symlink_to(self.source / "reference.wav")
                with self.assertRaisesRegex(ValueError, "普通WAV"):
                    build_site.stage_binaural_audio(source, self.root / (case + "-out"))

    def test_wrong_gain_and_incomplete_provenance_are_rejected(self):
        self.manifest["common_export_gain"] = .5
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "共同增益"):
            build_site.stage_binaural_audio(self.source, self.root / "wrong-gain")
        self.manifest["common_export_gain"] = 1
        self.manifest["source_sha256"].pop(self.SOURCES[0])
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "真实生成源"):
            build_site.stage_binaural_audio(self.source, self.root / "missing-source")

    def test_changed_generation_source_is_rejected(self):
        self.manifest["source_sha256"][self.SOURCES[0]] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "源摘要过期"):
            build_site.stage_binaural_audio(self.source, self.root / "out")

    def make_published_site(self):
        site = self.root / "site"
        (site / "research").mkdir(parents=True)
        shutil.copytree(self.source, site / "binaural_audio")
        for page, prefix in ((site / "01_problem-definition.html", ""),
                             (site / "research/05_exercises_and_audio.html", "../")):
            page.write_text("\n".join(
                f'<audio controls preload="none" aria-label="cue" src="{prefix}binaural_audio/{name}"></audio>'
                for name in self.NAMES) +
                f'\n<a href="{prefix}binaural_audio/MANIFEST.json">Manifest</a>', encoding="utf-8")
        return site

    def test_independent_quality_check_valid_and_corrupt_copy(self):
        site = self.make_published_site()
        with patch.object(quality_check, "BINAURAL_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_binaural_audio(errors)
            self.assertEqual(errors, [])
            (site / "binaural_audio/itd_only.wav").write_bytes(b"changed published copy")
            quality_check.check_binaural_audio(errors)
            self.assertTrue(any("副本不同" in error for error in errors))

    def test_independent_quality_requires_players_in_both_pages(self):
        site = self.make_published_site()
        (site / "research/05_exercises_and_audio.html").write_text("No cues here", encoding="utf-8")
        with patch.object(quality_check, "BINAURAL_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_binaural_audio(errors)
            self.assertTrue(any("缺少双耳线索播放器" in error for error in errors))

    def test_comments_and_players_without_controls_do_not_pass(self):
        for replacement, expected in (("comment", "集合"), ("controls", "控制")):
            with self.subTest(replacement=replacement):
                site = self.root / replacement
                shutil.copytree(self.make_published_site() if not (self.root / "site").exists() else self.root / "site", site)
                page = site / "01_problem-definition.html"
                text = page.read_text()
                page.write_text("<!--" + text + "-->" if replacement == "comment" else text.replace(" controls", ""))
                with patch.object(quality_check, "BINAURAL_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
                    errors = []
                    quality_check.check_binaural_audio(errors)
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_new_media_links_are_local_and_have_accessible_controls(self):
        text = '<p><a href="../codes/chapters/ch01/binaural_audio/itd_only.wav">Only ITD</a></p>'
        rendered = build_site.rewrite_site_links(text, build_site.SRC / "01_problem-definition.md")
        self.assertIn('href="binaural_audio/itd_only.wav"', rendered)
        self.assertIn('src="binaural_audio/itd_only.wav"', rendered)
        self.assertIn('preload="none"', rendered)
        self.assertIn('aria-label="Only ITD"', rendered)
        self.assertNotIn("autoplay", rendered)


if __name__ == "__main__":
    unittest.main()
