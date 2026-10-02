"""Independent PCM fixtures exercise known-component unitary focusing publication boundaries."""
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


class FocusPublicationTest(unittest.TestCase):
    NAMES = {"focus_reference.wav", "focus_delayed_source.wav", "focus_array.wav", "focus_known_focused.wav"}
    SOURCES = ("codes/chapters/ch04/core/focus_audio.py",
               "codes/chapters/ch04/examples/generate_focus_audio.py",
               "codes/chapters/ch03/core/geometry.py",
               "codes/chapters/ch02/core/conventions.py",
               "codes/chapters/ch00/core/audio_samples.py",
               'codes/chapters/ch00/io_contracts.py')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        payloads = {}
        channels = {"focus_reference.wav": 1, "focus_delayed_source.wav": 1, "focus_array.wav": 4, "focus_known_focused.wav": 4}
        for name in self.NAMES:
            data = io.BytesIO()
            with wave.open(data, "wb") as wav:
                wav.setnchannels(channels[name])
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x01\x00" * 32024 * channels[name])
            payloads[name] = data.getvalue()
            (self.source / name).write_bytes(payloads[name])
        self.manifest = {
            "sample_rate_hz": 16000, "common_export_gain": 1,
            "files": {name: {"channels": channels[name], "samples_per_channel": 32024,
                             "sha256": hashlib.sha256(payloads[name]).hexdigest()}
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
        self.assertEqual(build_site.stage_focus_audio(self.source, destination), expected)
        for name in expected:
            self.assertEqual((destination / name).read_bytes(), (self.source / name).read_bytes())

    def test_truncated_pcm_is_rejected_even_with_updated_digest(self):
        path = self.source / "focus_reference.wav"
        path.write_bytes(path.read_bytes()[:-16])
        self.manifest["files"]["focus_reference.wav"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "截断"):
            build_site.stage_focus_audio(self.source, self.root / "published")
        site = self.root / "site"
        (site / "research").mkdir(parents=True)
        shutil.copytree(self.source, site / "focus_audio")
        with patch.object(quality_check, "FOCUS_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_focus_audio(errors)
        self.assertTrue(any("截断" in error for error in errors), errors)

    def test_checksum_detects_a_changed_sample(self):
        path = self.source / "focus_reference.wav"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "摘要"):
            build_site.stage_focus_audio(self.source, self.root / "published")

    def test_format_is_checked_even_if_checksum_was_updated(self):
        path = self.source / "focus_reference.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0" * (32024 * 2))
        self.manifest["files"][path.name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "PCM格式"):
            build_site.stage_focus_audio(self.source, self.root / "published")

    def test_missing_extra_and_symlink_files_are_rejected(self):
        for case in ("missing", "extra", "symlink"):
            with self.subTest(case=case):
                source = self.root / case
                shutil.copytree(self.source, source)
                if case == "extra":
                    (source / "other.wav").write_bytes(b"not a WAV")
                else:
                    (source / "focus_reference.wav").unlink()
                    if case == "symlink":
                        (source / "focus_reference.wav").symlink_to(self.source / "focus_reference.wav")
                with self.assertRaisesRegex(ValueError, "普通WAV"):
                    build_site.stage_focus_audio(source, self.root / (case + "-out"))

    def test_wrong_gain_and_incomplete_provenance_are_rejected(self):
        self.manifest["common_export_gain"] = .5
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "共同增益"):
            build_site.stage_focus_audio(self.source, self.root / "wrong-gain")
        self.manifest["common_export_gain"] = 1
        self.manifest["source_sha256"].pop(self.SOURCES[0])
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "真实生成源"):
            build_site.stage_focus_audio(self.source, self.root / "missing-source")

    def test_changed_generation_source_is_rejected(self):
        self.manifest["source_sha256"][self.SOURCES[0]] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "源摘要过期"):
            build_site.stage_focus_audio(self.source, self.root / "out")

    def make_published_site(self):
        site = self.root / "site"
        (site / "research").mkdir(parents=True)
        shutil.copytree(self.source, site / "focus_audio")
        for page, prefix in ((site / "04_doa-estimation.html", ""),
                             (site / "research/05_exercises_and_audio.html", "../")):
            page.write_text("\n".join(
                f'<audio controls preload="none" aria-label="cue" src="{prefix}focus_audio/{name}"></audio>'
                for name in self.NAMES) +
                f'\n<a href="{prefix}focus_audio/MANIFEST.json">Manifest</a>', encoding="utf-8")
        return site

    def test_independent_quality_check_valid_and_corrupt_copy(self):
        site = self.make_published_site()
        with patch.object(quality_check, "FOCUS_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_focus_audio(errors)
            self.assertEqual(errors, [])
            (site / "focus_audio/focus_known_focused.wav").write_bytes(b"changed published copy")
            quality_check.check_focus_audio(errors)
            self.assertTrue(any("副本不同" in error for error in errors))

    def test_independent_quality_requires_players_in_both_pages(self):
        site = self.make_published_site()
        (site / "research/05_exercises_and_audio.html").write_text("No cues here", encoding="utf-8")
        with patch.object(quality_check, "FOCUS_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_focus_audio(errors)
            self.assertTrue(any("缺少已知酉聚焦播放器" in error for error in errors))

    def test_comments_and_players_without_controls_do_not_pass(self):
        for replacement, expected in (("comment", "集合"), ("controls", "控制")):
            with self.subTest(replacement=replacement):
                site = self.root / replacement
                shutil.copytree(self.make_published_site() if not (self.root / "site").exists() else self.root / "site", site)
                page = site / "04_doa-estimation.html"
                text = page.read_text()
                page.write_text("<!--" + text + "-->" if replacement == "comment" else text.replace(" controls", ""))
                with patch.object(quality_check, "FOCUS_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
                    errors = []
                    quality_check.check_focus_audio(errors)
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_manifest_link_inside_comment_does_not_pass(self):
        site = self.make_published_site()
        page = site / "04_doa-estimation.html"
        text = page.read_text()
        text = text.replace('<a href="focus_audio/MANIFEST.json">Manifest</a>',
                            '<!--<a href="focus_audio/MANIFEST.json">Manifest</a>-->')
        page.write_text(text)
        with patch.object(quality_check, "FOCUS_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_focus_audio(errors)
        self.assertTrue(any("缺少已知酉聚焦独立清单" in error for error in errors), errors)

    def test_inert_template_players_and_manifest_are_not_usable(self):
        site = self.make_published_site()
        for page in (site / "04_doa-estimation.html", site / "research/05_exercises_and_audio.html"):
            page.write_text("<template><template>" + page.read_text() + "</template></template>")
        with patch.object(quality_check, "FOCUS_AUDIO_ROOT", self.source), patch.object(quality_check, "SITE", site):
            errors = []
            quality_check.check_focus_audio(errors)
        self.assertTrue(any("缺少已知酉聚焦播放器" in error for error in errors), errors)

    def test_site_digest_binds_focus_pcm_and_independent_manifest(self):
        # A repository-local temporary tree is needed for relative hash paths.
        # Nothing is published, and existing assets remain untouched.
        with tempfile.TemporaryDirectory(prefix=".focus-digest-", dir=build_site.ROOT) as temporary:
            folder = Path(temporary)
            (folder / "sample.wav").write_bytes(b"original PCM fixture")
            manifest = folder / "MANIFEST.json"
            manifest.write_text('{"sample": 1}')
            with patch.object(build_site, "FOCUS_AUDIO_ROOT", folder), patch.object(quality_check, "FOCUS_AUDIO_ROOT", folder):
                before = build_site.source_digest()
                self.assertEqual(before, quality_check.site_source_digest())
                (folder / "sample.wav").write_bytes(b"changed PCM fixture")
                changed_pcm = build_site.source_digest()
                self.assertNotEqual(before, changed_pcm)
                self.assertEqual(changed_pcm, quality_check.site_source_digest())
                manifest.write_text('{"sample": 2}')
                changed_manifest = build_site.source_digest()
                self.assertNotEqual(changed_pcm, changed_manifest)
                self.assertEqual(changed_manifest, quality_check.site_source_digest())

    def test_new_media_links_are_local_and_have_accessible_controls(self):
        text = '<p><a href="../codes/chapters/ch04/focus_audio/focus_known_focused.wav">Four-channel known-component focus</a></p>'
        rendered = build_site.rewrite_site_links(text, build_site.SRC / "04_doa-estimation.md")
        self.assertIn('href="focus_audio/focus_known_focused.wav"', rendered)
        self.assertIn('src="focus_audio/focus_known_focused.wav"', rendered)
        self.assertIn('preload="none"', rendered)
        self.assertIn('aria-label="Four-channel known-component focus"', rendered)
        self.assertNotIn("autoplay", rendered)


if __name__ == "__main__":
    unittest.main()
