"""Independent audio publishing checks using isolated copies of real fixtures."""

import hashlib
import json
import shutil
import tempfile
import unittest
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock

from scripts import build_site, quality_check


ROOT = Path(__file__).resolve().parents[1]


class MediaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.players = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "audio":
            self.players.append(dict(attrs))
        elif tag == "a":
            self.links.append(dict(attrs).get("href"))


class AudioQualityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "codes/audio"
        self.site = self.root / "site"
        shutil.copytree(ROOT / "codes/audio", self.source)
        shutil.copytree(ROOT / "codes/audio", self.site / "audio")
        self.manifest_path = self.source / "MANIFEST.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for name in self.manifest["generator_inputs"]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.page = self.site / "research/05_exercises_and_audio.html"
        self.page.parent.mkdir(parents=True)
        self.page.write_text("\n".join(
            '<audio controls preload="none" aria-label="试听样本" '
            f'src="../audio/{escape(record["file"], quote=True)}"></audio>'
            for record in self.manifest["files"]
        ), encoding="utf-8")

    def errors(self):
        errors = []
        with mock.patch.object(quality_check, "ROOT", self.root), \
                mock.patch.object(quality_check, "SITE", self.site):
            quality_check.check_audio(errors)
        return errors

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def assert_rejected(self, text):
        self.assertTrue(any(text in error for error in self.errors()), text)

    def test_real_23_sample_fixture_passes_without_writes(self):
        self.assertEqual(len(self.manifest["files"]), 23)
        before = {p.relative_to(self.root): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.errors(), [])
        after = {p.relative_to(self.root): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_source_wav_is_rejected(self):
        (self.source / self.manifest["files"][0]["file"]).unlink()
        self.assert_rejected("文件集合不符")

    def test_missing_site_wav_is_rejected(self):
        (self.site / "audio" / self.manifest["files"][0]["file"]).unlink()
        self.assert_rejected("文件集合不符")

    def test_wrong_wav_digest_is_rejected(self):
        self.manifest["files"][0]["sha256"] = "0" * 64
        self.save_manifest()
        self.assert_rejected("摘要或站点副本不符")

    def test_changed_site_copy_is_rejected(self):
        path = self.site / "audio" / self.manifest["files"][0]["file"]
        path.write_bytes(path.read_bytes() + b"changed")
        self.assert_rejected("摘要或站点副本不符")

    def test_changed_generator_source_is_rejected(self):
        path = self.root / next(iter(self.manifest["generator_inputs"]))
        path.write_bytes(path.read_bytes() + b"\n# changed fixture\n")
        self.assert_rejected("音频生成源过期")

    def test_incorrect_frame_count_is_rejected(self):
        self.manifest["files"][0]["samples"] += 1
        self.save_manifest()
        self.assert_rejected("音频尺寸不符")

    def test_missing_generator_provenance_is_rejected(self):
        self.manifest["generator_inputs"] = {}
        self.save_manifest()
        self.assert_rejected("音频生成来源清单不完整")

    def test_incorrect_group_gain_is_rejected(self):
        self.manifest["files"][0]["common_export_gain"] *= .5
        self.save_manifest()
        self.assert_rejected("音频比较组增益不一致")

    def test_invalid_quantization_error_is_rejected(self):
        self.manifest["files"][0]["quantization_max_abs_error"] = 1 / 32768
        self.save_manifest()
        self.assert_rejected("音频量化误差超限")

    def test_missing_player_is_rejected(self):
        self.page.write_text("\n".join(self.page.read_text().splitlines()[1:]), encoding="utf-8")
        self.assert_rejected("试听控件集合不符")

    def test_autoplay_even_false_attribute_is_rejected(self):
        self.page.write_text(self.page.read_text().replace("<audio ", '<audio autoplay="false" ', 1), encoding="utf-8")
        self.assert_rejected("不自动播放或预加载")

    def test_missing_controls_label_or_wrong_preload_is_rejected(self):
        original = self.page.read_text()
        for before, after in ((" controls", ""), ('aria-label="试听样本"', 'aria-label=""'),
                              ('preload="none"', 'preload="auto"')):
            with self.subTest(attribute=before):
                self.page.write_text(original.replace(before, after, 1), encoding="utf-8")
                self.assert_rejected("试听控件必须有标签和控制")


class AudioLinkTest(unittest.TestCase):
    def rewrite(self, href, source, label="合成音"):
        html = build_site.rewrite_site_links(
            f'<a href="{escape(href, quote=True)}">{label}</a>', source)
        parsed = MediaParser()
        parsed.feed(html)
        return parsed

    def test_chapter_and_research_local_wavs_get_safe_controls(self):
        cases = [
            (ROOT / "chapters/01_problem-definition.md", "../codes/audio/spatial_reference.wav", "audio/spatial_reference.wav"),
            (ROOT / "codes/research/05_exercises_and_audio.md", "../audio/spatial_reference.wav", "../audio/spatial_reference.wav"),
        ]
        for source, href, output in cases:
            with self.subTest(source=source):
                parsed = self.rewrite(href, source, '目标 <strong>参考</strong> &amp; 对照')
                self.assertEqual(parsed.links, [output])
                self.assertEqual(len(parsed.players), 1)
                player = parsed.players[0]
                self.assertEqual(player["src"], output)
                self.assertEqual(player["aria-label"], "目标 参考 & 对照")
                self.assertEqual(player["preload"], "none")
                self.assertIn("controls", player)
                self.assertNotIn("autoplay", player)

    def test_external_wavs_are_unchanged_and_not_players(self):
        source = ROOT / "codes/research/05_exercises_and_audio.md"
        for href in ("https://example.org/audio/spatial_reference.wav",
                     "http://example.org/audio/spatial_reference.wav",
                     "//example.org/audio/spatial_reference.wav"):
            with self.subTest(href=href):
                parsed = self.rewrite(href, source)
                self.assertEqual(parsed.links, [href])
                self.assertEqual(parsed.players, [])

    def test_similar_but_disallowed_local_paths_are_not_players(self):
        source = ROOT / "codes/research/05_exercises_and_audio.md"
        for href in ("../audio/spatial_reference.wav?download=1",
                     "../audio/spatial_reference.wav#time", "../audio/MANIFEST.json",
                     "../audio/subdir/spatial_reference.wav", "../upstream/spatial_reference.wav",
                     "../audio/spatial-reference.wav", "../audio/spatial_reference.WAV",
                     "../../../../outside/audio/spatial_reference.wav"):
            with self.subTest(href=href):
                self.assertEqual(self.rewrite(href, source).players, [])


if __name__ == "__main__":
    unittest.main()
