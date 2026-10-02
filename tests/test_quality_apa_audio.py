"""Exercise the APA publication gate with real PCM and isolated source copies."""

import copy
import hashlib
import io
import json
import math
import struct
import tempfile
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts import quality_check


ROOT = Path(__file__).resolve().parents[1]
WAV_NAMES = {
    "apa_reference.wav", "apa_true_echo.wav", "apa_microphone.wav",
    "apa_nlms_residual.wav", "apa_apa2_residual.wav", "apa_apa4_residual.wav",
}
SOURCE_NAMES = {
    "codes/chapters/ch06/core/apa_audio.py",
    "codes/chapters/ch06/examples/generate_apa_audio.py",
    "codes/chapters/ch06/core/aec.py",
    "codes/chapters/ch06/core/aec_numeric.py",
    "codes/chapters/ch06/core/aec_affine_projection.py",
    "codes/chapters/ch02/core/conventions.py",
    "codes/chapters/ch00/core/audio_samples.py",
    "codes/chapters/ch00/io_contracts.py",
}


class APAQualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_root = ROOT / "codes/chapters/ch06/apa_audio"
        cls.assets = {name: (asset_root / name).read_bytes()
                      for name in WAV_NAMES | {"MANIFEST.json"}}
        cls.actual_sources = {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}
        cls.original_manifest = json.loads(cls.assets["MANIFEST.json"])

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / "codes/chapters/ch06/apa_audio"
        self.site = self.root / "site"
        self.published = self.site / "apa_audio"
        self.source.mkdir(parents=True)
        self.published.mkdir(parents=True)
        (self.site / "research").mkdir()
        for name, data in self.actual_sources.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.manifest = copy.deepcopy(self.original_manifest)
        for name, data in self.assets.items():
            (self.source / name).write_bytes(data)
            (self.published / name).write_bytes(data)
        self.pages = ((self.site / "06_aec.html", ""),
                      (self.site / "research/05_exercises_and_audio.html", "../"))
        for page, prefix in self.pages:
            page.write_text(self.valid_html(prefix), encoding="utf-8")
        self.stack.enter_context(patch.object(quality_check, "ROOT", self.root))
        self.stack.enter_context(patch.object(quality_check, "SITE", self.site))
        self.stack.enter_context(patch.object(quality_check, "APA_AUDIO_ROOT", self.source))

    @staticmethod
    def player(prefix, name):
        return (f'<audio controls preload="none" aria-label="APA {name}" '
                f'src="{prefix}apa_audio/{name}"></audio>')

    @classmethod
    def valid_html(cls, prefix):
        return "\n".join(cls.player(prefix, name) for name in sorted(WAV_NAMES)) + (
            f'\n<a href="{prefix}apa_audio/MANIFEST.json">Independent manifest</a>')

    def errors(self):
        errors = []
        quality_check.check_apa_audio(errors)
        return errors

    def rejected(self, message):
        errors = self.errors()
        self.assertTrue(errors, "Malformed APA fixture was accepted")
        self.assertTrue(any(message in error for error in errors), errors)

    def save_manifest(self):
        data = json.dumps(self.manifest).encode("utf-8")
        for folder in (self.source, self.published):
            (folder / "MANIFEST.json").write_bytes(data)

    def replace_pcm(self, name, data):
        for folder in (self.source, self.published):
            (folder / name).write_bytes(data)
        self.manifest["files"][name]["sha256"] = hashlib.sha256(data).hexdigest()
        self.save_manifest()

    def test_real_sources_pcm_and_integer_scores_form_a_valid_fixture(self):
        self.assertEqual(set(self.manifest["source_sha256"]), SOURCE_NAMES)
        for name, data in self.actual_sources.items():
            self.assertEqual(hashlib.sha256(data).hexdigest(),
                             self.manifest["source_sha256"][name])
        for name in WAV_NAMES:
            with self.subTest(name=name):
                with wave.open(io.BytesIO(self.assets[name]), "rb") as wav:
                    self.assertEqual((wav.getframerate(), wav.getnchannels(),
                                      wav.getsampwidth(), wav.getnframes()), (16000, 1, 2, 32013))
                    values = struct.unpack("<32013h", wav.readframes(32013))
                squared = sum(value * value for value in values[24000:32000])
                score = self.manifest["pcm_measurements"]["powers"][name[4:-4]]
                self.assertEqual(score["integer_squared_sum"], squared)
                self.assertEqual(score["sample_denominator"], 8000)
                self.assertEqual(score["mean_square"], squared / (8000 * 32768**2))
        self.assertEqual(self.errors(), [])

    def test_actual_published_html_passes_without_touching_repository_files(self):
        for page, _ in self.pages:
            page.write_bytes((ROOT / "site" / page.relative_to(self.site)).read_bytes())
        self.assertEqual(self.errors(), [])

    def test_check_preserves_temporary_bytes_and_mtimes(self):
        paths = [path for path in self.root.rglob("*") if path.is_file()]
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(self.errors(), [])
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}, before)

    def test_missing_generation_source_is_rejected(self):
        (self.root / "codes/chapters/ch06/core/aec_numeric.py").unlink()
        self.rejected("aec_numeric.py")

    def test_missing_provenance_member_is_rejected(self):
        self.manifest["source_sha256"].pop("codes/chapters/ch06/core/aec_numeric.py")
        self.save_manifest()
        self.rejected("真实源集合")

    def test_changed_real_source_and_forged_source_digest_are_rejected(self):
        name = "codes/chapters/ch06/core/apa_audio.py"
        path = self.root / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n# altered fixture source\n")
        self.rejected("真实源码摘要过期")
        path.write_bytes(original)
        self.manifest["source_sha256"][name] = "0" * 64
        self.save_manifest()
        self.rejected("真实源码摘要过期")

    def test_missing_extra_and_symlink_members_are_rejected_in_both_directories(self):
        for folder in (self.source, self.published):
            for case in ("missing", "extra", "symlink"):
                with self.subTest(folder=folder.name, case=case):
                    path = folder / "apa_reference.wav"
                    if case == "extra":
                        altered = folder / "additional.wav"
                        altered.write_bytes(self.assets["apa_reference.wav"])
                    else:
                        altered = path
                        path.unlink()
                        if case == "symlink":
                            path.symlink_to((self.published if folder == self.source else self.source)
                                            / "apa_reference.wav")
                    self.rejected("恰含六普通WAV")
                    if altered.exists() or altered.is_symlink():
                        altered.unlink()
                    path.write_bytes(self.assets[path.name])

    def test_modified_published_copy_is_rejected(self):
        path = self.published / "apa_apa2_residual.wav"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        self.rejected("网页副本不同")

    def test_modified_pcm_with_updated_sha_still_fails_actual_integer_scoring(self):
        name = "apa_apa2_residual.wav"
        data = bytearray(self.assets[name])
        # Last holdout sample in the canonical 44-byte PCM header fixture.
        self.assertEqual(data[36:40], b"data")
        offset = 44 + 2 * 31999
        value = struct.unpack_from("<h", data, offset)[0]
        struct.pack_into("<h", data, offset, value + 1 if value < 32767 else value - 1)
        self.replace_pcm(name, bytes(data))
        self.rejected("实际整数评分不符")

    def test_actual_format_and_truncation_are_checked_after_digest_update(self):
        name = "apa_reference.wav"
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0\0" * 32013)
        self.replace_pcm(name, data.getvalue())
        self.rejected("实际PCM格式不同")
        self.replace_pcm(name, self.assets[name][:-2])
        self.rejected("实际PCM截断")

    def test_window_decode_divisor_and_power_integer_denominators_are_checked(self):
        for case in ("window", "decode", "sample", "squared", "mean"):
            with self.subTest(case=case):
                self.manifest = copy.deepcopy(self.original_manifest)
                measurements = self.manifest["pcm_measurements"]
                expected = "实际整数评分不符"
                if case == "window":
                    measurements["holdout_interval_samples"] = [24000, 32013]
                    expected = "评分窗或PCM解码分母"
                elif case == "decode":
                    measurements["pcm_decode_divisor"] = 32767
                    expected = "评分窗或PCM解码分母"
                else:
                    score = measurements["powers"]["nlms_residual"]
                    field = {"sample": "sample_denominator", "squared": "integer_squared_sum",
                             "mean": "mean_square"}[case]
                    score[field] += 1
                self.save_manifest()
                self.rejected(expected)

    def test_ratio_numerator_denominator_zero_status_and_db_are_checked(self):
        for field in ("integer_numerator", "integer_denominator", "zero_output",
                      "microphone_to_residual_total_power_ratio_db"):
            with self.subTest(field=field):
                self.manifest = copy.deepcopy(self.original_manifest)
                ratio = self.manifest["pcm_measurements"]["ratios"]["nlms_residual"]
                ratio[field] = True if field == "zero_output" else ratio[field] + 1
                self.save_manifest()
                self.rejected("实际总功率比" if field.endswith("_db") else "实际功率比分母")

    def test_nonzero_output_ratio_rejects_nonfinite_and_non_numeric_reports(self):
        for key in ("nlms_residual", "apa2_residual", "apa4_residual"):
            ratio = self.manifest["pcm_measurements"]["ratios"][key]
            self.assertGreater(ratio["integer_denominator"], 0)
            for reported in (float("nan"), float("inf"), -float("inf"),
                             "22.0492", "NaN", None, True, False):
                with self.subTest(key=key, reported=reported):
                    self.manifest = copy.deepcopy(self.original_manifest)
                    self.manifest["pcm_measurements"]["ratios"][key][
                        "microphone_to_residual_total_power_ratio_db"] = reported
                    self.save_manifest()
                    self.rejected("nonfinite JSON" if type(reported) is float and not math.isfinite(reported)
                                  else "实际总功率比不同")

    def test_finite_ratio_reports_inside_absolute_tolerance_are_legal(self):
        sums = {}
        for name in ("apa_microphone.wav", "apa_nlms_residual.wav",
                     "apa_apa2_residual.wav", "apa_apa4_residual.wav"):
            with wave.open(io.BytesIO(self.assets[name]), "rb") as wav:
                values = struct.unpack("<32013h", wav.readframes(32013))
            sums[name[4:-4]] = sum(value * value for value in values[24000:32000])
        for offset in (-4e-13, 0.0, 4e-13):
            with self.subTest(offset=offset):
                for key in ("nlms_residual", "apa2_residual", "apa4_residual"):
                    expected = 10 * math.log10(sums["microphone"] / sums[key])
                    self.manifest["pcm_measurements"]["ratios"][key][
                        "microphone_to_residual_total_power_ratio_db"] = expected + offset
                self.save_manifest()
                self.assertEqual(self.errors(), [])

    def test_hidden_ancestors_templates_and_comments_do_not_supply_players(self):
        wrappers = (('<div hidden><section>', '</section></div>'),
                    ('<section inert><div>', '</div></section>'),
                    ('<div aria-hidden="true"><section>', '</section></div>'),
                    ('<template><template>', '</template></template>'), ('<!--', '-->'))
        for page, prefix in self.pages:
            original = self.valid_html(prefix)
            for opening, closing in wrappers:
                with self.subTest(page=page.name, wrapper=opening):
                    page.write_text(opening + original + closing, encoding="utf-8")
                    self.rejected("可见播放器缺失或重复")
                    page.write_text(original, encoding="utf-8")

    def test_hidden_player_itself_does_not_count(self):
        for attribute in ("hidden", "inert", 'aria-hidden="true"'):
            page, prefix = self.pages[0]
            page.write_text(self.valid_html(prefix).replace("<audio ", f"<audio {attribute} ", 1))
            self.rejected("可见播放器缺失或重复")

    def test_missing_duplicate_or_wrong_member_player_is_rejected_on_either_page(self):
        name = "apa_reference.wav"
        for page, prefix in self.pages:
            original = self.valid_html(prefix)
            player = self.player(prefix, name)
            for case, replacement in (("missing", ""), ("duplicate", player + player),
                                      ("wrong", self.player(prefix, "apa_not_reference.wav"))):
                with self.subTest(page=page.name, case=case):
                    page.write_text(original.replace(player, replacement), encoding="utf-8")
                    self.rejected("可见播放器缺失或重复")
                    page.write_text(original, encoding="utf-8")

    def test_controls_preload_label_and_autoplay_contract_is_enforced(self):
        page, prefix = self.pages[0]
        original = self.valid_html(prefix)
        for case, changed in (("controls", original.replace(" controls", "", 1)),
                              ("preload", original.replace('preload="none"', 'preload="auto"', 1)),
                              ("label", original.replace('aria-label="APA apa_apa2_residual.wav"', '', 1)),
                              ("autoplay", original.replace("<audio ", "<audio autoplay ", 1))):
            with self.subTest(case=case):
                page.write_text(changed, encoding="utf-8")
                self.rejected("标签或控件不同")

    def test_hidden_template_or_commented_manifest_link_is_rejected(self):
        for page, prefix in self.pages:
            original = self.valid_html(prefix)
            link = f'<a href="{prefix}apa_audio/MANIFEST.json">Independent manifest</a>'
            for opening, closing in (("<!--", "-->"), ("<template>", "</template>"),
                                     ("<div hidden>", "</div>"), ("", "")):
                with self.subTest(page=page.name, wrapper=opening):
                    replacement = opening + link + closing if opening else ""
                    page.write_text(original.replace(link, replacement), encoding="utf-8")
                    self.rejected("独立清单链接缺失")
                    page.write_text(original, encoding="utf-8")

    def test_nearby_non_apa_players_and_closed_hidden_content_are_legal(self):
        for page, prefix in self.pages:
            neighbors = ('<div hidden><audio src="audio/unrelated.wav"></audio></div>'
                         '<template><audio src="apa_audio/ignored.wav"></audio></template>'
                         f'<audio controls src="{prefix}audio/apa_reference.wav"></audio>'
                         f'<audio controls src="{prefix}apa_audio_archive/sample.wav"></audio>')
            page.write_text(neighbors + self.valid_html(prefix), encoding="utf-8")
        self.assertEqual(self.errors(), [])

    def test_exact_zero_residual_with_explicit_undefined_db_is_legal(self):
        name, key = "apa_apa4_residual.wav", "apa4_residual"
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 32013)
        measurements = self.manifest["pcm_measurements"]
        measurements["powers"][key].update(integer_squared_sum=0, mean_square=0.0)
        measurements["ratios"][key].update(integer_denominator=0, zero_output=True,
                                             microphone_to_residual_total_power_ratio_db=None)
        self.replace_pcm(name, data.getvalue())
        self.assertEqual(self.errors(), [])
        # With zero actual output power, only the explicit undefined value is valid.
        for reported in (0, True, False, float("nan"), float("inf"), -float("inf"), "None"):
            with self.subTest(reported=reported):
                measurements["ratios"][key]["microphone_to_residual_total_power_ratio_db"] = reported
                self.save_manifest()
                self.rejected("nonfinite JSON" if type(reported) is float and not math.isfinite(reported)
                              else "实际总功率比不同")

    def test_valid_zero_microphone_pcm_cannot_define_a_power_ratio(self):
        name = "apa_microphone.wav"
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 32013)
        self.manifest["pcm_measurements"]["powers"]["microphone"].update(
            integer_squared_sum=0, mean_square=0.0)
        self.replace_pcm(name, data.getvalue())
        self.rejected("麦克风评分分母须严格正")


if __name__ == "__main__":
    unittest.main()
