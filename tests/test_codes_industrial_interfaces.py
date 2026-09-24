"""Offline checks of the native experiment's independent acceptance criteria.

These tests never compile, download or execute external libraries.  The small
fixture is intentionally not copied from a measured report: its expectations
come from rate arithmetic, amplitude scaling and a seven-frame PCM sequence.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from codes.examples.run_industrial_interfaces import (
    DEFAULT_REPORT, ROOT, digest, record_final_source_status, validate_measurements,
    validate_variable_ratio_trace, verify_wave,
)


def valid_fixture():
    steady = {"output_frames": 16000, "before_flush_frames": 15900, "flush_frames": 100,
              "delay_after_flush_output_samples": 0, "partial_consumption_calls": 2}
    return {
        "src": {"whole": deepcopy(steady), "chunk127": deepcopy(steady), "chunk509": deepcopy(steady),
                "chunk127_max_abs_error": 0, "chunk509_max_abs_error": 0,
                "impulse_output_peak_frame": 4000, "sine_max_abs_error_frames_320_to_15680": 0,
                "clear_at_input_frame_24000_without_flush": {"output_frames": 15900},
                "reset_equal_index_overlap_max_abs_difference": 0.1},
        "loudness": {"full_lufs": -20, "half_lufs": -20 + 20 * math.log10(0.5),
                     "half_minus_full_lu": 20 * math.log10(0.5), "chunk127_minus_whole_lu": 0,
                     "full_sample_peak": 0.1, "half_sample_peak": 0.05,
                     "full_true_peak": 0.1, "half_true_peak": 0.05,
                     "silence_lufs": None, "silence_reason": "negative_infinity_no_gated_energy",
                     "silence_sample_peak": 0, "silence_true_peak": 0,
                     "intersample_12khz": {"frequency_hz": 12000, "fade_samples_each_end": 4800,
                                            "phase_rad": math.pi/4,
                                            "sample_peak": .95/math.sqrt(2),
                                            "true_peak": .94637, "chunk127_true_peak": .94637}},
        "variable_ratio": {
            "input_frames": 480060, "change_at_output_frame": 80000,
            "slew_output_frames": 160, "initial_input_output_ratio": 48004.8/16000,
            "final_input_output_ratio": 48007.2/16000,
            "marker_input_frames": [round(48004.8*s) if s <= 5 else
                                    240024+round(48007.2*(s-5)) for s in range(1, 10)],
            "constant_100ppm": {"consumed_input_frames": 480060, "output_frames": 160004,
                                "before_flush_frames": 160000, "flush_frames": 4,
                                "delay_after_flush_output_samples": 100,
                                "marker_output_frames": [16002,32002,48002,64002,80002,
                                                         96003,112004,128005,144005]},
            **{name: {"consumed_input_frames": 480060, "output_frames": 160000,
                      "before_flush_frames": 159900, "flush_frames": 100,
                      "delay_after_flush_output_samples": 100,
                      "marker_output_frames": [16002,32002,48002,64002,80002,
                                               96002,112002,128002,144002]}
               for name in ("step_150ppm", "slew_150ppm", "slew_150ppm_chunk127",
                            "slew320_150ppm")},
            "chunk127_max_abs_error": 0,
            "step_vs_slew_equal_index_max_abs_difference": .0006,
            "step_vs_slew320_equal_index_max_abs_difference": .0012,
        },
        "pcm16": {"frame_read_counts": [3, 3, 1, 0], "item_read_counts": [6, 6, 2, 0],
                  "pcm_interleaved": [-32768, 0, 0, 32767, 16384, -16384, 1, -1,
                                      12345, -23456, 0, 1000, 32767, -32768],
                  "float_max_abs_error": 0},
    }


class TestIndustrialAcceptance(unittest.TestCase):
    def test_independent_valid_fixture(self):
        validate_measurements(valid_fixture())

    def test_incomplete_flush_is_rejected(self):
        data = valid_fixture()
        data["src"]["chunk127"]["output_frames"] = 15900
        with self.assertRaisesRegex(ValueError, "duration"):
            validate_measurements(data)

    def test_wrong_chunk_state_is_rejected(self):
        data = valid_fixture()
        data["src"]["chunk509_max_abs_error"] = 0.01
        with self.assertRaisesRegex(ValueError, "chunking"):
            validate_measurements(data)

    def test_reset_control_must_actually_change_output(self):
        data = valid_fixture()
        data["src"]["clear_at_input_frame_24000_without_flush"]["output_frames"] = 16000
        data["src"]["reset_equal_index_overlap_max_abs_difference"] = 0
        with self.assertRaisesRegex(ValueError, "reset"):
            validate_measurements(data)

    def test_amplitude_and_power_decibels_cannot_be_swapped(self):
        data = valid_fixture()
        data["loudness"]["half_lufs"] = -20 + 10 * math.log10(0.5)
        with self.assertRaisesRegex(ValueError, "half-amplitude"):
            validate_measurements(data)

    def test_intersample_peak_needs_interpolation_and_continuous_state(self):
        for field, bad in (("sample_peak", .95), ("true_peak", .67),
                           ("chunk127_true_peak", .8)):
            with self.subTest(field=field):
                data = valid_fixture()
                data["loudness"]["intersample_12khz"][field] = bad
                with self.assertRaisesRegex(ValueError, "peak|sinusoid"):
                    validate_measurements(data)

    def test_variable_ratio_wrong_direction_or_missing_marker_fails(self):
        for field, bad in (("final_input_output_ratio", 1/3.00045),
                           ("marker_input_frames", [0]*9)):
            with self.subTest(field=field):
                data = valid_fixture()
                data["variable_ratio"][field] = bad
                with self.assertRaisesRegex(ValueError, "variable-ratio"):
                    validate_measurements(data)

    def test_variable_ratio_trace_reconciles_input_and_output(self):
        data = valid_fixture()
        for name in ("constant_100ppm", "step_150ppm", "slew_150ppm",
                     "slew_150ppm_chunk127", "slew320_150ppm"):
            data["variable_ratio"][name]["input_calls"] = 1
            data["variable_ratio"][name]["flush_calls"] = 1
        lines = ["case,phase,input_position_frames,idone_frames,odone_frames,delay_output_samples"]
        for name, out in (("vr_constant100ppm", 160004), ("vr_step150ppm", 160000),
                          ("vr_slew160", 160000), ("vr_slew160_chunk127", 160000),
                          ("vr_slew320", 160000)):
            flush = 4 if name == "vr_constant100ppm" else 100
            lines.extend((f"{name},input,0,480060,{out-flush},100",
                          f"{name},flush,480060,0,{flush},100"))
        validate_variable_ratio_trace("\n".join(lines), data)
        with self.assertRaisesRegex(ValueError, "trace totals"):
            validate_variable_ratio_trace("\n".join(lines).replace("480060,159900", "480059,159900"), data)

    def test_silence_cannot_be_zero_lufs_or_unexplained_null(self):
        for value, reason in ((0, "negative_infinity_no_gated_energy"), (None, "")):
            with self.subTest(value=value, reason=reason):
                data = valid_fixture()
                data["loudness"].update(silence_lufs=value, silence_reason=reason)
                with self.assertRaisesRegex(ValueError, "silence"):
                    validate_measurements(data)

    def test_nonfinite_json_numbers_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                data = valid_fixture()
                data["loudness"]["half_sample_peak"] = value
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    validate_measurements(data)

    def test_frames_and_items_cannot_be_interchanged(self):
        data = valid_fixture()
        data["pcm16"]["frame_read_counts"] = [6, 6, 2, 0]
        with self.assertRaisesRegex(ValueError, "frame/item"):
            validate_measurements(data)

    def test_pcm_channel_order_is_checked(self):
        data = valid_fixture()
        data["pcm16"]["pcm_interleaved"][:2] = [0, -32768]
        with self.assertRaisesRegex(ValueError, "interleaving"):
            validate_measurements(data)

    def test_pcm_signed_endpoint_normalization_is_exact(self):
        self.assertEqual(-32768 / 32768, -1)
        self.assertEqual(32767 / 32768, 0.999969482421875)
        data = valid_fixture()
        data["pcm16"]["float_max_abs_error"] = 1 / 32768
        with self.assertRaisesRegex(ValueError, "normalization"):
            validate_measurements(data)

    def test_standard_library_decoder_accepts_only_explicit_fixture(self):
        values = (-32768, 0, 0, 32767, 16384, -16384, 1, -1,
                  12345, -23456, 0, 1000, 32767, -32768)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.wav"
            for channels in (2, 1):
                with wave.open(str(path), "wb") as output:
                    output.setparams((channels, 2, 16000, 0, "NONE", "not compressed"))
                    output.writeframes(struct.pack("<14h", *values))
                if channels == 2:
                    self.assertEqual(verify_wave(path)["status"], "passed")
                else:
                    with self.assertRaises((ValueError, struct.error)):
                        verify_wave(path)

    def test_published_report_is_bound_to_current_sources(self):
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        for artifact in report["artifacts"].values():
            self.assertEqual(artifact["sha256"], digest(ROOT / artifact["path"]))
        lock = json.loads((ROOT / "codes/SOURCES.lock.json").read_text(encoding="utf-8"))
        revisions = {project["id"]: project["revision"] for project in lock["projects"]}
        for name, source in report["sources"].items():
            self.assertEqual(source["revision"], revisions[name])
            self.assertEqual(source["final_status"], "source_verified")
        validate_measurements(report["measurements"])
        self.assertNotIn("/Users/", json.dumps(report))
        self.assertTrue(all(command["returncode"] == 0 for command in report["commands"]))

    def test_post_build_source_failure_revokes_success(self):
        report = {"status": "passed", "sources": {"probe": {}}}
        with patch("codes.examples.run_industrial_interfaces.inspect_project",
                   return_value={"status": "source_selection_mismatch"}):
            errors = record_final_source_status(report, {"probe": {}}, Path("unused"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["sources"]["probe"]["final_status"], "source_selection_mismatch")
        self.assertEqual(len(errors), 1)

    def test_post_build_exception_preserves_primary_error(self):
        report = {"status": "failed", "failure": "original compilation failure", "sources": {"probe": {}}}
        with patch("codes.examples.run_industrial_interfaces.inspect_project",
                   side_effect=ValueError("dirty checkout")):
            errors = record_final_source_status(report, {"probe": {}}, Path("unused"))
        self.assertEqual(report["failure"], "original compilation failure")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["sources"]["probe"]["final_status"], "verification_failed")
        self.assertIn("dirty checkout", errors[0])


if __name__ == "__main__":
    unittest.main()
