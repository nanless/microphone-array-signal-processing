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
    DEFAULT_REPORT, ROOT, digest, record_final_source_status, validate_measurements, verify_wave,
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
                     "silence_sample_peak": 0, "silence_true_peak": 0},
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
