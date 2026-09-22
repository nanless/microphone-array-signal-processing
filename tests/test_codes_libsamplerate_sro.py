"""Independent clock arithmetic and malformed-trace checks for the SRC probe."""

import csv
from pathlib import Path
import shutil
import tempfile
import unittest

from codes.examples.run_libsamplerate_sro import (
    ROOT, clock_frame, inspect_trace, run, validate_clock_result,
)


class TestClockReference(unittest.TestCase):
    def test_piecewise_clock_indices_from_hand_arithmetic(self):
        # 16 kHz * 1.0001 = 16001.6; after 5 s, 16002.4 frames/s.
        self.assertEqual([clock_frame(t) for t in (1, 3, 4, 5, 6, 8, 9, 10)],
                         [16002, 48005, 64006, 80008, 96010, 128015, 144018, 160020])

    def test_expected_residual_is_not_generated_by_src(self):
        raw = {1: 2, 3: 5, 4: 6, 6: 10, 8: 15, 9: 18}
        fixed = {1: 0, 3: 0, 4: 0, 6: 0, 8: 0, 9: 0}
        measurement = validate_clock_result(
            {"marker_offsets_frames": raw, "output_frames": 160020},
            {"marker_offsets_frames": fixed, "output_frames": 160000},
        )
        self.assertEqual(measurement["independent_expected_span_frames"], 16)
        self.assertEqual(measurement["corrected_residual_span_frames"], 0)

    def test_wrong_correction_and_fabricated_raw_clock_are_rejected(self):
        raw = {1: 2, 3: 5, 4: 6, 6: 10, 8: 15, 9: 18}
        fixed = {1: 0, 3: 0, 4: 0, 6: 0, 8: 0, 9: 0}
        with self.assertRaisesRegex(ValueError, "residual drift"):
            validate_clock_result(
                {"marker_offsets_frames": raw, "output_frames": 160020},
                {"marker_offsets_frames": raw, "output_frames": 160000},
            )
        with self.assertRaisesRegex(ValueError, "residual drift"):
            validate_clock_result(
                {"marker_offsets_frames": fixed, "output_frames": 160020},
                {"marker_offsets_frames": fixed, "output_frames": 160000},
            )


class TestTraceProtocol(unittest.TestCase):
    def trace(self, rows):
        directory = tempfile.TemporaryDirectory(prefix="src-trace-test-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("phase", "input_start", "offered", "used", "generated", "eof", "ratio"))
            writer.writerows(rows)
        return path

    def valid_rows(self):
        return [
            ("input", 0, 80008, 80007, 64, 0, 16000 / 16001.6),
            ("input", 80007, 1, 1, 0, 0, 16000 / 16001.6),
            ("input", 80008, 80012, 80012, 64, 1, 16000 / 16002.4),
            ("drain", 160020, 0, 0, 20, 1, 16000 / 16002.4),
            ("drain", 160020, 0, 0, 0, 1, 16000 / 16002.4),
        ]

    def test_valid_trace_has_contiguous_input_and_drain(self):
        result = inspect_trace(self.trace(self.valid_rows()), 148, True)
        self.assertEqual(result["input_frames_used_total"], 160020)
        self.assertEqual(result["drain_output_frames"], 20)
        self.assertEqual(result["partial_input_calls"], 1)

    def test_dropped_input_is_rejected(self):
        rows = self.valid_rows()
        rows[1] = ("input", 80008, 1, 1, 0, 0, 16000 / 16001.6)
        with self.assertRaisesRegex(ValueError, "lost or duplicated"):
            inspect_trace(self.trace(rows), 148, True)

    def test_wrong_ratio_and_early_eof_are_rejected(self):
        rows = self.valid_rows()
        rows[2] = (*rows[2][:6], 16000 / 16001.6)
        with self.assertRaisesRegex(ValueError, "ratio"):
            inspect_trace(self.trace(rows), 148, True)
        rows = self.valid_rows()
        rows[0] = (*rows[0][:5], 1, rows[0][6])
        with self.assertRaisesRegex(ValueError, "end_of_input"):
            inspect_trace(self.trace(rows), 148, True)

    def test_final_drain_is_required(self):
        rows = self.valid_rows()[:-2]
        with self.assertRaisesRegex(ValueError, "drain"):
            inspect_trace(self.trace(rows), 128, True)


class TestNativeProbe(unittest.TestCase):
    @unittest.skipUnless((ROOT / "codes/upstream/_downloads/libsamplerate/src/samplerate.c").exists()
                         and shutil.which("cmake") and (shutil.which("clang") or shutil.which("cc")),
                         "locked local libsamplerate source and native build tools required")
    def test_native_two_clock_c_api(self):
        report = run()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["source"]["final_status"], "source_verified")
        self.assertEqual(report["independent_clock_check"]["independent_expected_span_frames"], 16)
        self.assertLessEqual(abs(report["independent_clock_check"]["corrected_residual_span_frames"]), 3)
        self.assertGreater(report["cases"]["corrected"]["trace"]["partial_input_calls"], 0)
        self.assertGreater(report["cases"]["corrected"]["trace"]["drain_output_frames"], 0)


if __name__ == "__main__":
    unittest.main()
