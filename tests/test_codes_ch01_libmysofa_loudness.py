"""Independent arithmetic, strict report binding and optional native execution."""
import json
import math
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch01.examples import audit_libmysofa_loudness as audit

REPORT = audit.ROOT / "codes/chapters/ch01/reports/libmysofa_loudness.json"


def strict_load(text):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON constant: {value}")
    return json.loads(text, parse_constant=reject_constant)


class LoudnessReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = strict_load(REPORT.read_text(encoding="utf-8"))

    def test_report_binds_current_audit_historical_lock_and_unchanged_used_project(self):
        r = self.report
        self.assertEqual(r["audit_source_sha256"], audit.sha256(Path(audit.__file__)))
        verify_lock_binding(r["source"]["lock_sha256"], ("libmysofa",), current_lock=audit.LOCK)
        self.assertEqual(r["source"]["revision"], audit.REVISION)
        self.assertTrue(r["source"]["worktree_clean"])
        self.assertFalse(r["scaffold"]["upstream_patched"])
        self.assertFalse(r["scaffold"]["temporary_binary_retained"])
        self.assertIn("Data.Delay", r["not_executed"])
        self.assertIn("SOFA parser", r["not_executed"])

    def test_independent_nonzero_energy_and_right_minus_left_ild(self):
        case = self.report["cases"][0]
        self.assertEqual(case["input_ir_left_right"], [1, .5])
        left, right = (item["value"] for item in case["output_ir_left_right"])
        # Normalize two receiver energies to total 2, derived without calling audit.evaluate.
        scale = (2 / (1 + 1/4)) ** .5
        self.assertAlmostEqual(left, scale, delta=2e-6)
        self.assertAlmostEqual(right, scale/2, delta=2e-6)
        self.assertAlmostEqual(left*left + right*right, 2, delta=2e-6)
        self.assertAlmostEqual(left/right, 2, delta=2e-6)
        ild = 10 * math.log10((right*right) / (left*left))
        self.assertAlmostEqual(ild, -6.020599913279624, delta=2e-6)
        self.assertAlmostEqual(case["output_ild_db"], ild, delta=2e-6)
        self.assertTrue(case["normalization_success"])

    def test_zero_is_explicitly_nonfinite_not_a_success(self):
        case = self.report["cases"][1]
        self.assertEqual(case["input_ir_left_right"], [0, 0])
        self.assertEqual(case["factor"], {"value": None, "classification": "positive_infinity"})
        self.assertEqual(case["output_ir_left_right"],
                         [{"value": None, "classification": "nan"}] * 2)
        self.assertFalse(case["normalization_success"])
        self.assertTrue(case["expected_behavior_observed"])
        strict_load(json.dumps(self.report, allow_nan=False))

    def test_wrong_head_and_dirty_tree_are_rejected_before_execution(self):
        if not (audit.UPSTREAM / ".git").exists():
            self.skipTest("Pinned optional source checkout is unavailable")
        with patch.object(audit, "_run", return_value="0" * 40 + "\n"):
            with self.assertRaisesRegex(ValueError, "Unexpected upstream HEAD"):
                audit.verify_sources()
        with patch.object(audit, "_run", side_effect=[audit.REVISION + "\n", " M src/hrtf/tools.c\n"]):
            with self.assertRaisesRegex(ValueError, "not clean"):
                audit.verify_sources()

    def test_optional_native_call_matches_independent_expectations(self):
        if shutil.which("cc") is None or not (audit.UPSTREAM / ".git").exists():
            self.skipTest("Optional C compiler or pinned source checkout is unavailable")
        r = audit.run_audit()
        self.assertTrue(r["expected_behavior_observed"])
        self.assertTrue(r["cases"][0]["normalization_success"])
        self.assertFalse(r["cases"][1]["normalization_success"])
        self.assertEqual(r["source"]["source_sha256"], self.report["source"]["source_sha256"])
        strict_load(json.dumps(r, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
