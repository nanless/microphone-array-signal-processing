"""锁定 SMP-PHAT 的隔离适配及独立数值报告。"""

import json
import unittest

from codes.examples import reproduce_smpphat_portable_overlay as overlay
from codes.examples import reproduce_smpphat_reference as reference


class PortableOverlayTests(unittest.TestCase):
    def test_patch_requires_exact_locked_expressions(self):
        source = (reference.ROOT / "upstream/_downloads/smpphat/src/system.c")
        if not source.is_file():
            self.skipTest("固定版外部源码未取得")
        original = source.read_text(encoding="utf-8")
        patched = overlay.patch_source(original)
        self.assertEqual(patched.count(overlay.NEW_LOOKUP), 2)
        self.assertNotIn(overlay.OLD_LOOKUP, patched)
        self.assertEqual(source.read_text(encoding="utf-8"), original)
        with self.assertRaises(ValueError):
            overlay.patch_source(original.replace(overlay.OLD_LOOKUP, "", 1))

    def test_saved_report_distinguishes_overlay_from_original(self):
        path = reference.ROOT / "reports/smpphat_portable_overlay.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed_patched_teaching_case")
        self.assertEqual(report["upstream_revision"], reference.UPSTREAM_REVISION)
        self.assertEqual(report["overlay"]["replacement_count"], 2)
        self.assertEqual([case["merged_group_count"] for case in report["cases"]], [4, 6])
        for case in report["cases"]:
            self.assertEqual(case["signed_lookup_mismatch_count"], 0)
            self.assertEqual(case["srp_peak_index"], reference.TRUE_INDEX)
            self.assertEqual(case["smp_peak_index"], reference.TRUE_INDEX)
            self.assertLess(case["srp_max_abs_error"], 2e-4)
            self.assertLess(case["smp_max_abs_error"], 2e-4)
            self.assertLess(case["srp_vs_smp_max_abs"], 2e-4)
        original = json.loads((reference.ROOT / "reports/smpphat_reference.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(original["numerical_status"], "failed_portability")


if __name__ == "__main__":
    unittest.main()
