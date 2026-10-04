"""Offline checks of recorded interface observations, never download upstream code."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch06.examples import audit_aec_upstream_interfaces as audit


class AecUpstreamInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.report = json.loads((root / "codes/chapters/ch06/reports/aec_upstream_interfaces.json").read_text())

    def test_report_binds_harness_sources_inputs_and_execution_scope(self):
        report = self.report
        relative = str(Path(audit.__file__).resolve().relative_to(audit.contracts.ROOT))
        spec = "621d727a63475e3ee8ad2a29d518889c42e92571:" + relative
        original = (audit.contracts.git(audit.contracts.ROOT, 'show', spec) + '\n').encode()
        self.assertEqual(len(original), int(audit.contracts.git(audit.contracts.ROOT, 'cat-file', '-s', spec)))
        self.assertEqual(report["script_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(report["binding_sha256"], audit.binding_sha256())
        self.assertEqual(report["sources"], audit.SOURCES)
        self.assertEqual(report["config"], audit.CONFIG)
        self.assertFalse(report["upstream_modified"])
        self.assertFalse(report["performance_evaluation"])
        self.assertEqual(report["environment"]["numpy"], "2.5.3")
        self.assertIn("original_source", report["results"]["pyaec"]["rls"]["execution_kind"])
        self.assertFalse(report["results"]["dtln"]["real_neural_inference"])
        self.assertFalse(report["results"]["dtln"]["real_audio_write"])

    def test_zero_reference_and_tail_have_independent_expected_indices(self):
        # An identically zero regressor produces zero estimated echo for every
        # sample regardless of the weight update: e[n] = d[n] - 0.
        for name in ("rls", "kalman"):
            row = self.report["results"]["pyaec"][name]
            self.assertEqual(row["actual_residual"], [1., 2., 3., 4.])
            self.assertEqual(row["analytic_zero_reference_residual"], [1., 2., 3., 4., 5., 6.])
            self.assertEqual(row["unprocessed_sample_indices"], [4, 5])
            self.assertEqual(row["returned_samples"], 4)
        for name in ("fdkf", "pfdkf"):
            row = self.report["results"]["pyaec"][name]
            self.assertEqual(row["exception_type"], "AttributeError")
            self.assertIn("'complex'", row["message"])
            self.assertFalse(row["output_computed"])

    def test_initial_res_output_from_exact_rational_four_point_transform(self):
        # rFFT([0,0,1,1]) = [2,-1+i,0]; its two nonzero-bin powers are 4,2.
        # With P=1 and m=power/2, W=1-power/(2*power+eps).
        epsilon = Fraction(1, 10**10)
        w0 = 1 - Fraction(4) / (8 + epsilon)
        w1 = 1 - Fraction(2) / (4 + epsilon)
        expected = float((w0 + w1) / 2)
        rows = self.report["results"]["echocatzh"]["rows"]
        self.assertEqual(rows[0]["error"], [1., 1.])
        self.assertEqual(rows[0]["reported_echo"], [0., 0.])
        np.testing.assert_allclose(rows[1]["error"], [expected] * 2, rtol=0, atol=1e-15)
        np.testing.assert_allclose(rows[1]["reported_echo"], [1-expected] * 2, rtol=0, atol=1e-15)
        for row in rows:
            self.assertTrue(row["filter_coefficients_remain_zero"])
            self.assertFalse(row["update_called"])

    def test_dtln_fake_overlap_fixture_is_not_neural_inference(self):
        rows = self.report["results"]["dtln"]["rows"]
        self.assertEqual(rows[0]["output_samples"], 640)
        self.assertEqual(rows[0]["minimum"], -4 * .5)
        self.assertEqual(rows[0]["maximum"], -4 * .5)
        self.assertEqual(rows[0]["samples_outside_unit_range"], 640)
        self.assertEqual(rows[1]["minimum"], .99)
        self.assertEqual(rows[1]["maximum"], .99)
        self.assertEqual(rows[1]["samples_outside_unit_range"], 0)
        self.assertIn("fake_interpreters", self.report["results"]["dtln"]["execution_kind"])
        for row in rows:
            # min(768,640)+2*384 samples, 512-point windows, 128-point hops.
            self.assertEqual(row["stage_invocation_counts"], [8, 8])

    def test_verifier_refuses_absent_or_mismatched_source_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                audit.verify_source(root, audit.SOURCES["pyaec"])
            with patch.object(audit.contracts, 'verify_project', side_effect=ValueError('origin mismatch')):
                with self.assertRaisesRegex(ValueError, 'origin mismatch'):
                    audit.verify_source(root, audit.SOURCES['pyaec'])
            with self.assertRaisesRegex(ValueError, 'declared fixed'):
                audit.verify_source(root, {'revision': 'unknown', 'files': {}})



if __name__ == "__main__":
    unittest.main()
