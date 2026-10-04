"""Offline report and independent analytic checks; never imports doatools/SciPy."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from codes.chapters.ch04.core import upstream_contracts as contracts

import numpy as np

from codes.chapters.ch04.examples import reproduce_doatools_esprit as experiment


class DoatoolsEspritReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((experiment.ROOT / "chapters/ch04/reports/doatools_esprit_reference.json").read_text())

    def test_saved_report_bound_to_harness_source_and_configuration(self):
        provenance = self.report["provenance"]
        self.assertEqual(provenance["revision"], experiment.REVISION)
        self.assertEqual(provenance["source_sha256"], experiment.SOURCE_HASHES)
        self.assertEqual(provenance["harness_sha256"], hashlib.sha256(contracts.historical_bytes(experiment.__file__)).hexdigest())
        self.assertEqual(provenance["configuration_sha256"], experiment.configuration_sha256())
        self.assertEqual(self.report["configuration"], experiment.CONFIG)
        self.assertFalse(provenance["upstream_modified"])
        self.assertIn("formulation='ls'", provenance["upstream_signature"])
        self.assertEqual(self.report["status"], "failed_default_row_weighting")

    def test_covariance_against_independent_scalar_trigonometry(self):
        # Fixed physical input: independent unit-power sources at -5 and +10
        # degrees and 0.1 white noise. No call to the harness to form expected R.
        expected = np.empty((8, 8), complex)
        for m in range(8):
            for n in range(8):
                phases = [math.pi * (m - n) * math.sin(math.radians(t)) for t in (-5, 10)]
                expected[m, n] = sum(complex(math.cos(p), math.sin(p)) for p in phases) + (0.1 if m == n else 0)
        reported = np.array(self.report["covariance_real"]) + 1j * np.array(self.report["covariance_imag"])
        np.testing.assert_allclose(reported, expected, atol=2e-15, rtol=0)
        np.testing.assert_allclose(experiment.population_input()[1], expected, atol=2e-15, rtol=0)

    def test_four_actual_calls_preserve_failure_despite_resolved_true(self):
        rows = self.report["upstream_results"]
        self.assertEqual([(r["formulation"], r["row_weights"]) for r in rows],
                         [("ls", "default"), ("ls", "none"), ("tls", "default"), ("tls", "none")])
        expected = [[-6.50854311582922, 11.527824709188026], [-5, 10],
                    [-6.478045116978704, 11.496900202412146], [-5, 10]]
        for row, angles in zip(rows, expected):
            with self.subTest(form=row["formulation"], weights=row["row_weights"]):
                np.testing.assert_allclose(row["angles_deg"], angles, atol=1e-7, rtol=0)
                error = max(abs(a - b) for a, b in zip(row["angles_deg"], (-5, 10)))
                self.assertAlmostEqual(error, row["max_abs_direction_error_deg"], places=12)
                self.assertTrue(row["resolved"])
                self.assertEqual(row["direction_check"], "failed" if row["row_weights"] == "default" else "passed")
                if row["row_weights"] == "default":
                    self.assertGreater(error, 1.4)
                else:
                    self.assertLess(error, 1e-8)

    def test_reference_on_quarter_cycle_basis_and_changed_coordinates(self):
        # Hand-calculated columns for +/-30 degrees: shift eigenvalues -j,+j.
        # A non-unitary invertible right transform changes coordinates, not DOAs.
        basis = np.array([[1, 1], [-1j, 1j], [-1, -1], [1j, -1j], [1, 1]])
        for change in (np.eye(2), np.array([[2, 1j], [0.3, 1]])):
            e = basis @ change
            saved = e.copy()
            for formulation in ("ls", "tls"):
                for weighted in (False, True):
                    with self.subTest(form=formulation, weighted=weighted):
                        result = experiment.safe_rotation(e, formulation, weighted)
                        np.testing.assert_allclose(result["angles_deg"], [-30, 30], atol=1e-12, rtol=0)
                        np.testing.assert_allclose(result["rotation_moduli"], [1, 1], atol=1e-12, rtol=0)
                        self.assertLess(result["subarray_residual_fro"], 1e-12)
                        self.assertFalse(result["subarrays_share_memory"])
                        np.testing.assert_array_equal(e, saved)

    def test_saved_references_match_physical_angles_and_analytic_rotation(self):
        expected_rotation = np.diag([complex(math.cos(math.pi * math.sin(math.radians(t))),
                                             math.sin(math.pi * math.sin(math.radians(t)))) for t in (-5, 10)])
        for group in ("safe_copy_eigenspace_references", "analytic_steering_basis_references"):
            for result in self.report[group].values():
                np.testing.assert_allclose(result["angles_deg"], [-5, 10], atol=1e-10, rtol=0)
                self.assertFalse(result["subarrays_share_memory"])
                self.assertLess(result["subarray_residual_fro"], 1e-12)
                if group == "analytic_steering_basis_references":
                    rotation = np.array(result["rotation_real"]) + 1j * np.array(result["rotation_imag"])
                    np.testing.assert_allclose(rotation, expected_rotation, atol=2e-14, rtol=0)

    def test_reference_rejects_invalid_or_rank_deficient_input(self):
        for invalid in (np.ones(3), np.ones((2, 2)), np.empty((4, 0)),
                        np.full((4, 1), np.nan), np.zeros((4, 1)), np.ones((4, 2))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                experiment.safe_rotation(invalid, "ls")
        with self.assertRaises(ValueError):
            experiment.safe_rotation(np.ones((4, 1)), "invalid")

    def test_missing_source_fails_without_download_or_substitute(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                experiment.run_experiment(Path(directory))


if __name__ == "__main__":
    unittest.main()
