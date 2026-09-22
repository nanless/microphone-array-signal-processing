"""Independent analytic checks and optional, offline locked-source execution."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from codes.examples import reproduce_sbl_reference as experiment


class SBLReferenceTests(unittest.TestCase):
    def test_saved_report_matches_the_harness_and_locked_source(self):
        report = json.loads((experiment.ROOT / "reports/sbl_reference.json").read_text())
        provenance = report["provenance"]
        self.assertEqual(provenance["revision"], experiment.REVISION)
        self.assertEqual(provenance["module_sha256"], experiment.MODULE_SHA256)
        self.assertEqual(provenance["harness_sha256"], hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest())
        self.assertEqual(len(report["cases"]), 5)

    def test_book_phase_convention_against_quarter_cycle_hand_calculation(self):
        expected = np.array([[1, 1], [-1j, 1j], [-1, -1], [1j, -1j]])
        np.testing.assert_allclose(experiment.steering(np.array([-30., 30.])), expected, atol=2e-15)

    def test_scalar_covariance_against_hand_calculation(self):
        y = np.array([[1, 1j], [2, 1]], dtype=complex)
        expected = np.array([[1, 1 + .5j], [1 - .5j, 2.5]])
        np.testing.assert_allclose(experiment.independent_covariance(y), expected, atol=0)

    def test_peak_selector_does_not_invent_missing_peaks(self):
        np.testing.assert_array_equal(experiment.independent_peaks(np.array([0., 1., 0.]), 2), [1])
        self.assertEqual(experiment.independent_peaks(np.zeros(4), 2).size, 0)
        self.assertEqual(experiment.independent_peaks(np.ones(4), 2).size, 0)
        np.testing.assert_array_equal(experiment.independent_peaks(np.array([3., 0., 2.]), 2), [0, 2])
        for bad in (np.array([1., np.nan]), np.array([-1., 2.]), np.ones((2, 2)), np.array([1j])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                experiment.independent_peaks(bad, 2)

    def test_population_spectrum_and_shared_inputs(self):
        cases = experiment.synthetic_cases()
        # The two quarter-cycle steering columns are orthogonal with norm^2=4.
        np.testing.assert_allclose(np.linalg.eigvalsh(cases[0]["population"]), [.02, .02, 4.02, 4.02], atol=3e-15)
        np.testing.assert_array_equal(cases[0]["y"], cases[4]["y"])
        self.assertEqual(cases[0]["y"].shape, (4, 200, 1))
        np.testing.assert_allclose(np.diag(cases[3]["population"]), [2.02] * 4)

    def test_invalid_arrays_and_source_count(self):
        a = experiment.steering(experiment.GRID)[:, :, None]
        y = experiment.synthetic_cases()[0]["y"]
        for k in (0, 4, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                experiment.validate_inputs(a, y, k)
        for bad in (y.transpose(0, 2, 1), np.zeros_like(y), np.full_like(y, np.nan)):
            with self.assertRaises(ValueError):
                experiment.validate_inputs(a, bad, 2)
        with self.assertRaises(ValueError):
            experiment.validate_inputs(np.zeros_like(a), y, 2)

    def test_missing_source_is_not_silently_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                experiment.load_locked_source(Path(directory))

    def test_actual_locked_source_when_present(self):
        source = experiment.ROOT / "upstream/_downloads/sbl"
        if not source.exists():
            self.skipTest("optional locked SBL checkout is absent; no network access attempted")
        result = experiment.run_experiment()
        cases = {case["case"]: case for case in result["cases"]}
        # Analytic true directions, not a stored output of SBL, define success.
        np.testing.assert_allclose(cases["on_grid"]["sbl_peak_angles_deg"], [-30, 30], atol=1)
        self.assertEqual(cases["on_grid"]["stop_reason"], "convergence_threshold")
        self.assertIsNone(cases["wrong_source_count"]["matched_signed_errors_deg"])
        self.assertEqual(cases["wrong_source_count"]["missing_peak_count_relative_to_truth"], 1)
        for case in cases.values():
            self.assertLess(case["scalar_vs_matrix_covariance_max_abs"], 1e-12)
            self.assertEqual(len(case["relative_update_trace"]), case["iterations_executed"])
            self.assertEqual(case["iterations_executed"], case["upstream_zero_based_iteration"] + 1)
            if case["stop_reason"] == "convergence_threshold":
                self.assertLess(case["final_relative_update"], 1e-5)
            else:
                self.assertEqual(case["iterations_executed"], 1000)
        json.dumps(result, allow_nan=False)

    def test_iteration_exhaustion_is_not_reported_as_convergence(self):
        if not (experiment.ROOT / "upstream/_downloads/sbl").exists():
            self.skipTest("optional locked SBL checkout is absent")
        for case in experiment.run_experiment(max_iterations=1)["cases"]:
            self.assertEqual(case["stop_reason"], "iteration_limit")
            self.assertEqual(case["iterations_executed"], 1)


if __name__ == "__main__":
    unittest.main()
