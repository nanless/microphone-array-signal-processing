"""Independent matrices and fixed-source evidence for the ch03 method audit."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch03.examples import audit_upstream_coarray as audit


REPORT = audit.ROOT / "codes/chapters/ch03/reports/upstream_coarray.json"
HAS_SOURCE = (audit.CACHE / "doatools/.git").exists()


def strict_json(text):
    def reject(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


def matrix(value):
    return np.asarray(value["real"]) + 1j*np.asarray(value["imag"])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = strict_json(REPORT.read_text(encoding="utf-8"))

    def test_current_script_lock_and_selected_source_bindings(self):
        self.assertEqual(self.report["audit_source_sha256"], audit.sha256(Path(audit.__file__)))
        self.assertEqual(self.report["sources"]["lock_sha256"], audit.sha256(audit.LOCK))
        entry = next(p for p in json.loads(audit.LOCK.read_text())["projects"] if p["id"] == "doatools")
        canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(self.report["sources"]["lock_entry_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        self.assertEqual(self.report["sources"]["revision"], "9469db201e0418aef6b97583ef54b6fec2769502")
        self.assertEqual(self.report["sources"]["license"], "MIT")
        records = self.report["sources"]["files"]
        self.assertEqual({r["path"] for r in records}, set(audit.DEFINITIONS) | {"LICENSE.md"})
        self.assertTrue(self.report["upstream_clean_before_and_after"])
        self.assertFalse(self.report["scaffold"]["full_package_import_executed"])
        self.assertFalse(self.report["scaffold"]["global_numpy_modified"])
        self.assertEqual(self.report["scaffold"]["local_numpy_aliases"],
                         {"float_": "float64", "complex_": "complex128"})

    def test_layout_channel_order_aperture_and_multiplicity(self):
        golden = (([0, 1, 2, 3, 7, 11], 23, 23, 12),
                  ([0, 3, 6, 9, 4, 8], 17, 13, 7),
                  ([0, 3, 6, 9, 4, 8, 12, 16, 20], 35, 29, 15))
        for result, (indices, distinct, central, virtual) in zip(self.report["results"]["layouts"], golden):
            self.assertEqual(result["indices_in_original_channel_order"], indices)
            self.assertEqual(len(result["signed_lags"]), distinct)
            self.assertEqual(result["central_signed_size"], central)
            self.assertEqual(result["virtual_size"], virtual)
            self.assertEqual(sum(result["multiplicity"]), len(indices)**2)
            self.assertEqual(result["multiplicity"][result["signed_lags"].index(0)], len(indices))
            np.testing.assert_allclose(result["positions_m"], np.array(indices)*.02, atol=1e-14, rtol=0)
        nested = self.report["results"]["layouts"][0]
        self.assertAlmostEqual(max(nested["positions_m"])-min(nested["positions_m"]), .22)

    def test_hand_matrices_and_single_snapshot_negative_eigenvalue(self):
        ideal, single = self.report["results"]["covariances"]
        ideal_T = np.array([[2.1, 1-1j, 0, 1+1j],
                            [1+1j, 2.1, 1-1j, 0],
                            [0, 1+1j, 2.1, 1-1j],
                            [1-1j, 0, 1+1j, 2.1]])
        single_T = np.array([[2/3, 0, 0, 1], [0, 2/3, 0, 0],
                             [0, 0, 2/3, 0], [1, 0, 0, 2/3]])
        # Hand SS coefficients: (2/3)^2/4 = 1/9; corners gain 1/4,
        # and offdiagonal = 2*(2/3)/4 = 1/3.
        single_SS = np.array([[13/36, 0, 0, 1/3], [0, 1/9, 0, 0],
                              [0, 0, 1/9, 0], [1/3, 0, 0, 13/36]])
        np.testing.assert_allclose(matrix(ideal["da"]), ideal_T, atol=1e-14, rtol=0)
        np.testing.assert_allclose(matrix(single["da"]), single_T, atol=1e-14, rtol=0)
        np.testing.assert_allclose(matrix(single["ss_default"]), single_SS, atol=1e-14, rtol=0)
        np.testing.assert_allclose(ideal["da_eigenvalues"], [.1, .1, 4.1, 4.1], atol=1e-14, rtol=0)
        np.testing.assert_allclose(ideal["ss_eigenvalues"], [.0025, .0025, 4.2025, 4.2025], atol=1e-14, rtol=0)
        np.testing.assert_allclose(single["da_eigenvalues"], [-1/3, 2/3, 2/3, 5/3], atol=1e-14, rtol=0)
        np.testing.assert_allclose(single["ss_eigenvalues"], [1/36, 1/9, 1/9, 25/36], atol=1e-14, rtol=0)
        self.assertTrue(single["input_psd"])
        self.assertFalse(single["da_psd"])
        self.assertTrue(single["ss_psd"])
        for result in (ideal, single):
            self.assertEqual((result["amplitude_scale"], result["physical_covariance_scale"],
                              result["da_scale"], result["ss_scale"]), (2, 4, 4, 16))
            self.assertLessEqual(result["scaled_ss_max_error"], 1e-12)

    def test_forward_models_are_given_parameters_not_estimators(self):
        gain, phase, coupling, location = self.report["results"]["forward_given_perturbations"]
        np.testing.assert_allclose(matrix(gain["output"]), [[1.1, 1.1j], [.8, -.8]], atol=1e-14, rtol=0)
        np.testing.assert_allclose(matrix(coupling["output"]), [[1.1, -.1+1j], [1.2, -1+.2j]], atol=1e-14, rtol=0)
        phases = np.array([[np.cos(.2)+1j*np.sin(.2), -np.sin(.2)+1j*np.cos(.2)],
                           [np.cos(.3)-1j*np.sin(.3), -np.cos(.3)+1j*np.sin(.3)]])
        np.testing.assert_allclose(matrix(phase["output"]), phases, atol=1e-14, rtol=0)
        np.testing.assert_allclose(location["output_m"], [[.001, .002], [.039, .003]], atol=1e-14, rtol=0)
        for result in (gain, phase, coupling, location):
            self.assertFalse(result["unknown_parameter_estimation_executed"])

    def test_original_shape_defects_are_not_described_as_valid_models(self):
        results = {r["case"]: r for r in self.report["results"]["negative_inputs"]}
        for key in ("GainErrors_list", "PhaseErrors_list"):
            self.assertEqual(results[key]["exception_type"], "AttributeError")
        self.assertEqual(results["MutualCoupling_2x3_constructor"]["value"], [2, 3])
        self.assertEqual(results["MutualCoupling_1d_constructor"]["exception_type"], "IndexError")
        bad = results["covariance_3x4_da"]
        self.assertEqual(matrix(bad["input"]).shape, (3, 4))
        np.testing.assert_array_equal(matrix(bad["input"])[:, 3], [99, 99, 99])
        self.assertTrue(bad["extra_column_silently_ignored"])
        np.testing.assert_allclose(matrix(bad["value"]), matrix(self.report["results"]["covariances"][0]["da"]),
                                   atol=1e-14, rtol=0)
        self.assertEqual(results["covariance_3x2_da"]["exception_type"], "IndexError")
        self.assertTrue(all(not r["valid_model_input"] for r in results.values()))


class GuardTests(unittest.TestCase):
    def test_own_guard_rejects_shape_nonfinite_nonhermitian_and_negative_energy(self):
        for bad in (np.ones((3, 4)), np.ones((3, 2)), np.eye(3)*np.nan,
                    [[1, 1j, 0], [1j, 1, 0], [0, 0, 1]], np.diag([1., 1., -.1])):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                audit.validate_physical_covariance(bad, 3)
        np.testing.assert_array_equal(audit.validate_physical_covariance(np.zeros((3, 3)), 3), np.zeros((3, 3)))

    def test_extraction_preserves_selected_decorator_and_skips_module_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fixture.py"
            source.write_text("raise RuntimeError('module must not run')\n"
                              "@identity\ndef selected(x):\n    return x + 7\n"
                              "def unselected():\n    raise RuntimeError('ignored')\n")
            original = source.read_bytes()
            ns = {"identity": lambda function: function}
            with patch.object(audit, "ROOT", root):
                records = audit.extract(source, ("selected",), ns)
            self.assertEqual(ns["selected"](3), 10)
            self.assertNotIn("unselected", ns)
            self.assertFalse(records[0]["body_or_decorators_changed"])
            self.assertEqual(source.read_bytes(), original)
            tree_node = ast.parse(original).body[1]
            self.assertEqual(records[0]["definition_ast_sha256"], hashlib.sha256(
                ast.dump(tree_node, include_attributes=False).encode()).hexdigest())

    def test_bad_head_dirty_checkout_or_blob_difference_rejects_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            checkout = cache / "doatools"
            (checkout / ".git").mkdir(parents=True)
            source = checkout / "fixture.py"
            source.write_bytes(b"fixed source\n")
            lock = cache / "lock.json"
            lock.write_text(json.dumps({"projects": [{"id": "doatools", "revision": audit.REVISION,
                                                       "license": "MIT", "url": audit.URL}]}))
            for responses in ([b"wrong-head\n"], [audit.REVISION.encode(), b" M fixture.py\n"],
                              [audit.REVISION.encode(), b"", b"different Git blob\n"]):
                with patch.object(audit, "DEFINITIONS", {"fixture.py": ("selected",)}), \
                     patch.object(audit, "_git", side_effect=responses), self.assertRaises(ValueError):
                    audit.verify_sources(cache, lock)
                self.assertEqual(source.read_bytes(), b"fixed source\n")

    def test_report_write_is_atomic_and_never_targets_upstream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "report.json"
            target.write_text('{"old": true}\n')
            with patch.object(audit.os, "replace", side_effect=OSError("interrupted publication")), \
                 self.assertRaises(OSError):
                audit.write_report(target, {"new": True})
            self.assertEqual(strict_json(target.read_text()), {"old": True})
            self.assertEqual(list(root.iterdir()), [target])
            with self.assertRaises(ValueError):
                audit.write_report(target, {"value": float("nan")})
            self.assertEqual(strict_json(target.read_text()), {"old": True})
            audit.write_report(target, {"new": True})
            self.assertEqual(strict_json(target.read_text()), {"new": True})
            with self.assertRaises(ValueError):
                audit.write_report(audit.CACHE / "doatools/forbidden-report.json", {})

    @unittest.skipUnless(HAS_SOURCE, "Existing pinned doatools checkout is absent; no download performed")
    def test_actual_methods_match_report_and_default_run_is_read_only(self):
        before = REPORT.read_bytes()
        global_aliases_before = {name: hasattr(np, name) for name in ("float_", "complex_")}
        result = audit.run_audit()
        current = strict_json(before)
        self.assertEqual(result["sources"], current["sources"])
        self.assertEqual(result["results"], current["results"])
        self.assertTrue(result["all_expected_behaviors_observed"])
        self.assertEqual(REPORT.read_bytes(), before)
        self.assertEqual({name: hasattr(np, name) for name in ("float_", "complex_")}, global_aliases_before)


if __name__ == "__main__":
    unittest.main()
