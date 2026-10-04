"""Independent matrices and fixed-source evidence for the ch03 method audit."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
from contextlib import redirect_stdout
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch00.core.source_history import verify_lock_binding
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

    def test_historical_script_lock_and_selected_source_bindings(self):
        source = self.report["audit_source"]
        revision = audit.HISTORICAL_SCRIPT_REVISION + ":" + source
        original = (audit.run_git(["show", revision], cwd=audit.ROOT) + "\n").encode()
        self.assertEqual(len(original), int(audit.run_git(["cat-file", "-s", revision], cwd=audit.ROOT)))
        self.assertEqual(hashlib.sha256(original).hexdigest(), audit.HISTORICAL_SCRIPT_SHA256)
        self.assertEqual(self.report["audit_source_sha256"], audit.HISTORICAL_SCRIPT_SHA256)
        report_revision = audit.HISTORICAL_SCRIPT_REVISION + ":" + str(REPORT.relative_to(audit.ROOT))
        original_report = (audit.run_git(["show", report_revision], cwd=audit.ROOT) + "\n").encode()
        self.assertEqual(len(original_report), int(audit.run_git(["cat-file", "-s", report_revision], cwd=audit.ROOT)))
        self.assertEqual(REPORT.read_bytes(), original_report)
        binding = verify_lock_binding(self.report["sources"]["lock_sha256"], ("doatools",),
                                      current_lock=audit.LOCK)
        entry = binding["records"]["doatools"]
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

    def test_report_write_is_atomic_and_never_targets_upstream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "report.json"
            target.write_text('{"old": true}\n')
            with patch("codes.chapters.ch00.io_contracts.os.replace", side_effect=OSError("interrupted publication")), \
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
        for report in (result, current):
            verify_lock_binding(report["sources"]["lock_sha256"], ("doatools",),
                                current_lock=audit.LOCK)
        for key in ("lock_entry_sha256", "url", "revision", "license", "implementation_identity"):
            self.assertEqual(result["sources"][key], current["sources"][key])
        for actual, historical in zip(result["sources"]["files"], current["sources"]["files"]):
            self.assertEqual({key: actual[key] for key in historical}, historical)
        self.assertEqual(result["results"], current["results"])
        self.assertTrue(result["all_expected_behaviors_observed"])
        self.assertEqual(REPORT.read_bytes(), before)
        self.assertEqual({name: hasattr(np, name) for name in ("float_", "complex_")}, global_aliases_before)

    def test_report_policy_rejects_before_original_methods_run(self):
        for target in (REPORT, audit.LOCK, audit.STATUS, audit.CACHE / "report.json",
                       audit.ROOT / "README.md", audit.CURRENT_REPORT.with_name("other.json")):
            with self.subTest(target=target), patch.object(audit, "run_audit") as run:
                with self.assertRaises(ValueError):
                    audit.main(["--report", str(target)])
                run.assert_not_called()

    def test_report_links_hardlinks_directories_and_traversal_reject_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            leaf = real / "original.json"
            leaf.write_text("{}")
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            symlink = root / "symbolic.json"
            symlink.symlink_to(leaf)
            hardlink = root / "hard.json"
            os.link(leaf, hardlink)
            for target in (linked / "new.json", symlink, hardlink, real, root / "x/../new.json"):
                with self.subTest(target=target), patch.object(audit, "run_audit") as run:
                    with self.assertRaises(ValueError):
                        audit.main(["--report", str(target)])
                    run.assert_not_called()
            self.assertEqual(leaf.read_text(), "{}")

    def test_execution_rejects_source_or_shared_dependency_change(self):
        for changed_part in ("sources", "dependencies"):
            with self.subTest(changed_part=changed_part), \
                 patch.object(audit, "source_dependencies", side_effect=[{"helper": "before"},
                    {"helper": "after" if changed_part == "dependencies" else "before"}]), \
                 patch.object(audit, "verify_sources", side_effect=[{"identity": "before"},
                    {"identity": "after" if changed_part == "sources" else "before"}]), \
                 patch.object(audit, "load_methods", return_value=({}, [])), \
                 patch.object(audit, "audit_layouts", return_value=[]), \
                 patch.object(audit, "audit_covariances", return_value=([], None, None)), \
                 patch.object(audit, "audit_perturbations", return_value=[]), \
                 patch.object(audit, "audit_invalid_inputs", return_value=[]):
                with self.assertRaisesRegex(ValueError, "changed during execution"):
                    audit.run_audit()

    def test_default_cli_stdout_is_read_only(self):
        with patch.object(audit, "run_audit", return_value={"finite": True}), \
             patch.object(audit, "write_report") as write, redirect_stdout(io.StringIO()) as output:
            audit.main([])
        self.assertTrue(audit.strict_json_loads(output.getvalue())["finite"])
        write.assert_not_called()


class SourceIntegrityTests(unittest.TestCase):
    """Real local Git fixtures exercise identity failures without network/cache writes."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.checkout = self.base / "doatools"
        audit.run_git(["init", str(self.checkout)])
        audit.run_git(["config", "user.name", "offline test"], cwd=self.checkout)
        audit.run_git(["config", "user.email", "offline@example.invalid"], cwd=self.checkout)
        self.source = self.checkout / "fixture.py"
        self.source.write_bytes(b"locked source\n")
        (self.checkout / "LICENSE.md").write_bytes(b"fixture licence\n")
        audit.run_git(["add", "fixture.py", "LICENSE.md"], cwd=self.checkout)
        audit.run_git(["commit", "-m", "offline fixture"], cwd=self.checkout)
        self.revision = audit.run_git(["rev-parse", "HEAD"], cwd=self.checkout)
        self.origin = "https://example.invalid/fixture.git"
        audit.run_git(["remote", "add", "origin", self.origin], cwd=self.checkout)
        self.entry = {"id": "doatools", "revision": self.revision, "url": self.origin,
                      "license": "MIT", "entrypoints": ["fixture.py", "LICENSE.md"]}
        self.lock = self.base / "lock.json"
        self.status = self.base / "status.json"
        self.lock.write_text(json.dumps({"schema_version": 1, "projects": [self.entry]}))
        self.status.write_text(json.dumps({"schema_version": 1, "lock_sha256": audit.sha256(self.lock),
            "projects": [{"id": "doatools", "revision": self.revision,
                          "status": "source_verified", "source_selection_verified": True}]}))
        for patcher in (patch.object(audit, "REVISION", self.revision),
                        patch.object(audit, "URL", self.origin),
                        patch.object(audit, "DEFINITIONS", {"fixture.py": ("selected",)})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def verify(self):
        return audit.verify_sources(self.base, self.lock, self.status)

    def test_actual_origin_head_dirty_and_blob_difference_fail(self):
        self.assertTrue(self.verify()["used_source_identity_verified"])
        audit.run_git(["remote", "set-url", "origin", "https://example.invalid/other.git"], cwd=self.checkout)
        with self.assertRaisesRegex(ValueError, "origin"):
            self.verify()
        audit.run_git(["remote", "set-url", "origin", self.origin], cwd=self.checkout)
        self.source.write_bytes(b"changed source\n")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.verify()
        original = audit._git
        def hide_dirty(checkout, *args):
            return "" if args[0] == "status" else original(checkout, *args)
        with patch.object(audit, "_git", side_effect=hide_dirty):
            with self.assertRaisesRegex(ValueError, "blob"):
                self.verify()
        self.assertEqual(self.source.read_bytes(), b"changed source\n")
        audit.run_git(["add", "fixture.py"], cwd=self.checkout)
        audit.run_git(["commit", "-m", "new head"], cwd=self.checkout)
        with self.assertRaisesRegex(ValueError, "HEAD"):
            self.verify()

    def test_inherited_git_routing_and_configuration_are_ignored(self):
        with patch.dict(os.environ, {"GIT_DIR": "/absent", "GIT_WORK_TREE": "/absent",
                "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "remote.origin.url",
                "GIT_CONFIG_VALUE_0": "https://example.invalid/forged.git"}):
            self.assertEqual(self.verify()["actual_origin"], self.origin)

    def test_git_newline_normalization_does_not_hide_different_source_bytes(self):
        audit.run_git(["config", "core.autocrlf", "true"], cwd=self.checkout)
        self.source.write_bytes(b"locked source\r\n")
        # Exercise blob verification independently of the cleanliness check.
        # A normal hash-object can hash normalized LF content under this config.
        self.assertEqual(audit.run_git(["hash-object", "--", str(self.source)], cwd=self.checkout),
                         audit.run_git(["rev-parse", "HEAD:fixture.py"], cwd=self.checkout))
        original = audit._git
        def hide_dirty(checkout, *args):
            return "" if args[0] == "status" else original(checkout, *args)
        with patch.object(audit, "_git", side_effect=hide_dirty):
            with self.assertRaisesRegex(ValueError, "blob"):
                self.verify()
        self.assertEqual(self.source.read_bytes(), b"locked source\r\n")

    def test_strict_source_documents_and_unbound_status_reject(self):
        original_lock, original_status = self.lock.read_bytes(), self.status.read_bytes()
        for invalid in (b'{"schema_version":1,"schema_version":1,"projects":[]}',
                        b'{"value":NaN}', b'{"value":1e999}', b'[]', b'\xff',
                        b'{"schema_version":true,"projects":[]}',
                        b'{"schema_version":1,"projects":[{"id":"x"},{"id":"x"}]}'):
            for path in (self.lock, self.status):
                with self.subTest(path=path, invalid=invalid):
                    path.write_bytes(invalid)
                    with self.assertRaises(ValueError):
                        self.verify()
                    self.lock.write_bytes(original_lock)
                    self.status.write_bytes(original_status)
        state = audit.strict_json_loads(original_status)
        state["lock_sha256"] = "0"*64
        self.status.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "bind"):
            self.verify()

    def test_linked_source_licence_checkout_and_metadata_reject(self):
        for path in (self.source, self.checkout / "LICENSE.md", self.lock, self.status):
            raw = path.read_bytes()
            outside = self.base / (path.name + ".outside")
            outside.write_bytes(raw)
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaises(ValueError):
                self.verify()
            path.unlink()
            path.write_bytes(raw)
        alias = self.base / "alias"
        alias.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            audit.verify_sources(alias, self.lock, self.status)

    def test_live_selection_mismatch_does_not_upgrade_used_file_identity(self):
        self.entry["source_paths"] = ["fixture.py", "LICENSE.md"]
        self.lock.write_text(json.dumps({"schema_version": 1, "projects": [self.entry]}))
        state = audit.strict_json_loads(self.status.read_bytes())
        state["lock_sha256"] = audit.sha256(self.lock)
        state["projects"][0].update(status="source_selection_mismatch", source_selection_verified=False)
        self.status.write_text(json.dumps(state))
        result = self.verify()
        self.assertTrue(result["used_source_identity_verified"])
        self.assertFalse(result["recorded_acquisition"]["source_selection_verified"])
        self.assertEqual(result["live_acquisition_scope"]["status"], "source_selection_mismatch")
        self.assertFalse(result["live_acquisition_scope"]["source_selection_verified"])


@unittest.skipUnless(audit.CURRENT_REPORT.is_file(), "Current report has not yet been generated")
class CurrentReportTests(unittest.TestCase):
    def test_current_dependencies_sources_and_historical_numeric_results(self):
        current = audit.strict_json_loads(audit.CURRENT_REPORT.read_bytes())
        historical = audit.strict_json_loads(REPORT.read_bytes())
        self.assertEqual(current["schema_version"], 2)
        self.assertEqual(current["audit_source_sha256"], audit.sha256(Path(audit.__file__)))
        self.assertEqual(current["source_sha256"], audit.source_dependencies())
        self.assertEqual(current["sources"]["lock_sha256"], audit.sha256(audit.LOCK))
        self.assertEqual(current["sources"]["status_sha256"], audit.sha256(audit.STATUS))
        self.assertEqual(current["sources"]["actual_origin"], audit.URL)
        self.assertTrue(current["sources"]["used_source_identity_verified"])
        self.assertEqual(current["sources"]["live_acquisition_scope"]["status"], "source_selection_mismatch")
        self.assertFalse(current["sources"]["live_acquisition_scope"]["source_selection_verified"])
        verification = current["source_verification"]
        self.assertTrue(verification["before_after_equal"])
        self.assertEqual(verification["canonical_sources_before_sha256"], verification["canonical_sources_after_sha256"])
        self.assertEqual(current["results"], historical["results"])


if __name__ == "__main__":
    unittest.main()
