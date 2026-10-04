"""Independent scalar expectations and source-integrity guards for ch02 audit."""
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
from contextlib import redirect_stdout
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch02.examples import audit_upstream_models as audit


REPORT = audit.ROOT / "codes/chapters/ch02/reports/upstream_models.json"
HAS_SOURCES = all((audit.CACHE / name / ".git").exists() for name in audit.PROJECTS)


class ReportTests(unittest.TestCase):
    def setUp(self):
        # parse_constant rejects JSON's optional non-standard NaN/Infinity.
        def reject(value):
            raise ValueError(value)
        self.report = json.loads(REPORT.read_text(), parse_constant=reject)

    def test_historical_source_and_lock_bindings(self):
        # Historical bytes bind their actual Git script, never today's script.
        source = self.report["audit_source"]
        revision = audit.HISTORICAL_SCRIPT_REVISION + ":" + source
        raw = (audit.run_git(["show", revision], cwd=audit.ROOT) + "\n").encode()
        # run_git strips terminal whitespace. Require the reconstructed byte
        # count to equal the original blob before accepting its byte digest.
        self.assertEqual(len(raw), int(audit.run_git(["cat-file", "-s", revision], cwd=audit.ROOT)))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), audit.HISTORICAL_SCRIPT_SHA256)
        self.assertEqual(self.report["audit_source_sha256"], audit.HISTORICAL_SCRIPT_SHA256)
        binding = verify_lock_binding(self.report["sources"]["lock_sha256"], audit.HISTORICAL_PROJECTS,
                                      current_lock=audit.LOCK)
        for name in audit.HISTORICAL_PROJECTS:
            spec = audit.PROJECTS[name]
            source = self.report["sources"]["projects"][name]
            self.assertEqual(source["revision"], spec["revision"])
            self.assertEqual(source["license"], spec["license"])
            entry = binding["records"][name]
            canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(source["lock_entry_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
            self.assertEqual(set(source["source_sha256"]), set(spec["files"]))

    def test_physical_near_response_has_distance_ratio_and_negative_phase(self):
        mode = self.report["results"]["pyroomacoustics"]["mode_vector"]
        phase = 2*math.pi*1000*.1/343
        physical = [complex(2/3*math.cos(phase), -2/3*math.sin(phase)),
                    1+0j, complex(2*math.cos(phase), 2*math.sin(phase))]
        actual = [complex(v["real"], v["imag"]) for v in mode["independent_physical_near_response"]]
        np.testing.assert_allclose(actual, physical, atol=1e-12, rtol=0)
        for case in mode["cases"]:
            z = [complex(v["real"], v["imag"]) for v in case["relative_response"]]
            np.testing.assert_allclose(np.abs(z), [1, 1, 1], atol=1e-12, rtol=0)
            self.assertGreater(z[0].imag if case["mode"] == "near" else -z[0].imag, .9)
        self.assertFalse(mode["upper_doa_constructor_executed"])

    def test_weight_responses_and_norms_are_not_all_unit(self):
        golden = {"classic": (11/9, 1/3), "inverse": (1, 7/18),
                  "true level": (1, 9/49), "true location": (7/math.sqrt(27), 1/3)}
        for case in self.report["results"]["acoular"]["cases"]:
            response, norm = golden[case["steer_type"]]
            self.assertAlmostEqual(case["target_response"]["real"], response, delta=2e-6)
            self.assertAlmostEqual(case["target_response"]["imag"], 0, delta=2e-6)
            self.assertAlmostEqual(case["squared_weight_norm"], norm, delta=2e-6)

    def test_eyring_wrong_monotonicity_is_not_reported_as_valid_physics(self):
        cases = self.report["results"]["pyroomacoustics"]["eyring"]["cases"]
        self.assertGreater(cases[1]["output_s"], cases[0]["output_s"])
        self.assertLess(cases[1]["independent_positive_absorption_reference_s"],
                        cases[0]["independent_positive_absorption_reference_s"])
        self.assertLess(cases[-1]["output_s"], 0)
        self.assertEqual(cases[-1]["classification"], "negative_time_invalid")

    def test_finite_zero_return_does_not_mean_valid_rt60(self):
        decay = self.report["results"]["pyroomacoustics"]["measure_rt60"]
        good, zero, short = decay["cases"]
        self.assertAlmostEqual(good["output_s"], .6, delta=1e-10)
        self.assertTrue(good["measurement_valid_for_fixture"])
        for bad in (zero, short):
            self.assertEqual(bad["output_s"], 0)
            self.assertFalse(bad["measurement_valid_for_fixture"])
        self.assertEqual(zero["zero_energy_edc_classification"], "undefined_log_and_normalization")
        self.assertTrue(any(w["category"] == "RuntimeWarning" for w in zero["warnings"]))

    def test_crb_units_against_hand_derived_position_second_moment(self):
        # Sum of squared centred positions = d^2 * M*(M^2-1)/12.
        second_moment = .014**2 * 6*35/12
        for case in self.report["results"]["doatools"]["cases"]:
            radians = math.radians(case["angle_deg"])
            fisher = 2*100*10*(2*math.pi/.343)**2*math.cos(radians)**2*second_moment
            variance_rad = 1/fisher
            expected = variance_rad * ((180/math.pi)**2 if case["source_unit"] == "deg" else 1)
            self.assertAlmostEqual(case["variance"] / expected, 1, delta=1e-12)
            self.assertAlmostEqual(case["standard_deviation_deg"],
                                   1.1941938389179 if case["angle_deg"] == 0 else 2.3883876778358,
                                   delta=1e-12)


class ExtractionAndGuardTests(unittest.TestCase):
    def test_only_selected_node_runs_and_only_requested_decorator_is_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fixture.py"
            source.write_text("raise RuntimeError('module body must not run')\n"
                              "@missing_jit\ndef selected(x):\n    return x + 3\n"
                              "def ignored():\n    raise RuntimeError('not selected')\n")
            ns = {}
            with patch.object(audit, "ROOT", root):
                evidence = audit.extract(source, ("selected",), ns, remove_decorators=("selected",))
            self.assertEqual(ns["selected"](7), 10)
            self.assertNotIn("ignored", ns)
            self.assertEqual(evidence[0]["removed_decorators"], ["missing_jit"])
            self.assertEqual(ast.parse(source.read_text()).body[1].decorator_list[0].id, "missing_jit")

    @unittest.skipUnless(HAS_SOURCES, "Independent fixed source checkouts are not installed")
    def test_actual_fixed_methods_and_git_blobs(self):
        result = audit.run_audit()
        self.assertTrue(result["all_expected_behaviors_observed"])
        old = audit.strict_json_loads(REPORT.read_bytes())
        for name in audit.HISTORICAL_PROJECTS:
            source = result["sources"]["projects"][name]
            self.assertEqual(source["source_sha256"], old["sources"]["projects"][name]["source_sha256"])
            self.assertEqual(result["results"][name], old["results"][name])
        for name, source in result["sources"]["projects"].items():
            self.assertEqual(source["actual_origin"], audit.ORIGINS[name])
            self.assertTrue(source["used_source_identity_verified"])
            self.assertTrue(source["worktree_clean"])
        for name in ("acoular", "doatools"):
            scope = result["sources"]["projects"][name]["live_acquisition_scope"]
            self.assertEqual(scope["status"], "source_selection_mismatch")
            self.assertFalse(scope["source_selection_verified"])
        for name in ("pyroomacoustics", "pyfar", "speed-of-sound-in-air"):
            self.assertEqual(result["sources"]["projects"][name]["live_acquisition_scope"]["status"], "source_verified")
        current = audit.strict_json_loads(audit.CURRENT_REPORT.read_bytes())
        self.assertEqual(result["results"]["pyfar"], current["results"]["pyfar"])


class PathAndJsonTests(unittest.TestCase):
    def test_all_protected_repository_paths_reject_before_execution(self):
        for path in (REPORT, Path(audit.__file__), audit.LOCK, audit.STATUS,
                     audit.CACHE / "pyfar/new.json", audit.ROOT / "site/new.json",
                     audit.ROOT / "dist/new.json", audit.ROOT / "reviews/new.json"):
            with self.subTest(path=path), patch.object(audit, "run_audit") as run:
                with self.assertRaises(ValueError):
                    audit.main(["--report", str(path)])
                run.assert_not_called()

    def test_links_and_directory_leaf_reject_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            victim = base / "victim.json"
            victim.write_bytes(b"preserved")
            leaf = base / "leaf.json"
            leaf.symlink_to(victim)
            parent = base / "alias"
            parent.symlink_to(base, target_is_directory=True)
            for path in (leaf, parent / "new.json", base):
                with self.subTest(path=path), patch.object(audit, "run_audit") as run:
                    with self.assertRaises(ValueError):
                        audit.main(["--report", str(path)])
                    run.assert_not_called()
            self.assertEqual(victim.read_bytes(), b"preserved")

    def test_stdout_is_read_only_and_explicit_report_is_strict(self):
        with patch.object(audit, "run_audit", return_value={"control": 1}), \
                patch.object(audit, "write_json_report") as write, redirect_stdout(io.StringIO()) as stream:
            audit.main([])
            self.assertEqual(audit.strict_json_loads(stream.getvalue())["control"], 1)
            write.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "new-parent/report.json"
            with patch.object(audit, "run_audit", return_value={"control": 1}), redirect_stdout(io.StringIO()):
                audit.main(["--report", str(target)])
            self.assertEqual(audit.strict_json_loads(target.read_bytes())["control"], 1)
            self.assertEqual(list(target.parent.iterdir()), [target])
            before = target.read_bytes()
            with patch.object(audit, "run_audit", return_value={"bad": float("nan")}):
                with self.assertRaises(ValueError):
                    audit.main(["--report", str(target)])
            self.assertEqual(target.read_bytes(), before)


class OfflineSourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.checkout = self.base / "fixture"
        audit.run_git(["init", str(self.checkout)])
        audit.run_git(["config", "user.name", "offline test"], cwd=self.checkout)
        audit.run_git(["config", "user.email", "offline@example.invalid"], cwd=self.checkout)
        self.source = self.checkout / "source.py"
        self.source.write_bytes(b"locked source\n")
        (self.checkout / "LICENSE").write_bytes(b"fixture licence\n")
        audit.run_git(["add", "source.py", "LICENSE"], cwd=self.checkout)
        audit.run_git(["commit", "-m", "offline fixture"], cwd=self.checkout)
        self.revision = audit.run_git(["rev-parse", "HEAD"], cwd=self.checkout)
        self.origin = "https://example.invalid/fixture.git"
        audit.run_git(["remote", "add", "origin", self.origin], cwd=self.checkout)
        self.config = {"fixture": {"revision": self.revision, "license": "MIT", "files": ("source.py", "LICENSE")}}
        self.entry = {"id": "fixture", "revision": self.revision, "url": self.origin, "license": "MIT", "entrypoints": ["source.py", "LICENSE"]}
        self.lock = self.base / "lock.json"
        self.status = self.base / "status.json"
        self.lock.write_text(json.dumps({"schema_version": 1, "projects": [self.entry]}))
        self.status.write_text(json.dumps({"schema_version": 1, "lock_sha256": audit.sha256(self.lock),
            "projects": [{"id": "fixture", "revision": self.revision, "status": "source_verified", "source_selection_verified": True}]}))
        self.addCleanup(patch.stopall)
        patch.object(audit, "PROJECTS", self.config).start()
        patch.object(audit, "ORIGINS", {"fixture": self.origin}).start()

    def verify(self):
        return audit.verify_sources(self.base, self.lock, self.status)

    def test_actual_origin_head_dirty_and_changed_blob(self):
        self.assertTrue(self.verify()["projects"]["fixture"]["used_source_identity_verified"])
        audit.run_git(["remote", "set-url", "origin", "https://example.invalid/other.git"], cwd=self.checkout)
        with self.assertRaisesRegex(ValueError, "origin"):
            self.verify()
        audit.run_git(["remote", "set-url", "origin", self.origin], cwd=self.checkout)
        self.source.write_bytes(b"changed source\n")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.verify()
        # Force only status to look clean: independent blob checks must still fail.
        original = audit._git
        def hide_dirty(checkout, *args):
            return "" if args[0] == "status" else original(checkout, *args)
        with patch.object(audit, "_git", side_effect=hide_dirty):
            with self.assertRaisesRegex(ValueError, "blob"):
                self.verify()
        self.assertEqual(self.source.read_bytes(), b"changed source\n")
        audit.run_git(["add", "source.py"], cwd=self.checkout)
        audit.run_git(["commit", "-m", "different head"], cwd=self.checkout)
        with self.assertRaisesRegex(ValueError, "HEAD"):
            self.verify()

    def test_outer_git_environment_cannot_redirect_or_inject_configuration(self):
        with patch.dict(os.environ, {"GIT_DIR": "/absent", "GIT_WORK_TREE": "/absent",
                "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "remote.origin.url",
                "GIT_CONFIG_VALUE_0": "https://example.invalid/forged.git"}):
            self.assertEqual(self.verify()["projects"]["fixture"]["actual_origin"], self.origin)

    def test_invalid_source_json_and_unbound_status(self):
        valid_lock, valid_status = self.lock.read_bytes(), self.status.read_bytes()
        for invalid in (b'{"schema_version":1,"schema_version":1,"projects":[]}',
                        b'{"value":NaN}', b'{"value":1e999}', b'[]', b'\xff'):
            for path in (self.lock, self.status):
                with self.subTest(invalid=invalid, path=path):
                    path.write_bytes(invalid)
                    with self.assertRaises(ValueError):
                        self.verify()
                    self.lock.write_bytes(valid_lock)
                    self.status.write_bytes(valid_status)
        state = audit.strict_json_loads(valid_status)
        state["lock_sha256"] = "0"*64
        self.status.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "bind"):
            self.verify()

    def test_linked_source_and_licence_are_rejected(self):
        for relative in ("source.py", "LICENSE"):
            path = self.checkout / relative
            raw = path.read_bytes()
            outside = self.base / (relative + ".outside")
            outside.write_bytes(raw)
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaises(ValueError):
                self.verify()
            path.unlink()
            path.write_bytes(raw)

    def test_selection_mismatch_is_retained_separately_from_valid_used_files(self):
        # Change only the declared selection, not any used file or Git tree.
        self.entry["source_paths"] = ["source.py", "LICENSE"]
        self.lock.write_text(json.dumps({"schema_version": 1, "projects": [self.entry]}))
        state = audit.strict_json_loads(self.status.read_bytes())
        state["lock_sha256"] = audit.sha256(self.lock)
        state["projects"][0].update(status="source_selection_mismatch", source_selection_verified=False)
        self.status.write_text(json.dumps(state))
        result = self.verify()["projects"]["fixture"]
        self.assertTrue(result["used_source_identity_verified"])
        self.assertFalse(result["recorded_acquisition"]["source_selection_verified"])
        self.assertEqual(result["live_acquisition_scope"]["status"], "source_selection_mismatch")
        self.assertFalse(result["live_acquisition_scope"]["source_selection_verified"])


class CurrentReportTests(unittest.TestCase):
    def setUp(self):
        self.report = audit.strict_json_loads(audit.CURRENT_REPORT.read_bytes())

    def test_current_dependencies_and_identity_are_bound(self):
        self.assertEqual(self.report["audit_source_sha256"], audit.sha256(Path(audit.__file__)))
        self.assertEqual(self.report["source_sha256"], audit.source_dependencies())
        self.assertEqual(self.report["sources"]["lock_sha256"], audit.sha256(audit.LOCK))
        self.assertEqual(self.report["sources"]["status_sha256"], audit.sha256(audit.STATUS))
        self.assertEqual(set(self.report["method_scopes"]), set(audit.PROJECTS))

    def test_regularized_four_point_control_against_scalar_fractions(self):
        rows = self.report["results"]["pyfar"]["cases"]
        first = rows[0]
        self.assertEqual(first["epsilon"], .25)
        # P=[9/4,5/4,1/4], H=[3/2,1/2,3/2] => [27/20,5/12,3/4].
        for actual, expected in zip(first["response"], (27/20, 5/12, 3/4)):
            self.assertAlmostEqual(actual["real"], expected, delta=1e-12)
            self.assertAlmostEqual(actual["imag"], 0, delta=1e-12)
        np.testing.assert_allclose(first["waveform"], [11/15, 3/20, 19/60, 3/20], rtol=0, atol=1e-12)
        noisy = rows[1]
        for actual, expected in zip(noisy["response"], (1.41, .45-1j/15, .65)):
            self.assertAlmostEqual(actual["real"], expected.real, delta=1e-12)
            self.assertAlmostEqual(actual["imag"], expected.imag, delta=1e-12)
        self.assertAlmostEqual(rows[3]["response"][0]["real"], 0., delta=1e-12)
        self.assertAlmostEqual(rows[3]["response"][0]["imag"], 0., delta=1e-12)
        for row, expected in ((rows[2], (9/10, 5/6, 1/2)), (rows[3], (0., 4/9, 24/17))):
            np.testing.assert_allclose([v["real"] for v in row["response"]], expected, rtol=0, atol=1e-12)

    def test_digital_ess_defaults_and_scope(self):
        control = self.report["results"]["pyfar"]
        ess = control["cases"][-1]
        self.assertEqual((ess["fs_hz"], ess["sweep_samples"], ess["record_samples"], ess["fft_length"]), (8000, 1024, 1027, 2048))
        self.assertLess(ess["unregularized_full_fir_max_error"], 2e-12)
        self.assertGreater(ess["default_fir_max_error"], 1e-5)
        self.assertGreater(ess["default_epsilon"], 0)
        self.assertEqual(len(control["warnings"]), 12)
        self.assertIn("original Signal class, arithmetic and FFT normalization machinery", control["not_executed"])


if __name__ == "__main__":
    unittest.main()
