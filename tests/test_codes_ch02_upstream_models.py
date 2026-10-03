"""Independent scalar expectations and source-integrity guards for ch02 audit."""
import ast
import hashlib
import json
import math
from pathlib import Path
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

    def test_current_source_and_lock_bindings(self):
        self.assertEqual(self.report["audit_source_sha256"], audit.sha256(Path(audit.__file__)))
        verify_lock_binding(self.report["sources"]["lock_sha256"], tuple(audit.PROJECTS),
                            current_lock=audit.LOCK)
        lock = json.loads(audit.LOCK.read_text())
        for name, spec in audit.PROJECTS.items():
            source = self.report["sources"]["projects"][name]
            self.assertEqual(source["revision"], spec["revision"])
            self.assertEqual(source["license"], spec["license"])
            entry = next(p for p in lock["projects"] if p["id"] == name)
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

    def test_head_dirty_and_blob_mismatch_reject_without_changing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            checkout = cache / "fixture"
            (checkout / ".git").mkdir(parents=True)
            source = checkout / "source.py"
            source.write_bytes(b"locked source\n")
            lock = cache / "lock.json"
            lock.write_text(json.dumps({"projects": [{"id": "fixture", "revision": "abc",
                                                       "license": "MIT", "url": "test"}]}))
            config = {"fixture": {"revision": "abc", "license": "MIT", "files": ("source.py",)}}
            for responses in ([b"other\n"], [b"abc\n", b" M source.py\n"],
                              [b"abc\n", b"", b"changed blob\n"]):
                with patch.object(audit, "PROJECTS", config), patch.object(audit, "_git", side_effect=responses):
                    with self.assertRaises(ValueError):
                        audit.verify_sources(cache, lock)
                self.assertEqual(source.read_bytes(), b"locked source\n")

    @unittest.skipUnless(HAS_SOURCES, "Independent fixed source checkouts are not installed")
    def test_actual_fixed_methods_and_git_blobs(self):
        result = audit.run_audit()
        self.assertTrue(result["all_expected_behaviors_observed"])
        for name, source in result["sources"]["projects"].items():
            self.assertEqual(source, json.loads(REPORT.read_text())["sources"]["projects"][name])


if __name__ == "__main__":
    unittest.main()
