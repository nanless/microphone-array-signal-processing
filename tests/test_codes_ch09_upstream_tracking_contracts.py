"""Independent small-model expectations; no download and no upstream writes."""
from fractions import Fraction
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch09.examples import audit_upstream_tracking_contracts as audit


@unittest.skipUnless(all((audit.CACHE / name).is_dir() for name in audit.REVISIONS),
                     "optional fixed upstream caches missing; no downloads")
class FixedTrackingContracts(unittest.TestCase):
    def test_current_selection_mismatch_keeps_method_identity_separate(self):
        result = audit.verify_sources()
        status = json.loads(audit.LOCK.with_name("SOURCE_STATUS.json").read_text())
        rows = {row["id"]: row for row in status["projects"]}
        for name, source in result.items():
            self.assertEqual(source["acquisition_record"], rows[name])
            self.assertIs(source["required_source_identity_verified"], True)
            self.assertIs(source["source_selection_verified"],
                          rows[name]["source_selection_verified"])
            self.assertTrue(source["worktree_clean"])
        # Independent Fraction expectations and native calls below must still
        # run when the current complete sparse-selection policy does not pass.
        for name in ("spatial-audio-framework", "filterpy", "stonesoup"):
            self.assertEqual(result[name]["acquisition_record"]["status"],
                             "source_selection_mismatch")
            self.assertIs(result[name]["source_selection_verified"], False)

    def test_original_q_against_integral_and_outer_product(self):
        audit.verify_sources()
        result = audit.python_contracts(audit.CACHE)["filterpy_q"]
        h, q = Fraction(1, 2), Fraction(2)
        # Continuous integral of [t,1][t,1]^T vs one constant acceleration
        # sample propagated by [h²/2,h]. These independent models differ.
        expected_continuous = [[q*h**3/3, q*h**2/2], [q*h**2/2, q*h]]
        impulse = [h*h/2, h]
        expected_discrete = [[q*x*y for y in impulse] for x in impulse]
        np.testing.assert_allclose(result["continuous"], np.array(expected_continuous, float), rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["discrete"], np.array(expected_discrete, float), rtol=0, atol=1e-15)
        self.assertNotEqual(result["continuous"], result["discrete"])

    def test_jpda_one_to_one_includes_two_misses(self):
        result = audit.python_contracts(audit.CACHE)["jpda_isvalid"]
        valid = {tuple(r["assignment"]) for r in result["events"] if r["valid"]}
        expected = {(-1,-1),(-1,0),(-1,1),(-1,2),(0,-1),(1,-1),(2,-1),
                    (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)}
        self.assertEqual(valid, expected)
        self.assertEqual(result["valid_count"], 13)

    @unittest.skipUnless(shutil.which("cc"), "optional compiler unavailable")
    def test_native_mean_covariance_and_delayed_state_contracts(self):
        before = audit.verify_sources()
        result = audit.c_contracts(audit.CACHE)
        odas = dict(zip(result["odas"]["output_order"], result["odas"]["actual"]))
        # FPF^T on the x/v block: [[1+h²,h],[h,1]]; only velocity gets .2².
        for key, expected in {"F03": .5, "Q00": 0, "Q33": .04, "R00": .01,
                              "direction_x": 1, "velocity_x": 0,
                              "P00": 1.25, "P03": .5, "P33": 1.04}.items():
            self.assertAlmostEqual(odas[key], expected, delta=1e-6)
        self.assertNotAlmostEqual(odas["direction_x"], 1.5)
        saf = result["saf"]["actual"]
        self.assertEqual(saf["empty1"], [1,0,9,1,101])
        self.assertEqual(saf["empty2"], [2,0,9,1,101])
        self.assertEqual(saf["resume"], [0,3,1,12,3])
        self.assertEqual(saf["twoobs"], [0,4,3,13,1,0])
        self.assertEqual(saf["copy"][:4], [404]*4)
        self.assertEqual(saf["copy"][4:8], [.25]*4)
        self.assertIn("original prediction/update", result["saf"]["not_run"])
        self.assertEqual(audit.verify_sources(), before)

    def test_wrong_digest_rejected_before_native_execution(self):
        wrong = copy.deepcopy(audit.FILES)
        wrong["odas"]["LICENSE"] = "0"*64
        with patch.object(audit, "FILES", wrong):
            with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
                audit.verify_sources()


class ReadOnlyContracts(unittest.TestCase):
    def test_current_report_binds_real_source_and_preserves_acquisition_states(self):
        path = audit.ROOT / "codes/chapters/ch09/reports/upstream_tracking_contracts.json"
        report = json.loads(path.read_text())
        self.assertEqual(report["tool_sha256"],
                         hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        self.assertEqual(report["source_lock_sha256"],
                         hashlib.sha256(audit.LOCK.read_bytes()).hexdigest())
        status_path = audit.LOCK.with_name("SOURCE_STATUS.json")
        self.assertEqual(report["source_status_sha256"],
                         hashlib.sha256(status_path.read_bytes()).hexdigest())
        rows = {row["id"]: row for row in json.loads(status_path.read_text())["projects"]}
        self.assertEqual(report["sources_before"], report["sources_after"])
        for name, source in report["sources_before"].items():
            self.assertEqual(source["acquisition_record"], rows[name])
            self.assertIs(source["required_source_identity_verified"], True)
            self.assertIs(source["source_selection_verified"], rows[name]["source_selection_verified"])

    def test_source_function_fragment_is_exact(self):
        if not (audit.CACHE / "spatial-audio-framework").is_dir():
            self.skipTest("optional SAF cache missing")
        source = audit.CACHE / "spatial-audio-framework/framework/modules/saf_tracker/saf_tracker.c"
        fragment = audit.extract_c_step(source)
        self.assertIn(fragment, source.read_text())
        self.assertIn("//resampstr", fragment)
        self.assertIn("pData->incrementTime++", fragment)
        self.assertEqual(audit.digest(fragment.encode()), hashlib.sha256(fragment.encode()).hexdigest())

    def test_strict_json_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            audit.stable_digest({"not_a_valid_result": float("nan")})


class SourceIdentityFixtures(unittest.TestCase):
    """Real temporary Git fixtures; no mutation of an acquired checkout."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="masp-tracking-identity-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.checkout = self.cache / "fixture"
        self.checkout.mkdir(parents=True)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git("init", "-q")
        self.origin = "https://example.invalid/official/fixture.git"
        self.git("remote", "add", "origin", self.origin)
        self.contents = {"LICENSE": b"Fixture license; not an upstream license\n",
                         "method.py": b"def original():\n    return 3\n"}
        for name, data in self.contents.items():
            (self.checkout / name).write_bytes(data)
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "identity fixture")
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.lock = self.root / "SOURCES.lock.json"
        self.lock.write_text(json.dumps({"projects": [{"id": "fixture",
            "revision": self.revision, "url": self.origin, "license": "Fixture-only"}]}))
        self.row = {"id": "fixture", "revision": self.revision,
                    "status": "source_selection_mismatch", "missing_entrypoints": [],
                    "source_selection_verified": False,
                    "observed_sparse_patterns": ["/*", "!*.wav"],
                    "expected_sparse_patterns": ["/*", "!*.wav", "!*.mat"]}
        self.status = {"lock_sha256": audit.digest(self.lock.read_bytes()),
                       "projects": [self.row]}
        self.save_status()
        for attribute, value in (("LOCK", self.lock),
                ("REVISIONS", {"fixture": self.revision}),
                ("FILES", {"fixture": {name: audit.digest(data)
                    for name, data in self.contents.items()}})):
            patcher = patch.object(audit, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.checkout), *args],
                                       env=self.env, text=True, stderr=subprocess.PIPE)

    def save_status(self):
        self.lock.with_name("SOURCE_STATUS.json").write_text(json.dumps(self.status))

    def test_method_identity_passes_without_relabeling_selection(self):
        before = self.lock.with_name("SOURCE_STATUS.json").read_bytes()
        result = audit.verify_sources(self.cache)["fixture"]
        self.assertEqual(result["acquisition_record"], self.row)
        self.assertIs(result["source_selection_verified"], False)
        self.assertIs(result["required_source_identity_verified"], True)
        self.assertEqual(result["origin"], self.origin)
        self.assertEqual(before, self.lock.with_name("SOURCE_STATUS.json").read_bytes())
        self.row.update(status="source_verified", source_selection_verified=True)
        self.save_status()
        self.assertIs(audit.verify_sources(self.cache)["fixture"]["source_selection_verified"], True)

    def test_failed_missing_stale_and_inconsistent_records_are_rejected(self):
        original = copy.deepcopy(self.row)
        changes = [{"status": status} for status in
                   ("failed", "index_only", "missing", "entrypoints_missing")]
        changes += [{"missing_entrypoints": ["method.py"]},
                    {"missing_entrypoints": None}, {"revision": "0"*40},
                    {"source_selection_verified": True},
                    {"source_selection_verified": 0}]
        for change in changes:
            with self.subTest(change=change):
                self.row.clear(); self.row.update(original); self.row.update(change)
                self.save_status()
                with self.assertRaisesRegex(ValueError, "source identity unavailable"):
                    audit.verify_sources(self.cache)
        self.row.clear(); self.row.update(original)
        self.status["lock_sha256"] = "0"*64
        self.save_status()
        with self.assertRaisesRegex(ValueError, "current lock"):
            audit.verify_sources(self.cache)

    def test_duplicate_status_and_lock_ids_are_rejected(self):
        self.status["projects"].append(copy.deepcopy(self.row))
        self.save_status()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.verify_sources(self.cache)
        self.status["projects"].pop()
        lock = json.loads(self.lock.read_text())
        lock["projects"].append(copy.deepcopy(lock["projects"][0]))
        self.lock.write_text(json.dumps(lock))
        self.status["lock_sha256"] = audit.digest(self.lock.read_bytes())
        self.save_status()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.verify_sources(self.cache)

    def test_wrong_origin_and_head_are_rejected(self):
        self.git("remote", "set-url", "origin", self.origin + "-nearby")
        with self.assertRaisesRegex(ValueError, "origin mismatch"):
            audit.verify_sources(self.cache)
        self.git("remote", "set-url", "origin", self.origin)
        with patch.object(audit, "REVISIONS", {"fixture": "0"*40}):
            self.row["revision"] = "0"*40
            self.save_status()
            with self.assertRaisesRegex(ValueError, "HEAD mismatch"):
                audit.verify_sources(self.cache)

    def test_tracked_and_untracked_changes_are_rejected(self):
        method = self.checkout / "method.py"
        method.write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "worktree changes"):
            audit.verify_sources(self.cache)
        method.write_bytes(self.contents["method.py"])
        (self.checkout / "untracked.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "worktree changes"):
            audit.verify_sources(self.cache)

    def test_original_file_and_license_digests_are_required(self):
        for name in self.contents:
            with self.subTest(name=name):
                files = copy.deepcopy(audit.FILES)
                files["fixture"][name] = "0"*64
                with patch.object(audit, "FILES", files):
                    with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
                        audit.verify_sources(self.cache)


if __name__ == "__main__":
    unittest.main()
