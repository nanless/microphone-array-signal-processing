"""Independent full-matrix word-edit oracles and read-only native/report contracts."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from codes.chapters.ch11.examples import audit_meeting_kernel_contracts as audit


def matrix_distance(ref, hyp, ref_times=None, hyp_times=None):
    # Independent rectangular DP; prohibited pairs can only insert/delete.
    matrix = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        matrix[i][0] = i
    for j in range(len(hyp) + 1):
        matrix[0][j] = j
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(hyp, 1):
            choices = [matrix[i-1][j] + 1, matrix[i][j-1] + 1]
            if ref_times is None or (ref_times[i-1][0] < hyp_times[j-1][1] and hyp_times[j-1][0] < ref_times[i-1][1]):
                choices.append(matrix[i-1][j-1] + int(a != b))
            matrix[i][j] = min(choices)
    return matrix[-1][-1]


class MeetingKernelContractTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.folder.cleanup)
        self.directory = Path(self.folder.name)

    def test_hand_cases_match_independent_full_matrix(self):
        expected = [(0, 2), (0, 2), (0, 0), (1, 1), (1, 2), (2, 2), (2, 2), (0, 0), (0, 4)]
        self.assertEqual(len(audit.CASES), 9)
        for case, pair in zip(audit.CASES, expected):
            with self.subTest(case=case[0]):
                _, ref, hyp, rt, ht, ordinary, timed = case
                self.assertEqual((ordinary, timed), pair)
                self.assertEqual(matrix_distance(ref, hyp), ordinary)
                self.assertEqual(matrix_distance(ref, hyp, rt, ht), timed)

    def test_whole_chinese_words_are_not_character_tokens(self):
        text, words = audit.driver_text(Path("/private/tmp/space path/header.h"))
        self.assertEqual(set(words), {"春天", "夏天", "abc", "b"})
        self.assertEqual(len(words), 4)
        self.assertIn('#include "/private/tmp/space path/header.h"', text)
        self.assertIn("levenshtein_distance_(r,h)", text)
        self.assertNotIn("namespace", text)

    def test_strict_json_rejects_nonfinite_and_overflow(self):
        for value in ["NaN", "Infinity", "-Infinity", "1e999"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.strict_loads('{"value":' + value + '}')
        self.assertEqual(audit.strict_loads('{"value":1e-99}'), {"value": 1e-99})

    def test_strict_json_rejects_duplicate_fields_at_any_depth(self):
        for value in ['{"revision":"wrong","revision":"fixed"}',
                      '{"rows":[{"ordinary":2,"ordinary":0}]}',
                      '{"name":1,"na\\u006de":2}']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "duplicate"):
                audit.strict_loads(value)

    def test_equal_names_in_separate_objects_and_similar_names_are_legal(self):
        value = '{"rows":[{"ordinary":0},{"ordinary":2}],"Ordinary":3,"ordinary_count":4}'
        self.assertEqual(audit.strict_loads(value), {"rows": [{"ordinary": 0}, {"ordinary": 2}],
                                                  "Ordinary": 3, "ordinary_count": 4})

    def test_atomic_report_ordinary_new_and_existing_files(self):
        path = self.directory / "report.json"
        audit.write_report(path, {"a": 1})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})
        audit.write_report(path, {"a": 2})
        self.assertEqual(json.loads(path.read_text()), {"a": 2})
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ["report.json"])

    def test_nonfinite_output_does_not_change_old_file(self):
        path = self.directory / "report.json"
        path.write_text("original")
        with self.assertRaises(ValueError):
            audit.write_report(path, {"bad": float("nan")})
        self.assertEqual(path.read_text(), "original")

    def test_reject_output_symlink_and_parent_symlink(self):
        original = self.directory / "original.json"
        original.write_text("safe")
        link = self.directory / "link.json"
        link.symlink_to(original)
        folder_link = self.directory / "folder-link"
        folder_link.symlink_to(self.directory, target_is_directory=True)
        for target in [link, folder_link / "other.json"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                audit.write_report(target, {"a": 1})
        self.assertEqual(original.read_text(), "safe")
        self.assertFalse((self.directory / "other.json").exists())

    def test_reject_lexical_parent_even_if_link_would_be_cancelled(self):
        link = self.directory / "linked"
        link.symlink_to(self.directory, target_is_directory=True)
        for target in [link / ".." / "report.json", self.directory / ".." / "report.json"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                audit.write_report(target, {"a": 1})

    def test_reject_directory_missing_parent_and_cache_output(self):
        for target in [self.directory, self.directory / "missing" / "report.json", self.directory / "upstream.json"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                audit.write_report(target, {}, cache=self.directory)

    def test_near_cache_name_is_legal(self):
        cache = self.directory / "upstream"
        cache.mkdir()
        neighbor = self.directory / "upstream-neighbor.json"
        audit.write_report(neighbor, {"ok": True}, cache=cache)
        self.assertEqual(json.loads(neighbor.read_text()), {"ok": True})

    def test_cli_preflight_rejects_before_native_run(self):
        with mock.patch.object(audit, "run_audit") as run:
            with self.assertRaises(ValueError):
                audit.main(["--report", str(self.directory / ".." / "report.json")])
            run.assert_not_called()

    def test_cli_default_stdout_has_no_report_write(self):
        with mock.patch.object(audit, "run_audit", return_value={"ok": True}), mock.patch.object(audit, "write_report") as write, mock.patch("builtins.print") as output:
            audit.main([])
            write.assert_not_called()
            self.assertEqual(json.loads(output.call_args.args[0]), {"ok": True})

    def test_source_verification_uses_shared_fixed_contract(self):
        identity = {"head": audit.REVISION, "origin": audit.ORIGIN,
                    "checkout": str(self.directory / "meeteval"), "lock_sha256": "a" * 64,
                    "used_files": {name: {"sha256": sha} for name, sha in audit.FILES.items()}}
        with mock.patch.object(audit.upstream, "verify_project", return_value=identity) as verify, \
             mock.patch.object(audit.upstream, "source_documents", return_value=({"projects": [1]}, {}, "", "")):
            result = audit.verify_sources(self.directory)
        verify.assert_called_once_with("meeteval", self.directory / "meeteval", audit.FILES)
        self.assertIs(result["source_identity"], identity)

    def test_fixed_digest_failure_is_not_upgraded_by_verified_selection(self):
        identity = {"used_files": {name: {"sha256": "0" * 64} for name in audit.FILES}}
        with mock.patch.object(audit.upstream, "verify_project", return_value=identity), self.assertRaises(ValueError):
            audit.verify_sources(self.directory)

    def test_source_contract_failure_propagates_without_native_run(self):
        for failure in ["wrong origin", "wrong HEAD", "dirty tree", "nonmatching original blob"]:
            with self.subTest(failure=failure), mock.patch.object(audit.upstream, "verify_project", side_effect=ValueError(failure)), self.assertRaisesRegex(ValueError, failure):
                audit.verify_sources(self.directory)

    def test_historical_native_report_binds_real_git_tool(self):
        relative = "codes/chapters/ch11/examples/audit_meeting_kernel_contracts.py"
        historical = audit.subprocess.check_output(["git", "show", "801bfb9e7cb6e2cb56a2e23920c991d44e2acdc6:" + relative])
        report = json.loads((audit.ROOT / "codes/chapters/ch11/reports/meeting_kernel_contracts.json").read_bytes())
        self.assertEqual(hashlib.sha256(historical).hexdigest(), report["tool_sha256"])
        self.assertEqual(hashlib.sha256((audit.ROOT / "codes/chapters/ch11/reports/meeting_kernel_contracts.json").read_bytes()).hexdigest(),
                         "f272a118f944050022f5772dadfc051ca9a36b373ed63461f32c7e23456d9bde")

    def test_fixed_include_digest_and_history_scope(self):
        self.assertEqual(audit.FILES["meeteval/wer/matching/levenshtein.h"], "04e013ba7265a6df8f261f793624ba425b57b7beee9ec9b572f4dd4f1e79ce06")
        self.assertEqual(audit.REVISION, "6e3dc81284f2d6928f7ef9e620fd3b6906daa429")

    def test_current_points_use_strict_inequalities_not_intersection_length(self):
        expected = [0, 2, 2, 0]
        self.assertEqual(len(audit.EXTRA_POINT_CASES), 4)
        for case, count in zip(audit.EXTRA_POINT_CASES, expected):
            _, ref, hyp, rt, ht, ordinary, timed = case
            self.assertEqual(matrix_distance(ref, hyp), ordinary)
            self.assertEqual(matrix_distance(ref, hyp, rt, ht), count)
            self.assertEqual(timed, count)

    def test_native_fixed_original_kernels_optional_cache(self):
        if not (audit.CACHE / "meeteval").exists() or shutil.which("c++") is None:
            self.skipTest("optional fixed MeetEval checkout/C++ compiler is absent")
        report = audit.run_audit()
        self.assertTrue(report["before_clean"] and report["after_clean"])
        self.assertEqual(report["tool_sha256"], hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        self.assertEqual(report["source_lock_sha256"], hashlib.sha256(audit.LOCK.read_bytes()).hexdigest())
        self.assertFalse(report["execution"]["source_patch"])
        self.assertEqual(report["execution"]["algorithm_substitutes"], [])
        self.assertEqual(len(report["cases"]), 13)
        for row in report["cases"]:
            self.assertEqual(row["ordinary"], matrix_distance(row["reference_words"], row["hypothesis_words"]))
            self.assertEqual(row["timed"], matrix_distance(row["reference_words"], row["hypothesis_words"], row["reference_intervals_s"], row["hypothesis_intervals_s"]))
        self.assertIn("cpWER/ORC-WER/tcpWER", report["execution"]["not_executed"][0])


if __name__ == "__main__":
    unittest.main()
