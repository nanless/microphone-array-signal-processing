"""Offline independent oracles; no downloaded source or scientific packages."""
import itertools
import hashlib
import subprocess
import json
from pathlib import Path
import unittest

from codes.chapters.ch11.examples import audit_meeting_scoring_interfaces as audit

ROOT = Path(__file__).resolve().parents[1]


def edit_distance(reference, hypothesis):
    # Independent full matrix, rather than the upstream one-row algorithm.
    matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        matrix[i][0] = i
    for j in range(len(hypothesis) + 1):
        matrix[0][j] = j
    for i, a in enumerate(reference, 1):
        for j, b in enumerate(hypothesis, 1):
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + (a != b))
    return matrix[-1][-1]


class MeetingScoringReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "codes/chapters/ch11/reports/meeting_scoring_interfaces.json").read_text(),
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def test_binding_and_execution_scopes(self):
        r = self.report
        historical = subprocess.check_output(["git", "show", "801bfb9e7cb6e2cb56a2e23920c991d44e2acdc6:codes/chapters/ch11/examples/audit_meeting_scoring_interfaces.py"])
        self.assertEqual(r["harness_sha256"], hashlib.sha256(historical).hexdigest())
        self.assertEqual(hashlib.sha256((ROOT / "codes/chapters/ch11/reports/meeting_scoring_interfaces.json").read_bytes()).hexdigest(), "0d12839fd2db92d4baefd6f67077f670b858b4b03be90b4a6802cd733c1f7bf5")
        self.assertEqual(r["source_config_sha256"], audit.binding_sha256())
        self.assertEqual(r["sources"], audit.SOURCES)
        self.assertEqual(r["config"], audit.CONFIG)
        self.assertFalse(r["chime"]["challenge_pipeline_executed"])
        self.assertFalse(r["chime"]["official_normalizer_executed"])
        self.assertIn("not production compiled kernels", r["documentation"]["execution"])
        self.assertIn("no WER scoring", r["package"]["multifile_execution"])

    def test_slot_reuse_against_independent_exhaustive_assignment(self):
        # Three speaker references but only two reused CSS output streams.
        refs, hyps = ["a", "b", "c"], ["ac", "b", ""]
        cp = min(sum(edit_distance(r, h) for r, h in zip(refs, permutation))
                 for permutation in itertools.permutations(hyps))
        costs = []
        for assignment in itertools.product(range(2), repeat=3):
            streams = ["".join(r for r, a in zip(refs, assignment) if a == i) for i in range(2)]
            costs.append(sum(edit_distance(r, h) for r, h in zip(streams, hyps)))
        self.assertEqual(cp, 2)
        self.assertEqual(min(costs), 0)
        docs = self.report["documentation"]
        self.assertEqual(docs["cp_errors"], cp)
        self.assertEqual(docs["orc_errors"], min(costs))
        self.assertAlmostEqual(docs["cp_rate"], 2 / 3)
        self.assertEqual(docs["identity_cp_errors"], 0)

    def test_utterances_cannot_be_split_to_claim_zero_orc(self):
        # Enumerate the four permitted assignments of whole utterances a / bc.
        expected = [sum(edit_distance(r, h) for r, h in zip(refs, ["ab", "c"]))
                    for refs in [("abc", ""), ("a", "bc"), ("bc", "a"), ("", "abc")]]
        self.assertEqual(expected, [2, 2, 3, 4])
        self.assertEqual(self.report["documentation"]["orc_indivisible_utterance_errors"], min(expected))

    def test_documentation_typo_is_not_production_algorithm_result(self):
        r = self.report["documentation"]["di_cp_documentation_call"]
        self.assertEqual(r["exception_type"], "NameError")
        self.assertIn("hypothesis", r["exception_message"])
        self.assertIn("static only", self.report["static_di_cp"]["execution"])
        self.assertEqual(self.report["static_di_cp"]["normalization"], "original reference word count")

    def test_production_cp_failure_preserved(self):
        r = self.report["package"]["cp_word_error_rate"]
        self.assertEqual(r["status"], "exception")
        self.assertEqual(r["exception_type"], "ModuleNotFoundError")
        self.assertIn("cy_levenshtein", r["exception_message"])
        self.assertEqual([s["speaker"] for s in r["reference"]], ["A", "B", "C"])
        self.assertEqual([s["speaker"] for s in r["hypothesis"]], ["slot0", "slot1", "slot0"])

    def test_original_multifile_dispatcher_boundary(self):
        rows = {r["case"]: r for r in self.report["package"]["multifile_cases"]}
        one = rows["one_missing_of_ten"]
        self.assertEqual(one["status"], "returned")
        self.assertEqual(len(one["value"]), 10)
        self.assertEqual(one["value"]["9"], {"reference_segments": 1, "hypothesis_segments": 0})
        for name in ["one_missing_of_two", "extra_hypothesis", "empty_reference"]:
            self.assertEqual(rows[name]["exception_type"], "RuntimeError")
        partial = rows["partial_extra_hypothesis"]
        self.assertEqual(list(partial["value"]), ["0"])
        self.assertTrue(partial["partial"])

    def test_chime_missing_files_not_silently_treated_as_success(self):
        cases = {r["case"]: r for r in self.report["chime"]["cases"]}
        for name in ["first_reference_missing", "first_hypothesis_missing"]:
            self.assertEqual(cases[name]["exception_type"], "UnboundLocalError")
            self.assertEqual(cases[name]["yielded"], [])
        for name, field in [("second_reference_missing", "reference"),
                            ("second_hypothesis_missing", "hypothesis")]:
            row = cases[name]["yielded"][1]
            self.assertEqual(row["scenario"], "mixer6")
            self.assertEqual(row[field][0]["session_id"], "chime6_session")
            self.assertTrue(cases[name]["logs"])
        self.assertEqual([r["scenario"] for r in cases["skip_second_reference"]["yielded"]],
                         ["chime6", "dipco", "notsofar1"])
        self.assertEqual(len(cases["complete"]["yielded"]), 4)

    def test_non_idempotent_fake_normalizer_stops_before_fixed_point(self):
        case = next(r for r in self.report["chime"]["cases"] if r["case"] == "non_idempotent_normalizer")
        # Independent repeated deletion reaches the empty string in 3 calls.
        word = "aaa"
        for _ in range(3):
            word = word[1:]
        self.assertEqual(word, "")
        self.assertEqual(case["status"], "returned")
        self.assertEqual(case["yielded"][0]["reference"][0]["words"], "aa")
        self.assertNotEqual(case["yielded"][0]["reference"][0]["words"], word)


if __name__ == "__main__":
    unittest.main()
