"""Independent contract expectations; no network, models or audio devices."""
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from codes.chapters.ch10.examples import audit_industrial_contracts as audit


class IndustrialContractOfflineTest(unittest.TestCase):
    def test_q15_floor_is_distinct_from_nearest_even(self):
        self.assertEqual(audit.fir_integer_oracle([1,0,-1,0,1], [16384,16384,0,0]),
                         [0,0,-1,-1,0])
        self.assertEqual(audit.fir_integer_oracle([-32768]*4, [-32768,-32768,0,0]),
                         [32767]*4)

    def test_direct_fir_fraction_expectation(self):
        x, h = [8,16,-8,24,-16,0,32], [8192,-4096,2048,1024]
        # A separate rational model of the represented Q15 multiplication.
        expected = []
        for n in range(len(x)):
            value = sum((Fraction(h[k], 32768)*x[n-k]
                         for k in range(4) if n >= k), Fraction(0))
            expected.append(max(-32768, min(32767, math.floor(value))))
        self.assertEqual(audit.fir_integer_oracle(x, h), expected)

    def test_function_extractor_selects_declaration_not_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.c"
            path.write_text("void earlier(void){target();}\nint target(void) { return 3; }\n")
            fragment, metadata = audit.c_fragment(path, "target")
            self.assertEqual(fragment, "int target(void) { return 3; }")
            self.assertEqual(metadata["first_line"], 2)
            self.assertEqual(metadata["sha256"], audit.digest(fragment.encode()))
            with self.assertRaisesRegex(ValueError, "declaration not found"):
                audit.c_fragment(path, "missing")

    def test_report_rejects_nonstandard_json(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.strict_json({"measurement": value})

    def test_report_refuses_cache_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / "upstream"
            cache.mkdir()
            with self.assertRaisesRegex(ValueError, "forbidden|protected"):
                audit.write_report(cache / "output.json", {"status": "fixture"}, cache)
            outside = root / "outside.json"
            outside.write_text("preserved")
            link = root / "linked.json"
            link.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                audit.write_report(link, {"status": "fixture"}, cache)
            self.assertEqual(outside.read_text(), "preserved")
            linked_directory = root / "linked-directory"
            directory = root / "outside-directory"
            directory.mkdir()
            linked_directory.symlink_to(directory, target_is_directory=True)
            for traversal in [linked_directory / ".." / "outside.json",
                              root / "ordinary" / ".." / "outside.json"]:
                with self.subTest(path=traversal), self.assertRaisesRegex(ValueError, "parent traversal"):
                    audit.write_report(traversal, {"status": "fixture"}, cache)
            self.assertEqual(outside.read_text(), "preserved")
            self.assertEqual(list(directory.iterdir()), [])
            target = root / "regular.json"
            audit.write_report(target, {"status": "fixture"}, cache)
            self.assertEqual(json.loads(target.read_text()), {"status": "fixture"})

    def test_fixed_license_boundaries_are_explicit(self):
        self.assertIn("demo file header BSD-2-Clause", audit.SOURCES["rnnoise"]["license"])
        self.assertIn("weights/data not obtained", audit.SOURCES["fastenhancer"]["license"])
        self.assertEqual(len(audit.SOURCES), 5)


class IndustrialContractActualCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((audit.DEFAULT_CACHE / name).is_dir() for name in audit.SOURCES):
            raise unittest.SkipTest("optional fixed upstream cache is absent; no download attempted")
        if shutil.which("cc") is None:
            raise unittest.SkipTest("optional compiler cc is absent; no installation attempted")
        cls.report = audit.run_audit()

    def test_actual_binding_and_immutable_sources(self):
        r = self.report
        self.assertEqual(r["tool_sha256"], audit.file_sha(audit.__file__))
        self.assertEqual(r["source_lock_sha256"], audit.file_sha(audit.LOCK))
        for name, record in r["sources_before"].items():
            self.assertTrue(record["clean_before"])
            self.assertFalse(record["clean_after"])
            after = r["sources_after"][name]
            self.assertTrue(after["clean_after"])
            self.assertEqual(record["head"], audit.SOURCES[name]["revision"])
            self.assertEqual(record["used_files"], after["used_files"])
            self.assertEqual(record["recorded_complete_selection"], after["recorded_complete_selection"])
            for source in record["used_files"].values():
                self.assertEqual(source["git_blob"], source["actual_blob"])
        self.assertEqual(r["sources_after"]["webrtc"]["live_complete_selection"]["status"], "source_selection_mismatch")
        json.loads(audit.strict_json(r), parse_constant=lambda x: self.fail(x))

    def test_q15_original_whole_vs_chunks_and_rational_model(self):
        cases = {c["name"]: c for c in self.report["contracts"]["cmsis_q15"]["cases"]}
        self.assertEqual(cases["half_lsb"]["observed_q15"], [0,0,-1,-1,0])
        self.assertEqual(cases["saturation"]["observed_q15"], [32767]*4)
        whole = cases["asymmetric_whole"]
        h, x = whole["impulse_q15"], whole["input_q15"]
        expected = [max(-32768,min(32767,math.floor(sum(
            (Fraction(h[k]*x[n-k],32768) for k in range(4) if n>=k), Fraction(0)))))
                    for n in range(len(x))]
        self.assertEqual(whole["observed_q15"], expected)
        self.assertEqual(cases["asymmetric_chunks"]["observed_q15"], expected)
        self.assertEqual(whole["stored_coefficients_q15"], [1024,2048,-4096,8192])

    def test_speex_two_tracker_and_guarded_noise_contract(self):
        result = self.report["contracts"]["speex_noise_helpers"]
        self.assertEqual(result["observed_rows"], [[0.,1.,3.,2.], [2.5,1.75],
                                                 [3.,7.,3.,4.], [3.,2.,9.,2.]])
        self.assertEqual(len(result["fragments"]), 4)
        self.assertIn("MCRA/IMCRA algorithm", result["not_executed"])

    def test_vad_routes_are_not_classifier_accuracy(self):
        r = self.report["contracts"]["webrtc_vad_router"]
        self.assertEqual(r["observed_legal_rows"], [[fs,ms,0,int(fs!=8000)]
                         for fs in [8000,16000,32000,48000] for ms in [10,20,30]])
        self.assertEqual(r["negative_outputs"], [-1]*6)
        self.assertEqual(r["classifier_route_calls"], [3]*4)
        self.assertIn("original CalcVad classifiers", r["not_executed"])

    def test_rnnoise_wrapper_actual_short_read_and_amplitude(self):
        rows = self.report["contracts"]["rnnoise_demo"]["cases"]
        self.assertEqual([r["output_samples"] for r in rows], [960,960,0,0])
        self.assertEqual(rows[0]["process_first_values"], [1000.,1001.,1002.])
        self.assertEqual(rows[0]["first_output"], 1001)
        self.assertEqual(rows[1]["dropped_partial_samples"], 17)
        self.assertEqual(rows[2]["process_first_values"], [])

    def test_fast_wrapper_source_denominator_and_crop(self):
        rows = {r["source_samples"]: r for r in self.report["contracts"]["fastenhancer_wrapper"]["cases"]}
        self.assertEqual(len(rows[1000]["calls"]), 5)
        self.assertEqual(rows[1000]["wrapper_denominator_samples"], 1280)
        self.assertEqual(rows[1000]["wrapper_printed_rtf"], .125)
        self.assertEqual(rows[1000]["source_time_rtf"], .16)
        self.assertEqual(rows[16000]["wrapper_denominator_samples"], 16384)
        self.assertEqual(rows[16000]["source_time_rtf"], .01)
        for length, row in rows.items():
            self.assertEqual(row["saved_output_samples"], length)
            self.assertEqual(row["saved_rate_hz"], 16000)
            self.assertEqual(row["saved_dtype"], "float32")
            self.assertEqual(row["first_saved_value"], 256/65536 if length else None)
        self.assertIsNone(rows[0]["source_time_rtf"])
        self.assertEqual(rows[0]["source_time_classification"], "undefined_zero_source_duration")

    def test_all_five_scopes_are_limited(self):
        contracts = self.report["contracts"]
        self.assertEqual(len(contracts), 5)
        for result in contracts.values():
            self.assertTrue(result["not_executed"])
            self.assertTrue(result["substitutions"])


if __name__ == "__main__":
    unittest.main()
