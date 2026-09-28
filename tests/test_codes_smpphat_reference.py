"""Offline checks for the saved SMP-PHAT C reproduction and analytic baseline."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from codes.chapters.ch04.examples import reproduce_smpphat_reference as experiment


REPORT = experiment.ROOT / "chapters/ch04/reports/smpphat_reference.json"
PROVIDED_FFTW_HEADER_SHA256 = "0dd1b988b681d5797aaf10c2c57c107f7dca510695e4b94b3855eb52550d282c"
PROVIDED_FFTW_LIBRARY_SHA256 = "7d5247b5e7a20d6dcb5969cddb2b80ac85dd9ba0348174f03df0e69bafb3de99"


class SMPPHATReferenceTests(unittest.TestCase):
    def test_fixed_phat_and_grouping_have_independent_expected_values(self):
        directions = experiment.fixed_directions()
        cases = experiment.fixed_geometries()
        self.assertEqual(experiment.TRUE_INDEX, 3)
        np.testing.assert_array_equal(directions[0], [0, 1, 0])
        np.testing.assert_allclose(directions[6], [1, 0, 0], atol=5e-8)
        np.testing.assert_allclose(directions[3], [2 ** -0.5, 2 ** -0.5, 0], atol=5e-8)
        expected_counts = [4, 6]
        for (_, geometry, _), expected_count in zip(cases, expected_counts):
            phat = experiment.make_phat(geometry, directions[experiment.TRUE_INDEX])
            np.testing.assert_allclose(np.linalg.norm(phat[:, :-1], axis=-1), 1, atol=8e-8)
            np.testing.assert_array_equal(phat[:, -1], 0)
            groups, polarities = experiment.merge_plan(geometry)
            self.assertEqual(int(groups.max()) + 1, expected_count)
            self.assertTrue(np.all(np.isin(polarities, [-1, 1])))

    def test_direct_c2r_has_known_impulse_for_zero_delay(self):
        spectrum = np.ones(experiment.FRAME_SIZE // 2 + 1, dtype=complex)
        correlation = experiment._unscaled_c2r(spectrum)
        self.assertEqual(correlation.shape, (experiment.FRAME_SIZE * experiment.INTERPOLATION_RATE,))
        self.assertEqual(correlation[0], experiment.FRAME_SIZE + 1)
        expected_at_one = 1 + 2 * sum(
            math.cos(2 * math.pi * k / (experiment.FRAME_SIZE * experiment.INTERPOLATION_RATE))
            for k in range(1, experiment.FRAME_SIZE // 2 + 1))
        self.assertAlmostEqual(correlation[1], expected_at_one, places=12)

    def test_saved_report_is_from_current_sources_and_exact_lock(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        provenance = report["provenance"]
        self.assertEqual(provenance["upstream_revision"], experiment.UPSTREAM_REVISION)
        fftw = provenance["fftw"]
        self.assertEqual(fftw["runtime_version"], f"fftw-{experiment.FFTW_VERSION}")
        if fftw["source"] == "checksum_verified_official_archive":
            self.assertEqual(fftw["version"], experiment.FFTW_VERSION)
            self.assertEqual(fftw["archive_url"], experiment.FFTW_URL)
            self.assertEqual(fftw["archive_sha256"], experiment.FFTW_SHA256)
            self.assertRegex(fftw["header_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(fftw["library_sha256"], r"^[0-9a-f]{64}$")
        elif fftw["source"] == "provided_prefix":
            # The prefix's actual header and static library are pinned. Their
            # bytes do not prove that an official source archive was verified.
            self.assertEqual(fftw["version"], "not verified from an archive")
            self.assertIsNone(fftw["archive_url"])
            self.assertIsNone(fftw["archive_sha256"])
            self.assertEqual(fftw["header_sha256"], PROVIDED_FFTW_HEADER_SHA256)
            self.assertEqual(fftw["library_sha256"], PROVIDED_FFTW_LIBRARY_SHA256)
        else:
            self.fail(f"unknown FFTW provenance: {fftw['source']}")
        self.assertEqual(
            provenance["runner_sha256"], hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest())
        harness = Path(experiment.__file__).with_name("reproduce_smpphat_harness.c")
        self.assertEqual(provenance["harness_sha256"], hashlib.sha256(harness.read_bytes()).hexdigest())
        self.assertFalse(report["build"]["upstream_cmake_used"])
        self.assertFalse(report["build"]["algorithm_source_modified"])
        self.assertFalse(report["build"]["fftw_backend_replaced"])
        source = experiment.ROOT / "chapters/ch00/upstream/_downloads/smpphat"
        # The saved-report tests remain usable in a clean clone without the
        # ignored upstream downloads. Verify actual bytes when they are present.
        if source.is_dir():
            for relative, expected in provenance["upstream_file_sha256"].items():
                self.assertEqual(experiment.sha256(source / relative), expected)

    def test_saved_c_scores_match_direct_dft_and_srp_matches_smp(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual([case["upstream_c"]["groups_count"] for case in report["cases"]], [4, 6])
        directions = experiment.fixed_directions()
        for case, (_, fixed_geometry, _) in zip(report["cases"], experiment.fixed_geometries()):
            upstream = case["upstream_c"]
            expected = case["independent_signed_lookup_direct_dft"]
            comparison = case["comparison"]
            geometry = np.asarray(case["geometry_m"], dtype="<f4")
            np.testing.assert_array_equal(geometry, fixed_geometry)
            phat = experiment.make_phat(geometry, directions[experiment.TRUE_INDEX])
            self.assertEqual(hashlib.sha256(phat.tobytes()).hexdigest(), case["phat_float32_sha256"])
            recalculated = experiment.independent_scores(geometry, directions, phat)
            diagnosed = experiment.direct_scores_at_upstream_lookups(phat, upstream)
            for key in ("groups", "polarities", "srp_ranges", "smp_ranges",
                        "srp_lookups", "smp_lookups", "srp_peak_index", "smp_peak_index"):
                self.assertEqual(expected[key], recalculated[key])
            np.testing.assert_allclose(expected["srp_scores"], recalculated["srp_scores"], atol=0)
            np.testing.assert_allclose(expected["smp_scores"], recalculated["smp_scores"], atol=0)
            np.testing.assert_allclose(case["direct_dft_at_upstream_lookups"]["srp_scores"],
                                       diagnosed["srp_scores"], atol=0)
            np.testing.assert_allclose(case["direct_dft_at_upstream_lookups"]["smp_scores"],
                                       diagnosed["smp_scores"], atol=0)
            self.assertEqual(upstream["srp_status"], 0)
            self.assertEqual(upstream["smp_status"], 0)
            self.assertEqual(upstream["pairs_count"], 6)
            self.assertEqual(upstream["groups"], expected["groups"])
            np.testing.assert_array_equal(upstream["polarities"], expected["polarities"])
            self.assertEqual(upstream["srp_peak_index"], experiment.TRUE_INDEX)
            self.assertEqual(upstream["smp_peak_index"], experiment.TRUE_INDEX)
            self.assertTrue(comparison["same_peak"])
            self.assertLess(comparison["c_srp_vs_direct_dft_at_upstream_lookup_max_abs"], 2e-4)
            self.assertLess(comparison["c_smp_vs_direct_dft_at_upstream_lookup_max_abs"], 2e-4)
            self.assertLess(comparison["independent_srp_vs_smp_max_abs"], 1e-9)
            self.assertTrue(comparison["diagnosis_verified"])
            self.assertTrue(comparison["intended_signed_lookup_failed"])
            actual_srp = np.asarray(upstream["srp_scores"])
            actual_smp = np.asarray(upstream["smp_scores"])
            self.assertAlmostEqual(
                comparison["c_srp_vs_intended_signed_lookup_max_abs"],
                float(np.max(np.abs(actual_srp - recalculated["srp_scores"]))))
            self.assertAlmostEqual(
                comparison["c_smp_vs_intended_signed_lookup_max_abs"],
                float(np.max(np.abs(actual_smp - recalculated["smp_scores"]))))
            mismatches = np.count_nonzero(
                np.asarray(upstream["srp_lookups"]) != np.asarray(recalculated["srp_lookups"]))
            self.assertEqual(case["lookup_diagnosis"]["srp_lookup_mismatch_count"], mismatches)
        self.assertTrue(report["cases"][0]["comparison"]["expected_equivalence_failed"])
        self.assertFalse(report["cases"][1]["comparison"]["expected_equivalence_failed"])
        self.assertEqual(report["numerical_status"], "failed_portability")

    def test_default_path_does_not_download_missing_fftw(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "explicitly opt in"):
                experiment.ensure_fftw(Path(directory), download=False)

    def test_wrong_fftw_archive_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"fftw-{experiment.FFTW_VERSION}.tar.gz"
            path.write_bytes(b"not the pinned FFTW archive")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                experiment.ensure_fftw(Path(directory), download=False)

    def test_optional_local_c_rerun(self):
        prefix = os.environ.get("SMPPHAT_FFTW_PREFIX")
        if prefix is None:
            self.skipTest("set SMPPHAT_FFTW_PREFIX to rerun the locked C implementation offline")
        with tempfile.TemporaryDirectory() as directory:
            report = experiment.run_experiment(
                fftw_prefix=Path(prefix), work_dir=Path(directory), download_fftw=False)
        self.assertEqual([case["upstream_c"]["groups_count"] for case in report["cases"]], [4, 6])

    def test_cli_saves_diagnosis_but_returns_distinct_failure_status(self):
        failure = {"numerical_status": "failed_portability"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with mock.patch.object(experiment, "run_experiment", return_value=failure), \
                    mock.patch("sys.argv", ["runner", "--output", str(output)]), \
                    self.assertRaises(SystemExit) as caught:
                experiment.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(json.loads(output.read_text()), failure)

    def test_harness_checks_call_status_before_reading_results(self):
        source = Path(experiment.__file__).with_name("reproduce_smpphat_harness.c").read_text()
        self.assertLess(source.index("srp_status = srp_call"), source.index("if (srp_status != 0)"))
        self.assertLess(source.index("if (srp_status != 0)"), source.index("srp_scores(srp"))
        self.assertLess(source.index("smp_status = smp_call"), source.index("if (smp_status != 0)"))
        self.assertLess(source.index("if (smp_status != 0)"), source.index("smp_scores(smp"))


if __name__ == "__main__":
    unittest.main()
