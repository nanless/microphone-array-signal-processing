"""Independent analytic and existing-PCM checks for Appendix B E13-03..08."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from codes.chapters.appendix_b.appendix_b_experiments import (
    ROOM, _read_results, evidence_claims, four_mic_drr_and_convolution,
    paired_room_comparison, room_pcm_readback, run_exercises,
    t20_from_edc_points, two_mic_srp_phase,
)
from codes.chapters.appendix_b.examples.room_srp_exercise import plot_results, write_results


class AppendixBExperimentsTest(unittest.TestCase):
    def test_published_room_result_binds_source_plot_manifest_and_six_rows(self):
        result = _read_results()
        self.assertEqual(result["status"], "pyroomacoustics_simulation_executed")
        self.assertEqual(result["actual_max_order"], 40)
        self.assertEqual(len(result["results"]), 6)
        self.assertNotIn("elapsed_s", result)
        self.assertNotIn("audio_export", result)
        self.assertTrue(all("elapsed_s" not in row for row in result["results"]))
        self.assertEqual(result["assets"]["figure"]["file"], "ROOM_RESULTS.png")
        self.assertTrue(result["convergence_check"]["within_threshold"])

    def test_e13_03_fixed_pairs_and_seeded_cases_do_not_support_monotonic_claim(self):
        result = paired_room_comparison(_read_results()["results"])
        for pair in result["fixed_pairs"].values():
            self.assertEqual((pair["near_distance_m"], pair["far_distance_m"]), (1.0, 2.5))
            self.assertAlmostEqual(pair["report_unrounded_drr_far_minus_near_db"],
                                   -7.86863427, places=7)
            self.assertAlmostEqual(pair["published_3dp_drr_far_minus_near_db"],
                                   -7.868, places=12)
            self.assertEqual(pair["doa_error_far_minus_near_deg"], 1.0)
        self.assertLess(result["seeded_positions"][0]["distance_m"],
                        result["seeded_positions"][1]["distance_m"])
        self.assertGreater(result["seeded_positions"][0]["absolute_doa_error_deg"],
                           result["seeded_positions"][1]["absolute_doa_error_deg"])
        bad = copy.deepcopy(_read_results()["results"])
        next(row for row in bad if row["name"] == "fixed_far_left")["true_azimuth_deg"] = 0.0
        with self.assertRaisesRegex(ValueError, "retain angle"):
            paired_room_comparison(bad)

    def test_e13_04_median_is_not_ratio_of_pooled_energy(self):
        result = four_mic_drr_and_convolution()
        one_to_four_db = 10 * math.log10(4)
        np.testing.assert_allclose(result["per_mic_drr_db"],
                                   [0, one_to_four_db, -one_to_four_db, one_to_four_db],
                                   atol=1e-14)
        self.assertAlmostEqual(result["median_drr_db"], one_to_four_db / 2)
        self.assertAlmostEqual(result["pooled_energy_drr_db"], 10 * math.log10(10 / 7))
        self.assertNotAlmostEqual(result["median_drr_db"], result["pooled_energy_drr_db"])
        self.assertEqual(result["direct_output"], [1., 1., 0.])
        self.assertEqual(result["reflected_output"], [0., .5, .5])
        self.assertEqual(result["cross_term_in_full_energy"], 1.0)
        self.assertEqual(result["full_output_energy"], 3.5)
        self.assertEqual(result["full_output_energy"], result["direct_output_energy"] +
                         result["reflected_output_energy"] + result["cross_term_in_full_energy"])

    def test_e13_05_extrapolation_is_time_shift_invariant_and_rejects_bad_edc(self):
        original = t20_from_edc_points([.05, .15, .25], [-5, -15, -25])
        shifted = t20_from_edc_points([.15, .25, .35], [-5, -15, -25])
        self.assertAlmostEqual(original["slope_db_per_s"], -100)
        self.assertAlmostEqual(original["t20_s"], .2)
        self.assertAlmostEqual(original["t60_extrapolated_s"], .6)
        self.assertAlmostEqual(shifted["t60_extrapolated_s"], .6)
        for times, decay in (([0, .1], [-5, -25]),
                             ([0, .1, .2], [-5, -15, -14]),
                             ([0, .1, .2], [-4, -14, -24]),
                             ([0, .1, .1], [-5, -15, -25])):
            with self.assertRaises(ValueError):
                t20_from_edc_points(times, decay)

    def test_e13_06_srp_phase_has_correct_sign_and_analytic_scores(self):
        result = two_mic_srp_phase()
        phi = 2 * math.pi * 1000 * .02 / 343
        self.assertAlmostEqual(result["tau_12_us"], .02 / 343 * 1e6)
        self.assertAlmostEqual(result["cross_spectrum_phase_rad"], -phi)
        np.testing.assert_allclose(result["srp_scores"],
                                   [math.cos(2 * phi), math.cos(phi), 1.], atol=1e-14)
        self.assertEqual(result["candidate_azimuth_deg"], [-30., 0., 30.])

    def test_e13_07_existing_pcm_readback_exposes_library_delay(self):
        manifest = json.loads((ROOM / "MANIFEST.json").read_text(encoding="utf-8"))
        result = room_pcm_readback(manifest, _read_results())
        self.assertEqual(result["shapes_frames_channels"], {
            "source": [16000, 1], "full": [38497, 4], "direct": [38497, 4]})
        self.assertEqual(result["measured_source_to_direct_lag_samples"], [87, 89, 85, 86])
        for observed, geometric in zip(result["measured_source_to_direct_lag_samples"],
                                       result["geometric_propagation_samples"]):
            self.assertLess(abs(observed - geometric - 40), .5)
        self.assertEqual(result["library_fractional_delay_filter_length"], 81)
        self.assertAlmostEqual(result["common_export_gain_from_manifest"],
                               .1516968997435788)
        self.assertLess(max(result["pcm_peak_absolute"].values()), .801)
        corrupted = copy.deepcopy(manifest)
        next(item for item in corrupted["files"] if item["case"] == "fixed_near_left"
             and item["role"] == "source")["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest differs"):
            room_pcm_readback(corrupted, _read_results())

    def test_e13_08_evidence_levels_do_not_skip_stages(self):
        result = run_exercises()
        self.assertEqual(set(result), {f"E13-{number:02d}" for number in range(3, 9)})
        stages = result["E13-08"]
        self.assertTrue(stages["cases_are_hypothetical"])
        self.assertEqual([len(stages[name]) for name in "ABC"], [2, 4, 5])
        self.assertNotIn("source obtained", stages["A"])
        self.assertNotIn("specified scoring completed", stages["B"])
        with self.assertRaisesRegex(ValueError, "skip a prerequisite"):
            evidence_claims(locked=True, license_checked=True, obtained=False,
                            executed=True, scored=False)
        with self.assertRaisesRegex(ValueError, "booleans"):
            evidence_claims(locked=1, license_checked=True, obtained=False,
                            executed=False, scored=False)

    def test_existing_plot_and_result_paths_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plot = directory / "figure.png"
            plot.write_bytes(b"user figure")
            with self.assertRaisesRegex(ValueError, "already exists"):
                plot_results({}, plot)
            self.assertEqual(plot.read_bytes(), b"user figure")
            manifest = directory / "MANIFEST.json"
            manifest.write_text("{}", encoding="utf-8")
            result_path = directory / "RESULTS.json"
            result_path.write_bytes(b"user results")
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_results({"status": "pyroomacoustics_simulation_executed"},
                              plot, directory, result_path)
            self.assertEqual(result_path.read_bytes(), b"user results")


if __name__ == "__main__":
    unittest.main()
