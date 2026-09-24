"""Independent checks for the pinned-source blind separation experiment."""

import unittest

import numpy as np

from codes.examples.reproduce_auxiva_reference import (
    MIXING, SOURCE_TREE, independent_sources, pinned_auxiva_class, run_experiment,
)
from codes.examples.gss_activity_error_demo import run_experiment as run_activity_experiment


class AuxIVAReferenceExperimentTests(unittest.TestCase):
    def test_source_generation_is_deterministic_and_full_rank(self):
        first = independent_sources()
        self.assertEqual(first.shape, (2, 32000))
        np.testing.assert_array_equal(first, independent_sources())
        self.assertEqual(np.linalg.matrix_rank(MIXING), 2)

    @unittest.skipUnless((SOURCE_TREE / "ssspy/bss/iva.py").is_file(), "pinned upstream source not present")
    def test_true_blind_fit_and_harmonic_counterexample(self):
        self.assertTrue(pinned_auxiva_class().__module__.startswith("ssspy."))
        report = run_experiment()
        score = report["score"]
        self.assertEqual(sorted(score["output_to_reference"]), [0, 1])
        self.assertGreater(score["mean_si_sdri_db"], 25.0)
        self.assertLess(report["analysis"]["padded_stft_roundtrip_max_abs_error"], 1e-10)
        self.assertEqual(report["objective"]["recorded_points"], 31)
        self.assertLess(report["objective"]["final"], report["objective"]["initial"])
        for gain in score["output_to_reference_rms_ratios"]:
            self.assertLess(abs(gain - 1.0), 0.05)
        self.assertEqual(report["rank_deficient_boundary"]["rank"], 1)
        self.assertEqual(report["rank_deficient_boundary"]["upstream_run"]["status"], "failed_numerically")
        self.assertLess(report["harmonic_counterexample"]["score"]["mean_output_si_sdr_db"], 5.0)


class GSSActivityErrorTests(unittest.TestCase):
    def test_analytic_activity_gate_cases(self):
        report = run_activity_experiment()
        self.assertAlmostEqual(report["speaker_1_first_frame_posterior_correct"], 16 / 17)
        self.assertEqual(report["speaker_1_first_frame_posterior_when_missed"], 0.0)
        self.assertEqual(report["speaker_1_silent_frame_posterior_correct"], 0.0)
        self.assertAlmostEqual(report["speaker_1_silent_frame_posterior_when_falsely_active"], 2 / 3)
        for scenario in report["posteriors"].values():
            for frame in scenario:
                self.assertAlmostEqual(sum(frame), 1.0)


if __name__ == "__main__":
    unittest.main()
