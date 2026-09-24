"""Independent checks for guided cACGMM and the two-source teaching run."""

import unittest
import numpy as np

from codes.array_tutorial.gss_teaching import guided_cacgmm_mvdr
from codes.array_tutorial.separation import si_sdr
from codes.examples.gss_teaching_demo import run_experiment


class TestGSSTeaching(unittest.TestCase):
    def test_activity_gates_and_scm(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(4, 2, 12)) + 1j * rng.normal(size=(4, 2, 12))
        activity = np.zeros((12, 2), dtype=int)
        activity[:4, 0] = 1
        activity[4:8, 1] = 1
        result = guided_cacgmm_mvdr(x, activity, iterations=3)
        np.testing.assert_allclose(result["posterior"].sum(axis=1), 1, atol=1e-14)
        np.testing.assert_array_equal(result["posterior"][:, 0, 8:], 0)
        np.testing.assert_allclose(result["target_scm"],
                                   result["target_scm"].conj().transpose(0, 2, 1), atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(result["target_scm"]) >= -1e-12))

    def test_full_synthetic_chain_and_missed_label(self):
        report, arrays = run_experiment()
        self.assertGreater(report["si_sdr_db"]["correct_activity_output"],
                           report["si_sdr_db"]["reference_mic0"])
        self.assertGreater(report["activity_error"]["correct_output_scored_si_sdr_db"],
                           report["activity_error"]["missed_output_scored_si_sdr_db"])
        self.assertEqual(arrays["posterior"].shape[1], 3)
        self.assertFalse(report["correct_shape_resets"])
        left, right = (int(t * report["sample_rate_hz"]) for t in report["scored_interval_seconds"])
        self.assertEqual(right - left, 20000)
        self.assertAlmostEqual(
            report["si_sdr_db"]["correct_activity_output"],
            si_sdr(arrays["enhanced_correct"][0, left:right], arrays["source_1"][0, left:right]),
            places=12,
        )

    def test_rejects_nonbinary_activity(self):
        with self.assertRaises(ValueError):
            guided_cacgmm_mvdr(np.ones((2, 2, 4), complex), np.full((4, 1), 0.5))


if __name__ == "__main__":
    unittest.main()
