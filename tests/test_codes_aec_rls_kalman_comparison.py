"""Independent model, boundary, metric and scope checks for the AEC experiment."""

import json
import unittest

import numpy as np

from codes.examples.aec_rls_kalman_comparison import (
    LENGTH, PATH_A, PATH_B, SEGMENTS, make_signals, run_demo, run_experiment,
)


class TestRLSKalmanComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = run_experiment()

    def test_known_components_and_half_open_segments(self):
        signals = make_signals()
        self.assertEqual(LENGTH, 512)
        self.assertEqual(list(SEGMENTS.values()),
                         [(0, 128), (128, 256), (256, 384), (384, 512)])
        np.testing.assert_array_equal(
            signals["microphone"],
            signals["echo"] + signals["near_end"] + signals["noise"],
        )
        x = signals["render"]
        independent_echo = (signals["path"][:, 0] * x
                            + signals["path"][:, 1] * np.r_[0., x[:-1]])
        np.testing.assert_array_equal(signals["echo"], independent_echo)
        np.testing.assert_array_equal(signals["path"][:384],
                                      np.tile(PATH_A, (384, 1)))
        np.testing.assert_array_equal(signals["path"][384:],
                                      np.tile(PATH_B, (128, 1)))
        np.testing.assert_array_equal(signals["near_end"][:128], 0.)
        self.assertGreater(np.linalg.norm(signals["near_end"][128:256]), 0.)
        np.testing.assert_array_equal(signals["near_end"][256:], 0.)

    def test_runs_use_same_input_and_prior_output_convention(self):
        signals = self.experiment["signals"]
        for name, trace in self.experiment["runs"].items():
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    trace["prior_residual"],
                    signals["microphone"] - trace["prior_echo"],
                    rtol=0, atol=2e-15,
                )
                self.assertEqual(trace["posterior_path"].shape, (LENGTH, 2))
        # Before the oracle interval all settings of each method coincide.
        rls = self.experiment["runs"]
        np.testing.assert_array_equal(
            rls["rls_continuous"]["prior_echo"][:128],
            rls["rls_oracle_freeze"]["prior_echo"][:128],
        )
        for other in ("kalman_oracle_variance", "kalman_oracle_freeze"):
            np.testing.assert_array_equal(
                rls["kalman_fixed_variance"]["prior_echo"][:128],
                rls[other]["prior_echo"][:128],
            )

    def test_oracle_controls_freeze_or_softly_reduce_update(self):
        runs = self.experiment["runs"]
        b, e = SEGMENTS["known_near_end_injection"]
        rls_frozen = runs["rls_oracle_freeze"]["posterior_path"]
        np.testing.assert_array_equal(
            rls_frozen[b:e], np.tile(rls_frozen[b - 1], (e - b, 1))
        )
        kalman_frozen = runs["kalman_oracle_freeze"]
        np.testing.assert_array_equal(kalman_frozen["gain_l2_norm"][b:e], 0.)
        np.testing.assert_array_equal(
            kalman_frozen["posterior_path"][b:e],
            np.tile(kalman_frozen["posterior_path"][b - 1], (e - b, 1)),
        )
        soft = runs["kalman_oracle_variance"]["gain_l2_norm"][b:e]
        fixed = runs["kalman_fixed_variance"]["gain_l2_norm"][b:e]
        self.assertTrue(np.all(soft > 0.))
        self.assertLess(float(np.mean(soft)), float(np.mean(fixed)))

    def test_metrics_are_independently_recomputed_from_traces(self):
        signals = self.experiment["signals"]
        for name, trace in self.experiment["runs"].items():
            for segment, (b, e) in SEGMENTS.items():
                with self.subTest(run=name, segment=segment):
                    row = self.experiment["summary"][name][segment]
                    echo_mse = sum(float(v * v) for v in
                                   (signals["echo"][b:e]
                                    - trace["prior_echo"][b:e])) / (e - b)
                    path_delta = (signals["path"][b:e]
                                  - trace["posterior_path"][b:e])
                    path_rmse = np.sqrt(sum(float(v * v) for v in
                                            path_delta.flat) / (2 * (e - b)))
                    self.assertAlmostEqual(row["echo_error_mse"], echo_mse)
                    self.assertAlmostEqual(row["posterior_path_rmse"], path_rmse)

    def test_report_is_json_safe_and_does_not_claim_erle_or_detection(self):
        report = run_demo()
        serialized = json.dumps(report, allow_nan=False)
        self.assertIn("oracle controls", serialized)
        self.assertIn("not a detector", serialized)
        self.assertIn("no audio, ERLE", serialized)
        self.assertEqual(set(report["summary"]), set(self.experiment["runs"]))


if __name__ == "__main__":
    unittest.main()
