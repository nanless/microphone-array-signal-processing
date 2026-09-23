"""No-network, no-audio-write tests for the pinned real-pair PBFDAF adapter."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from codes.examples.aec_pbfdaf_real_pair_compare import (
    evaluate_arrays, power_change_db, reference_conditions, run,
)


class PBFDAFRealPairCompareTests(unittest.TestCase):
    def test_power_is_energy_ratio_not_amplitude_ratio(self) -> None:
        self.assertAlmostEqual(power_change_db(np.array([2., -2.]),
                                               np.array([1., -1.])),
                               6.020599913279624, places=12)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            power_change_db(np.ones(2), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "finite"):
            power_change_db(np.array([1., np.nan]), np.ones(2))

    def test_late_control_is_exactly_one_second_late(self) -> None:
        reference = np.arange(1., 9.)
        controls = reference_conditions(reference, rate=2)
        np.testing.assert_array_equal(controls["paired"], reference)
        np.testing.assert_array_equal(controls["zero_reference"], np.zeros(8))
        np.testing.assert_array_equal(controls["late_reference_1s"],
                                      [0., 0., 1., 2., 3., 4., 5., 6.])
        with self.assertRaises(ValueError):
            reference_conditions(reference[:2], rate=2)

    def test_tiny_pcm_fixture_has_exact_zero_control_and_all_score_samples(self) -> None:
        # Independent oracle: a zero reference and zero initial weights must
        # leave d untouched, so every before/after ratio is exactly 0 dB.
        reference = np.zeros(9, dtype=np.int16)
        microphone = np.array([100, -200, 300, -400, 500, -600,
                               700, -800, 900,], dtype=np.int16)
        result = evaluate_arrays(reference, microphone, score_start=2,
                                 score_stop=7, rate=2, block_length=2,
                                 filter_length=4, step_size=.3, epsilon=.05)
        self.assertEqual(result["common_complete_block_samples"], 8)
        self.assertEqual(result["discarded_tail_samples"], {"lpb": 1, "mic": 1})
        self.assertEqual(result["score_interval_half_open"], [2, 7])
        for condition in result["controls"].values():
            self.assertEqual(condition["input_output_total_power_change_db"], 0.)
            self.assertEqual([item["sample_interval_half_open"]
                              for item in condition["score_windows"]],
                             [[2, 4], [4, 6], [6, 7]])
            self.assertEqual([item["samples"] for item in condition["score_windows"]],
                             [2, 2, 1])
            self.assertEqual([item["input_output_total_power_change_db"]
                              for item in condition["score_windows"]], [0., 0., 0.])
            expected = hashlib.sha256((microphone[:8].astype(np.float64)
                                       / 32768).astype("<f8").tobytes()).hexdigest()
            self.assertEqual(condition["whole_output_float64_le_sha256"], expected)
        self.assertEqual(result["paired_output_relative_to_zero_reference_db"], 0.)

    def test_nonzero_reference_changes_output_but_zero_control_does_not(self) -> None:
        reference = np.array([1000, 0, 0, 0, 1000, 0, 0, 0], dtype=np.int16)
        microphone = np.array([1000, 100, 200, 100, 1000, 100, 200, 100], dtype=np.int16)
        result = evaluate_arrays(reference, microphone, score_start=2,
                                 score_stop=8, rate=2, block_length=2,
                                 filter_length=4, step_size=.3, epsilon=.05)
        controls = result["controls"]
        self.assertEqual(controls["zero_reference"]["input_output_total_power_change_db"], 0.)
        self.assertNotEqual(controls["paired"]["whole_output_float64_le_sha256"],
                            controls["zero_reference"]["whole_output_float64_le_sha256"])
        self.assertAlmostEqual(result["paired_output_relative_to_zero_reference_db"],
                               controls["paired"]["input_output_total_power_change_db"])

    def test_invalid_input_and_unpinned_pair_rejected(self) -> None:
        valid = np.ones(8, dtype=np.int16)
        with self.assertRaisesRegex(ValueError, "PCM16"):
            evaluate_arrays(valid.astype(float), valid, score_start=2,
                            score_stop=8, rate=2, block_length=2)
        with self.assertRaisesRegex(ValueError, "score interval"):
            evaluate_arrays(valid, valid, score_start=2, score_stop=9,
                            rate=2, block_length=2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            evaluate_arrays(valid, valid, score_start=2, score_stop=8,
                            rate=3, block_length=2)
        with self.assertRaisesRegex(ValueError, "pair must"):
            run("unverified_pair")


if __name__ == "__main__":
    unittest.main()
