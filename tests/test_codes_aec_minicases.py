"""Independent hand or analytic expectations for four chapter 6 mini-cases."""

import json
import unittest

from codes.examples.aec_algorithm_minicases import (
    geigel_two_boundaries,
    ipnlms_two_tap_case,
    overlap_save_two_partitions,
    run_demo,
    signed_delay_polarity_case,
)


class TestAECAlgorithmMiniCases(unittest.TestCase):
    def test_overlap_save_matches_hand_convolution_and_discards_aliasing(self):
        result = overlap_save_two_partitions()
        self.assertEqual(result["block_length"], 2)
        self.assertEqual(result["fft_length"], 4)
        self.assertEqual(result["fft_reference_blocks"], [[0, 0, 1, 2], [1, 2, 3, 4]])
        # Hand linear convolution: 1; 2+2=4; 3+4+3=10; 4+6+6+4=20.
        self.assertEqual(result["valid_output"], [1, 4, 10, 20])
        # The current second block's first two circular points wrap around;
        # its valid p=0 tail is [7, 10], while delayed p=1 contributes [3, 10].
        self.assertEqual(result["circular_terms_by_block_and_partition"][1],
                         [[9, 4, 7, 10], [8, 0, 3, 10]])
        self.assertEqual(result["discarded_first_halves"][1], [17, 4])

    def test_ipnlms_allocation_is_hand_reproducible(self):
        result = ipnlms_two_tap_case()
        self.assertAlmostEqual(result["prior_echo"], 1.)
        self.assertAlmostEqual(result["prior_residual"], .2)
        cases = result["kappa_cases"]
        self.assertEqual(cases["-1"]["tap_allocation"], [.5, .5])
        self.assertEqual(cases["-1"]["delta_weights"], [.1, .1])
        self.assertEqual(cases["0"]["tap_allocation"], [.65, .35])
        self.assertEqual(cases["0"]["delta_weights"], [.13, .07])
        self.assertEqual(cases["1"]["tap_allocation"], [.8, .2])
        self.assertEqual(cases["1"]["delta_weights"], [.16, .04])
        for case in cases.values():
            self.assertAlmostEqual(case["normalization"], 1.)
            self.assertAlmostEqual(sum(case["delta_weights"]), .2)
        self.assertFalse(result["pure_proportionate_zero_start"]
                         ["update_possible_without_a_tap_floor"])

    def test_geigel_multipath_false_alarm_and_zero_reference(self):
        result = geigel_two_boundaries()
        # No near-end signal: two .4 taps each see reference 1, so d=.8.
        self.assertEqual(result["microphone"], .8)
        self.assertEqual(result["peak_ratio"], .8)
        self.assertTrue(result["bare_rule_says_double_talk"])
        self.assertFalse(result["ground_truth_double_talk"])
        self.assertIsNone(result["zero_reference_case"]["peak_ratio"])
        self.assertIsNone(result["zero_reference_case"]["decision"])

    def test_signed_delay_peak_fails_for_negative_polarity(self):
        result = signed_delay_polarity_case()
        self.assertEqual(result["signed_correlations"], [0, 0, -1, 0])
        self.assertEqual(result["largest_signed_peak_lag"], 0)
        self.assertEqual(result["largest_absolute_peak_lag"], 2)
        self.assertEqual(result["true_delay_samples"], 2)

    def test_demo_is_finite_serializable_and_stable(self):
        result = run_demo()
        self.assertEqual(set(result), {"overlap_save", "ipnlms", "geigel", "delay_polarity"})
        json.dumps(result, allow_nan=False)
        self.assertEqual(result, run_demo())


if __name__ == "__main__":
    unittest.main()
