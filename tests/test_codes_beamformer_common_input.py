"""Independent checks for the Chapter 5 common-input comparison."""

import math
import unittest

from codes.chapters.ch05.beamformer_common_input_demo import run_experiment


class BeamformerCommonInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_experiment()
        cls.rows = {row["method"]: row for row in cls.report["results"]}

    def test_same_model_and_reference_for_every_method(self):
        model = self.report["model"]
        self.assertEqual(set(self.rows), {"DSB", "MVDR", "MVDR_loaded", "LCMV"})
        self.assertEqual(model["absolute_diagonal_loading"], 11.0)
        self.assertAlmostEqual(self.report["reference_mic_sinr_db"],
                               10 * math.log10(1 / 11), places=12)
        for row in self.rows.values():
            self.assertAlmostEqual(row["nominal_target_response_abs"], 1.0, places=12)
            self.assertTrue(math.isfinite(row["true_target_output_sinr_db"]))
            self.assertGreater(row["interference_plus_noise_output_power"], 0.0)

    def test_dsb_independent_geometric_series_and_wng(self):
        # At broadside the four DSB weights are 1/4. The spatial phase step
        # for an interferer at 40 degrees follows directly from geometry.
        phase = 2 * math.pi * 2000 * 0.04 * math.sin(math.radians(40)) / 343
        response = abs(sum(complex(math.cos(m * phase), math.sin(m * phase))
                           for m in range(4)) / 4)
        dsb = self.rows["DSB"]
        self.assertAlmostEqual(dsb["interferer_response_abs"], response, places=12)
        self.assertAlmostEqual(dsb["white_noise_gain_db"], 10 * math.log10(4), places=12)
        expected_power = 10 * response**2 + 1 / 4
        self.assertAlmostEqual(dsb["interference_plus_noise_output_power"],
                               expected_power, places=12)

    def test_nominal_and_true_white_noise_gain_use_their_respective_response(self):
        self.assertEqual(self.report["model"]["white_noise_gain_direction_deg"], 0)
        for row in self.rows.values():
            self.assertEqual(row["white_noise_gain_reference"], "nominal design steering at 0 degrees")
            # The ratio of true to nominal WNG is the target power gain.
            self.assertAlmostEqual(row["true_target_white_noise_gain_db"],
                                   row["white_noise_gain_db"] + row["true_target_gain_db"], places=12)
        phase = 2*math.pi*2000*.04*math.sin(math.radians(10))/343
        response = abs(sum(complex(math.cos(m*phase), math.sin(m*phase)) for m in range(4))/4)
        self.assertAlmostEqual(self.rows["DSB"]["true_target_white_noise_gain_db"],
                               10*math.log10(4*response**2), places=12)

    def test_null_and_loading_tradeoff_are_not_single_metric_rankings(self):
        self.assertLess(self.rows["LCMV"]["interferer_response_abs"], 1e-12)
        self.assertGreater(self.rows["MVDR_loaded"]["white_noise_gain_db"],
                           self.rows["MVDR"]["white_noise_gain_db"])
        self.assertGreater(self.rows["MVDR_loaded"]["interferer_response_abs"],
                           self.rows["MVDR"]["interferer_response_abs"])
        self.assertLess(self.rows["MVDR"]["true_target_response_abs"],
                        self.rows["DSB"]["true_target_response_abs"])
        self.assertLess(self.rows["MVDR_loaded"]["true_target_output_sinr_db"],
                        self.rows["MVDR"]["true_target_output_sinr_db"])


if __name__ == "__main__":
    unittest.main()
