"""Independent clock arithmetic and chunk-boundary checks for the SRO lesson."""

import json
import unittest

from codes.examples.sro_closed_loop_demo import (
    StatefulLinearClockCorrector,
    recover_device_indices,
    run_experiment,
)


class SROClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_experiment()

    def test_estimate_separates_offset_slope_and_deleted_sample(self):
        report = self.report
        true_ppm = 150.0
        exact_delay_slope = (true_ppm * 1e-6) / (1 + true_ppm * 1e-6)
        self.assertAlmostEqual(report["estimate"]["first_order_sro_ppm"],
                               exact_delay_slope * 1e6, places=7)
        self.assertAlmostEqual(report["estimate"]["exact_sro_ppm"], true_ppm, places=7)
        self.assertAlmostEqual(report["estimate"]["initial_offset_ms"], 2.0, places=9)
        self.assertEqual(report["estimate"]["detected_gaps"], [
            {"after_received_index": 11999, "missing_device_samples": 1},
        ])
        json.dumps(report, allow_nan=False)

    def test_residuals_are_scored_on_both_sides_of_gap(self):
        report = self.report
        self.assertEqual(report["compensation"]["invalid_output_indices"], [12002, 12003])
        self.assertEqual(report["estimate"]["last_calibration_received_index"], 4000)
        self.assertEqual(report["compensation"]["first_reference_output_index"], 4500)
        for name, length in (("before_drop", 6000), ("after_drop", 9000)):
            score = report["segments"][name]
            self.assertEqual(score["reference_samples"], length)
            self.assertEqual(score["corrected_valid_samples"], length)
            self.assertLess(score["corrected_mse"], 1e-5)
            self.assertGreater(score["uncorrected_mse"], 0.01)

    def test_chunk_size_does_not_reset_phase_or_hide_gap(self):
        original = self.report
        alternate = run_experiment(block_size=509)
        self.assertEqual(original["estimate"], alternate["estimate"])
        self.assertEqual(original["segments"], alternate["segments"])
        self.assertEqual(original["compensation"]["invalid_output_indices"],
                         alternate["compensation"]["invalid_output_indices"])

    def test_small_linear_clock_and_gap_across_chunks(self):
        # Missing physical index 4 makes the j=2 output invalid, even though
        # the next input block contains samples on both sides of that gap.
        corrector = StatefulLinearClockCorrector(2.0, 4.0, 0.0, 5)
        first = corrector.push([0, 1, 2, 3], [0., .25, .5, .75])
        second = corrector.push([5, 6, 7, 8], [1.25, 1.5, 1.75, 2.])
        self.assertEqual(first + second,
                         [(0, 0.0), (1, 0.5), (2, None), (3, 1.5), (4, 2.0)])

        # At 3 Hz input and 2 Hz output, j=1 maps to physical position 1.5.
        # Its interpolation pair straddles the push boundary.
        fractional = StatefulLinearClockCorrector(2.0, 3.0, 0.0, 4)
        first = fractional.push([0, 1], [0., 1 / 3])
        second = fractional.push([2, 3, 4, 5], [2 / 3, 1., 4 / 3, 5 / 3])
        self.assertEqual([j for j, _ in first + second], [0, 1, 2, 3])
        for (_, value), expected in zip(first + second, [0., .5, 1., 1.5]):
            self.assertAlmostEqual(value, expected)

    def test_bad_timestamp_and_index_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            recover_device_indices([0., .5, .75, 1.5])
        with self.assertRaises(ValueError):
            recover_device_indices([0., .5, .5, 1.])
        with self.assertRaises(ValueError):
            run_experiment(block_size=True)
        corrector = StatefulLinearClockCorrector(2.0, 2.0, 0.0, 4)
        corrector.push([0, 1], [0., 1.])
        with self.assertRaises(ValueError):
            corrector.push([1, 2], [1., 2.])


if __name__ == "__main__":
    unittest.main()
