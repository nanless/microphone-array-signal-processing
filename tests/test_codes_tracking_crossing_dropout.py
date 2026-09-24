"""Independent hand checks for the Chapter 9 synthetic tracking sequence."""

from __future__ import annotations

import unittest

from codes.examples.tracking_crossing_dropout_demo import (
    _minimum_cost_assignment,
    run_experiment,
)


class TrackingCrossingDropoutTest(unittest.TestCase):
    def test_tie_order_and_missing_detection_boundaries(self):
        self.assertEqual(_minimum_cost_assignment((40.0, 40.0), (39.0, 41.0)),
                         {0: 0, 1: 1})
        self.assertEqual(_minimum_cost_assignment((40.0, 40.0), (41.0, 39.0)),
                         {0: 0, 1: 1})
        self.assertEqual(_minimum_cost_assignment((40.0, 40.0), ()), {})
        self.assertEqual(_minimum_cost_assignment((40.0, 40.0), (80.0,)), {})
        self.assertEqual(_minimum_cost_assignment((40.0, 80.0), (41.0, 130.0)),
                         {0: 0})
        with self.assertRaises(ValueError):
            _minimum_cost_assignment((40.0, 40.0), (39.0, 40.0, 41.0))

    def test_equal_cost_crossing_can_swap_labels_without_changing_position_set(self):
        report = run_experiment()
        crossing = report["minimum_cost_assignment"][1]
        diagnostic = report["truth_association_diagnostic"][1]

        # Independent cost table for predictions (40, 40), detections
        # (B at 39, A at 41): every squared residual is 1 deg^2.
        cost = ((1.0, 1.0), (1.0, 1.0))
        self.assertEqual(cost[0][0] + cost[1][1], 2.0)
        self.assertEqual(cost[0][1] + cost[1][0], 2.0)
        self.assertEqual(crossing["assigned_detection_label"], {"A": "B", "B": "A"})
        self.assertEqual(crossing["wrong_detection_labels"], 2)
        self.assertEqual(diagnostic["wrong_detection_labels"], 0)

        # With labels discarded, both assignments use the exact truth set
        # {39, 41}. Thus set-position error is zero although IDs are wrong.
        observed_positions = sorted(item["angle_deg"] for item in crossing["observations"])
        truth_positions = sorted(report["truth_deg"][2].values())
        self.assertEqual(observed_positions, truth_positions)
        self.assertEqual(sum(abs(a - b) for a, b in zip(observed_positions, truth_positions)), 0.0)

        # Hand covariance: P_1^-=[[2.1,1],[1,1.01]], after the exact
        # t=1 observation P_1[0,0]=2.1/3.1, P_1[0,1]=1/3.1,
        # P_1[1,1]=1.01-1/3.1. Hence P_2^-[0,0]=2.11 and K=2.11/3.11.
        gain = 2.11 / 3.11
        self.assertAlmostEqual(crossing["state_angle_deg"]["A"], 40.0 - gain)
        self.assertAlmostEqual(diagnostic["state_angle_deg"]["A"], 40.0 + gain)
        self.assertAlmostEqual(crossing["absolute_angle_error_deg"]["A"], 1.0 + gain)
        self.assertAlmostEqual(diagnostic["absolute_angle_error_deg"]["A"], 1.0 - gain)

    def test_missing_and_reappearing_target_is_predicted_then_reassociated(self):
        rows = run_experiment()["minimum_cost_assignment"]
        self.assertEqual(rows[2]["frame"], 3)
        self.assertEqual(rows[2]["missing_tracks"], ["A"])
        self.assertIsNone(rows[2]["assigned_detection_label"]["A"])
        self.assertGreater(rows[2]["angle_variance_deg2"]["A"],
                           rows[1]["angle_variance_deg2"]["A"])
        self.assertEqual(rows[3]["frame"], 4)
        self.assertEqual(rows[3]["assigned_detection_label"]["A"], "A")
        self.assertLess(rows[3]["angle_variance_deg2"]["A"],
                        rows[2]["angle_variance_deg2"]["A"])
        self.assertEqual([row["wrong_detection_labels"] for row in rows], [0, 2, 0, 0])

    def test_two_prediction_only_frames_and_steering_limit_match_hand_calculation(self):
        rows = run_experiment()["dropout_control"]
        self.assertEqual([row["observation"] for row in rows], [None, None])
        self.assertEqual([row["predicted_angle_deg"] for row in rows], [30.0, 40.0])
        # P_1^-=F I F^T+Q, P_1^-[0,0]=2.1. Predicting once more without
        # updating gives 2.1+2*1+1.01+0.1=5.21 deg^2.
        self.assertAlmostEqual(rows[0]["angle_variance_deg2"], 2.1)
        self.assertAlmostEqual(rows[1]["angle_variance_deg2"], 5.21)
        self.assertEqual([row["steer_angle_deg"] for row in rows], [25.0, 30.0])
        self.assertEqual([row["control_lag_deg"] for row in rows], [5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
