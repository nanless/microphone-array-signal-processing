"""Independent checks for the co-array and repeated two-source exercises."""

import unittest

import numpy as np

from codes.examples.coarray_covariance_exercise import (
    average_ordered_lags, run_exercise as coarray_exercise, virtual_toeplitz,
)
from codes.examples.doa_resolution_trials import (
    classify_peaks, run_experiment, steering_rows, two_peaks,
)


class CoarrayExerciseTest(unittest.TestCase):
    def test_ideal_lags_and_virtual_psd_from_hand_calculation(self):
        report = coarray_exercise()
        lags = report["ideal_lags_real_imag"]
        self.assertEqual(lags["0"], [2.1, 0.0])
        self.assertEqual(lags["1"], [1.0, 1.0])
        self.assertEqual(lags["-1"], [1.0, -1.0])
        self.assertEqual(lags["2"], [0.0, 0.0])
        self.assertEqual(lags["3"], [1.0, -1.0])
        np.testing.assert_allclose(report["ideal_virtual_eigenvalues"], [0.1, 0.1, 4.1, 4.1])

    def test_psd_physical_snapshot_can_yield_indefinite_toeplitz(self):
        report = coarray_exercise()
        np.testing.assert_allclose(report["sample_physical_eigenvalues"], [0, 0, 2])
        np.testing.assert_allclose(report["sample_virtual_eigenvalues"],
                                   [-1/3, 2/3, 2/3, 5/3])

    def test_rejects_missing_lag_and_nonhermitian_input(self):
        with self.assertRaises(ValueError):
            virtual_toeplitz({0: 1.0, 1: 1.0}, 3)
        with self.assertRaises(ValueError):
            average_ordered_lags(np.array([[1, 1], [0, 1]], complex), np.array([0, 1]))


class ResolutionExerciseTest(unittest.TestCase):
    def test_half_wavelength_positive_angle_phase(self):
        row = steering_rows(np.array([30.0]), channels=3)[0]
        np.testing.assert_allclose(row, [1, 1j, -1], atol=1e-15)

    def test_peak_selection_and_one_to_one_matching(self):
        grid = np.arange(-5, 6, dtype=float)
        scores = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0], float)
        self.assertEqual(two_peaks(scores, grid, minimum_separation_deg=3), [-4.0, 4.0])
        self.assertEqual(classify_peaks([-4.0, 4.0], (-4.0, 4.0)), "matched")
        self.assertEqual(classify_peaks([0.0, 4.0], (-4.0, 4.0)), "wrong_location")
        self.assertEqual(classify_peaks([4.0], (-4.0, 4.0)), "fewer_than_two_peaks")

    def test_repeated_trial_histograms_keep_every_trial(self):
        first = run_experiment(trials=5, seed=910)
        second = run_experiment(trials=5, seed=910)
        self.assertEqual(first, second)
        self.assertEqual(len(first["cases"]), 9)
        for case in first["cases"]:
            self.assertEqual(sum(case["outcome_counts"].values()), 5)
            self.assertEqual(case["success_rate"], case["outcome_counts"]["matched"] / 5)
            self.assertLessEqual(case["wilson_95_interval"][0], case["success_rate"])
            self.assertGreaterEqual(case["wilson_95_interval"][1], case["success_rate"])

    def test_rejects_invalid_trial_configuration(self):
        for kwargs in ({"trials": 0}, {"seed": -1}, {"snapshots": 7},
                       {"array_snr_db": float("nan")}, {"trials": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_experiment(**kwargs)


if __name__ == "__main__":
    unittest.main()
