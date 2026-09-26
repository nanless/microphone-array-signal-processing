"""Analytic oracles for Fourier power, whitening and constrained beamforming."""

import unittest

import numpy as np

from codes.examples.spatial_model_exercises import (
    colored_noise_music, loading_from_wng, parseval_power, rfft_mean_square,
    run_exercises, singular_mvdr_counterexample,
)


class SpatialModelExercisesTest(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(set(run_exercises()), {"E02-08", "E04-11", "E05-07", "E12-05"})

    def test_parseval_endpoints_padding_and_odd_length(self):
        for row, expected in zip(parseval_power()["cases"], (0.5, 1, 1, 0.5, 2/3)):
            self.assertAlmostEqual(row["rfft_mean_square"], expected)
            self.assertAlmostEqual(row["time_mean_square"], expected)
        # An independently specified real sequence with DC and paired bins.
        for length in (1, 3, 4, 9, 16):
            x = np.arange(length, dtype=float) - 0.25
            expected = float(np.dot(x, x) / length)
            for nfft in (length, length + 1, 2 * length):
                self.assertAlmostEqual(rfft_mean_square(x, nfft), expected)

    def test_rfft_rejects_silent_projection_or_cropping(self):
        for signal, nfft in (([1+1j], None), ([], None), ([np.nan], None),
                             ([[1, 2]], None), ([1, 2], 1), ([1], True), ([1], 2.5)):
            with self.subTest(signal=signal, nfft=nfft), self.assertRaises(ValueError):
                rfft_mean_square(signal, nfft)

    def test_whitener_matches_rank_one_projector_and_target(self):
        report = colored_noise_music()
        packed = report["whitener"]
        actual = np.array(packed["real"]) + 1j * np.array(packed["imag"])
        b = np.array([1, 1j, -1])
        expected = np.eye(3) + (1/np.sqrt(28) - 1) * np.outer(b, b.conj()) / 3
        np.testing.assert_allclose(actual, expected, atol=2e-15)
        np.testing.assert_allclose(report["noise_eigenvalues"], [1, 1, 28], atol=1e-13)
        np.testing.assert_allclose(report["whitened_total_eigenvalues"], [1, 1, 103/28], atol=1e-13)
        self.assertLess(report["whitening_identity_max_error"], 1e-13)
        peaks = [row["peak_degrees"] for row in report["cases"]]
        np.testing.assert_allclose(peaks, [29.1, -4.8, 0.0], atol=0.05)
        self.assertLess(report["cases"][2]["minimum_projection_energy"], 1e-24)
        self.assertGreater(report["cases"][1]["minimum_projection_energy"], 0.06)

    def test_load_budget_and_original_objective(self):
        report = loading_from_wng()
        self.assertAlmostEqual(report["minimum_absolute_load"], 9)
        for row in report["cases"]:
            delta = row["absolute_load"]
            t = 5/(11 + delta)
            np.testing.assert_allclose(row["weights"]["real"], [0.5, 0.5])
            np.testing.assert_allclose(row["weights"]["imag"], [t, -t])
            self.assertAlmostEqual(row["wng_linear"], 1/(0.5 + 2*t*t))
            self.assertAlmostEqual(row["interference_power_gain"], 2*(0.5-t)**2)
        self.assertLess(report["cases"][1]["wng_linear"], 1.6)
        self.assertAlmostEqual(report["cases"][2]["wng_linear"], 1.6)
        self.assertAlmostEqual(report["cases"][2]["original_covariance_output_power"], 1.875)

    def test_singular_counterexample_feasibility_and_zero_cost(self):
        report = singular_mvdr_counterexample()
        np.testing.assert_array_equal(report["pseudoinverse_candidate"], [0, 1])
        self.assertEqual(report["candidate_constraint"], 1)
        self.assertEqual(report["candidate_output_power"], 1)
        np.testing.assert_array_equal(report["optimal_weights"], [1, 0])
        self.assertEqual(report["optimal_output_power"], 0)
        for row in report["loaded_cases"]:
            delta = row["absolute_load"]
            np.testing.assert_allclose(row["weights"], [(1+delta)/(1+2*delta), delta/(1+2*delta)])
            self.assertAlmostEqual(row["original_covariance_output_power"], (delta/(1+2*delta))**2)


if __name__ == "__main__":
    unittest.main()
