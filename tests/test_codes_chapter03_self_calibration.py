"""Independent gauge algebra and bounded-fit checks for chapter 3."""

import json
import math
import unittest

import numpy as np

from codes.chapters.ch03.core.calibration import fit_anchored_ula, ula_prediction
from codes.chapters.ch03.examples.self_calibration_demo import run_demo


class AnchoredSelfCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.spacing = 0.04
        self.frequency = 1000.0
        self.speed = 343.0
        self.kd = 2 * math.pi * self.frequency * self.spacing / self.speed
        self.angles = np.array([-40.0, 0.0, 35.0])
        self.gains = np.array([1.0, 1.08, 0.93 * np.exp(0.12j),
                               1.03 * np.exp(-0.08j)])
        rng = np.random.default_rng(17)
        self.sources = (rng.normal(size=(3, 32)) + 1j * rng.normal(size=(3, 32))) / math.sqrt(2)
        self.clean = ula_prediction(
            self.angles, self.gains, self.sources,
            spacing_m=self.spacing, frequency_hz=self.frequency)

    def test_independent_scalar_gauge_identity(self):
        shift = 0.15
        alternative_angles = np.rad2deg(np.arcsin(np.sin(np.deg2rad(self.angles)) + shift))
        alternative_gains = self.gains * np.exp(-1j * self.kd * np.arange(4) * shift)
        self.assertEqual(alternative_gains[0], 1.0)
        self.assertGreater(np.max(abs(alternative_angles - self.angles)), 8.0)
        self.assertGreater(abs(np.angle(alternative_gains[1])), 0.1)
        # Evaluate each cell without the core prediction helper.  The common
        # shift cancels algebraically in the full complex observation.
        for q in range(3):
            for t in range(32):
                for m in range(4):
                    other = (self.sources[q, t] * alternative_gains[m]
                             * np.exp(1j * self.kd * m
                                      * math.sin(math.radians(alternative_angles[q]))))
                    self.assertAlmostEqual(abs(other - self.clean[q, t, m]), 0.0, places=13)

    def test_exact_external_anchor_recovers_noiseless_parameters(self):
        result = fit_anchored_ula(
            self.clean, spacing_m=self.spacing, frequency_hz=self.frequency,
            measured_relative_phase_rad=0.0)
        self.assertEqual(result["status"], "converged")
        np.testing.assert_allclose(result["angles_deg"], self.angles, atol=1e-7)
        np.testing.assert_allclose(result["relative_gains"], self.gains, atol=1e-9)
        self.assertLess(result["objective_history"][-1], 1e-12)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(
            result["objective_history"], result["objective_history"][1:])))

    def test_wrong_anchor_has_exact_fit_but_biased_directions(self):
        wrong_phase = math.radians(5)
        result = fit_anchored_ula(
            self.clean, spacing_m=self.spacing, frequency_hz=self.frequency,
            measured_relative_phase_rad=wrong_phase)
        expected = np.rad2deg(np.arcsin(
            np.sin(np.deg2rad(self.angles)) - wrong_phase / self.kd))
        self.assertEqual(result["status"], "converged")
        np.testing.assert_allclose(result["angles_deg"], expected, atol=2e-6)
        self.assertAlmostEqual(result["angles_deg"][1], -6.840003585, places=5)
        self.assertAlmostEqual(np.angle(result["relative_gains"][1]), wrong_phase, places=12)
        self.assertLess(result["objective_history"][-1], 1e-12)

    def test_noisy_example_reports_scope_and_monotone_objective(self):
        report = run_demo()
        json.dumps(report, allow_nan=False)
        gauge = report["unanchored_gauge"]
        fit = report["correct_external_anchor"]
        wrong = report["wrong_external_anchor_on_noiseless_data"]
        self.assertLess(gauge["max_same_sample_prediction_difference"], 1e-12)
        self.assertEqual(fit["status"], "converged")
        self.assertLess(max(abs(v) for v in fit["angle_errors_deg"]), 0.1)
        self.assertEqual(wrong["status"], "converged")
        self.assertLess(wrong["last_objective"], 1e-22)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(
            fit["objective_history"], fit["objective_history"][1:])))

    def test_invalid_information_and_nonconvergence_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "reference channel"):
            fit_anchored_ula(np.zeros_like(self.clean), spacing_m=self.spacing,
                             frequency_hz=self.frequency, measured_relative_phase_rad=0)
        with self.assertRaisesRegex(ValueError, "aliases"):
            fit_anchored_ula(self.clean, spacing_m=1.0,
                             frequency_hz=self.frequency, measured_relative_phase_rad=0)
        with self.assertRaisesRegex(ValueError, "outside"):
            fit_anchored_ula(self.clean, spacing_m=self.spacing,
                             frequency_hz=self.frequency, measured_relative_phase_rad=1.0)
        with self.assertRaisesRegex(ValueError, "initial angles"):
            fit_anchored_ula(self.clean, spacing_m=self.spacing,
                             frequency_hz=self.frequency, measured_relative_phase_rad=0,
                             initial_angles_deg=[0, 0, 80])
        rng = np.random.default_rng(31)
        noisy = self.clean + 0.01 * (rng.normal(size=self.clean.shape)
                                     + 1j * rng.normal(size=self.clean.shape))
        early = fit_anchored_ula(
            noisy, spacing_m=self.spacing, frequency_hz=self.frequency,
            measured_relative_phase_rad=0, max_iterations=1)
        self.assertEqual(early["status"], "max_iterations")
        self.assertEqual(early["iterations"], 1)


if __name__ == "__main__":
    unittest.main()
