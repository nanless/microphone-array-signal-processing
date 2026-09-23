"""Independent arithmetic checks for controlled AEC double-talk scoring."""

from __future__ import annotations

import unittest

import numpy as np

from codes.examples.aec_controlled_doubletalk import (
    SYNTH_SHA256, checked_pcm_add, increment_metrics, interval_mask, read_pinned_pcm,
)
from codes.examples.aec_real_pair_experiment import ROOT


class ControlledDoubleTalkTests(unittest.TestCase):
    def test_published_pcm_fixture_is_exactly_additive_after_near_removal(self) -> None:
        directory = ROOT / "codes/audio"
        near = read_pinned_pcm(directory / "aec_near.wav", SYNTH_SHA256["aec_near.wav"])
        mixed = read_pinned_pcm(directory / "aec_microphone.wav", SYNTH_SHA256["aec_microphone.wav"])
        base = mixed.astype(np.int32) - near.astype(np.int32)
        self.assertTrue(np.all((base >= -32768) & (base <= 32767)))
        self.assertTrue(np.array_equal(checked_pcm_add(base.astype(np.int16), near), mixed))
        self.assertTrue(np.all(near[:19200] == 0))
        self.assertTrue(np.all(near[28800:] == 0))
        self.assertGreater(np.max(np.abs(near[19200:28800].astype(np.int32))), 0)

    def test_fixed_sample_increment_rejects_gain_and_shift(self) -> None:
        base = np.zeros(8, dtype=np.int16)
        near = np.array([0, 1, 2, -1, 0, 0, 0, 0], dtype=np.int16)
        perfect = increment_metrics(base, near, near, 1, 5)
        self.assertEqual(perfect["increment_projection_gain_no_delay_fit"], 1.0)
        self.assertEqual(perfect["fixed_sample_relative_squared_error"], 0.0)
        halved = increment_metrics(base.astype(float), near.astype(float) / 2, near, 1, 5)
        self.assertAlmostEqual(halved["increment_projection_gain_no_delay_fit"], 0.5)
        self.assertAlmostEqual(halved["fixed_sample_relative_squared_error"], 0.25)
        shifted = np.zeros_like(near)
        shifted[2:] = near[:-2]
        self.assertGreater(increment_metrics(base, shifted, near, 1, 5)
                           ["fixed_sample_relative_squared_error"], 0.0)

    def test_pcm_add_rejects_clipping(self) -> None:
        with self.assertRaisesRegex(ValueError, "clip"):
            checked_pcm_add(np.array([32767], dtype=np.int16),
                            np.array([1], dtype=np.int16))

    def test_known_interval_includes_pcm_zero_crossings(self) -> None:
        mask = interval_mask(8, 2, 6)
        self.assertEqual(mask.tolist(), [False, False, True, True, True, True, False, False])

    def test_increment_scoring_is_scale_safe(self) -> None:
        base = np.zeros(2)
        near = np.array([1e308, -1e308])
        result = increment_metrics(base, near, near, 0, 2)
        self.assertEqual(result["increment_projection_gain_no_delay_fit"], 1.0)
        self.assertEqual(result["fixed_sample_relative_squared_error"], 0.0)
        with self.assertRaisesRegex(ValueError, "numeric scale"):
            increment_metrics(base, near, np.array([1e-308, -1e-308]), 0, 2)


if __name__ == "__main__":
    unittest.main()
