"""Offline checks for the pinned-pair AEC experiment's validation logic."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from codes.examples.aec_real_pair_experiment import (
    load_pinned_pair, peak_lag, power_ratio_db, speex_linear_aec,
)
from codes.examples.aec_doubletalk_experiment import load_doubletalk
from codes.examples.aec3_offline_compare import command, run as run_aec3


class RealPairExperimentTests(unittest.TestCase):
    def test_peak_lag_sign_and_normalization(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.integers(-1000, 1000, 5000, dtype=np.int16)
        microphone = np.zeros_like(reference)
        microphone[37:] = reference[:-37] // 2
        lag, corr = peak_lag(reference, microphone, start=200, stop=4500,
                             max_lag=100)
        self.assertEqual(lag, 37)
        self.assertGreater(corr, 0.99)

    def test_power_ratio_is_power_not_amplitude_db(self) -> None:
        before = np.array([1000, -1000], dtype=np.int16)
        after = np.array([500, -500], dtype=np.int16)
        self.assertAlmostEqual(power_ratio_db(before, after), 6.020599913, places=8)

    def test_lfs_pointer_or_unpinned_audio_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_lpb.wav"
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                load_pinned_pair(Path(directory))

    def test_partial_frame_rejected_before_loading_library(self) -> None:
        with self.assertRaisesRegex(ValueError, "whole 10 ms frames"):
            speex_linear_aec(Path("/nonexistent/libspeexdsp.dylib"),
                             np.zeros(161, dtype=np.int16),
                             np.zeros(161, dtype=np.int16))

    def test_doubletalk_rejects_unpinned_or_lfs_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "-2jLGNCgf0WDpKMY2iup7g_doubletalk_lpb.wav"
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                load_doubletalk(Path(directory))

    def test_aec3_adapter_pins_order_and_separate_taps(self) -> None:
        args = command(Path("/tmp/audioproc_f"), Path("/tmp/mic.wav"),
                       Path("/tmp/lpb.wav"), Path("/tmp/final.wav"),
                       Path("/tmp/linear.wav"), Path("/tmp/rc.txt"))
        self.assertIn("--custom_call_order_file=/tmp/rc.txt", args)
        self.assertIn("--linear_aec_output=/tmp/linear.wav", args)
        self.assertIn("--o=/tmp/final.wav", args)
        self.assertIn("--stream_delay=0", args)

    def test_aec3_adapter_does_not_fake_missing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_aec3(Path(directory) / "missing-audioproc_f", Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
