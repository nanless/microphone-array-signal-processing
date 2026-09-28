"""Offline checks for the pinned-pair AEC experiment's validation logic."""

from __future__ import annotations

from pathlib import Path
import tempfile
import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np

from codes.chapters.ch06.examples.aec_real_pair_experiment import (
    load_pinned_pair, peak_lag, power_ratio_db, speex_linear_aec,
)
from codes.chapters.ch06.aec_doubletalk_experiment import load_doubletalk
from codes.chapters.ch06.examples.aec3_offline_compare import command, run as run_aec3


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

    def test_c_adapter_materializes_strided_pcm_views(self) -> None:
        captured = []
        def ctl(_state, request, pointer):
            if request == 25:
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 16000
            return 0
        def cancel(_state, rec_pointer, play_pointer, out_pointer):
            rec = np.ctypeslib.as_array((ctypes.c_int16 * 160).from_address(rec_pointer))
            play = np.ctypeslib.as_array((ctypes.c_int16 * 160).from_address(play_pointer))
            out = np.ctypeslib.as_array((ctypes.c_int16 * 160).from_address(out_pointer))
            captured.append((rec.copy(), play.copy()))
            out[:] = rec - play
        fake = SimpleNamespace(speex_echo_state_init=Mock(return_value=1),
            speex_echo_ctl=Mock(side_effect=ctl),
            speex_echo_cancellation=Mock(side_effect=cancel), speex_echo_state_destroy=Mock())
        storage = np.arange(640, dtype=np.int16)
        with tempfile.NamedTemporaryFile() as library:
            for view in (storage[::2], storage[1::2], storage[:320][::-1]):
                microphone = (storage + 1000)[::2]
                captured.clear()
                with patch('codes.chapters.ch06.examples.aec_real_pair_experiment.ctypes.CDLL', return_value=fake):
                    out, rate = speex_linear_aec(Path(library.name), view, microphone)
                self.assertEqual(rate, 16000)
                np.testing.assert_array_equal(out, microphone - view)
                np.testing.assert_array_equal(np.concatenate([p for _, p in captured]), view)
                np.testing.assert_array_equal(np.concatenate([r for r, _ in captured]), microphone)
                self.assertTrue(out.flags.c_contiguous)
        # Non-native byte order is rejected, never reinterpreted by the C ABI.
        swapped = storage[:320].astype(np.dtype(np.int16).newbyteorder('S'))
        with self.assertRaises(ValueError):
            speex_linear_aec(Path('/missing'), swapped, storage[:320])

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
