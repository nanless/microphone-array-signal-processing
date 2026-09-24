"""Independent checks for Appendix B exercise 16's offline and metric paths."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codes.examples.room_srp_exercise import (
    SAMPLE_RATE_HZ, _make_room, _write_pcm16, configuration, drr_db,
    export_audio, measured_t60_from_t20, run_experiment, validate_configuration,
)


class RoomSrpExerciseTest(unittest.TestCase):
    def test_geometry_and_sabine_inputs_match_independent_hand_calculation(self):
        report = configuration()
        self.assertEqual(report["status"], "geometry_checked_only")
        self.assertEqual(report["room_volume_m3"], 720.0)
        self.assertEqual(report["room_surface_m2"], 504.0)
        self.assertAlmostEqual(report["sabine_energy_absorption"],
                               24 * math.log(10) * 720 / (343 * 504 * 0.6))
        self.assertEqual(report["inverse_sabine_suggested_max_order"], 40)
        microphones = np.asarray(report["microphones_m"])
        farthest_pair_m = max(np.linalg.norm(microphones[i] - microphones[j])
                              for i in range(4) for j in range(i + 1, 4))
        self.assertAlmostEqual(farthest_pair_m, 0.06 * math.sqrt(2))
        self.assertLess(farthest_pair_m, 343 / (2 * 2000))
        self.assertEqual(len(report["cases"]), 6)
        self.assertEqual([r["distance_m"] for r in report["cases"][:4]], [1, 1, 2.5, 2.5])
        self.assertEqual([r["true_azimuth_deg"] for r in report["cases"][:4]],
                         [-30, 30, -30, 30])
        self.assertEqual(report, configuration(), "seeded positions must reproduce exactly")
        self.assertNotIn("results", report)

    def test_geometry_rejects_coordinate_label_mismatch(self):
        report = configuration()
        report["cases"][0]["source_m"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "distance and coordinates disagree"):
            validate_configuration(report)

    def test_drr_subtracts_direct_component_before_power(self):
        # Direct impulse 1; a separate reflected impulse 0.5 has 1/4 power.
        self.assertAlmostEqual(drr_db(np.array([1.0, 0.5]), np.array([1.0])),
                               10 * math.log10(4))
        # Negative reflected amplitude still contributes positive energy.
        self.assertAlmostEqual(drr_db(np.array([1.0, -0.5]), np.array([1.0, 0.0])),
                               10 * math.log10(4))
        with self.assertRaisesRegex(ValueError, "nonzero direct and reflected"):
            drr_db(np.array([1.0]), np.array([1.0]))

    def test_t20_extrapolation_on_known_exponential_energy_decay(self):
        # If h(t) = exp(-a t), squared-energy decay has slope -20a/ln(10)
        # dB/s and T60 = 3 ln(10)/a. The 2 s tail is > 3 target T60s.
        target_t60_s = 0.6
        a = 3 * math.log(10) / target_t60_s
        time_s = np.arange(2 * SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
        rir = np.exp(-a * time_s)
        self.assertAlmostEqual(measured_t60_from_t20(rir), target_t60_s, delta=0.002)
        with self.assertRaisesRegex(ValueError, "usable -5 to -25 dB"):
            measured_t60_from_t20(np.ones(100))

    def test_offline_cli_emits_finite_json_without_simulation_result(self):
        completed = subprocess.run(
            [sys.executable, "-m", "codes.examples.room_srp_exercise", "--check"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "geometry_checked_only")
        self.assertEqual(len(report["cases"]), 6)
        self.assertNotIn("actual_max_order", report)
        self.assertNotIn("results", report)
        json.dumps(report, allow_nan=False)

    def test_pcm_writer_preserves_channel_order_and_rejects_clipping(self):
        signal = np.array([[0.0, 0.25, -0.25, 0.5],
                           [0.5, -0.5, 0.25, -0.25]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "four_mics.wav"
            _write_pcm16(path, signal)
            with wave.open(str(path), "rb") as stream:
                self.assertEqual((stream.getnchannels(), stream.getframerate(),
                                  stream.getsampwidth(), stream.getnframes()),
                                 (4, SAMPLE_RATE_HZ, 2, 2))
                decoded = np.frombuffer(stream.readframes(2), dtype="<i2").reshape(2, 4)
            np.testing.assert_array_equal(decoded, np.rint(signal * 32767).astype("<i2"))
            with self.assertRaisesRegex(ValueError, "clip"):
                _write_pcm16(path, np.array([[1.01]]))
            with self.assertRaisesRegex(ValueError, "finite"):
                _write_pcm16(path, np.array([[float("nan")]]))

    def test_actual_room_direct_component_and_audio_export_when_dependency_available(self):
        try:
            import pyroomacoustics as pra
            from pyroomacoustics.utilities import design_highpass_filter_sos
            from scipy.signal import sosfiltfilt
        except ImportError:
            self.skipTest("pyroomacoustics 0.10.0 and scipy are optional")
        if pra.__version__ != "0.10.0":
            self.skipTest("this numeric check is pinned to pyroomacoustics 0.10.0")
        config = configuration()
        case = config["cases"][0]
        full = _make_room(pra, case["source_m"], config["microphones_m"],
                          config["sabine_energy_absorption"], 2)
        direct = _make_room(pra, case["source_m"], config["microphones_m"], 1.0, 2)
        source = full.sources[0]
        mask = (full.visibility[0][0] > 0) & (source.orders == 0)
        self.assertEqual(int(np.sum(mask)), 1)
        raw_direct = pra.simulation.compute_ism_rir(
            source, full.mic_array.R[:, 0], full.mic_array.directivity[0],
            source.directions[0], mask, pra.constants.get("frac_delay_length"),
            full.c, full.fs, full.octave_bands, min_phase=full.min_phase,
            air_abs_coeffs=full.air_absorption)
        raw_direct = np.pad(raw_direct, (0, len(full.rir[0][0]) - len(raw_direct)))
        if pra.constants.get("rir_hpf_enable"):
            sos = design_highpass_filter_sos(
                full.fs, pra.constants.get("rir_hpf_fc"),
                **pra.constants.get("rir_hpf_kwargs"))
            raw_direct = sosfiltfilt(sos, raw_direct)
        np.testing.assert_allclose(direct.rir[0][0], raw_direct, atol=1e-7, rtol=0)
        self.assertEqual(len(full.rir[0][0]), len(direct.rir[0][0]))
        self.assertGreater(drr_db(full.rir[0][0], direct.rir[0][0]), 0.0)

        report = run_experiment()
        self.assertEqual(report["actual_max_order"], 40)
        self.assertTrue(report["convergence_check"]["within_threshold"])
        self.assertEqual(len(report["results"]), 6)
        self.assertTrue(all(row["elapsed_s"] > 0 for row in report["results"]))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "audio"
            manifest = export_audio(report, directory)
            self.assertEqual(len(manifest["files"]), 18)
            self.assertLessEqual(max(row["peak_after_gain"]
                                     for row in manifest["files"]), 0.8 + 1e-12)
            for file in manifest["files"]:
                with wave.open(str(directory / file["file"]), "rb") as stream:
                    self.assertEqual(stream.getnchannels(), file["channels"])
                    self.assertEqual(stream.getnframes(), file["frames"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_audio(report, directory)


if __name__ == "__main__":
    unittest.main()
