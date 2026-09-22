import json
from pathlib import Path
import unittest

import numpy as np

from codes.array_tutorial.engineering import (
    HysteresisVAD,
    PeakProtectAGC,
    RingBuffer,
    TELEMETRY_FIELDS,
    estimate_sro_ppm,
    q15_dequantize,
    q15_dot,
    q15_quantize,
    resample_sro_to_reference,
    simulate_deadline_queue,
    validate_telemetry,
)


ROOT = Path(__file__).resolve().parents[1]


class EngineeringCodeTests(unittest.TestCase):
    def test_sro_fit_separates_offset_and_rate(self):
        times = np.arange(0.0, 101.0, 10.0)
        delays = 0.003 - 75e-6 * times
        ppm, offset = estimate_sro_ppm(times, delays)
        self.assertAlmostEqual(ppm, -75.0, places=10)
        self.assertAlmostEqual(offset, 0.003, places=12)

    def test_sro_resampling_length_and_channels(self):
        signal = np.vstack([np.arange(10001.0), 2.0 * np.arange(10001.0)])
        result = resample_sro_to_reference(signal, 100.0)
        expected = int(np.floor((signal.shape[-1] - 1) / 1.0001)) + 1
        self.assertEqual(result.shape, (2, expected))
        np.testing.assert_allclose(result[1], 2.0 * result[0])

    def test_sro_resampling_direction_for_positive_and_negative_ppm(self):
        reference_rate = 16000.0
        tone_hz = 421.0
        for ppm in (-200.0, 200.0):
            ratio = 1.0 + ppm * 1e-6
            device_rate = reference_rate * ratio
            count = int(np.floor(0.5 * device_rate)) + 1
            device = np.sin(2.0 * np.pi * tone_hz * np.arange(count) / device_rate)
            aligned = resample_sro_to_reference(device, ppm)
            expected = np.sin(2.0 * np.pi * tone_hz * np.arange(aligned.size) / reference_rate)
            self.assertLess(float(np.max(np.abs(aligned - expected))), 0.004)

    def test_sro_rejects_nonpositive_rate(self):
        with self.assertRaises(ValueError):
            resample_sro_to_reference(np.arange(4.0), -1_000_000.0)

    def test_vad_hysteresis_and_hangover(self):
        vad = HysteresisVAD(0.04, 0.01, hangover_frames=2)
        amplitudes = [0.1, 0.3, 0.05, 0.0, 0.0, 0.0]
        states = [vad.update(np.full(32, value)) for value in amplitudes]
        self.assertEqual(states, [False, True, True, True, False, False])

    def test_agc_never_clips_and_releases_slowly(self):
        agc = PeakProtectAGC(target_peak=0.8, max_gain=4.0, attack=0.2, release=0.1, gain=4.0)
        _, raised_gain = agc.process(np.array([-0.05, 0.05]))
        self.assertEqual(raised_gain, 4.0)
        loud, loud_gain = agc.process(np.array([-2.0, 1.0]))
        quiet, quiet_gain = agc.process(np.array([-0.1, 0.1]))
        self.assertLessEqual(np.max(np.abs(loud)), 1.0)
        self.assertAlmostEqual(loud_gain, 0.5)
        self.assertLess(loud_gain, raised_gain)
        self.assertGreater(quiet_gain, loud_gain)
        self.assertLess(quiet_gain, 4.0)
        self.assertLessEqual(np.max(np.abs(quiet)), 1.0)

    def test_ring_buffer_drops_oldest(self):
        ring = RingBuffer(4)
        ring.write(np.arange(6.0))
        np.testing.assert_array_equal(ring.read(4), np.array([2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(ring.dropped, 2)
        self.assertEqual(len(ring), 0)

    def test_ring_buffer_preroll_across_wrap(self):
        ring = RingBuffer(5)
        ring.write(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(ring.read(2), np.array([0.0, 1.0]))
        ring.write(np.array([3.0, 4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(ring.read(5), np.array([2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_deadline_queue_reports_overload(self):
        result = simulate_deadline_queue(np.full(8, 25.0), 10.0, capacity_frames=2)
        self.assertGreater(result["deadline_misses"], 0)
        self.assertGreater(result["dropped"], 0)
        self.assertEqual(result["processed"] + result["dropped"], result["frames"])
        self.assertLessEqual(result["queue_high_water"], 2)

    def test_q15_saturates_and_round_trips(self):
        encoded = q15_quantize(np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
        np.testing.assert_array_equal(
            encoded,
            np.array([-32768, -32768, -16384, 16384, 32767, 32767], dtype=np.int16),
        )
        decoded = q15_dequantize(encoded)
        self.assertAlmostEqual(decoded[2], -0.5)
        self.assertAlmostEqual(decoded[3], 0.5)
        self.assertEqual(q15_dot(np.array([-32768], dtype=np.int16), np.array([-32768], dtype=np.int16)), 32767)
        self.assertEqual(q15_dot(np.array([16384, 16384], dtype=np.int16), np.array([16384, 16384], dtype=np.int16)), 16384)
        with self.assertRaises(ValueError):
            q15_dequantize(np.array([32768], dtype=np.int32))

    def test_telemetry_schema_matches_validator(self):
        schema_path = ROOT / "codes" / "engineering" / "telemetry_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(TELEMETRY_FIELDS))
        record = {
            "timestamp_ns": 1,
            "frame_index": 0,
            "sample_rate_hz": 16000,
            "queue_depth": 0,
            "xrun_count": 0,
            "dropped_samples": 0,
            "clipping_fraction": 0.0,
            "sro_ppm": 4.5,
            "rtf": 0.2,
            "vad_active": True,
            "deadline_miss": False,
            "agc_gain": 1.0,
            "model_version": "baseline-1",
        }
        self.assertEqual(validate_telemetry(record), [])
        record["clipping_fraction"] = 1.2
        record.pop("model_version")
        errors = validate_telemetry(record)
        self.assertIn("clipping_fraction must be in [0, 1]", errors)
        self.assertIn("missing field: model_version", errors)

        invalid = {name: record.get(name, 0) for name in TELEMETRY_FIELDS}
        invalid.update(sample_rate_hz=0, queue_depth=-1, sro_ppm=float("nan"), rtf=-0.1, model_version=" ")
        invalid_errors = validate_telemetry(invalid)
        self.assertIn("sample_rate_hz must be positive", invalid_errors)
        self.assertIn("queue_depth must be non-negative", invalid_errors)
        self.assertIn("sro_ppm must be finite", invalid_errors)
        self.assertIn("rtf must be non-negative", invalid_errors)
        self.assertIn("model_version must be non-empty", invalid_errors)


if __name__ == "__main__":
    unittest.main()
