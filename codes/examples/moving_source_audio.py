"""Create a separate free-field moving-source listening fixture and truth file.

This synthetic example is not part of the 60-WAV main manifest and is not a
room recording. All exported WAVs receive one common gain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from codes.array_tutorial.audio_samples import pcm16_bytes
from codes.array_tutorial.moving_source import free_field_array, synthetic_source


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "codes" / "moving_audio"


def build_fixture(sample_rate: int = 16000, duration_seconds: float = 2.0) -> tuple[dict[str, np.ndarray], dict]:
    if sample_rate != 16000 or duration_seconds != 2.0:
        raise ValueError("this fixed experiment uses 16 kHz and 2 s")
    times = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    microphones = np.array([[-0.05, 0.0], [0.05, 0.0]])
    start = (-0.8, 1.5)
    velocity = (0.8, 0.0)
    moving, emission = free_field_array(times, microphones,
                                        source_start_xy=start, source_velocity_xy=velocity)
    static, _ = free_field_array(times, microphones,
                                 source_start_xy=start, source_velocity_xy=(0.0, 0.0))
    source = synthetic_source(times)[None, :]
    signals = {"source": source, "static_array": static, "moving_array": moving}
    peak = max(float(np.max(np.abs(x))) for x in signals.values())
    common_gain = 0.70 / peak
    signals = {key: value * common_gain for key, value in signals.items()}
    frame_times = np.arange(0, times.size, 160) / sample_rate
    positions_x = start[0] + velocity[0] * frame_times
    distances = np.sqrt((positions_x[:, None] - microphones[None, :, 0]) ** 2 + start[1] ** 2)
    signed_tdoa = (distances[:, 1] - distances[:, 0]) / 343.0
    angle = np.degrees(np.arctan2(positions_x, start[1]))
    metadata = {
        "model": "continuous moving point source in 2-D free field; exact retarded emission time and 1/r pressure; no reflection, HRTF, microphone directivity or measured noise",
        "sample_rate_hz": sample_rate,
        "duration_seconds": duration_seconds,
        "sound_speed_m_per_s": 343.0,
        "microphones_xy_m": microphones.tolist(),
        "source_start_xy_m": list(start),
        "source_velocity_xy_m_per_s": list(velocity),
        "source_signal": "deterministic 220 and 320 Hz sinusoids with 20 ms raised onset; not speech",
        "common_export_gain": common_gain,
        "truth_step_samples": 160,
        "truth": {"time_seconds": frame_times.tolist(), "angle_degrees_from_positive_y": angle.tolist(),
                  "mic1_minus_mic0_travel_time_seconds": signed_tdoa.tolist()},
        "retarded_equation_max_residual_seconds": float(np.max(np.abs(
            emission + np.linalg.norm(
                np.asarray(start)[None, None, :] + emission[:, :, None] * np.asarray(velocity) - microphones[:, None, :], axis=2
            ) / 343.0 - times[None, :]
        ))),
        "limits": "free-field point-source teaching fixture, not a room, speech, hardware recording, or measured tracking benchmark",
    }
    return signals, metadata


def generate(out_dir: Path = OUT) -> dict:
    signals, metadata = build_fixture()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for stem, waveform in signals.items():
        name = f"{stem}.wav"
        data = pcm16_bytes(waveform, metadata["sample_rate_hz"])
        (out_dir / name).write_bytes(data)
        files[name] = {"channels": int(waveform.shape[0]), "samples_per_channel": int(waveform.shape[1]),
                       "sha256": hashlib.sha256(data).hexdigest()}
    metadata["files"] = files
    (out_dir / "MANIFEST.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
