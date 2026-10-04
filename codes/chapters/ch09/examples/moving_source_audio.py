"""Create a separate free-field moving-source listening fixture and truth file.

This synthetic example is separate from the main audio manifest and is not a
room recording. All exported WAVs receive one common gain.
"""

from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))

import argparse
import hashlib
import platform
import json
from pathlib import Path

import numpy as np
from codes.chapters.ch00.io_contracts import same_metadata, strict_json_loads, validate_asset_directory

from codes.chapters.ch00.core.audio_samples import pcm16_bytes
from codes.chapters.ch09.core.moving_source import free_field_array, synthetic_source
from codes.chapters.ch02.core.conventions import finite_real_scalar


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "codes/chapters/ch09/moving_audio"


def build_fixture(sample_rate: int = 16000, duration_seconds: float = 2.0) -> tuple[dict[str, np.ndarray], dict]:
    sample_rate = finite_real_scalar(sample_rate, "sample_rate")
    duration_seconds = finite_real_scalar(duration_seconds, "duration_seconds")
    if sample_rate != 16000 or duration_seconds != 2.0:
        raise ValueError("this fixed experiment uses 16 kHz and 2 s")
    sample_rate = int(sample_rate)
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


SOURCE_PATHS = (
    'codes/chapters/ch09/examples/moving_source_audio.py', 'codes/chapters/ch09/core/moving_source.py',
    'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py', 'codes/chapters/ch02/core/conventions.py',
)


# Reviewed literal values/types, independent of build_fixture return metadata.
REQUIRED_MODEL = {'model': 'continuous moving point source in 2-D free field; exact retarded emission time and 1/r '
          'pressure; no reflection, HRTF, microphone directivity or measured noise',
 'sample_rate_hz': 16000,
 'duration_seconds': 2.0,
 'sound_speed_m_per_s': 343.0,
 'microphones_xy_m': [[-0.05, 0.0], [0.05, 0.0]],
 'source_start_xy_m': [-0.8, 1.5],
 'source_velocity_xy_m_per_s': [0.8, 0.0],
 'source_signal': 'deterministic 220 and 320 Hz sinusoids with 20 ms raised onset; not speech',
 'truth_step_samples': 160,
 'limits': 'free-field point-source teaching fixture, not a room, speech, hardware recording, or '
           'measured tracking benchmark'}

def _validate_model(signals, metadata):
    """Reject internal model drift before writing any existing asset.

    This is a trusted-code regression contract, not a hostile-code sandbox.
    The shared physical kernels retain the sole waveform implementation.
    """
    if type(metadata) is not dict or set(metadata) != set(REQUIRED_MODEL) | {
            'truth', 'common_export_gain', 'retarded_equation_max_residual_seconds'}:
        raise ValueError('moving model metadata fields differ')
    if not same_metadata({key: metadata[key] for key in REQUIRED_MODEL}, REQUIRED_MODEL):
        raise ValueError('moving fixed model values or true types differ')
    if type(signals) is not dict or set(signals) != {'source', 'static_array', 'moving_array'}:
        raise ValueError('three declared moving waveform members required')
    for key, signal in signals.items():
        if (not isinstance(signal, np.ndarray) or signal.dtype.kind != 'f'
                or signal.shape != (1 if key == 'source' else 2, 32000)
                or not np.isfinite(signal).all()):
            raise ValueError('fixed finite real floating waveform required: ' + key)
    times = np.arange(32000) / 16000
    microphones = np.array([[-.05, 0.], [.05, 0.]])
    moving, emission = free_field_array(times, microphones,
        source_start_xy=(-.8, 1.5), source_velocity_xy=(.8, 0.))
    static, _ = free_field_array(times, microphones,
        source_start_xy=(-.8, 1.5), source_velocity_xy=(0., 0.))
    source = synthetic_source(times)[None, :]
    gain = .70 / max(float(np.max(abs(x))) for x in (source, moving, static))
    expected_signals = {'source': source * gain, 'static_array': static * gain, 'moving_array': moving * gain}
    if any(not np.array_equal(signals[key], value) for key, value in expected_signals.items()):
        raise ValueError('moving waveform differs from the declared shared-kernel model')
    frame_times = np.arange(0, 32000, 160) / 16000
    px = -.8 + .8 * frame_times
    distances = np.sqrt((px[:, None] - microphones[None, :, 0]) ** 2 + 1.5 ** 2)
    expected_truth = {'time_seconds': frame_times.tolist(),
        'angle_degrees_from_positive_y': np.degrees(np.arctan2(px, 1.5)).tolist(),
        'mic1_minus_mic0_travel_time_seconds': ((distances[:, 1]-distances[:, 0])/343.).tolist()}
    residual = float(np.max(abs(emission + np.linalg.norm(
        np.array([-.8, 1.5])[None, None, :] + emission[:, :, None] * np.array([.8, 0.])
        - microphones[:, None, :], axis=2) / 343. - times[None, :])))
    for key, expected in {'truth': expected_truth, 'common_export_gain': gain,
            'retarded_equation_max_residual_seconds': residual}.items():
        if not same_metadata(metadata[key], expected):
            raise ValueError('moving model truth, gain or numerical metadata differs: ' + key)


def _validate_directory(directory, expected, *, check):
    """Local preflight only; no concurrency or crash-persistence guarantee."""
    return validate_asset_directory(directory, expected, check=check)


def generate(out_dir: Path = OUT, *, check: bool = False) -> dict:
    """Generate assets or strictly check them in memory without writing."""
    out_dir = Path(out_dir)
    _validate_directory(out_dir, ("source.wav", "static_array.wav", "moving_array.wav", "MANIFEST.json"), check=check)
    signals, metadata = build_fixture()
    _validate_model(signals, metadata)
    assets, files = {}, {}
    for stem, waveform in signals.items():
        name = f"{stem}.wav"
        data = pcm16_bytes(waveform, metadata["sample_rate_hz"])
        assets[name] = data
        files[name] = {"channels": int(waveform.shape[0]), "samples_per_channel": int(waveform.shape[1]),
                       "sha256": hashlib.sha256(data).hexdigest()}
    metadata["files"] = files
    metadata["source_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    metadata["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    metadata["truth_time_axis"] = "emission-event clock: path position at u and both microphone travel times for this same u; not simultaneous receiver samples"
    metadata["propagation_scope"] = "assigned retarded-time 1/r pressure model; no moving-monopole radiation-amplitude correction"
    assets['MANIFEST.json'] = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if check:
        recorded = strict_json_loads((out_dir/'MANIFEST.json').read_bytes())
        if not same_metadata(recorded, metadata):
            raise ValueError('manifest differs from current fixed model, source identities or scores')
        for name, data in assets.items():
            path = out_dir/name
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f'missing or stale moving-source asset: {path}')
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in assets.items():
            (out_dir/name).write_bytes(data)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = generate(args.output, check=args.check)
    print(json.dumps({'mode': 'verified' if args.check else 'generated',
                      'files': list(report['files'])}, ensure_ascii=False, indent=2))
