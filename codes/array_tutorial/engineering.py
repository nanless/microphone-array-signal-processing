"""Deterministic engineering baselines for Chapter 10.

These functions are deliberately small enough to audit against the equations in
the tutorial.  They are teaching references, not audio-driver or production DSP
replacements.  Time is always the last array axis unless stated otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


def _finite_1d(values: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def estimate_sro_ppm(times_s: np.ndarray, delays_s: np.ndarray) -> tuple[float, float]:
    """Fit ``delay = intercept + slope * time`` and return first-order SRO ppm.

    ``delays_s`` uses ``reference time - device time``.  A positive slope
    therefore corresponds to ``f_device > f_reference``, matching
    :func:`resample_sro_to_reference`.  Fixed initial offset is returned
    separately and is not mistaken for sampling-rate offset.
    """

    times = _finite_1d(times_s, "times_s")
    delays = _finite_1d(delays_s, "delays_s")
    if times.size != delays.size or times.size < 2:
        raise ValueError("times_s and delays_s must have the same length >= 2")
    centered = times - np.mean(times)
    denominator = float(centered @ centered)
    if denominator == 0.0:
        raise ValueError("times_s must contain at least two distinct times")
    slope = float(centered @ (delays - np.mean(delays)) / denominator)
    intercept = float(np.mean(delays) - slope * np.mean(times))
    return slope * 1e6, intercept


def resample_sro_to_reference(samples: np.ndarray, relative_sro_ppm: float) -> np.ndarray:
    """Align samples from ``f_device=f_reference*(1+ppm*1e-6)``.

    Linear interpolation is used so the arithmetic is visible.  Production
    sample-rate conversion needs a band-limited filter and streaming state.
    """

    signal = np.asarray(samples, dtype=float)
    if signal.ndim not in (1, 2) or signal.shape[-1] < 2:
        raise ValueError("samples must have shape (time,) or (channel, time), with time >= 2")
    ratio = 1.0 + float(relative_sro_ppm) * 1e-6
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("relative_sro_ppm gives a non-positive or non-finite rate")
    output_length = int(np.floor((signal.shape[-1] - 1) / ratio)) + 1
    source_positions = np.arange(output_length, dtype=float) * ratio
    source_index = np.arange(signal.shape[-1], dtype=float)
    if signal.ndim == 1:
        return np.interp(source_positions, source_index, signal)
    return np.vstack(
        [np.interp(source_positions, source_index, channel) for channel in signal]
    )


@dataclass
class HysteresisVAD:
    """Frame-energy VAD with separate start/continue thresholds and hangover.

    A frame below ``off_threshold`` consumes one hangover count.  Therefore
    ``hangover_frames=2`` keeps the first two below-threshold frames active.
    Pre-roll audio is a separate buffering concern; see :class:`RingBuffer`.
    """

    on_threshold: float
    off_threshold: float
    hangover_frames: int
    active: bool = False
    _remaining: int = 0

    def __post_init__(self) -> None:
        if self.on_threshold < self.off_threshold or self.off_threshold < 0.0:
            raise ValueError("thresholds require on_threshold >= off_threshold >= 0")
        if self.hangover_frames < 0:
            raise ValueError("hangover_frames must be non-negative")

    def update(self, frame: np.ndarray) -> bool:
        samples = np.asarray(frame, dtype=float)
        if samples.size == 0 or not np.all(np.isfinite(samples)):
            raise ValueError("frame must be non-empty and finite")
        energy = float(np.mean(samples * samples))
        if not self.active:
            if energy >= self.on_threshold:
                self.active = True
                self._remaining = self.hangover_frames
            return self.active
        if energy >= self.off_threshold:
            self._remaining = self.hangover_frames
        elif self._remaining > 0:
            self._remaining -= 1
        else:
            self.active = False
        return self.active


@dataclass
class PeakProtectAGC:
    """Block AGC whose safety ceiling prevents output clipping.

    ``attack`` is used when gain must decrease; ``release`` is used when gain
    may increase.  Both are smoothing fractions in ``(0, 1]``.
    """

    target_peak: float = 0.8
    max_gain: float = 4.0
    attack: float = 1.0
    release: float = 0.1
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_peak <= 1.0:
            raise ValueError("target_peak must be in (0, 1]")
        if self.max_gain <= 0.0 or self.gain <= 0.0:
            raise ValueError("gains must be positive")
        if not 0.0 < self.attack <= 1.0 or not 0.0 < self.release <= 1.0:
            raise ValueError("attack and release must be in (0, 1]")

    def process(self, block: np.ndarray) -> tuple[np.ndarray, float]:
        samples = np.asarray(block, dtype=float)
        if samples.size == 0 or not np.all(np.isfinite(samples)):
            raise ValueError("block must be non-empty and finite")
        peak = float(np.max(np.abs(samples)))
        if peak == 0.0:
            desired = self.max_gain
            safety_ceiling = self.max_gain
        else:
            desired = min(self.max_gain, self.target_peak / peak)
            safety_ceiling = 1.0 / peak
        coefficient = self.attack if desired < self.gain else self.release
        smoothed = self.gain + coefficient * (desired - self.gain)
        self.gain = min(smoothed, safety_ceiling, self.max_gain)
        return samples * self.gain, self.gain


class RingBuffer:
    """Fixed-capacity FIFO that drops the oldest samples on overflow."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data = np.empty(self.capacity, dtype=float)
        self._start = 0
        self._size = 0
        self.dropped = 0

    def __len__(self) -> int:
        return self._size

    def write(self, values: np.ndarray) -> int:
        samples = _finite_1d(values, "values")
        for value in samples:
            if self._size == self.capacity:
                self._start = (self._start + 1) % self.capacity
                self._size -= 1
                self.dropped += 1
            end = (self._start + self._size) % self.capacity
            self._data[end] = value
            self._size += 1
        return samples.size

    def read(self, count: int) -> np.ndarray:
        if count < 0 or count > self._size:
            raise ValueError("count must be between zero and the current size")
        indices = (self._start + np.arange(count)) % self.capacity
        result = self._data[indices].copy()
        self._start = (self._start + count) % self.capacity
        self._size -= count
        return result


def simulate_deadline_queue(
    processing_ms: np.ndarray,
    frame_period_ms: float,
    capacity_frames: int,
) -> dict[str, float | int]:
    """Simulate one serial worker with a finite in-flight frame capacity.

    ``queue_high_water`` counts every accepted but unfinished frame, including
    the frame currently being processed.  It is not the wait-only
    ``queue_depth`` field used by the telemetry schema.
    """

    durations = _finite_1d(processing_ms, "processing_ms")
    if np.any(durations < 0.0):
        raise ValueError("processing durations must be non-negative")
    if frame_period_ms <= 0.0 or capacity_frames <= 0:
        raise ValueError("frame_period_ms and capacity_frames must be positive")
    finish_times: list[float] = []
    missed = 0
    dropped = 0
    high_water = 0
    max_wait = 0.0
    for index, duration in enumerate(durations):
        arrival = index * float(frame_period_ms)
        finish_times = [finish for finish in finish_times if finish > arrival]
        if len(finish_times) >= capacity_frames:
            dropped += 1
            continue
        start = max(arrival, finish_times[-1] if finish_times else arrival)
        finish = start + float(duration)
        finish_times.append(finish)
        high_water = max(high_water, len(finish_times))
        max_wait = max(max_wait, start - arrival)
        if finish > arrival + frame_period_ms:
            missed += 1
    return {
        "frames": int(durations.size),
        "processed": int(durations.size - dropped),
        "dropped": dropped,
        "deadline_misses": missed,
        "queue_high_water": high_water,
        "max_wait_ms": max_wait,
    }


def q15_quantize(values: np.ndarray) -> np.ndarray:
    """Round to nearest-even and saturate to signed Q1.15 integers."""

    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    scaled = np.rint(array * 32768.0)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def q15_dequantize(values: np.ndarray) -> np.ndarray:
    integers = np.asarray(values)
    if not np.issubdtype(integers.dtype, np.integer):
        raise ValueError("values must have an integer dtype")
    if np.any(integers < -32768) or np.any(integers > 32767):
        raise ValueError("values must be valid signed Q1.15 integers")
    return integers.astype(float) / 32768.0


def q15_dot(left: np.ndarray, right: np.ndarray) -> np.int16:
    """Q1.15 dot product with a wide accumulator and saturated output.

    Products are accumulated exactly in signed 64-bit integer space, scaled
    back by 2**15 with nearest-even rounding, and saturated once at the output.
    A target DSP may use another rounding point or accumulator width; record it
    before requiring bit-exact agreement.
    """

    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("left and right must have the same non-empty shape")
    for name, values in (("left", a), ("right", b)):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must have an integer dtype")
        if np.any(values < -32768) or np.any(values > 32767):
            raise ValueError(f"{name} must contain valid signed Q1.15 integers")
    accumulator = int(np.sum(a.astype(np.int64) * b.astype(np.int64), dtype=np.int64))
    rounded = int(np.rint(accumulator / 32768.0))
    return np.int16(min(32767, max(-32768, rounded)))


TELEMETRY_FIELDS = {
    "timestamp_ns": int,
    "frame_index": int,
    "sample_rate_hz": int,
    "queue_depth": int,
    "xrun_count": int,
    "dropped_samples": int,
    "clipping_fraction": float,
    "sro_ppm": float,
    "rtf": float,
    "vad_active": bool,
    "deadline_miss": bool,
    "agc_gain": float,
    "model_version": str,
}


def validate_telemetry(record: Mapping[str, object]) -> list[str]:
    """Return human-readable errors for one engineering telemetry record."""

    errors: list[str] = []
    for name, expected in TELEMETRY_FIELDS.items():
        if name not in record:
            errors.append(f"missing field: {name}")
            continue
        value = record[name]
        if expected is bool:
            valid_type = isinstance(value, (bool, np.bool_))
        elif expected is int:
            valid_type = isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        elif expected is float:
            valid_type = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, expected)
        if not valid_type:
            errors.append(f"wrong type for {name}")
    for name in ("timestamp_ns", "frame_index", "sample_rate_hz", "queue_depth", "xrun_count", "dropped_samples"):
        if name in record and isinstance(record[name], (int, np.integer)) and record[name] < 0:
            errors.append(f"{name} must be non-negative")
    if isinstance(record.get("sample_rate_hz"), (int, np.integer)) and record["sample_rate_hz"] == 0:
        errors.append("sample_rate_hz must be positive")
    for name in ("sro_ppm", "rtf", "agc_gain", "clipping_fraction"):
        value = record.get(name)
        if isinstance(value, (int, float, np.integer, np.floating)) and not math.isfinite(float(value)):
            errors.append(f"{name} must be finite")
    clipping = record.get("clipping_fraction")
    if isinstance(clipping, (int, float, np.integer, np.floating)) and not 0.0 <= float(clipping) <= 1.0:
        errors.append("clipping_fraction must be in [0, 1]")
    for name in ("rtf", "agc_gain"):
        value = record.get(name)
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) < 0.0:
            errors.append(f"{name} must be non-negative")
    if isinstance(record.get("model_version"), str) and not record["model_version"].strip():
        errors.append("model_version must be non-empty")
    return errors
