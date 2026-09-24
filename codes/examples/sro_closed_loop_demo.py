"""Controlled SRO estimation and stateful correction without an audio device.

The capture timestamps here are exact synthetic metadata. The interpolator is a
teaching tool, not a band-limited asynchronous sample-rate converter.
"""

from __future__ import annotations

from bisect import bisect_left
import json
import math
from statistics import median

def synthetic_signal(times_s: list[float]) -> list[float]:
    """Low-frequency tones plus smooth, known-time markers."""
    return [
        0.35 * math.sin(2 * math.pi * 47 * t)
        + 0.2 * math.sin(2 * math.pi * 113 * t)
        + math.fsum(0.35 * math.exp(-0.5 * ((t - center) / 0.003) ** 2)
                    for center in (2.0, 5.0, 8.0, 10.0))
        for t in times_s
    ]


def recover_device_indices(timestamps_s: list[float]) -> tuple[list[int], list[dict[str, int]]]:
    """Count isolated missing samples from exact, monotonic capture timestamps.

    This controlled rule assumes one stable sample clock and timestamp jitter
    much smaller than a sample period. It does not infer gaps from audio alone.
    """
    times = list(timestamps_s)
    if len(times) < 3 or not all(math.isfinite(t) for t in times):
        raise ValueError("timestamps must be a finite sequence of length >= 3")
    steps = [right - left for left, right in zip(times, times[1:])]
    if any(step <= 0 for step in steps):
        raise ValueError("timestamps must strictly increase")
    nominal_step = median(steps)
    rounded_steps = [round(step / nominal_step) for step in steps]
    if any(count < 1 or abs(step / nominal_step - count) > 0.05
           for step, count in zip(steps, rounded_steps)):
        raise ValueError("timestamp intervals do not fit a stable sample clock")
    indices = [0]
    for count in rounded_steps:
        indices.append(indices[-1] + count)
    gaps = [
        {"after_received_index": i, "missing_device_samples": count - 1}
        for i, count in enumerate(rounded_steps) if count > 1
    ]
    return indices, gaps


def fit_delay_ppm(reference_times_s: list[float], delays_s: list[float]) -> tuple[float, float]:
    """Fit delay = intercept + slope * reference time, with no package dependency."""
    if (len(reference_times_s) != len(delays_s) or len(delays_s) < 2
            or not all(math.isfinite(v) for v in reference_times_s + delays_s)):
        raise ValueError("fit inputs must have equal finite lengths >= 2")
    mean_time = math.fsum(reference_times_s) / len(reference_times_s)
    mean_delay = math.fsum(delays_s) / len(delays_s)
    centered_times = [t - mean_time for t in reference_times_s]
    denominator = math.fsum(t * t for t in centered_times)
    if denominator == 0:
        raise ValueError("reference times must vary")
    slope = math.fsum(t * (d - mean_delay)
                      for t, d in zip(centered_times, delays_s)) / denominator
    return slope * 1e6, mean_delay - slope * mean_time


class StatefulLinearClockCorrector:
    """Map one continuous device stream onto reference-clock samples.

    The phase (``next_output_index``) and last device sample survive each push.
    A missing device sample creates invalid outputs rather than being silently
    bridged. The caller must keep the same instance across chunks.
    """

    def __init__(self, reference_rate_hz: float, device_rate_hz: float,
                 device_start_s: float, reference_length: int,
                 output_start_index: int = 0):
        if not all(math.isfinite(v) for v in (reference_rate_hz, device_rate_hz, device_start_s)):
            raise ValueError("clock parameters must be finite")
        if reference_rate_hz <= 0 or device_rate_hz <= 0 or reference_length <= 0:
            raise ValueError("rates and reference length must be positive")
        if (isinstance(output_start_index, bool) or not isinstance(output_start_index, int)
                or not 0 <= output_start_index < reference_length):
            raise ValueError("output_start_index must be within the reference signal")
        self.reference_rate_hz = reference_rate_hz
        self.device_rate_hz = device_rate_hz
        self.device_start_s = device_start_s
        self.reference_length = reference_length
        self.next_output_index = max(output_start_index,
                                     math.ceil(device_start_s * reference_rate_hz - 1e-12))
        self.previous: tuple[int, float] | None = None

    def push(self, device_indices: list[int], samples: list[float]) -> list[tuple[int, float | None]]:
        indices = list(device_indices)
        values = list(samples)
        if (len(indices) != len(values)
                or any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
                or not all(math.isfinite(v) for v in values)):
            raise ValueError("indices and samples must be equal-length finite sequences")
        if len(indices) == 0:
            return []
        if any(right <= left for left, right in zip(indices, indices[1:])) or (self.previous and indices[0] <= self.previous[0]):
            raise ValueError("device indices must strictly increase across calls")
        known_indices = ([self.previous[0]] if self.previous else []) + indices
        known_values = ([self.previous[1]] if self.previous else []) + values
        emitted: list[tuple[int, float | None]] = []
        while self.next_output_index < self.reference_length:
            j = self.next_output_index
            position = (j / self.reference_rate_hz - self.device_start_s) * self.device_rate_hz
            if position > known_indices[-1] + 1e-10:
                break
            right = bisect_left(known_indices, position)
            if right < len(known_indices) and abs(known_indices[right] - position) < 1e-10:
                corrected: float | None = known_values[right]
            elif right == 0:
                corrected = None
            else:
                left = right - 1
                if right == len(known_indices) or known_indices[right] - known_indices[left] != 1:
                    corrected = None
                else:
                    fraction = position - known_indices[left]
                    corrected = known_values[left] * (1 - fraction) + known_values[right] * fraction
            emitted.append((j, corrected))
            self.next_output_index += 1
        self.previous = (int(indices[-1]), float(values[-1]))
        return emitted


def run_experiment(block_size: int = 257) -> dict[str, object]:
    """Estimate one clock, flag a deleted sample, and score before/after it."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    fs = 2000
    duration_s = 12
    true_ppm = 150.0
    true_start_s = 0.002
    true_device_rate = fs * (1 + true_ppm * 1e-6)
    missing_device_index = 12000
    reference_times = [n / fs for n in range(fs * duration_s)]
    device_indices_true = [n for n in range(math.ceil((duration_s - true_start_s)
                                                         * true_device_rate))
                           if n != missing_device_index]
    capture_times = [true_start_s + n / true_device_rate for n in device_indices_true]
    reference = synthetic_signal(reference_times)
    received = synthetic_signal(capture_times)

    recovered_indices, gaps = recover_device_indices(capture_times)
    # Before the gap, received index and physical device sample index agree.
    pre_gap_end = gaps[0]["after_received_index"] + 1 if gaps else len(received)
    # The first 2 s are calibration data. Do not estimate from the later gap
    # or score samples that precede this calibration window.
    anchor_indices = list(range(0, min(pre_gap_end, 4001), 500))
    times_for_fit = [n / fs for n in anchor_indices]
    measured_delay = [t - capture_times[n] for t, n in zip(times_for_fit, anchor_indices)]
    first_order_ppm, intercept_s = fit_delay_ppm(times_for_fit, measured_delay)
    slope = first_order_ppm * 1e-6
    estimated_ppm = slope / (1 - slope) * 1e6
    estimated_start_s = -intercept_s
    estimated_device_rate = fs * (1 + estimated_ppm * 1e-6)

    correction_start_index = 4500  # 2.25 s; after the final 2.0 s anchor.
    corrector = StatefulLinearClockCorrector(
        fs, estimated_device_rate, estimated_start_s, len(reference),
        output_start_index=correction_start_index,
    )
    corrected: dict[int, float | None] = {}
    for start in range(anchor_indices[-1] + 1, len(received), block_size):
        for output_index, value in corrector.push(
            recovered_indices[start:start + block_size], received[start:start + block_size],
        ):
            corrected[output_index] = value

    segments = {"before_drop": (5000, 11000), "after_drop": (13000, 22000)}
    scores = {}
    for name, (start, stop) in segments.items():
        valid_indices = [j for j in range(start, stop) if corrected.get(j) is not None]
        corrected_error = [float(corrected[j]) - reference[j] for j in valid_indices]
        raw_error = [received[j] - reference[j] for j in range(start, stop)]
        scores[name] = {
            "reference_samples": stop - start,
            "corrected_valid_samples": len(valid_indices),
            "uncorrected_mse": math.fsum(error * error for error in raw_error) / len(raw_error),
            "corrected_mse": math.fsum(error * error for error in corrected_error) / len(corrected_error),
        }
    invalid = [j for j, value in corrected.items() if value is None]
    return {
        "model": {"reference_rate_hz": fs, "duration_s": duration_s,
                  "true_device_sro_ppm": true_ppm, "true_initial_offset_ms": true_start_s * 1000,
                  "deleted_device_sample_index": missing_device_index,
                  "timestamp_model": "exact synthetic per-sample capture times"},
        "estimate": {"fit_anchor_count": len(anchor_indices),
                     "last_calibration_received_index": anchor_indices[-1],
                     "first_order_sro_ppm": first_order_ppm, "exact_sro_ppm": estimated_ppm,
                     "initial_offset_ms": estimated_start_s * 1000,
                     "detected_gaps": gaps},
        "compensation": {"method": "stateful linear interpolation; no anti-alias filter",
                         "block_size": block_size,
                         "first_reference_output_index": correction_start_index,
                         "invalid_output_indices": invalid,
                         "emitted_output_count": len(corrected)},
        "segments": scores,
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))
