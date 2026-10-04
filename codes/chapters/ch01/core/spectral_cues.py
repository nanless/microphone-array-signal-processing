"""E01-10: two artificial FIR responses, not measured HRIRs or directions.

Known-frequency complex projections use the negative Fourier exponent and
2/N scaling on the declared steady window. This is not blind HRTF estimation.
No files, directories, experiments or network operations run at import time.
"""
from __future__ import annotations

import math
import numpy as np

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
OUTPUT_SAMPLES = 32001
WINDOW = (1600, 30400)
FREQUENCIES = (1000, 7000)
CASES = {"flat": (.1, .1), "tilted": (.12, .04)}


def build_cases() -> dict[str, np.ndarray]:
    """Return 1x32000 sources and 2x32001 complete causal FIR outputs.

All four mathematical records use export gain one and the same 20 ms linear
edge envelope. Channel zero is left; h_L=[1,.5], h_R=[1,-.5].
"""
    time = np.arange(SOURCE_SAMPLES) / SAMPLE_RATE
    envelope = np.minimum(1., np.minimum(time / .02, (2. - time) / .02))
    records = {}
    for name, amplitudes in CASES.items():
        source = sum(amplitude * np.sin(2 * np.pi * frequency * time)
                     for amplitude, frequency in zip(amplitudes, FREQUENCIES)) * envelope
        stereo = np.zeros((2, OUTPUT_SAMPLES))
        stereo[:, :SOURCE_SAMPLES] = source
        stereo[0, 1:] += .5 * source
        stereo[1, 1:] -= .5 * source
        records[name + "_source"] = source[None, :]
        records[name + "_stereo"] = stereo
    return records


def _real_record(record: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(record)
    if values.shape != shape or values.dtype.kind not in "iuf":
        raise ValueError(f"Expected finite real numeric array with shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Samples must be representable finite real numbers")
    return values


def _representable_power(amplitude: float) -> float:
    try:
        power = amplitude * amplitude
    except OverflowError as error:
        raise ValueError("Power is not representable as a positive finite float") from error
    if not math.isfinite(power) or power <= 0:
        raise ValueError("Power is not representable as a positive finite float")
    return power


def _window_metrics(record: np.ndarray) -> tuple[float, list[complex]]:
    start, stop = WINDOW
    values = record[start:stop]
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        raise ValueError("Steady window must have nonzero energy")
    normalized = values / scale
    rms = scale * math.sqrt(float(np.mean(normalized**2)))
    power = _representable_power(rms)
    index = np.arange(start, stop)
    coefficients = []
    for frequency in FREQUENCIES:
        projected = 2 * np.mean(normalized * np.exp(-2j * np.pi * frequency * index / SAMPLE_RATE))
        # A mathematically absent line can leave rounding-sized trigonometric
        # residue. Reject it rather than form a ratio from numerical leakage.
        if abs(projected) <= 64 * np.finfo(float).eps:
            raise ValueError("Known-frequency projection has no resolvable nonzero support")
        coefficient = complex(projected * scale)
        if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)):
            raise ValueError("Complex projection is not representable")
        if abs(coefficient) == 0:
            raise ValueError("Complex projection underflowed")
        coefficients.append(coefficient)
    return power, coefficients


def measure_response(source: np.ndarray, stereo: np.ndarray) -> dict:
    """Measure uncentered powers and two known-frequency response ratios.

Input shapes are source=(1,32000), stereo=(2,32001), real and finite. The
half-open 28800-point window excludes fades and the retained final FIR tail.
Complex coefficients are 2/N sum(x[n] exp(-j omega n)), encoded [real,imag].
They are sinusoidal amplitude phasors, not RMS amplitudes or calibrated Pa.
ILD and IPD are right minus left; IPD uses [-pi,pi). Finite metrics require
positive representable powers and nonzero source/left/right line support.
Projections <=64 machine eps relative to that window's peak are rejected.
Unrepresentable powers/gains are rejected, never silently replaced by zero,
infinity or a floor. Scaling is internal arithmetic, not gain compensation.
"""
    source = _real_record(source, (1, SOURCE_SAMPLES))
    stereo = _real_record(stereo, (2, OUTPUT_SAMPLES))
    source_power, source_coefficients = _window_metrics(source[0])
    left_power, left_coefficients = _window_metrics(stereo[0])
    right_power, right_coefficients = _window_metrics(stereo[1])
    spectral = []
    for frequency, src, left, right in zip(FREQUENCIES, source_coefficients,
                                           left_coefficients, right_coefficients):
        source_amplitude, left_amplitude, right_amplitude = abs(src), abs(left), abs(right)
        left_gain = _representable_power(left_amplitude / source_amplitude)
        right_gain = _representable_power(right_amplitude / source_amplitude)
        phase = (math.atan2(right.imag, right.real) - math.atan2(left.imag, left.real)
                 + math.pi) % (2 * math.pi) - math.pi
        spectral.append({
            "frequency_hz": frequency,
            "source_complex": [src.real, src.imag],
            "left_complex": [left.real, left.imag],
            "right_complex": [right.real, right.imag],
            "left_power_gain": left_gain, "right_power_gain": right_gain,
            "ild_right_minus_left_db": 20 * (math.log10(right_amplitude) - math.log10(left_amplitude)),
            "ipd_right_minus_left_rad": phase,
            "ipd_right_minus_left_deg": math.degrees(phase),
        })
    return {
        "source_mean_square": source_power,
        "left_mean_square": left_power, "right_mean_square": right_power,
        "ild_right_minus_left_db": 10 * (math.log10(right_power) - math.log10(left_power)),
        "spectral": spectral,
    }
