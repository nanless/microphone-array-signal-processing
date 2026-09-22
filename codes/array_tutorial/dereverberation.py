"""Minimal offline WPE implementations for complex STFT arrays."""

from __future__ import annotations

import numpy as np


def offline_wpe(
    spectrum: np.ndarray,
    *,
    taps: int,
    delay: int,
    iterations: int = 3,
    diagonal_loading: float = 1e-6,
    power_floor: float = 1e-5,
) -> np.ndarray:
    """Apply a compact single- or multi-channel offline WPE baseline.

    Input is ``(frequency, frame)`` or ``(frequency, channel, frame)``.
    Output has the same shape.  ``taps=0`` and records shorter than the first
    valid regression frame are passed through.  This batch routine uses the
    entire recording and is therefore not causal.
    """

    original = np.asarray(spectrum)
    if original.ndim not in (2, 3) or not np.iscomplexobj(original):
        raise ValueError("spectrum must be a complex array shaped (F,T) or (F,M,T)")
    if any(size == 0 for size in original.shape) or not np.all(np.isfinite(original)):
        raise ValueError("spectrum dimensions must be non-empty and values finite")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (taps, delay, iterations)):
        raise ValueError("taps, delay and iterations must be integers")
    if taps < 0 or delay < 1 or iterations < 0:
        raise ValueError("require taps >= 0, delay >= 1, iterations >= 0")
    if not np.isfinite(diagonal_loading) or not np.isfinite(power_floor) or diagonal_loading < 0 or power_floor <= 0:
        raise ValueError("loading must be non-negative and power_floor positive")
    if taps == 0 or iterations == 0:
        return original.copy()

    squeeze = original.ndim == 2
    y = original[:, None, :] if squeeze else original
    y = np.asarray(y, dtype=np.complex128)
    frequencies, channels, frames = y.shape
    first = delay + taps - 1
    if frames <= first:
        return original.copy()

    output = y.copy()
    dimension = channels * taps
    for frequency in range(frequencies):
        raw = y[frequency]
        # A common amplitude scale leaves the WPE solution unchanged. Divide
        # components separately: complex division can overflow its reciprocal
        # even when every desired ratio is bounded (subnormal input levels).
        input_scale = float(max(np.max(np.abs(raw.real)), np.max(np.abs(raw.imag))))
        if input_scale == 0.0:
            continue
        observed = np.empty_like(raw)
        observed.real = raw.real / input_scale
        observed.imag = raw.imag / input_scale
        current = observed.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                initial_scale = float(np.max(np.mean(np.abs(observed) ** 2, axis=0)))
                floor = power_floor * initial_scale
                if not np.isfinite(floor) or floor <= 0.0:
                    raise ValueError("relative WPE power floor is not representable")
                power = np.maximum(np.mean(np.abs(current) ** 2, axis=0), floor)
        except FloatingPointError as error:
            raise ValueError("WPE power estimate exceeds the float64 range") from error

        histories = []
        for frame in range(first, frames):
            # [x[t-delay], x[t-delay-1], ...], with all channels per lag.
            histories.append(
                np.concatenate([observed[:, frame - delay - lag] for lag in range(taps)])
            )
        history = np.asarray(histories, dtype=np.complex128)
        targets = observed[:, first:].T

        for _ in range(iterations):
            # Multiplying every inverse-power weight by the same positive
            # value does not change either the normal equations or relative
            # diagonal loading, and keeps the weights bounded by one.
            valid_power = np.maximum(power[first:], floor)
            weights = np.min(valid_power) / valid_power
            try:
                with np.errstate(over="raise", invalid="raise"):
                    correlation = np.einsum("t,ti,tj->ij", weights, history, history.conj())
                    cross = np.einsum("t,ti,tm->im", weights, history, targets.conj())
            except FloatingPointError as error:
                raise ValueError("WPE normal equations exceed the float64 range") from error
            trace = float(np.trace(correlation).real)
            if not np.isfinite(trace) or not np.all(np.isfinite(cross)):
                raise ValueError("WPE normal equations exceed the float64 range")
            if trace == 0.0 and np.all(history == 0):
                current = observed.copy()
                break
            if trace <= np.finfo(float).tiny:
                raise ValueError("WPE history energy is too small to solve reliably")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    loaded = correlation + diagonal_loading * trace / dimension * np.eye(dimension)
            except FloatingPointError as error:
                raise ValueError("WPE diagonal loading exceeds the float64 range") from error
            try:
                predictor = np.linalg.solve(loaded, cross)
            except np.linalg.LinAlgError:
                predictor = np.linalg.lstsq(loaded, cross, rcond=None)[0]
            if not np.all(np.isfinite(predictor)):
                raise ValueError("WPE predictor is not finite")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    prediction = history @ predictor.conj()
            except FloatingPointError as error:
                raise ValueError("WPE prediction exceeds the float64 range") from error
            current = observed.copy()
            current[:, first:] = (targets - prediction).T
            if not np.all(np.isfinite(current)):
                raise ValueError("WPE residual is not finite")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    power = np.maximum(np.mean(np.abs(current) ** 2, axis=0), floor)
            except FloatingPointError as error:
                raise ValueError("WPE residual power exceeds the float64 range") from error
        try:
            with np.errstate(over="raise", invalid="raise"):
                output[frequency].real = current.real * input_scale
                output[frequency].imag = current.imag * input_scale
        except FloatingPointError as error:
            raise ValueError("WPE output exceeds the float64 range") from error
        if not np.all(np.isfinite(output[frequency])):
            raise ValueError("WPE output exceeds the float64 range")

    return output[:, 0, :] if squeeze else output
