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
        observed = y[frequency]
        current = output[frequency]
        observed_scale = float(np.max(np.mean(np.abs(observed) ** 2, axis=0)))
        if observed_scale <= np.finfo(float).tiny:
            output[frequency] = observed
            continue
        initial_scale = observed_scale
        floor = power_floor * initial_scale
        power = np.maximum(np.mean(np.abs(current) ** 2, axis=0), floor)

        histories = []
        for frame in range(first, frames):
            # [x[t-delay], x[t-delay-1], ...], with all channels per lag.
            histories.append(
                np.concatenate([observed[:, frame - delay - lag] for lag in range(taps)])
            )
        history = np.asarray(histories, dtype=np.complex128)
        targets = observed[:, first:].T

        for _ in range(iterations):
            weights = 1.0 / np.maximum(power[first:], floor)
            correlation = np.einsum("t,ti,tj->ij", weights, history, history.conj())
            cross = np.einsum("t,ti,tm->im", weights, history, targets.conj())
            trace = float(np.trace(correlation).real)
            if trace <= np.finfo(float).tiny:
                current = observed.copy()
                break
            loaded = correlation + diagonal_loading * trace / dimension * np.eye(dimension)
            try:
                predictor = np.linalg.solve(loaded, cross)
            except np.linalg.LinAlgError:
                predictor = np.linalg.lstsq(loaded, cross, rcond=None)[0]
            prediction = history @ predictor.conj()
            current = observed.copy()
            current[:, first:] = (targets - prediction).T
            power = np.maximum(np.mean(np.abs(current) ** 2, axis=0), floor)
        output[frequency] = current

    return output[:, 0, :] if squeeze else output
