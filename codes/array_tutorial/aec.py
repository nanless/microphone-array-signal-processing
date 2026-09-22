"""Small, auditable acoustic echo cancellation baselines.

These routines are teaching references, not replacements for a production AEC.
Signals are real-valued, one-dimensional NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def nlms(
    reference: np.ndarray,
    microphone: np.ndarray,
    filter_length: int,
    *,
    step_size: float = 0.5,
    epsilon: float = 1e-8,
    freeze: np.ndarray | None = None,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run sample-wise NLMS and return ``(residual, echo_hat, weights)``.

    ``freeze[n]`` prevents the update at sample ``n`` while still producing an
    output.  A zero-energy reference also skips the update, avoiding ``0 / 0``.
    The final weights use the ordering ``[x[n], x[n-1], ...]``.
    """

    x = np.asarray(reference, dtype=float)
    d = np.asarray(microphone, dtype=float)
    if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
        raise ValueError("reference and microphone must be equal-length 1-D arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(d)):
        raise ValueError("reference and microphone must be finite")
    if isinstance(filter_length, (bool, np.bool_)) or not isinstance(filter_length, (int, np.integer)) or filter_length <= 0:
        raise ValueError("filter_length must be a positive integer")
    if not 0.0 <= step_size < 2.0:
        raise ValueError("step_size must satisfy 0 <= step_size < 2")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")

    frozen = np.zeros(x.size, dtype=bool) if freeze is None else np.asarray(freeze, dtype=bool)
    if frozen.shape != x.shape:
        raise ValueError("freeze must have the same shape as reference")
    if initial_weights is None:
        weights = np.zeros(filter_length, dtype=float)
    else:
        weights = np.asarray(initial_weights, dtype=float).copy()
        if weights.shape != (filter_length,):
            raise ValueError("initial_weights has the wrong length")
        if not np.all(np.isfinite(weights)):
            raise ValueError("initial_weights must be finite")

    padded = np.pad(x, (filter_length - 1, 0))
    residual = np.empty_like(d)
    echo_hat = np.empty_like(d)
    for n in range(x.size):
        regression = padded[n : n + filter_length][::-1]
        echo_hat[n] = weights @ regression
        residual[n] = d[n] - echo_hat[n]
        energy = float(regression @ regression)
        if not frozen[n] and energy > 0.0:
            weights += step_size * residual[n] * regression / (energy + epsilon)
    return residual, echo_hat, weights


def erle_db(
    microphone: np.ndarray,
    residual: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    double_talk_mask: np.ndarray | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Return ERLE in dB over valid far-end single-talk samples only.

    ``double_talk_mask=True`` samples are always excluded.  The caller remains
    responsible for excluding convergence transients and near-end-only regions.
    """

    d = np.asarray(microphone, dtype=float)
    e = np.asarray(residual, dtype=float)
    if d.ndim != 1 or d.shape != e.shape:
        raise ValueError("microphone and residual must be equal-length 1-D arrays")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(e)):
        raise ValueError("microphone and residual must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    valid = np.ones(d.size, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
    if valid.shape != d.shape:
        raise ValueError("valid_mask must have the same shape as microphone")
    if double_talk_mask is not None:
        double_talk = np.asarray(double_talk_mask, dtype=bool)
        if double_talk.shape != d.shape:
            raise ValueError("double_talk_mask must have the same shape as microphone")
        valid &= ~double_talk
    if not np.any(valid):
        raise ValueError("ERLE requires at least one valid far-end single-talk sample")
    input_power = float(np.mean(d[valid] ** 2))
    residual_power = float(np.mean(e[valid] ** 2))
    if input_power <= epsilon:
        raise ValueError("ERLE is undefined without far-end input energy")
    return float(10.0 * np.log10((input_power + epsilon) / (residual_power + epsilon)))
