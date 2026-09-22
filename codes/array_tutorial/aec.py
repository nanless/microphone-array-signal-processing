"""Small, auditable acoustic echo cancellation baselines.

These routines are teaching references, not replacements for a production AEC.
Signals are real-valued, one-dimensional NumPy arrays.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


def _scaled_dot(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a real dot product without overflowing removable scale factors."""

    left_scale = float(np.max(np.abs(left)))
    right_scale = float(np.max(np.abs(right)))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    unit_dot = float((left / left_scale) @ (right / right_scale))
    left_mantissa, left_exponent = np.frexp(left_scale)
    right_mantissa, right_exponent = np.frexp(right_scale)
    return float(np.ldexp(
        unit_dot * left_mantissa * right_mantissa,
        int(left_exponent + right_exponent),
    ))


def _log10_mean_square_plus_floor(values: np.ndarray, floor: float) -> tuple[float, float]:
    """Return ``log10(mean(values**2) + floor)`` and the unfloored log power.

    Scaling before squaring keeps the calculation defined for finite float64
    inputs close to either end of the representable range.
    """

    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return float(np.log10(floor)), -np.inf
    normalized_power = float(np.mean((values / scale) ** 2))
    log_power = 2.0 * float(np.log10(scale)) + float(np.log10(normalized_power))
    log_floor = float(np.log10(floor))
    larger = max(log_power, log_floor)
    log_total = larger + float(np.log10(1.0 + 10.0 ** (-abs(log_power - log_floor))))
    return log_total, log_power


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

    x = finite_real_array(reference, "reference")
    d = finite_real_array(microphone, "microphone")
    if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
        raise ValueError("reference and microphone must be equal-length 1-D arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(d)):
        raise ValueError("reference and microphone must be finite")
    if isinstance(filter_length, (bool, np.bool_)) or not isinstance(filter_length, (int, np.integer)) or filter_length <= 0:
        raise ValueError("filter_length must be a positive integer")
    step_size = finite_real_scalar(step_size, "step_size")
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if not 0.0 <= step_size < 2.0:
        raise ValueError("step_size must satisfy 0 <= step_size < 2")
    if epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")

    frozen = np.zeros(x.size, dtype=bool) if freeze is None else np.asarray(freeze, dtype=bool)
    if frozen.shape != x.shape:
        raise ValueError("freeze must have the same shape as reference")
    if initial_weights is None:
        weights = np.zeros(filter_length, dtype=float)
    else:
        weights = finite_real_array(initial_weights, "initial_weights").copy()
        if weights.shape != (filter_length,):
            raise ValueError("initial_weights has the wrong length")
        if not np.all(np.isfinite(weights)):
            raise ValueError("initial_weights must be finite")

    padded = np.pad(x, (filter_length - 1, 0))
    residual = np.empty_like(d)
    echo_hat = np.empty_like(d)
    for n in range(x.size):
        regression = padded[n : n + filter_length][::-1]
        scale = float(np.max(np.abs(regression)))
        if scale == 0.0:
            echo_hat[n] = 0.0
            residual[n] = d[n]
            continue

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                echo_hat[n] = _scaled_dot(weights, regression)
                residual[n] = d[n] - echo_hat[n]
        except FloatingPointError as error:
            raise ValueError("NLMS prediction exceeds the float64 range") from error

        if frozen[n] or step_size == 0.0:
            continue
        normalized = regression / scale
        normalized_energy = float(normalized @ normalized)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            normalized_epsilon = (epsilon / scale) / scale
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                if np.isinf(normalized_epsilon):
                    update = step_size * (residual[n] * regression) / epsilon
                else:
                    update = (step_size * (residual[n] / scale) * normalized
                              / (normalized_energy + normalized_epsilon))
                candidate = weights + update
        except FloatingPointError as error:
            raise ValueError("NLMS weight update exceeds the float64 range") from error
        if not np.all(np.isfinite(candidate)):
            raise ValueError("NLMS weight update exceeds the float64 range")
        weights = candidate
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

    d = finite_real_array(microphone, "microphone")
    e = finite_real_array(residual, "residual")
    if d.ndim != 1 or d.shape != e.shape:
        raise ValueError("microphone and residual must be equal-length 1-D arrays")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(e)):
        raise ValueError("microphone and residual must be finite")
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if epsilon <= 0.0:
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
    input_total_log, input_power_log = _log10_mean_square_plus_floor(d[valid], epsilon)
    residual_total_log, _ = _log10_mean_square_plus_floor(e[valid], epsilon)
    if input_power_log <= np.log10(epsilon):
        raise ValueError("ERLE is undefined without far-end input energy")
    return float(10.0 * (input_total_log - residual_total_log))
