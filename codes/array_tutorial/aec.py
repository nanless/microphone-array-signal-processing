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


def _boolean_mask(value: np.ndarray | None, shape: tuple[int, ...], name: str,
                  *, default: bool) -> np.ndarray:
    """Validate a mask without silently converting numbers or strings to bool."""

    if value is None:
        return np.full(shape, default, dtype=bool)
    mask = np.asarray(value)
    if mask.dtype.kind != "b" or mask.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return mask.copy()


class NLMSState:
    """Sample-wise NLMS with state preserved across arbitrary input blocks.

    ``history`` holds the previous ``filter_length - 1`` reference samples in
    oldest-to-newest order.  ``weights`` uses the current-first regression
    order ``[x[n], x[n-1], ...]``. Both properties return copies. ``reset()``
    restores the constructor's initial weights and history. Processing an empty
    block leaves the state unchanged; invalid blocks are rejected before update.

    This NumPy teaching implementation does not provide a real-time audio or
    thread-safe interface, a double-talk detector, or reference-delay tracking.
    """

    def __init__(
        self,
        filter_length: int,
        *,
        step_size: float = 0.5,
        epsilon: float = 1e-8,
        initial_weights: np.ndarray | None = None,
        initial_history: np.ndarray | None = None,
    ) -> None:
        if (isinstance(filter_length, (bool, np.bool_))
                or not isinstance(filter_length, (int, np.integer))
                or filter_length <= 0):
            raise ValueError("filter_length must be a positive integer")
        step_size = finite_real_scalar(step_size, "step_size")
        epsilon = finite_real_scalar(epsilon, "epsilon")
        if not 0.0 <= step_size < 2.0:
            raise ValueError("step_size must satisfy 0 <= step_size < 2")
        if epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")

        if initial_weights is None:
            weights = np.zeros(filter_length, dtype=float)
        else:
            weights = finite_real_array(initial_weights, "initial_weights").copy()
            if weights.shape != (filter_length,):
                raise ValueError("initial_weights has the wrong length")
        if initial_history is None:
            history = np.zeros(filter_length - 1, dtype=float)
        else:
            history = finite_real_array(initial_history, "initial_history").copy()
            if history.shape != (filter_length - 1,):
                raise ValueError("initial_history has the wrong length")

        self.filter_length = int(filter_length)
        self.step_size = step_size
        self.epsilon = epsilon
        self._initial_weights = weights
        self._initial_history = history
        self._weights = weights.copy()
        self._history = history.copy()

    @property
    def weights(self) -> np.ndarray:
        """Return a copy of the current filter coefficients."""

        return self._weights.copy()

    @property
    def history(self) -> np.ndarray:
        """Return a copy of recent reference samples, oldest first."""

        return self._history.copy()

    def reset(self) -> None:
        """Restore the initial coefficients and reference history."""

        self._weights = self._initial_weights.copy()
        self._history = self._initial_history.copy()

    def process(
        self,
        reference: np.ndarray,
        microphone: np.ndarray,
        *,
        freeze: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(residual, echo_hat)`` and retain the resulting state.

        The prediction for each sample uses the old weights; ``freeze[n]``
        suppresses only that sample's update, not its prediction or history.
        """

        x = finite_real_array(reference, "reference")
        d = finite_real_array(microphone, "microphone")
        if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
            raise ValueError("reference and microphone must be equal-length 1-D arrays")
        frozen = _boolean_mask(freeze, x.shape, "freeze", default=False)

        history_length = self.filter_length - 1
        concatenated = np.concatenate((self._history, x))
        residual = np.empty_like(d)
        echo_hat = np.empty_like(d)
        weights = self._weights.copy()
        for n in range(x.size):
            regression = concatenated[n : n + self.filter_length][::-1]
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

            if frozen[n] or self.step_size == 0.0:
                continue
            normalized = regression / scale
            normalized_energy = float(normalized @ normalized)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                normalized_epsilon = (self.epsilon / scale) / scale
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    if np.isinf(normalized_epsilon):
                        update = self.step_size * (residual[n] * regression) / self.epsilon
                    else:
                        update = (self.step_size * (residual[n] / scale) * normalized
                                  / (normalized_energy + normalized_epsilon))
                    candidate = weights + update
            except FloatingPointError as error:
                raise ValueError("NLMS weight update exceeds the float64 range") from error
            if not np.all(np.isfinite(candidate)):
                raise ValueError("NLMS weight update exceeds the float64 range")
            weights = candidate

        self._weights = weights
        if history_length:
            self._history = concatenated[-history_length:].copy()
        return residual, echo_hat


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

    state = NLMSState(filter_length, step_size=step_size, epsilon=epsilon,
                      initial_weights=initial_weights)
    residual, echo_hat = state.process(reference, microphone, freeze=freeze)
    return residual, echo_hat, state.weights


def erle_db(
    microphone: np.ndarray,
    residual: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    double_talk_mask: np.ndarray | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Return floor-regularized ERLE in dB over far-end single-talk samples.

    ``double_talk_mask=True`` samples are always excluded.  The caller remains
    responsible for excluding convergence transients and near-end-only regions.
    The reported ratio adds ``epsilon`` to both mean-square powers, so perfect
    cancellation returns a finite, epsilon-dependent value rather than the
    ideal infinite ERLE. It is not a device noise-floor measurement.
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
    valid = _boolean_mask(valid_mask, d.shape, "valid_mask", default=True)
    if double_talk_mask is not None:
        double_talk = _boolean_mask(double_talk_mask, d.shape, "double_talk_mask",
                                    default=False)
        valid &= ~double_talk
    if not np.any(valid):
        raise ValueError("ERLE requires at least one valid far-end single-talk sample")
    input_total_log, input_power_log = _log10_mean_square_plus_floor(d[valid], epsilon)
    residual_total_log, _ = _log10_mean_square_plus_floor(e[valid], epsilon)
    if input_power_log <= np.log10(epsilon):
        raise ValueError("ERLE is undefined without far-end input energy")
    return float(10.0 * (input_total_log - residual_total_log))
