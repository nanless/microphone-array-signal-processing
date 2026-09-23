"""Small real, full-covariance Kalman echo-path teaching baseline.

The state is an ``L``-tap real FIR echo path.  One call consumes one render
sample ``x[n]`` and one microphone sample ``d[n]`` and predicts the *causal*
echo ``taps[n] @ prior_weights`` before adapting.  Taps are ordered newest to
oldest.  The model is ``h[n] = a h[n-1] + q[n]`` and
``d[n] = taps[n] @ h[n] + v[n]``.  ``Q`` and ``R`` are supplied, known
per-sample covariance and variance in squared coefficient and squared signal
units, respectively; neither is estimated from audio here.

This time-domain, real, short-filter baseline is not FDKF, PBFDKF, a
double-talk detector, or an industrial echo canceller.  In particular it has
no FFT, overlap-save constraint, delay estimator or residual suppressor.
The full covariance and its per-step validity checks are deliberately costly
and not a real-time implementation strategy for a long acoustic path.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


def _covariance(value: object, name: str, length: int) -> np.ndarray:
    matrix = finite_real_array(value, name)
    if matrix.shape != (length, length):
        raise ValueError(f"{name} must have shape ({length}, {length})")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * matrix + 0.5 * matrix.T
    scale = float(np.max(np.abs(symmetric)))
    if scale and np.min(np.linalg.eigvalsh(symmetric / scale)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric.copy()


class KalmanAECState:
    """Keep a real FIR path estimate and its *full* covariance across calls.

    ``freeze=True`` skips the measurement update, but still propagates the
    state with ``a`` and the covariance with ``Q`` and advances render history.
    Thus it is an *adaptation freeze*, not a bitwise hold of weights when
    ``a != 1``.  ``reset()`` restores initial weights, covariance and history.
    """

    def __init__(
        self,
        filter_length: int,
        *,
        transition: float,
        process_covariance: object,
        observation_variance: float,
        initial_covariance: object,
        initial_weights: object | None = None,
        initial_history: object | None = None,
    ) -> None:
        if isinstance(filter_length, (bool, np.bool_)) or not isinstance(filter_length, (int, np.integer)) or filter_length < 1:
            raise ValueError("filter_length must be a positive integer")
        self.filter_length = int(filter_length)
        self.transition = finite_real_scalar(transition, "transition")
        self.observation_variance = finite_real_scalar(observation_variance, "observation_variance")
        if self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        self.process_covariance = _covariance(process_covariance, "process_covariance", self.filter_length)
        self._initial_covariance = _covariance(initial_covariance, "initial_covariance", self.filter_length)
        self._initial_weights = finite_real_array(
            np.zeros(self.filter_length) if initial_weights is None else initial_weights,
            "initial_weights",
        ).copy()
        self._initial_history = finite_real_array(
            np.zeros(self.filter_length - 1) if initial_history is None else initial_history,
            "initial_history",
        ).copy()
        if self._initial_weights.shape != (self.filter_length,):
            raise ValueError("initial_weights must have shape (filter_length,)")
        if self._initial_history.shape != (self.filter_length - 1,):
            raise ValueError("initial_history must have shape (filter_length - 1,)")
        self.reset()

    def reset(self) -> None:
        """Restore the complete state, including previously seen render samples."""
        self.weights = self._initial_weights.copy()
        self.covariance = self._initial_covariance.copy()
        self.history = self._initial_history.copy()

    def step(self, render_sample: float, microphone_sample: float, *, freeze: bool = False) -> dict:
        """Return causal prior echo/error and one posterior state.

        ``posterior_echo`` uses the current microphone sample during the
        update, so it is diagnostic only and must not replace ``prior_echo``
        when measuring the causal AEC output.
        """
        x = finite_real_scalar(render_sample, "render_sample")
        d = finite_real_scalar(microphone_sample, "microphone_sample")
        if not isinstance(freeze, (bool, np.bool_)):
            raise ValueError("freeze must be a bool")
        taps = np.concatenate(([x], self.history))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                prior_weights = self.transition * self.weights
                prior_covariance = (
                    (self.transition * self.transition) * self.covariance
                    + self.process_covariance
                )
                prior_covariance = _covariance(prior_covariance, "predicted covariance", self.filter_length)
                prior_echo = float(taps @ prior_weights)
                prior_error = d - prior_echo
                projected = prior_covariance @ taps
                innovation_variance = float(taps @ projected + self.observation_variance)
                if innovation_variance <= 0.0 or not np.isfinite(innovation_variance):
                    raise ValueError("innovation variance is not positive and finite")
                if freeze:
                    gain = np.zeros(self.filter_length)
                    posterior_weights = prior_weights.copy()
                    posterior_covariance = prior_covariance.copy()
                else:
                    gain = projected / innovation_variance
                    posterior_weights = prior_weights + gain * prior_error
                    residual_map = np.eye(self.filter_length) - np.outer(gain, taps)
                    # Joseph form remains symmetric/PSD under roundoff better
                    # than subtracting two nearly equal covariance matrices.
                    posterior_covariance = (
                        residual_map @ prior_covariance @ residual_map.T
                        + self.observation_variance * np.outer(gain, gain)
                    )
                    posterior_covariance = _covariance(
                        posterior_covariance, "posterior covariance", self.filter_length
                    )
                posterior_echo = float(taps @ posterior_weights)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("Kalman intermediate exceeds float64 range") from error
        if not all(np.all(np.isfinite(item)) for item in (
            prior_weights, prior_covariance, prior_echo, prior_error,
            projected, gain, posterior_weights, posterior_covariance,
            posterior_echo,
        )):
            raise ValueError("Kalman intermediate exceeds float64 range")
        self.weights = posterior_weights.copy()
        self.covariance = posterior_covariance.copy()
        self.history = taps[:-1].copy()
        return {
            "taps": taps.copy(),
            "prior_echo": prior_echo,
            "prior_error": prior_error,
            "prior_weights": prior_weights.copy(),
            "prior_covariance": prior_covariance.copy(),
            "innovation_variance": innovation_variance,
            "gain": gain.copy(),
            "posterior_weights": posterior_weights.copy(),
            "posterior_covariance": posterior_covariance.copy(),
            "posterior_echo": posterior_echo,
        }

    def process_block(
        self,
        render: object,
        microphone: object,
        *,
        freeze_mask: object | None = None,
    ) -> dict:
        """Process equal-length 1-D blocks; split blocks give identical results.

        An invalid input or failed intermediate leaves the original state
        untouched, including render history.
        """
        x = finite_real_array(render, "render")
        d = finite_real_array(microphone, "microphone")
        if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
            raise ValueError("render and microphone must be equal-length 1-D arrays")
        if freeze_mask is None:
            frozen = np.zeros(x.size, dtype=bool)
        else:
            frozen = np.asarray(freeze_mask)
            if frozen.dtype.kind != "b" or frozen.shape != x.shape:
                raise ValueError("freeze_mask must be a bool array matching the blocks")
        saved = (self.weights.copy(), self.covariance.copy(), self.history.copy())
        try:
            steps = [self.step(xi, di, freeze=bool(fi)) for xi, di, fi in zip(x, d, frozen)]
        except (ValueError, OverflowError):
            self.weights, self.covariance, self.history = saved
            raise
        return {
            "prior_echo": np.asarray([item["prior_echo"] for item in steps]),
            "prior_error": np.asarray([item["prior_error"] for item in steps]),
            "gain": np.asarray([item["gain"] for item in steps]).reshape(-1, self.filter_length),
            "posterior_weights": self.weights.copy(),
            "posterior_covariance": self.covariance.copy(),
        }
