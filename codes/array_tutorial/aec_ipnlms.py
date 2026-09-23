"""Real, sample-wise IPNLMS teaching baseline for a single playback reference.

The per-tap gains follow Benesty and Huang, EUSIPCO 2004, Table 1. This is
not a complete AEC: the caller must align the reference and supply any
double-talk freeze decisions. It is not a real-time or fixed-point interface.
The current-first FIR and prior-output conventions match ``aec.NLMSState``.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


class IPNLMSState:
    """Keep FIR taps and the preceding ``L-1`` reference samples.

    ``kappa=-1`` allocates equal gains. To compare it step-for-step with
    ``NLMSState``, use ``NLMSState.epsilon = L * denominator_floor``.
    Positive ``gain_floor`` makes the gain computation defined at zero taps,
    but pure proportional allocation (``kappa=1``) still cannot start there.
    At ``kappa=1``, zero taps remain zero; use a smaller value to start from
    zero. The boundary is retained to make this failure mode testable.
    """

    def __init__(
        self,
        filter_length: int,
        *,
        step_size: float = 0.5,
        kappa: float = 0.0,
        denominator_floor: float = 1e-8,
        gain_floor: float = 1e-8,
        initial_weights: np.ndarray | None = None,
        initial_history: np.ndarray | None = None,
    ) -> None:
        if (isinstance(filter_length, (bool, np.bool_))
                or not isinstance(filter_length, (int, np.integer))
                or filter_length <= 0):
            raise ValueError("filter_length must be a positive integer")
        mu = finite_real_scalar(step_size, "step_size")
        kappa = finite_real_scalar(kappa, "kappa")
        delta = finite_real_scalar(denominator_floor, "denominator_floor")
        gain_floor = finite_real_scalar(gain_floor, "gain_floor")
        if not 0.0 <= mu < 2.0:
            raise ValueError("step_size must satisfy 0 <= mu < 2")
        if not -1.0 <= kappa <= 1.0:
            raise ValueError("kappa must satisfy -1 <= kappa <= 1")
        if delta <= 0.0 or gain_floor <= 0.0:
            raise ValueError("both floors must be positive")

        length = int(filter_length)
        weights = (np.zeros(length, dtype=float) if initial_weights is None
                   else finite_real_array(initial_weights, "initial_weights").copy())
        history = (np.zeros(length - 1, dtype=float) if initial_history is None
                   else finite_real_array(initial_history, "initial_history").copy())
        if weights.shape != (length,):
            raise ValueError("initial_weights must have shape (filter_length,)")
        if history.shape != (length - 1,):
            raise ValueError("initial_history must have shape (filter_length-1,)")

        self.filter_length = length
        self.step_size = mu
        self.kappa = kappa
        self.denominator_floor = delta
        self.gain_floor = gain_floor
        self._initial_weights = weights
        self._initial_history = history
        self.reset()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def history(self) -> np.ndarray:
        """Previous ``L-1`` reference samples, oldest first."""

        return self._history.copy()

    def reset(self) -> None:
        self._weights = self._initial_weights.copy()
        self._history = self._initial_history.copy()

    def tap_gains(self) -> np.ndarray:
        """Return current nonnegative diagonal gains of equation (6-4)."""

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                magnitude = np.abs(self._weights)
                norm = float(np.sum(magnitude))
                if not np.isfinite(norm):
                    raise ValueError("IPNLMS coefficient norm exceeds float64 range")
                gains = ((1.0 - self.kappa) / (2.0 * self.filter_length)
                         + (1.0 + self.kappa) * magnitude
                         / (2.0 * norm + self.gain_floor))
        except FloatingPointError as exc:
            raise ValueError("IPNLMS gain computation exceeds float64 range") from exc
        if not np.all(np.isfinite(gains)):
            raise ValueError("IPNLMS gains are not finite")
        return gains

    def process(
        self,
        reference: np.ndarray,
        microphone: np.ndarray,
        *,
        freeze: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return prior ``(residual, echo_hat)`` and retain updated state.

        A frozen sample still advances reference history. The call is
        transactional on invalid input or non-finite intermediate results.
        """

        x = finite_real_array(reference, "reference")
        d = finite_real_array(microphone, "microphone")
        if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
            raise ValueError("reference and microphone must be equal-length 1-D arrays")
        if freeze is None:
            frozen = np.zeros(x.shape, dtype=bool)
        else:
            frozen = np.asarray(freeze)
            if frozen.dtype.kind != "b" or frozen.shape != x.shape:
                raise ValueError("freeze must be a Boolean array matching reference")

        length = self.filter_length
        joined = np.concatenate((self._history, x))
        weights = self._weights.copy()
        errors = np.empty_like(d)
        predictions = np.empty_like(d)
        for n in range(x.size):
            u = joined[n:n + length][::-1]
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise",
                                 under="ignore"):
                    prediction = float(weights @ u)
                    error = float(d[n] - prediction)
                    if not frozen[n] and self.step_size and np.any(u):
                        magnitude = np.abs(weights)
                        norm = float(np.sum(magnitude))
                        if not np.isfinite(norm):
                            raise ValueError("IPNLMS coefficient norm exceeds float64 range")
                        gains = ((1.0 - self.kappa) / (2.0 * length)
                                 + (1.0 + self.kappa) * magnitude
                                 / (2.0 * norm + self.gain_floor))
                        denominator = float(u @ (gains * u) + self.denominator_floor)
                        if not (np.all(np.isfinite(gains))
                                and np.isfinite(denominator) and denominator > 0.0):
                            raise ValueError("IPNLMS normalization exceeds float64 range")
                        candidate = weights + (self.step_size * error / denominator) * gains * u
            except FloatingPointError as exc:
                raise ValueError("IPNLMS intermediate exceeds float64 range") from exc
            if not np.isfinite(prediction) or not np.isfinite(error):
                raise ValueError("IPNLMS prediction exceeds float64 range")
            predictions[n] = prediction
            errors[n] = error
            if not frozen[n] and self.step_size and np.any(u):
                if not np.all(np.isfinite(candidate)):
                    raise ValueError("IPNLMS update exceeds float64 range")
                weights = candidate

        self._weights = weights
        if length > 1:
            self._history = joined[-(length - 1):].copy()
        return errors, predictions


def ipnlms(
    reference: np.ndarray,
    microphone: np.ndarray,
    filter_length: int,
    *,
    step_size: float = 0.5,
    kappa: float = 0.0,
    denominator_floor: float = 1e-8,
    gain_floor: float = 1e-8,
    freeze: np.ndarray | None = None,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one stream and return ``(prior_residual, prior_echo, final_taps)``."""

    state = IPNLMSState(
        filter_length, step_size=step_size, kappa=kappa,
        denominator_floor=denominator_floor, gain_floor=gain_floor,
        initial_weights=initial_weights,
    )
    residual, echo_hat = state.process(reference, microphone, freeze=freeze)
    return residual, echo_hat, state.weights
