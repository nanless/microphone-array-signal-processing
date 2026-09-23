"""Small real-valued, full-matrix RLS baseline for acoustic echo cancellation.

This is an arithmetic teaching reference, not a real-time AEC stack. There is
no delay estimator, double-talk detector, residual suppressor, or device I/O.
The caller supplies an external Boolean freeze mask when one is appropriate.

For accepted samples indexed 0..n, ``R[-1] = delta * I`` and
``P[-1] = R[-1]**-1`` give the exactly matched objective

    sum_i lambda**(n-i) * (d[i] - w.T @ u[i])**2
        + lambda**(n+1) * delta * ||w - w_initial||**2.

Here ``u[i] = [x[i], x[i-1], ...]`` includes the reference history. A frozen
sample advances the reference history and produces a *prior* prediction, but
does not enter the least-squares objective or advance its forgetting clock.
An unfrozen zero-reference sample does enter that clock: ``P <- P/lambda``.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


class RLSState:
    """Streaming full-matrix RLS with current-first FIR taps and sample updates.

    ``initial_regularization`` is the scale of the initial normal matrix,
    not the initial inverse covariance. Its reciprocal is ``P[-1]``'s diagonal.
    This teaching implementation uses ``O(L**2)`` storage. Its RLS algebra
    takes ``O(L**2)`` work per accepted sample, but the explicit Cholesky
    positive-definiteness check below costs ``O(L**3)``. It is not a
    long-filter real-time implementation.
    ``reset()`` restores the constructor's coefficients, inverse matrix and
    reference history. Public array properties return copies.
    """

    def __init__(
        self,
        filter_length: int,
        *,
        forgetting_factor: float = 0.999,
        initial_regularization: float = 1.0,
        initial_weights: np.ndarray | None = None,
        initial_history: np.ndarray | None = None,
    ) -> None:
        if (isinstance(filter_length, (bool, np.bool_))
                or not isinstance(filter_length, (int, np.integer))
                or filter_length <= 0):
            raise ValueError("filter_length must be a positive integer")
        lam = finite_real_scalar(forgetting_factor, "forgetting_factor")
        delta = finite_real_scalar(initial_regularization, "initial_regularization")
        if not 0.0 < lam <= 1.0:
            raise ValueError("forgetting_factor must satisfy 0 < lambda <= 1")
        if delta <= 0.0:
            raise ValueError("initial_regularization must be positive")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inverse_delta = 1.0 / delta
        if not np.isfinite(inverse_delta):
            raise ValueError("initial_regularization has an unrepresentable reciprocal")

        length = int(filter_length)
        if initial_weights is None:
            weights = np.zeros(length, dtype=float)
        else:
            weights = finite_real_array(initial_weights, "initial_weights").copy()
            if weights.shape != (length,):
                raise ValueError("initial_weights must have shape (filter_length,)")
        if initial_history is None:
            history = np.zeros(length - 1, dtype=float)
        else:
            history = finite_real_array(initial_history, "initial_history").copy()
            if history.shape != (length - 1,):
                raise ValueError("initial_history must have shape (filter_length-1,)")

        self.filter_length = length
        self.forgetting_factor = lam
        self.initial_regularization = delta
        self._initial_weights = weights
        self._initial_history = history
        self._initial_inverse_covariance = np.eye(length, dtype=float) * inverse_delta
        self.reset()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def inverse_covariance(self) -> np.ndarray:
        """Return ``P = R**-1``; this is not the sample covariance itself."""

        return self._inverse_covariance.copy()

    @property
    def history(self) -> np.ndarray:
        """Return the last ``L-1`` reference samples, oldest first."""

        return self._history.copy()

    def reset(self) -> None:
        self._weights = self._initial_weights.copy()
        self._inverse_covariance = self._initial_inverse_covariance.copy()
        self._history = self._initial_history.copy()

    def process(
        self,
        reference: np.ndarray,
        microphone: np.ndarray,
        *,
        freeze: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(prior_residual, prior_echo_hat)`` and retain state.

        The update uses the error predicted with the *old* weights. For a
        frozen sample, both ``P`` and ``w`` stay fixed while FIR history moves.
        The entire call is transactional: invalid input or a non-finite
        intermediate raises ``ValueError`` without changing object state.
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

        history_length = self.filter_length - 1
        concatenated = np.concatenate((self._history, x))
        residual = np.empty_like(d)
        echo_hat = np.empty_like(d)
        weights = self._weights.copy()
        p = self._inverse_covariance.copy()

        for n in range(x.size):
            u = concatenated[n:n + self.filter_length][::-1]
            should_update = not frozen[n]
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise",
                                 under="ignore"):
                    prediction = float(weights @ u)
                    error = float(d[n] - prediction)
                    if should_update:
                        q = p @ u
                        denominator = float(self.forgetting_factor + u @ q)
                        if denominator <= 0.0:
                            raise ValueError("RLS innovation denominator is not positive")
                        gain = q / denominator
                        candidate_weights = weights + gain * error
                        candidate_p = (p - np.outer(q, q) / denominator) / self.forgetting_factor
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError("RLS intermediate exceeds the float64 range") from exc

            if not np.isfinite(prediction) or not np.isfinite(error):
                raise ValueError("RLS prediction exceeds the float64 range")
            echo_hat[n] = prediction
            residual[n] = error
            if should_update:
                if not (np.all(np.isfinite(candidate_weights))
                        and np.all(np.isfinite(candidate_p))):
                    raise ValueError("RLS update exceeds the float64 range")
                # Algebraically symmetric. Averaging only removes round-off
                # asymmetry; it is not a positive-definiteness repair.
                candidate_p = 0.5 * candidate_p + 0.5 * candidate_p.T
                # Positive diagonal entries do not imply that the whole
                # inverse correlation matrix is positive definite. This
                # costly check is deliberate in a short-filter teaching
                # implementation; it does not repair an unstable update.
                try:
                    np.linalg.cholesky(candidate_p)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(
                        "RLS inverse covariance lost positive definiteness"
                    ) from exc
                weights = candidate_weights
                p = candidate_p

        self._weights = weights
        self._inverse_covariance = p
        if history_length:
            self._history = concatenated[-history_length:].copy()
        return residual, echo_hat


def rls(
    reference: np.ndarray,
    microphone: np.ndarray,
    filter_length: int,
    *,
    forgetting_factor: float = 0.999,
    initial_regularization: float = 1.0,
    freeze: np.ndarray | None = None,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one real-valued RLS stream; return residual, echo estimate, taps."""

    state = RLSState(
        filter_length, forgetting_factor=forgetting_factor,
        initial_regularization=initial_regularization,
        initial_weights=initial_weights,
    )
    residual, echo_hat = state.process(reference, microphone, freeze=freeze)
    return residual, echo_hat, state.weights
