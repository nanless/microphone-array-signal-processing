"""Causal real affine-projection arithmetic for chapter 6, not a complete AEC.

All window errors are recomputed with the SAME pre-update weights. Columns
are current-first regressors, ordered newest observation to oldest. The FIR
history is [x[n-1], x[n-2], ...]. Startup uses only actual observations; it
does not invent zero microphone observations to fill the projection window.

Regularization has reference-amplitude-squared units. At zero regularization
an update requires a full-column-rank window at float64 precision. Positive
regularization makes a deficient window computable, not identifiable. This
small Gram solve is not optimized for a long filter or a large order.
Finite input alone does not guarantee that every normalized intermediate is
representable. A change of scale that erases a nonzero matrix/error entry is
explicitly rejected, even if a more general solver could find a finite final
answer. This interface never declares such a lost entry mathematically zero.

Freeze skips adaptation but retains both render and microphone histories.
Consequently resumed adaptation can still consume contaminated observations.
reset_projection_history() explicitly removes those observations WITHOUT
erasing the FIR render history or weights. Every step, and an entire block,
is atomic on failure. No sample is truncated and no future sample is read.
"""

from __future__ import annotations

import math

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar
from codes.chapters.ch06.core.aec import _scaled_dot


def _positive_integer(value: object, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_divide(values: np.ndarray, divisor: object, name: str) -> np.ndarray:
    """Do not silently erase a nonzero input in a change of numerical scale."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result = values / divisor
    if not np.all(np.isfinite(result)) or np.any((values != 0.) & (result == 0.)):
        raise ValueError(f"{name} is not representable at float64 precision")
    return result


def _regularized_increment(U: np.ndarray, error: np.ndarray, delta: float) -> np.ndarray:
    """Solve the small projection system without first squaring a huge scale."""
    if delta > 0. and not np.any(U):
        # The exact increment is zero regardless of the microphone magnitude;
        # no artificial error rescaling is needed for a zero reference window.
        return np.zeros(U.shape[0])
    if delta == 0.:
        # Independent column scales matter: a tiny but independent column
        # must not be called rank-deficient merely because another is huge.
        scales = np.max(np.abs(U), axis=0)
        if np.any(scales == 0.) or U.shape[1] > U.shape[0]:
            raise ValueError("zero-regularization window is not full column rank")
        Z = _checked_divide(U, scales, "normalized projection columns")
        singular = np.linalg.svd(Z, compute_uv=False)
        tolerance = np.finfo(float).eps * max(Z.shape) * singular[0]
        if singular[-1] <= tolerance:
            raise ValueError("zero-regularization window is not full column rank at float64 precision")
        rhs = _checked_divide(error, scales, "normalized window errors")
        rhs_scale = float(np.max(np.abs(rhs)))
        if rhs_scale == 0.:
            return np.zeros(U.shape[0])
        # The column rescaling changes the constraints, not the Euclidean
        # norm of the unknown increment: this is still the minimum-norm solve.
        normalized_rhs = _checked_divide(rhs, rhs_scale, "normalized solve right-hand side")
        increment, _, rank, _ = np.linalg.lstsq(Z.T, normalized_rhs, rcond=None)
        if rank != U.shape[1]:
            raise ValueError("projection solve lost column rank")
        return increment*rhs_scale

    scale = max(float(np.max(np.abs(U))), math.sqrt(delta))
    Z = _checked_divide(U, scale, "normalized projection matrix")
    dm, de = math.frexp(delta)
    sm, se = math.frexp(scale)
    beta = math.ldexp(dm / (sm * sm), de - 2 * se)
    if beta <= 0. or not math.isfinite(beta):
        raise ValueError("scaled regularization is not positive and representable")
    rhs = _checked_divide(error, scale, "normalized window errors")
    rhs_scale = float(np.max(np.abs(rhs)))
    if rhs_scale == 0.:
        return np.zeros(U.shape[0])
    gram = np.array([[_scaled_dot(Z[:, i], Z[:, j])
                      for j in range(U.shape[1])] for i in range(U.shape[1])])
    system = gram + beta * np.eye(U.shape[1])
    # Reject an unresolvable regularizer rather than silently using a singular
    # unregularized inverse. Cholesky also checks this solve's positive floor.
    np.linalg.cholesky(system)
    # Normalize the RHS too: solving directly with subnormal entries can
    # erase each alpha although their combined increment is representable.
    normalized_rhs = _checked_divide(rhs, rhs_scale, "normalized solve right-hand side")
    alpha = np.linalg.solve(system, normalized_rhs)
    return np.array([_scaled_dot(row, alpha) for row in Z])*rhs_scale


class APAState:
    """Stateful real APA with prior causal outputs and explicit window control.

    ``projection_order`` is the maximum number of actual recent observations.
    ``step_size`` satisfies 0 <= mu < 2; this interface range is not a general
    double-talk/tracking stability guarantee. ``regularization >= 0`` permits
    the exact full-rank projection boundary; a positive floor is recommended.
    weights/history properties and all returned arrays are independent copies.
    """

    def __init__(self, filter_length: int, projection_order: int, *,
                 step_size: float = 0.5, regularization: float = 1e-3,
                 initial_weights: object | None = None,
                 initial_history: object | None = None) -> None:
        self.filter_length = _positive_integer(filter_length, "filter_length")
        self.projection_order = _positive_integer(projection_order, "projection_order")
        self.step_size = finite_real_scalar(step_size, "step_size")
        self.regularization = finite_real_scalar(regularization, "regularization")
        if not 0. <= self.step_size < 2.:
            raise ValueError("step_size must satisfy 0 <= mu < 2")
        if self.regularization < 0.:
            raise ValueError("regularization must be nonnegative")
        self._initial_weights = finite_real_array(
            np.zeros(self.filter_length) if initial_weights is None else initial_weights,
            "initial_weights").copy()
        self._initial_history = finite_real_array(
            np.zeros(self.filter_length - 1) if initial_history is None else initial_history,
            "initial_history").copy()
        if self._initial_weights.shape != (self.filter_length,):
            raise ValueError("initial_weights must have shape (filter_length,)")
        if self._initial_history.shape != (self.filter_length - 1,):
            raise ValueError("initial_history must have shape (filter_length-1,)")
        self.reset()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def history(self) -> np.ndarray:
        return self._history.copy()

    def reset_projection_history(self) -> None:
        """Clear only past projection observations; preserve taps/render history."""
        self._columns = np.empty((self.filter_length, 0), dtype=float)
        self._desired = np.empty(0, dtype=float)

    def reset(self) -> None:
        """Restore constructor taps/render history and an empty observation window."""
        self._weights = self._initial_weights.copy()
        self._history = self._initial_history.copy()
        self.reset_projection_history()

    def step(self, render_sample: float, microphone_sample: float, *,
             freeze: bool = False) -> dict:
        """Consume one actual sample; report the whole same-weight error window."""
        x = finite_real_scalar(render_sample, "render_sample")
        d = finite_real_scalar(microphone_sample, "microphone_sample")
        if not isinstance(freeze, (bool, np.bool_)):
            raise ValueError("freeze must be Boolean")
        u = np.concatenate(([x], self._history))
        U = np.column_stack((u, self._columns))[:, :self.projection_order]
        desired = np.concatenate(([d], self._desired))[:self.projection_order]
        old_weights = self._weights.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                estimates = np.array([_scaled_dot(U[:, j], old_weights)
                                      for j in range(U.shape[1])])
                errors = desired - estimates
                if not np.all(np.isfinite(errors)):
                    raise ValueError("window prior error exceeds float64 range")
                new_weights = old_weights.copy()
                if not freeze and self.step_size != 0.:
                    # At delta=0 a zero/deficient window is rejected, even if
                    # its error is zero: the specified inverse is undefined.
                    increment = _regularized_increment(U, errors, self.regularization)
                    adopted = self.step_size * increment
                    if np.any((increment != 0.) & (adopted == 0.)):
                        raise ValueError("APA adopted increment is not representable at float64 precision")
                    new_weights = old_weights + adopted
                if not np.all(np.isfinite(new_weights)):
                    raise ValueError("APA weights exceed float64 range")
                post_errors = desired - np.array([
                    _scaled_dot(U[:, j], new_weights) for j in range(U.shape[1])])
                if not np.all(np.isfinite(post_errors)):
                    raise ValueError("window posterior error exceeds float64 range")
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise ValueError("APA solve is outside its float64 numerical range") from error
        result = {
            "prior_echo": float(estimates[0]), "prior_error": float(errors[0]),
            "projection_matrix": U.copy(), "desired_window": desired.copy(),
            "window_prior_errors": errors.copy(),
            "posterior_weights": new_weights.copy(),
            "window_posterior_errors": post_errors.copy(),
            "effective_projection_order": U.shape[1], "frozen": bool(freeze),
        }
        # Commit only after validation of the complete current observation.
        self._weights = new_weights
        self._history = u[:-1].copy()
        self._columns = U.copy()
        self._desired = desired.copy()
        return result

    def process_block(self, render: object, microphone: object, *,
                      freeze_mask: object | None = None) -> dict:
        """Return prior_echo/prior_error/final_weights, atomically for ALL samples."""
        x = finite_real_array(render, "render")
        d = finite_real_array(microphone, "microphone")
        if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
            raise ValueError("render and microphone must be matching one-dimensional arrays")
        if freeze_mask is None:
            frozen = np.zeros(x.size, dtype=bool)
        else:
            frozen = np.asarray(freeze_mask)
            if frozen.dtype.kind != "b" or frozen.shape != x.shape:
                raise ValueError("freeze_mask must be a Boolean array matching render")
        # Work on independent state. If any later sample fails, the receiver
        # retains even its pre-block observation and render histories.
        working = object.__new__(type(self))
        working.filter_length = self.filter_length
        working.projection_order = self.projection_order
        working.step_size = self.step_size
        working.regularization = self.regularization
        working._weights = self._weights.copy()
        working._history = self._history.copy()
        working._columns = self._columns.copy()
        working._desired = self._desired.copy()
        echo = np.empty(x.size)
        error = np.empty(x.size)
        for n in range(x.size):
            result = working.step(x[n], d[n], freeze=frozen[n])
            echo[n] = result["prior_echo"]
            error[n] = result["prior_error"]
        self._weights = working._weights
        self._history = working._history
        self._columns = working._columns
        self._desired = working._desired
        return {"prior_echo": echo, "prior_error": error,
                "final_weights": self.weights}
