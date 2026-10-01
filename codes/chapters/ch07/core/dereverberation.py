"""Minimal offline WPE implementations for complex STFT arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_scalar


@dataclass(frozen=True)
class WPEFrequencyDiagnostic:
    """Numerical status for one frequency across all requested iterations.

    ``min_rank`` and ``max_condition_number`` refer to ``matrix_kind``:
    the loaded normal matrix or the augmented design matrix. Their condition
    numbers are not interchangeable. They do not describe speech quality.
    ``None`` means that no normal equation was solved at this frequency.
    """

    frequency: int
    min_rank: int | None
    max_condition_number: float | None
    least_squares_count: int
    bypass_reason: str | None
    solver: str = "normal"
    matrix_kind: str = "loaded_normal"


@dataclass(frozen=True)
class PredictionSolveDiagnostic:
    """Status of the actual matrix passed to the prediction solver.

    ``used_lstsq`` is true whenever least squares was actually called: always
    for ``design_lstsq``, or after a failed normal-equation direct solve.
    WPEFrequencyDiagnostic.least_squares_count instead counts only normal
    equation fallbacks, so that count stays zero for a chosen design solve.
    """

    solver: str
    matrix_kind: str
    rank: int
    condition_number: float
    absolute_loading: float
    used_lstsq: bool


def _check_product_range(*factors: np.ndarray, stage: str) -> None:
    """Conservative component-product support bound, not a mathematical limit.

    Even a nonzero term below the subnormal range may contribute to a sum
    that becomes representable after restoring the input scale. Reject such
    ranges instead of silently calling the term zero. This sufficient bound
    can reject products that never coincide or eventually cancel.
    """
    exponent_sum = 0.0
    for factor in factors:
        values = np.asarray(factor)
        components = np.concatenate((np.abs(values.real).ravel(),
                                     np.abs(values.imag).ravel()))
        nonzero = components[components != 0]
        if not nonzero.size:
            return
        exponent_sum += float(np.log2(np.min(nonzero)))
    if exponent_sum < -1074:
        raise ValueError(f"WPE {stage} is outside the supported float64 product range; "
                         "the final mathematical result may still be representable")


def _checked_component_divide(value: np.ndarray, scale: float, stage: str) -> np.ndarray:
    result = np.empty(np.shape(value), dtype=np.complex128)
    with np.errstate(under="ignore"):
        result.real = np.asarray(value).real / scale
        result.imag = np.asarray(value).imag / scale
    if (np.any((np.asarray(value).real != 0) & (result.real == 0)) or
            np.any((np.asarray(value).imag != 0) & (result.imag == 0))):
        raise ValueError(f"WPE {stage} loses a nonzero component outside the supported float64 range")
    return result


def _smoothed_power(power: np.ndarray, context: int) -> np.ndarray:
    if context == 0:
        return power
    # Each boundary mean divides by its actual number of available frames.
    return np.asarray([np.mean(power[max(0, t-context):min(len(power), t+context+1)])
                       for t in range(len(power))])


def _normal_solve(loaded: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    solve_matrix, solve_cross = loaded, cross
    matrix_scale = float(max(np.max(np.abs(loaded.real)), np.max(np.abs(loaded.imag))))
    if matrix_scale > np.sqrt(np.finfo(float).max):
        solve_matrix = _checked_component_divide(loaded, matrix_scale, "matrix scaling")
        solve_cross = _checked_component_divide(cross, matrix_scale, "right-hand-side scaling")
    used_lstsq = False
    try:
        predictor = np.linalg.solve(solve_matrix, solve_cross)
    except np.linalg.LinAlgError:
        used_lstsq = True
        predictor = np.linalg.lstsq(solve_matrix, solve_cross, rcond=None)[0]
    if not np.all(np.isfinite(predictor)):
        raise ValueError("WPE predictor is not finite")
    if np.any(cross != 0) and np.all(predictor == 0):
        raise ValueError("WPE solve loses a nonzero right-hand side outside the supported float64 range")
    return predictor, solve_matrix, used_lstsq


def solve_prediction_design(
    design: np.ndarray, target: np.ndarray, *, diagonal_loading: float = 0.0,
    solver: str = "design_lstsq", return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, PredictionSolveDiagnostic]:
    """Solve ||A G-B||_F² + delta||G||_F² for supplied regression snapshots.

    A is (frames, parameters); B is (frames,) or (frames, outputs).
    WPE rows are sqrt(weight)*q^H and sqrt(weight)*X^H, respectively.
    delta=diagonal_loading*||A||_F²/parameters. ``design_lstsq`` sends
    [A;sqrt(delta)I] directly to NumPy least squares (SVD), without forming
    its Gram matrix. ``normal`` solves A^H A+delta I and falls back to
    minimum-norm least squares of that matrix. Complex target shape is kept.
    Extreme component-product ranges are deliberately unsupported; a finite
    input alone does not promise that every intermediate or final fit works.
    """
    if not isinstance(solver, str) or solver not in ("normal", "design_lstsq"):
        raise ValueError("solver must be 'normal' or 'design_lstsq'")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("return_diagnostics must be a boolean")
    epsilon = finite_real_scalar(diagonal_loading, "diagonal_loading")
    if epsilon < 0:
        raise ValueError("diagonal_loading must be non-negative")
    a0, b0 = np.asarray(design), np.asarray(target)
    if a0.dtype.kind not in "fciu" or b0.dtype.kind not in "fciu":
        raise ValueError("design and target must be numeric arrays")
    if (a0.ndim != 2 or min(a0.shape) == 0 or b0.ndim not in (1, 2)
            or b0.shape[0] != a0.shape[0] or b0.size == 0
            or not np.all(np.isfinite(a0)) or not np.all(np.isfinite(b0))):
        raise ValueError("require finite non-empty design (T,D) and target (T,) or (T,M)")
    a, b = np.asarray(a0, dtype=np.complex128), np.asarray(b0, dtype=np.complex128)
    _check_product_range(a, a, stage="design energy")
    with np.errstate(over="raise", invalid="raise"):
        try:
            energy = float(np.sum(np.abs(a)**2))
            loading = epsilon * (energy / a.shape[1])
            if not np.isfinite(loading):
                raise ValueError("WPE diagonal loading exceeds the float64 range")
            if epsilon != 0 and energy != 0 and loading == 0:
                raise ValueError("WPE diagonal loading is outside the supported float64 range")
            if solver == "normal":
                _check_product_range(a, b, stage="design cross products")
                matrix = a.conj().T @ a + loading*np.eye(a.shape[1])
                cross = a.conj().T @ b
                result, matrix, fallback = _normal_solve(matrix, cross)
                kind = "loaded_normal"
            else:
                matrix = np.vstack((a, np.sqrt(loading)*np.eye(a.shape[1]))) if loading else a
                rhs = np.concatenate((b, np.zeros((a.shape[1],)+b.shape[1:], complex))) if loading else b
                result = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
                fallback, kind = True, "augmented_design"
        except FloatingPointError as error:
            raise ValueError("WPE design solve exceeds the float64 range") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("WPE design solution is not finite")
    if not return_diagnostics:
        return result
    status = PredictionSolveDiagnostic(solver, kind, int(np.linalg.matrix_rank(matrix)),
                                       float(np.linalg.cond(matrix)), loading, fallback)
    return result, status


def offline_wpe(
    spectrum: np.ndarray,
    *,
    taps: int,
    delay: int,
    iterations: int = 3,
    diagonal_loading: float = 1e-6,
    power_floor: float = 1e-5,
    power_context: int = 0,
    solver: str = "normal",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[WPEFrequencyDiagnostic, ...]]:
    """Apply a compact single- or multi-channel offline WPE baseline.

    Input is ``(frequency, frame)`` or ``(frequency, channel, frame)``.
    Output has the same shape.  ``taps=0`` and records shorter than the first
    valid regression frame are passed through.  This batch routine uses the
    entire recording and is therefore not causal. By default the return value
    remains the output array. With ``return_diagnostics=True`` it is
    ``(output, diagnostics)`` with one status record per frequency. A singular
    loaded normal matrix falls back to a minimum-norm least-squares solution;
    that numerical fit does not establish that the removed signal was reverb.
    ``power_context`` is a centered mean radius: 2 includes up to two future
    frames; 0 preserves the unsmoothed default. ``solver='design_lstsq'``
    bypasses Gram formation and diagnoses the augmented design matrix.
    Finite inputs with unsupported component-product ranges raise ValueError
    rather than silently dropping nonzero terms. This is a conservative
    float64 implementation boundary, not a claim about exact mathematics.
    """

    original = np.asarray(spectrum)
    if original.ndim not in (2, 3) or not np.iscomplexobj(original):
        raise ValueError("spectrum must be a complex array shaped (F,T) or (F,M,T)")
    if any(size == 0 for size in original.shape) or not np.all(np.isfinite(original)):
        raise ValueError("spectrum dimensions must be non-empty and values finite")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (taps, delay, iterations, power_context)):
        raise ValueError("taps, delay, iterations and power_context must be integers")
    if taps < 0 or delay < 1 or iterations < 0 or power_context < 0:
        raise ValueError("require taps >= 0, delay >= 1, iterations >= 0, power_context >= 0")
    if not isinstance(solver, str) or solver not in ("normal", "design_lstsq"):
        raise ValueError("solver must be 'normal' or 'design_lstsq'")
    diagonal_loading = finite_real_scalar(diagonal_loading, "diagonal_loading")
    power_floor = finite_real_scalar(power_floor, "power_floor")
    if diagonal_loading < 0 or power_floor <= 0:
        raise ValueError("loading must be non-negative and power_floor positive")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("return_diagnostics must be a boolean")
    if taps == 0 or iterations == 0:
        matrix_kind = "loaded_normal" if solver == "normal" else "augmented_design"
        result = original.copy()
        if not return_diagnostics:
            return result
        reason = "zero_taps" if taps == 0 else "zero_iterations"
        return result, tuple(WPEFrequencyDiagnostic(f, None, None, 0, reason, solver, matrix_kind)
                             for f in range(original.shape[0]))

    squeeze = original.ndim == 2
    y = original[:, None, :] if squeeze else original
    y = np.asarray(y, dtype=np.complex128)
    frequencies, channels, frames = y.shape
    first = delay + taps - 1
    matrix_kind = "loaded_normal" if solver == "normal" else "augmented_design"
    if frames <= first:
        result = original.copy()
        if not return_diagnostics:
            return result
        return result, tuple(WPEFrequencyDiagnostic(f, None, None, 0, "short_record", solver, matrix_kind)
                             for f in range(frequencies))

    output = y.copy()
    diagnostics: list[WPEFrequencyDiagnostic] = []
    dimension = channels * taps
    for frequency in range(frequencies):
        raw = y[frequency]
        min_rank: int | None = None
        max_condition_number: float | None = None
        least_squares_count = 0
        bypass_reason: str | None = None
        # A common amplitude scale leaves the WPE solution unchanged. Divide
        # components separately: complex division can overflow its reciprocal
        # even when every desired ratio is bounded (subnormal input levels).
        input_scale = float(max(np.max(np.abs(raw.real)), np.max(np.abs(raw.imag))))
        if input_scale == 0.0:
            if return_diagnostics:
                diagnostics.append(WPEFrequencyDiagnostic(frequency, None, None, 0, "zero_input", solver, matrix_kind))
            continue
        observed = _checked_component_divide(raw, input_scale, "input normalization")
        _check_product_range(observed, observed, stage="power products")
        current = observed.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                initial_scale = float(np.max(np.mean(np.abs(observed) ** 2, axis=0)))
                floor = power_floor * initial_scale
                if not np.isfinite(floor) or floor <= 0.0:
                    raise ValueError("relative WPE power floor is not representable")
                power = np.maximum(_smoothed_power(np.mean(np.abs(current) ** 2, axis=0), power_context), floor)
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
            if np.any(weights == 0):
                raise ValueError("WPE positive weights are outside the supported float64 range")
            _check_product_range(weights, history, history, stage="history statistics")
            _check_product_range(weights, history, targets, stage="cross statistics")
            if solver == "design_lstsq":
                weighted_history = np.sqrt(weights)[:, None]*history.conj()
                weighted_targets = np.sqrt(weights)[:, None]*targets.conj()
                if np.all(history == 0):
                    current = observed.copy()
                    bypass_reason = "zero_history"
                    break
                predictor, status = solve_prediction_design(
                    weighted_history, weighted_targets, diagonal_loading=diagonal_loading,
                    solver="design_lstsq", return_diagnostics=True)
                if return_diagnostics:
                    min_rank = status.rank if min_rank is None else min(min_rank, status.rank)
                    max_condition_number = (status.condition_number if max_condition_number is None
                                            else max(max_condition_number, status.condition_number))
                _check_product_range(history, predictor, stage="prediction products")
                prediction = history @ predictor.conj()
                current = observed.copy()
                current[:, first:] = (targets-prediction).T
                if not np.all(np.isfinite(current)):
                    raise ValueError("WPE residual is not finite")
                _check_product_range(current, current, stage="residual power products")
                power = np.maximum(_smoothed_power(np.mean(np.abs(current)**2, axis=0), power_context), floor)
                continue
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
                bypass_reason = "zero_history"
                break
            if trace <= np.finfo(float).tiny:
                raise ValueError("WPE history energy is too small to solve reliably")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    # Average first: loading*trace can overflow even when
                    # loading*(trace/dimension) and the final matrix fit.
                    loading = diagonal_loading * (trace / dimension)
                    if diagonal_loading != 0 and loading == 0:
                        raise ValueError("WPE diagonal loading is outside the supported float64 range")
                    loaded = correlation + loading * np.eye(dimension)
                    if not np.all(np.isfinite(loaded)):
                        raise ValueError("WPE diagonal loading exceeds the float64 range")
            except FloatingPointError as error:
                raise ValueError("WPE diagonal loading exceeds the float64 range") from error
            predictor, solve_matrix, fallback = _normal_solve(loaded, cross)
            least_squares_count += int(fallback)
            if return_diagnostics:
                rank = int(np.linalg.matrix_rank(solve_matrix))
                condition = float(np.linalg.cond(solve_matrix))
                min_rank = rank if min_rank is None else min(min_rank, rank)
                max_condition_number = (condition if max_condition_number is None
                                        else max(max_condition_number, condition))
            _check_product_range(history, predictor, stage="prediction products")
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
                    _check_product_range(current, current, stage="residual power products")
                    power = np.maximum(_smoothed_power(np.mean(np.abs(current) ** 2, axis=0), power_context), floor)
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
        if (np.any((current.real != 0) & (output[frequency].real == 0)) or
                np.any((current.imag != 0) & (output[frequency].imag == 0))):
            raise ValueError("WPE output restoration loses a nonzero component outside the supported float64 range")
        # No regression is defined on startup frames. Preserve the raw values
        # exactly, rather than re-rounding them through normalization.
        output[frequency, :, :first] = raw[:, :first]
        if bypass_reason == "zero_history":
            output[frequency] = raw
        if return_diagnostics:
            diagnostics.append(WPEFrequencyDiagnostic(frequency, min_rank,
                                                       max_condition_number,
                                                       least_squares_count,
                                                       bypass_reason, solver,
                                                       "loaded_normal" if solver == "normal" else "augmented_design"))

    result = output[:, 0, :] if squeeze else output
    return (result, tuple(diagnostics)) if return_diagnostics else result
