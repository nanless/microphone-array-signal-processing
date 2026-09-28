"""Minimal fixed and adaptive-beamforming building blocks for Chapter 5."""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_scalar, finite_real_array, hermitian_part, validate_cft, validate_frequencies, validate_positions


def dsb_weights(steering: np.ndarray) -> np.ndarray:
    """Return distortionless delay-and-sum weights ``a / (a.H @ a)``."""
    a = np.asarray(steering, dtype=complex)
    if a.ndim not in (1, 2) or min(a.shape) < 1 or not np.all(np.isfinite(a)):
        raise ValueError("steering must be channels or frequency x channels")
    scale = np.max(np.maximum(np.abs(a.real), np.abs(a.imag)), axis=-1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError("steering vector must be non-zero")
    # Divide real and imaginary parts separately: complex division can form
    # an overflowing reciprocal even when both components are representable.
    normalized = a.real / scale + 1j * (a.imag / scale)
    denominator = np.sum(np.abs(normalized) ** 2, axis=-1, keepdims=True)
    reduced = normalized / denominator
    with np.errstate(over="ignore", invalid="ignore"):
        result = reduced.real / scale + 1j * (reduced.imag / scale)
    if not np.all(np.isfinite(result)):
        raise ValueError("DSB weights exceed floating-point range")
    return result


def apply_beamformer(spectra: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply frequency-wise weights to channels x frequency x frames spectra."""
    x = validate_cft(spectra)
    w = np.asarray(weights, dtype=complex)
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if w.ndim == 1:
        if w.shape != (x.shape[0],):
            raise ValueError("one-dimensional weights must match channel count")
        contraction = "c,cft->ft"
    else:
        if w.shape != (x.shape[1], x.shape[0]):
            raise ValueError("weights must have shape frequency x channels")
        contraction = "fc,cft->ft"
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.einsum(contraction, w.conj(), x)
    if not np.all(np.isfinite(output)):
        raise ValueError("beamformer output exceeds floating-point range")
    return output


def diffuse_coherence(
    positions: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Return the 3-D isotropic diffuse-field sinc coherence matrices."""
    microphones = validate_positions(positions)
    frequencies = validate_frequencies(frequencies_hz)
    if not np.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be positive")
    separations = np.linalg.norm(
        microphones[:, None, :] - microphones[None, :, :], axis=-1
    )
    return np.sinc(2.0 * frequencies[:, None, None] * separations[None, :, :] / sound_speed)


def mvdr_weights(
    covariance: np.ndarray,
    steering: np.ndarray,
    *,
    relative_diagonal_loading: float = 0.0,
    condition_limit: float = 1e12,
) -> np.ndarray:
    """Return MVDR weights for one matrix or a frequency-indexed matrix stack."""
    matrices = np.asarray(covariance, dtype=complex)
    vectors = np.asarray(steering, dtype=complex)
    single = matrices.ndim == 2
    if single:
        matrices = matrices[None, :, :]
        vectors = vectors[None, :] if vectors.ndim == 1 else vectors
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("covariance must be channels x channels or frequency x channels x channels")
    if vectors.shape != (matrices.shape[0], matrices.shape[1]):
        raise ValueError("steering must match the covariance frequency and channel axes")
    if min(matrices.shape) < 1 or not np.all(np.isfinite(vectors)):
        raise ValueError("covariance must be non-empty and steering must be finite")
    result = np.empty_like(vectors)
    for index, (matrix, vector) in enumerate(zip(matrices, vectors)):
        loaded = _load_covariance(matrix, relative_diagonal_loading, condition_limit)
        with np.errstate(over="ignore", invalid="ignore"):
            solved = np.linalg.solve(loaded, vector)
            denominator = np.vdot(vector, solved)
        if not np.all(np.isfinite(solved)) or not np.isfinite(denominator):
            raise ValueError("MVDR solve or normalization exceeds floating-point range")
        # A covariance rescaling also rescales this quadratic form. Judge its
        # round-off imaginary component relatively, not in absolute units.
        if (denominator.real <= 0.0
                or abs(denominator.imag) > 1e-8 * abs(denominator.real)):
            raise np.linalg.LinAlgError("MVDR normalization is not positive real")
        with np.errstate(over="ignore", invalid="ignore"):
            result[index] = solved / denominator.real
        if not np.all(np.isfinite(result[index])):
            raise ValueError("MVDR weights exceed floating-point range")
    return result[0] if single else result


def superdirective_weights(
    diffuse_field_coherence: np.ndarray,
    steering: np.ndarray,
    *,
    relative_diagonal_loading: float,
    condition_limit: float = 1e12,
) -> np.ndarray:
    """Return loaded superdirective weights from diffuse-field coherence."""
    relative_diagonal_loading = finite_real_scalar(relative_diagonal_loading, "relative_diagonal_loading")
    if relative_diagonal_loading <= 0.0:
        raise ValueError("superdirective examples require an explicit positive loading")
    return mvdr_weights(
        diffuse_field_coherence,
        steering,
        relative_diagonal_loading=relative_diagonal_loading,
        condition_limit=condition_limit,
    )


def lcmv_weights(
    covariance: np.ndarray,
    constraints: np.ndarray,
    responses: np.ndarray,
    *,
    relative_diagonal_loading: float = 0.0,
    condition_limit: float = 1e12,
) -> np.ndarray:
    """Return the minimum-power vector satisfying ``C.H @ w = f``.

    The output responses ``w.H @ C`` are therefore ``conj(f)``.  Pass the
    conjugate of the desired responses when they are not real-valued.
    """
    matrix = _load_covariance(covariance, relative_diagonal_loading, condition_limit)
    c = np.asarray(constraints, dtype=complex)
    f = np.asarray(responses, dtype=complex)
    if c.ndim != 2 or c.shape[0] != matrix.shape[0]:
        raise ValueError("constraints must have shape channels x constraints")
    if f.shape != (c.shape[1],):
        raise ValueError("responses must have one value per constraint")
    if c.shape[1] < 1 or not np.all(np.isfinite(c)) or not np.all(np.isfinite(f)):
        raise ValueError("constraints and responses must be finite and non-empty")
    if c.shape[1] > c.shape[0] or np.any(np.all(c == 0, axis=0)):
        raise np.linalg.LinAlgError("constraint columns must be linearly independent")
    # A positive column rescaling changes neither C.H w=f nor the solution,
    # provided f is rescaled too. This avoids squaring tiny/huge columns.
    scales = np.max(np.maximum(np.abs(c.real), np.abs(c.imag)), axis=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c = c.real / scales + 1j * (c.imag / scales)
        f = f.real / scales + 1j * (f.imag / scales)
    if not np.all(np.isfinite(f)):
        raise ValueError("LCMV normalized responses exceed floating-point range")
    if np.linalg.matrix_rank(c) != c.shape[1]:
        raise np.linalg.LinAlgError("constraint columns must be linearly independent")
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = np.linalg.solve(matrix, c)
        gram = c.conj().T @ whitened
    if not np.all(np.isfinite(whitened)) or not np.all(np.isfinite(gram)):
        raise ValueError("LCMV solve or Gram matrix exceeds floating-point range")
    condition = np.linalg.cond(gram)
    if not np.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError("constraint Gram matrix is ill-conditioned")
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = np.linalg.solve(gram, f)
        result = whitened @ coefficients
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(result)):
        raise ValueError("LCMV weights exceed floating-point range")
    # Verify C.H w=f without squaring huge norms or overflowing complex abs.
    result_peak = float(np.max(np.maximum(np.abs(result.real), np.abs(result.imag))))
    if result_peak == 0.0:
        if np.any(f != 0):
            raise np.linalg.LinAlgError("LCMV final constraints are not satisfied")
    else:
        reduced = result.real / result_peak + 1j * (result.imag / result_peak)
        expected = f.real / result_peak + 1j * (f.imag / result_peak)
        actual = c.conj().T @ reduced
        error = float(np.max(np.abs(actual-expected)))
        response_scale = float(np.max(np.abs(expected)))
        if not np.isfinite(error) or error > 1e-8 * response_scale:
            raise np.linalg.LinAlgError("LCMV final constraint residual exceeds relative tolerance 1e-8")
    return result


def blocking_matrix(constraints: np.ndarray, *, rtol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis ``B`` for the null space of ``C.H``."""
    if not np.isfinite(rtol) or not 0.0 <= rtol < 1.0:
        raise ValueError("rtol must be finite and in [0, 1)")
    c = np.asarray(constraints, dtype=complex)
    if c.ndim == 1:
        c = c[:, None]
    if c.ndim != 2 or c.shape[1] < 1 or not np.all(np.isfinite(c)):
        raise ValueError("constraints must be a finite channels x constraints matrix")
    left, singular_values, _ = np.linalg.svd(c, full_matrices=True)
    threshold = rtol * singular_values[0] if singular_values.size else 0.0
    rank = int(np.sum(singular_values > threshold))
    if rank == 0:
        raise np.linalg.LinAlgError("at least one non-zero constraint is required")
    return left[:, rank:]


def wiener_gain(
    output_power: np.ndarray,
    noise_power: np.ndarray,
    *,
    gain_floor: float = 0.0,
    power_floor: float = 1e-12,
) -> np.ndarray:
    """Return ``max(1 - noise/output, gain_floor)`` element by element."""
    output, noise = np.broadcast_arrays(
        finite_real_array(output_power, "output_power"), finite_real_array(noise_power, "noise_power")
    )
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(noise)):
        raise ValueError("power arrays must be finite")
    if np.any(output < 0.0) or np.any(noise < 0.0):
        raise ValueError("power arrays must be non-negative")
    if not 0.0 <= gain_floor <= 1.0:
        raise ValueError("gain_floor must be in [0, 1]")
    if not np.isfinite(power_floor) or power_floor <= 0.0:
        raise ValueError("power_floor must be finite and positive")
    return np.maximum(1.0 - noise / np.maximum(output, power_floor), gain_floor)


def _load_covariance(
    covariance: np.ndarray,
    relative_diagonal_loading: float,
    condition_limit: float,
) -> np.ndarray:
    """Validate a covariance, then require a positive-definite loaded solve.

    Relative Hermitian and original-PSD tolerances are 1e-10. Small negative
    round-off is NOT silently clipped: the matrix actually solved must still
    be strictly positive definite. A documented positive load may make it so.
    """
    relative_diagonal_loading = finite_real_scalar(relative_diagonal_loading, "relative_diagonal_loading")
    condition_limit = finite_real_scalar(condition_limit, "condition_limit")
    raw = np.asarray(covariance, dtype=complex)
    if (raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[0] != raw.shape[1]
            or not np.all(np.isfinite(raw))):
        raise ValueError("covariance must be one finite square matrix")
    if not np.isfinite(condition_limit) or condition_limit < 1.0:
        raise ValueError("condition_limit must be finite and at least one")
    if not np.isfinite(relative_diagonal_loading) or relative_diagonal_loading < 0.0:
        raise ValueError("relative_diagonal_loading must be non-negative")
    peak = float(np.max(np.maximum(np.abs(raw.real), np.abs(raw.imag))))
    if peak == 0.0:
        raise np.linalg.LinAlgError("zero covariance has no invertible noise model")
    normalized = raw.real / peak + 1j * (raw.imag / peak)
    if np.max(np.abs(normalized - normalized.conj().T)) > 1e-10:
        raise ValueError("covariance must be Hermitian within relative tolerance 1e-10")
    normalized = hermitian_part(normalized)
    original_values = np.linalg.eigvalsh(normalized)
    spectral_scale = float(np.max(np.abs(original_values)))
    if spectral_scale == 0.0 or original_values[0] < -1e-10 * spectral_scale:
        raise np.linalg.LinAlgError("original covariance must be positive semidefinite")
    matrix = hermitian_part(raw)
    # Divide before summing, so a valid near-limit diagonal does not overflow.
    scale = float(np.sum(matrix.diagonal().real / matrix.shape[0]))
    if relative_diagonal_loading:
        if scale <= 0.0:
            raise np.linalg.LinAlgError("non-positive covariance scale cannot be loaded relatively")
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = matrix + relative_diagonal_loading * scale * np.eye(matrix.shape[0])
        if not np.all(np.isfinite(matrix)):
            raise ValueError("loaded covariance exceeds floating-point range")
    loaded_peak = float(np.max(np.maximum(np.abs(matrix.real), np.abs(matrix.imag))))
    reduced = matrix.real / loaded_peak + 1j * (matrix.imag / loaded_peak)
    values = np.linalg.eigvalsh(reduced)
    if values[0] <= 0.0:
        raise np.linalg.LinAlgError("loaded covariance must be strictly positive definite")
    np.linalg.cholesky(reduced)
    condition = np.linalg.cond(reduced)
    if not np.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError(
            "covariance is singular or ill-conditioned; use documented diagonal loading"
        )
    return matrix
