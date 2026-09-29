"""Shared covariance validation and relative diagonal loading for Chapters 4–5."""

from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_scalar, hermitian_part


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
