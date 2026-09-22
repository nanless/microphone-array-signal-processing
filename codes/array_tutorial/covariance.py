"""Spatial second-moment estimators for channel-frequency-frame spectra."""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar, hermitian_part, validate_cft


def spatial_covariance(
    spectra: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    demean: bool = False,
    relative_diagonal_loading: float = 0.0,
    denominator_floor: float = 1e-12,
) -> np.ndarray:
    """Estimate one channels x channels matrix per frequency.

    ``weights`` may have shape frames or frequency x frames.  A zero-weight
    frequency is rejected explicitly.  The result has shape frequency x
    channels x channels and is symmetrized after accumulation. Weights are
    divided by their per-frequency maximum before summing; denominator_floor
    applies to this dimensionless rescaled sum, not the original mask scale.
    """
    x = validate_cft(spectra)
    denominator_floor = finite_real_scalar(denominator_floor, "denominator_floor")
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")
    channels, frequencies, frames = x.shape
    if weights is None:
        weight = np.ones((frequencies, frames), dtype=float)
    else:
        weight = finite_real_array(weights, "weights")
        if weight.shape == (frames,):
            weight = np.broadcast_to(weight, (frequencies, frames))
        if weight.shape != (frequencies, frames):
            raise ValueError("weights must have shape frames or frequency x frames")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("weights must be finite and non-negative")
    maximum = np.max(weight, axis=1)
    if np.any(maximum == 0.0):
        raise ValueError("each frequency needs positive total weight")
    weight = weight / maximum[:, None]
    denominator = np.sum(weight, axis=1)
    if np.any(denominator <= denominator_floor):
        raise ValueError("each frequency needs positive total weight")
    centered = x
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if demean:
            mean = np.einsum("cft,ft->cf", x, weight / denominator[:, None])
            centered = x - mean[:, :, None]
        covariance = np.einsum(
            "cft,dft,ft->fcd", centered, centered.conj(), weight, optimize=True
        ) / denominator[:, None, None]
    if not np.all(np.isfinite(covariance)):
        raise ValueError("second-moment accumulation exceeds floating-point range")
    covariance = hermitian_part(covariance)
    relative_diagonal_loading = finite_real_scalar(relative_diagonal_loading, "relative_diagonal_loading")
    if not np.isfinite(relative_diagonal_loading) or relative_diagonal_loading < 0.0:
        raise ValueError("relative_diagonal_loading must be non-negative")
    if relative_diagonal_loading:
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.sum(np.diagonal(covariance, axis1=1, axis2=2).real / channels, axis=1)
            covariance += (
                relative_diagonal_loading
                * scale[:, None, None]
                * np.eye(channels, dtype=complex)[None, :, :]
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("diagonal loading exceeds floating-point range")
    return covariance


def recursive_covariance(
    previous: np.ndarray,
    snapshot: np.ndarray,
    *,
    forgetting_factor: float,
) -> np.ndarray:
    """Update frequency-wise second moments from one channels x frequency snapshot."""
    old = np.asarray(previous, dtype=complex)
    x = np.asarray(snapshot, dtype=complex)
    if old.ndim != 3 or old.shape[1] != old.shape[2]:
        raise ValueError("previous must have shape frequency x channels x channels")
    if x.shape != (old.shape[1], old.shape[0]):
        raise ValueError("snapshot must have shape channels x frequency")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(x)):
        raise ValueError("inputs contain NaN or infinity")
    forgetting_factor = finite_real_scalar(forgetting_factor, "forgetting_factor")
    if not np.isfinite(forgetting_factor) or not 0.0 <= forgetting_factor < 1.0:
        raise ValueError("forgetting_factor must be in [0, 1)")
    with np.errstate(over="ignore", invalid="ignore"):
        instant = np.einsum("cf,df->fcd", x, x.conj(), optimize=True)
        updated = forgetting_factor * old + (1.0 - forgetting_factor) * instant
    if not np.all(np.isfinite(updated)):
        raise ValueError("recursive second moment exceeds floating-point range")
    return hermitian_part(updated)
