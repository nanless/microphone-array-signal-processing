"""Spatial second-moment estimators for channel-frequency-frame spectra."""

from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar, hermitian_part, validate_cft


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
    Zero-weight frames are excluded before products. Ordinary representable
    inputs retain direct weighted accumulation. If that accumulation overflows
    or a weight ratio underflows, square-root normalized weights are applied
    before outer products. A positive
    effective diagonal power that underflows to zero is rejected explicitly.
    Demeaning a constant effective input legitimately returns zero.
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
    active = weight > 0
    scaled_weight = weight / maximum[:, None]
    denominator = np.sum(scaled_weight, axis=1)
    if np.any(denominator <= denominator_floor):
        raise ValueError("each frequency needs positive total weight")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        used = np.where(active[None, :, :], x, 0)
        positive_input = np.any((x != 0) & active[None, :, :], axis=2)
        centered = used
        if demean:
            first = np.argmax(active, axis=1)
            anchor = x[:, np.arange(frequencies), first]
            varying = np.any((x != anchor[:, :, None]) & active[None, :, :], axis=2)
            # Exact constants remain zero, including extreme finite constants;
            # weighted summation roundoff must not invent a tiny variance.
            mean = np.einsum("cft,ft->cf", used, scaled_weight / denominator[:, None])
            mean = np.where(varying, mean, anchor)
            centered = np.where(active[None, :, :] & varying[:, :, None], used-mean[:, :, None], 0)
            positive_input = varying
        covariance = np.einsum(
            "cft,dft,ft->fcd", centered, centered.conj(), scaled_weight, optimize=True
        ) / denominator[:, None, None]
        diagonal = np.diagonal(covariance, axis1=1, axis2=2).real
        unsafe = (np.any(active & (scaled_weight == 0), axis=1)
                  | ~np.all(np.isfinite(covariance), axis=(1, 2))
                  | np.any((diagonal == 0) & positive_input.T, axis=1))
        if np.any(unsafe):
            # A separate scale route protects representable weighted powers
            # without changing ordinary sums or forcing rounded results.
            root_weight = np.sqrt(weight) / np.sqrt(maximum[:, None])
            factor = root_weight / np.sqrt(np.sum(root_weight * root_weight, axis=1)[:, None])
            normalized = used * factor[None, :, :]
            if demean:
                mean = np.einsum("cft,ft->cf", normalized, factor)
                mean = np.where(varying, mean, anchor)
                normalized -= mean[:, :, None] * factor[None, :, :]
                normalized = np.where(varying[:, :, None], normalized, 0)
            safe_covariance = np.einsum("cft,dft->fcd", normalized, normalized.conj(), optimize=True)
            covariance = np.where(unsafe[:, None, None], safe_covariance, covariance)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("second-moment accumulation exceeds floating-point range")
    diagonal = np.diagonal(covariance, axis1=1, axis2=2).real
    if np.any((diagonal == 0) & positive_input.T):
        raise ValueError("positive effective diagonal power underflows supported numerical range")
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
    """Update frequency-wise second moments from one channels x frequency snapshot.

    The caller must supply a Hermitian positive-semidefinite previous state,
    normally zero or an earlier valid estimate.  Shape/finite checks and final
    symmetrization do not validate or repair a negative old eigenvalue.  With a
    valid state and 0 <= forgetting_factor < 1, both terms are PSD and so is
    their convex combination, apart from floating-point roundoff.
    Ordinary representable inputs retain the direct convex update. If its
    intermediate power overflows, apply square-root update weights before
    squaring. A zero forgetting factor
    discards the old values, and a nonzero positive diagonal that underflows
    is rejected rather than interpreted as a physically silent channel.
    """
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
        updated = instant if forgetting_factor == 0 else forgetting_factor * old + (1.0 - forgetting_factor) * instant
        unsafe = ~np.all(np.isfinite(updated), axis=(1, 2))
        if np.any(unsafe):
            weighted = np.sqrt(1.0 - forgetting_factor) * x
            instant = np.einsum("cf,df->fcd", weighted, weighted.conj(), optimize=True)
            safe_updated = instant if forgetting_factor == 0 else forgetting_factor * old + instant
            updated = np.where(unsafe[:, None, None], safe_updated, updated)
    if not np.all(np.isfinite(updated)):
        raise ValueError("recursive second moment exceeds floating-point range")
    positive_input = (x.T != 0)
    if forgetting_factor:
        positive_input |= np.diagonal(old, axis1=1, axis2=2).real > 0
    if np.any((np.diagonal(updated, axis1=1, axis2=2).real == 0) & positive_input):
        raise ValueError("positive recursive diagonal power underflows supported numerical range")
    return hermitian_part(updated)
