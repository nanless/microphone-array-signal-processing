"""Small separation metrics and mask-beamforming baselines."""

from __future__ import annotations

import itertools
import numpy as np


def si_sdr(estimate: np.ndarray, reference: np.ndarray, *, zero_mean: bool = True, epsilon: float = 1e-12) -> float:
    """Compute scale-invariant SDR for equal-length one-dimensional signals."""

    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if estimate.ndim != 1 or estimate.shape != reference.shape or estimate.size == 0:
        raise ValueError("estimate and reference must be equal-length 1-D arrays")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(reference)):
        raise ValueError("estimate and reference must be finite")
    if zero_mean:
        estimate = estimate - np.mean(estimate)
        reference = reference - np.mean(reference)
    reference_energy = float(reference @ reference)
    estimate_energy = float(estimate @ estimate)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if reference_energy <= epsilon:
        raise ValueError("SI-SDR is undefined for a silent reference")
    if estimate_energy <= epsilon:
        raise ValueError("SI-SDR is undefined for a silent estimate")
    target = (float(estimate @ reference) / reference_energy) * reference
    noise = estimate - target
    return float(10.0 * np.log10((float(target @ target) + epsilon) / (float(noise @ noise) + epsilon)))


def pit_permutation(estimates: np.ndarray, references: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Find the output-to-reference permutation maximizing summed SI-SDR."""

    estimates = np.asarray(estimates, dtype=float)
    references = np.asarray(references, dtype=float)
    if estimates.ndim != 2 or estimates.shape != references.shape:
        raise ValueError("estimates and references must have shape (source, sample)")
    count = estimates.shape[0]
    if count == 0:
        raise ValueError("PIT requires at least one source")
    scores = np.empty((count, count))
    for output in range(count):
        for reference in range(count):
            scores[output, reference] = si_sdr(estimates[output], references[reference])
    best_permutation: tuple[int, ...] | None = None
    best_score = -np.inf
    for permutation in itertools.permutations(range(count)):
        score = float(sum(scores[output, reference] for output, reference in enumerate(permutation)))
        if score > best_score:
            best_score = score
            best_permutation = permutation
    assert best_permutation is not None
    return best_permutation, best_score / count


def masked_spatial_covariance(spectrum: np.ndarray, mask: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Estimate ``(F,M,M)`` SCMs from ``spectrum=(F,M,T)`` and ``mask=(F,T)``."""

    x = np.asarray(spectrum)
    weights = np.asarray(mask, dtype=float)
    if x.ndim != 3 or weights.shape != (x.shape[0], x.shape[2]):
        raise ValueError("expected spectrum (F,M,T) and mask (F,T)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("spectrum must be finite and mask weights finite and non-negative")
    numerator = np.einsum("ft,fmt,fnt->fmn", weights, x, x.conj())
    denominator = np.sum(weights, axis=1)[:, None, None]
    return numerator / np.maximum(denominator, epsilon)


def mask_mvdr_2x2(
    spectrum: np.ndarray,
    target_mask: np.ndarray,
    interference_mask: np.ndarray,
    *,
    diagonal_loading: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a two-microphone mask-MVDR and return ``(output, weights)``.

    The target steering vector is the principal eigenvector of the target SCM,
    normalized to reference microphone 0.  Frequencies with unusable statistics
    fall back to selecting microphone 0.
    """

    x = np.asarray(spectrum)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("spectrum must have shape (F,2,T)")
    if not np.isfinite(diagonal_loading) or diagonal_loading < 0.0:
        raise ValueError("diagonal_loading must be finite and non-negative")
    target_weights = np.asarray(target_mask, dtype=float)
    interference_weights = np.asarray(interference_mask, dtype=float)
    target = masked_spatial_covariance(x, target_mask)
    interference = masked_spatial_covariance(x, interference_mask)
    output = np.empty((x.shape[0], x.shape[2]), dtype=np.complex128)
    beam_weights = np.empty((x.shape[0], 2), dtype=np.complex128)
    fallback = np.array([1.0 + 0j, 0j])
    target_mass = np.sum(target_weights, axis=1)
    interference_mass = np.sum(interference_weights, axis=1)

    for frequency in range(x.shape[0]):
        if target_mass[frequency] <= 1e-12 or interference_mass[frequency] <= 1e-12:
            beam_weights[frequency] = fallback
            output[frequency] = x[frequency, 0]
            continue
        values, vectors = np.linalg.eigh(target[frequency])
        if values[-1].real <= np.finfo(float).tiny:
            beam_weights[frequency] = fallback
            output[frequency] = x[frequency, 0]
            continue
        steering = vectors[:, int(np.argmax(values.real))]
        if abs(steering[0]) < 1e-10:
            weight = fallback
        else:
            steering = steering / steering[0]
            noise = interference[frequency]
            scale = float(np.trace(noise).real / 2.0)
            if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
                beam_weights[frequency] = fallback
                output[frequency] = x[frequency, 0]
                continue
            loaded = noise + diagonal_loading * scale * np.eye(2)
            try:
                inverse_steering = np.linalg.solve(loaded, steering)
                denominator = np.vdot(steering, inverse_steering)
                weight = inverse_steering / denominator if abs(denominator) > 1e-12 else fallback
            except np.linalg.LinAlgError:
                weight = fallback
        beam_weights[frequency] = weight
        output[frequency] = weight.conj() @ x[frequency]
    return output, beam_weights
