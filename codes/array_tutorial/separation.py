"""Small separation metrics and mask-beamforming baselines."""

from __future__ import annotations

import itertools
import numpy as np

from .conventions import finite_real_array, finite_real_scalar


MAX_EXHAUSTIVE_PIT_SOURCES = 8


def guided_activity_posterior(
    mixture_weights: np.ndarray,
    spatial_likelihoods: np.ndarray,
    speaker_activity: np.ndarray,
) -> np.ndarray:
    """One guided-mixture E step with an always-active background class.

    ``mixture_weights`` has shape ``(speakers + 1,)`` and ends with the
    background weight. ``spatial_likelihoods`` has shape ``(frames, speakers
    + 1)``; ``speaker_activity`` has shape ``(frames, speakers)`` with binary
    values. The result is a posterior per frame and class. This checks only
    activity gating and normalization, not cACGMM density estimation or EM.
    """

    weights = finite_real_array(mixture_weights, "mixture_weights")
    likelihoods = finite_real_array(spatial_likelihoods, "spatial_likelihoods")
    activity_input = np.asarray(speaker_activity)
    activity = (activity_input.astype(float) if activity_input.dtype.kind == "b"
                else finite_real_array(activity_input, "speaker_activity"))
    if weights.ndim != 1 or weights.size < 2:
        raise ValueError("mixture_weights must contain speakers and one background class")
    if likelihoods.ndim != 2 or likelihoods.shape[1] != weights.size or likelihoods.shape[0] == 0:
        raise ValueError("spatial_likelihoods must have shape (frames, classes)")
    if activity.shape != (likelihoods.shape[0], weights.size - 1):
        raise ValueError("speaker_activity must have shape (frames, speakers)")
    if (np.any(weights < 0) or weights[-1] <= 0 or
            np.any(likelihoods <= 0) or np.any((activity != 0) & (activity != 1))):
        raise ValueError("weights must be non-negative with positive background; likelihoods positive; activity binary")

    allowed = np.concatenate((activity.astype(bool), np.ones((activity.shape[0], 1), dtype=bool)), axis=1)
    # Log scores avoid overflowing when a valid finite prior and likelihood
    # are multiplied. A zero speaker prior remains an inactive component.
    log_scores = np.full(likelihoods.shape, -np.inf)
    positive = weights > 0
    log_scores[:, positive] = np.log(weights[positive])[None, :] + np.log(likelihoods[:, positive])
    log_scores[~allowed] = -np.inf
    maximum = np.max(log_scores, axis=1, keepdims=True)
    unnormalized = np.exp(log_scores - maximum)
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def si_sdr(estimate: np.ndarray, reference: np.ndarray, *, zero_mean: bool = True, epsilon: float = 1e-12) -> float:
    """Compute SI-SDR with a relative energy floor and no time alignment.

    Peak scaling before centering avoids absolute-level dependence. ``epsilon``
    sets finite caps of approximately +/- ``-10*log10(epsilon)`` dB. Zero signals
    and signals constant after centering are rejected; perfect estimates return
    the positive cap, not infinity. See Le Roux et al. (ICASSP 2019), eq. (3--5).
    """

    estimate = finite_real_array(estimate, "estimate")
    reference = finite_real_array(reference, "reference")
    if estimate.ndim != 1 or estimate.shape != reference.shape or estimate.size == 0:
        raise ValueError("estimate and reference must be equal-length 1-D arrays")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(reference)):
        raise ValueError("estimate and reference must be finite")
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be finite and between zero and one")
    estimate_peak = float(np.max(np.abs(estimate)))
    reference_peak = float(np.max(np.abs(reference)))
    if estimate_peak == 0.0 or reference_peak == 0.0:
        raise ValueError("SI-SDR is undefined for a silent signal")
    estimate = estimate / estimate_peak
    reference = reference / reference_peak
    if zero_mean:
        estimate = estimate - np.mean(estimate)
        reference = reference - np.mean(reference)
    reference_energy = float(reference @ reference)
    estimate_energy = float(estimate @ estimate)
    if reference_energy == 0.0:
        raise ValueError("SI-SDR is undefined for a silent reference")
    if estimate_energy == 0.0:
        raise ValueError("SI-SDR is undefined for a silent estimate")
    target = (float(estimate @ reference) / reference_energy) * reference
    noise = estimate - target
    # Normalize energies before flooring: epsilon * energy can underflow for
    # valid subnormal epsilon, and dividing by that floor can overflow even
    # when the final logarithmic score is representable.
    target_fraction = max(float(target @ target) / estimate_energy, epsilon)
    noise_fraction = max(float(noise @ noise) / estimate_energy, epsilon)
    return float(10.0 * (np.log10(target_fraction) - np.log10(noise_fraction)))


def pit_permutation(estimates: np.ndarray, references: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Find the best SI-SDR permutation by enumeration for at most eight sources."""

    estimates = finite_real_array(estimates, "estimates")
    references = finite_real_array(references, "references")
    if estimates.ndim != 2 or estimates.shape != references.shape or estimates.shape[1] == 0:
        raise ValueError("estimates and references must have shape (source, sample)")
    count = estimates.shape[0]
    if count == 0:
        raise ValueError("PIT requires at least one source")
    if count > MAX_EXHAUSTIVE_PIT_SOURCES:
        raise ValueError(
            f"teaching PIT exhaustively enumerates at most {MAX_EXHAUSTIVE_PIT_SOURCES} sources"
        )
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
    """Estimate ``(F,M,M)`` SCMs from ``spectrum=(F,M,T)`` and ``mask=(F,T)``.

    The denominator is ``max(sum(mask), epsilon)``, not an additive epsilon.
    Thus ordinary mask rescaling is invariant only above the denominator floor.
    """

    x = np.asarray(spectrum)
    weights = finite_real_array(mask, "mask")
    if x.ndim != 3 or weights.shape != (x.shape[0], x.shape[2]):
        raise ValueError("expected spectrum (F,M,T) and mask (F,T)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("spectrum must be finite and mask weights finite and non-negative")
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = np.einsum("ft,fmt,fnt->fmn", weights, x, x.conj())
        denominator = np.sum(weights, axis=1)[:, None, None]
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("masked SCM accumulation exceeds the float64 range")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        covariance = numerator / np.maximum(denominator, epsilon)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("masked SCM result exceeds the float64 range")
    return covariance


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
    fall back to selecting microphone 0. Statistics use a common per-frequency
    input scale, which cancels from the MVDR weights; output retains input scale.
    Each non-empty mask is scaled by its own per-frequency maximum before its
    SCM is formed, so a common positive change in mask values does not by itself
    trigger the absolute denominator floor used by ``masked_spatial_covariance``.
    There is no confidence threshold on the original mask mass: even uniformly
    tiny but nonzero masks can produce weights. A zero mask, zero usable target
    or interference statistic, or failed solve selects reference microphone 0.
    """

    x = np.asarray(spectrum)
    if x.ndim != 3 or x.shape[1] != 2 or any(size == 0 for size in x.shape):
        raise ValueError("spectrum must have shape (F,2,T)")
    if not np.all(np.isfinite(x)):
        raise ValueError("spectrum must be finite")
    diagonal_loading = finite_real_scalar(diagonal_loading, "diagonal_loading")
    if diagonal_loading < 0.0:
        raise ValueError("diagonal_loading must be finite and non-negative")
    target_weights = finite_real_array(target_mask, "target_mask")
    interference_weights = finite_real_array(interference_mask, "interference_mask")
    expected_mask_shape = (x.shape[0], x.shape[2])
    if target_weights.shape != expected_mask_shape or interference_weights.shape != expected_mask_shape:
        raise ValueError("target and interference masks must have shape (F,T)")
    if np.any(target_weights < 0) or np.any(interference_weights < 0):
        raise ValueError("target and interference masks must be non-negative")
    target_peak = np.max(target_weights, axis=1)
    interference_peak = np.max(interference_weights, axis=1)
    target_weights = target_weights / np.where(target_peak > 0, target_peak, 1.0)[:, None]
    interference_weights = interference_weights / np.where(interference_peak > 0, interference_peak, 1.0)[:, None]
    # Avoid squaring extreme but finite STFT amplitudes. Real/imaginary peak
    # also avoids overflow in abs(complex) near the floating-point limit.
    input_scale = np.maximum(np.max(np.abs(x.real), axis=(1, 2)),
                             np.max(np.abs(x.imag), axis=(1, 2)))
    common_scale = np.where(input_scale > 0, input_scale, 1.0)[:, None, None]
    # Complex division may form 1/common_scale internally, overflowing for
    # subnormal scales even though each desired component ratio is bounded.
    # Divide the real components directly so no reciprocal is materialized.
    normalized = np.empty(x.shape, dtype=np.complex128)
    normalized.real = x.real / common_scale
    normalized.imag = x.imag / common_scale
    target = masked_spatial_covariance(normalized, target_weights)
    interference = masked_spatial_covariance(normalized, interference_weights)
    output = np.empty((x.shape[0], x.shape[2]), dtype=np.complex128)
    beam_weights = np.empty((x.shape[0], 2), dtype=np.complex128)
    fallback = np.array([1.0 + 0j, 0j])
    target_mass = np.sum(target_weights, axis=1)
    interference_mass = np.sum(interference_weights, axis=1)

    for frequency in range(x.shape[0]):
        if target_mass[frequency] == 0.0 or interference_mass[frequency] == 0.0:
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
            loaded = noise / scale + diagonal_loading * np.eye(2)
            try:
                inverse_steering = np.linalg.solve(loaded, steering)
                denominator = np.vdot(steering, inverse_steering)
                weight = (inverse_steering / denominator
                          if np.isfinite(denominator) and denominator.real > 0
                          else fallback)
            except np.linalg.LinAlgError:
                weight = fallback
        beam_weights[frequency] = weight
        try:
            with np.errstate(over="raise", invalid="raise"):
                output[frequency] = (weight.conj() @ normalized[frequency]) * input_scale[frequency]
        except FloatingPointError as error:
            raise ValueError("MVDR output exceeds the float64 range") from error
        if not np.all(np.isfinite(output[frequency])):
            raise ValueError("MVDR output exceeds the float64 range")
    return output, beam_weights
