"""Small separation metrics and mask-beamforming baselines."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


MAX_EXHAUSTIVE_PIT_SOURCES = 8


def _finite_numeric_spectrum(value, name="spectrum"):
    """Keep ordinary numeric arrays; reject text/bool before any conversion."""
    array = np.asarray(value)
    if array.dtype.kind not in "iufc" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real or complex numbers")
    return array.astype(np.float64) if array.dtype.kind in "iu" else array


def _checked_scale_components(value, scale, name):
    """Divide components without a complex reciprocal or silently lost support.

    This is a float64 teaching boundary, not an arbitrary-precision solver.
    A mathematically nonzero component rounded to zero is explicitly refused.
    """
    out = np.empty(np.shape(value), dtype=np.complex128)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out.real = np.asarray(value).real / scale
        out.imag = np.asarray(value).imag / scale
    if (not np.all(np.isfinite(out)) or
            np.any((np.asarray(value).real != 0) & (out.real == 0)) or
            np.any((np.asarray(value).imag != 0) & (out.imag == 0))):
        raise ValueError(f"{name} normalization loses nonzero components outside float64 support")
    return out


def _posterior_from_log_scores(log_scores):
    """Normalize active log scores; refuse positive probabilities lost to zero."""
    maximum = np.max(log_scores, axis=-1, keepdims=True)
    with np.errstate(under="ignore"):
        scores = np.exp(log_scores - maximum)
        posterior = scores / np.sum(scores, axis=-1, keepdims=True)
    if np.any(np.isfinite(log_scores) & (posterior == 0)):
        raise ValueError("positive posterior underflows float64 support; it is not an inactive class")
    return posterior


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
    A mathematically positive active posterior that cannot be represented in
    float64 is explicitly refused, unlike a disabled class whose probability
    is exactly zero.
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
    return _posterior_from_log_scores(log_scores)


def si_sdr(estimate: np.ndarray, reference: np.ndarray, *, zero_mean: bool = True, epsilon: float = 1e-12) -> float:
    """Compute SI-SDR with a relative energy floor and no time alignment.

    Peak scaling before centering avoids absolute-level dependence. ``epsilon``
    sets finite caps of approximately +/- ``-10*log10(epsilon)`` dB. Zero signals
    and signals constant after centering are rejected; perfect estimates return
    the positive cap, not infinity. See Le Roux et al. (ICASSP 2019), eq. (3--5).
    """

    if not isinstance(zero_mean, (bool, np.bool_)):
        raise ValueError("zero_mean must be boolean")
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
    Exceptional exponent products use a separate accumulation route. A detected
    nonzero covariance entry outside float64's final range is explicitly refused;
    an exactly cancelled entry or a genuinely zero masked channel remains zero.
    """

    x = _finite_numeric_spectrum(spectrum)
    weights = finite_real_array(mask, "mask")
    if x.ndim != 3 or any(size == 0 for size in x.shape) or weights.shape != (x.shape[0], x.shape[2]):
        raise ValueError("expected spectrum (F,M,T) and mask (F,T)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("spectrum must be finite and mask weights finite and non-negative")
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    # Retain the ordinary arithmetic order (and existing PCM fixtures). The
    # scaled route is for exponents at which the raw numerator can overflow
    # or underflow even though the final weighted mean is representable.
    peak = np.maximum(np.max(np.abs(x.real), axis=(1, 2)), np.max(np.abs(x.imag), axis=(1, 2)))
    mask_peak = np.max(weights, axis=1)
    ordinary = ((peak == 0) | ((peak >= 1e-75) & (peak <= 1e75))) & ((mask_peak == 0) | ((mask_peak >= 1e-75) & (mask_peak <= 1e75)))
    if not np.all(ordinary):
        result = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=np.complex128)
        for f in range(x.shape[0]):
            if peak[f] == 0 or mask_peak[f] == 0:
                continue
            mass = float((weights[f] / mask_peak[f]).sum())
            pm, pe = math.frexp(float(mask_peak[f]))
            sm, se = math.frexp(mass)
            denominator_mantissa, denominator_exponent = pm * sm, pe + se
            if mask_peak[f] < epsilon / mass:
                denominator_mantissa, denominator_exponent = math.frexp(epsilon)
            wm, we = np.frexp(weights[f])
            real_m, real_e = np.frexp(np.asarray(x[f].real, dtype=float))
            imag_m, imag_e = np.frexp(np.asarray(x[f].imag, dtype=float))

            def sum_products(parts, *, positive=False):
                # Accumulate weighted complex outer products in exponent form.
                # Zero-weight outliers and tiny positive weights cannot erase
                # smaller contributing samples by selecting a global peak.
                mantissas, exponents = [], []
                for am, ae, bm, be, sign in parts:
                    term = sign * wm * am * bm / denominator_mantissa
                    use = term != 0
                    mantissas.extend(term[use])
                    exponents.extend((we + ae + be - denominator_exponent)[use])
                if not mantissas:
                    return 0.
                if positive:
                    # A positive diagonal must be rounded only AFTER its
                    # terms have been summed: two individually sub-half-ulp
                    # powers can together produce a representable subnormal.
                    exponent = max(exponents)
                    total = math.fsum(math.ldexp(float(m), int(e-exponent))
                                      for m, e in zip(mantissas, exponents))
                    return math.ldexp(total, int(exponent))
                # Ordinary restored terms retain the historical summation.
                # At subnormal term scales, round only their COMPLETE sum;
                # exponent alignment alone can lose a tiny cancellation
                # residual. Fraction is deliberately confined to this rare
                # path and retains each already-normalized binary term.
                restored = [math.ldexp(float(m), int(e))
                            for m, e in zip(mantissas, exponents)]
                if any(abs(value) < np.finfo(float).tiny for value in restored):
                    total = Fraction()
                    for m, e in zip(mantissas, exponents):
                        term = Fraction.from_float(float(m))
                        total += term * (2 ** int(e) if e >= 0 else Fraction(1, 2 ** -int(e)))
                    value = float(total)
                    if total != 0 and value == 0:
                        raise ValueError("nonzero masked SCM cross term underflows float64 range")
                    return value
                return math.fsum(restored)
            try:
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    # Diagonals are nonnegative sums. Check their range first;
                    # valid diagonals also bound the cross-term magnitudes.
                    pairs = [(m, m) for m in range(x.shape[1])]
                    pairs += [(m, n) for m in range(x.shape[1])
                              for n in range(m + 1, x.shape[1])]
                    for m, n in pairs:
                        re = sum_products([(real_m[m], real_e[m], real_m[n], real_e[n], 1),
                                           (imag_m[m], imag_e[m], imag_m[n], imag_e[n], 1)], positive=m == n)
                        if m == n and re == 0 and np.any((weights[f] > 0) & (x[f, m] != 0)):
                            raise ValueError("positive masked SCM diagonal underflows float64 range")
                        im = 0. if m == n else sum_products([(imag_m[m], imag_e[m], real_m[n], real_e[n], 1),
                                           (real_m[m], real_e[m], imag_m[n], imag_e[n], -1)])
                        result[f, m, n] = complex(re, im)
                        result[f, n, m] = complex(re, -im)
            except (FloatingPointError, OverflowError) as error:
                raise ValueError("masked SCM result exceeds the float64 range") from error
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = np.einsum("ft,fmt,fnt->fmn", weights, x, x.conj())
        denominator = np.sum(weights, axis=1)[:, None, None]
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("masked SCM accumulation exceeds the float64 range")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        covariance = numerator / np.maximum(denominator, epsilon)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("masked SCM result exceeds the float64 range")
    active = np.any((weights[:, None, :] > 0) & (x != 0), axis=-1)
    if np.any(active & (covariance.diagonal(axis1=1, axis2=2).real == 0)):
        raise ValueError("positive masked SCM diagonal underflows float64 range")
    return covariance


def mask_mvdr_2x2(
    spectrum: np.ndarray,
    target_mask: np.ndarray,
    interference_mask: np.ndarray,
    *,
    diagonal_loading: float = 1e-6,
    return_diagnostics: bool = False,
) -> tuple:
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
    With ``return_diagnostics=True`` a third dictionary reports the per-bin
    route and target eigenvalue gap; a finite output is not proof of separation.
    Cross-scale masks/input components that disappear during normalization
    raise ValueError rather than being labeled a zero physical statistic.
    The float64 eigensolver, constraint division and beam sum also explicitly
    refuse detected loss of a nonzero component; this is a finite teaching
    support boundary, not a guarantee for every finite float64 combination.
    A restored component may round to zero after cancellation at the summation
    roundoff scale. ``return_diagnostics`` reports its count and positions as
    ``rounding_underflow_components/positions``: those zeros cannot distinguish
    an exact null from a strictly nonzero roundoff-scale residual. Other detected
    restoration loss raises ValueError. No claim of an exact mathematical zero
    follows from a finite output.
    """

    x = _finite_numeric_spectrum(spectrum)
    if x.ndim != 3 or x.shape[1] != 2 or any(size == 0 for size in x.shape):
        raise ValueError("spectrum must have shape (F,2,T)")
    if not np.all(np.isfinite(x)):
        raise ValueError("spectrum must be finite")
    diagonal_loading = finite_real_scalar(diagonal_loading, "diagonal_loading")
    if diagonal_loading < 0.0:
        raise ValueError("diagonal_loading must be finite and non-negative")
    if not isinstance(return_diagnostics, bool):
        raise ValueError("return_diagnostics must be boolean")
    target_weights = finite_real_array(target_mask, "target_mask")
    interference_weights = finite_real_array(interference_mask, "interference_mask")
    expected_mask_shape = (x.shape[0], x.shape[2])
    if target_weights.shape != expected_mask_shape or interference_weights.shape != expected_mask_shape:
        raise ValueError("target and interference masks must have shape (F,T)")
    if np.any(target_weights < 0) or np.any(interference_weights < 0):
        raise ValueError("target and interference masks must be non-negative")
    target_peak = np.max(target_weights, axis=1)
    interference_peak = np.max(interference_weights, axis=1)
    old_target, old_interference = target_weights, interference_weights
    with np.errstate(under="ignore"):
        target_weights = target_weights / np.where(target_peak > 0, target_peak, 1.0)[:, None]
        interference_weights = interference_weights / np.where(interference_peak > 0, interference_peak, 1.0)[:, None]
    if (np.any((old_target > 0) & (target_weights == 0)) or
            np.any((old_interference > 0) & (interference_weights == 0))):
        raise ValueError("mask normalization loses positive weights outside float64 support")
    # Avoid squaring extreme but finite STFT amplitudes. Real/imaginary peak
    # also avoids overflow in abs(complex) near the floating-point limit.
    input_scale = np.maximum(np.max(np.abs(x.real), axis=(1, 2)),
                             np.max(np.abs(x.imag), axis=(1, 2)))
    common_scale = np.where(input_scale > 0, input_scale, 1.0)[:, None, None]
    # Complex division may form 1/common_scale internally, overflowing for
    # subnormal scales even though each desired component ratio is bounded.
    # Divide the real components directly so no reciprocal is materialized.
    normalized = _checked_scale_components(x, common_scale, "MVDR input")
    target = masked_spatial_covariance(normalized, target_weights)
    interference = masked_spatial_covariance(normalized, interference_weights)
    output = np.empty((x.shape[0], x.shape[2]), dtype=np.complex128)
    beam_weights = np.empty((x.shape[0], 2), dtype=np.complex128)
    fallback = np.array([1.0 + 0j, 0j])
    target_mass = np.sum(target_weights, axis=1)
    interference_mass = np.sum(interference_weights, axis=1)
    reasons = []
    rounding_positions = []
    gaps = np.zeros(x.shape[0])

    for frequency in range(x.shape[0]):
        if target_mass[frequency] == 0.0 or interference_mass[frequency] == 0.0:
            reasons.append("empty_target_mask" if target_mass[frequency] == 0 else "empty_interference_mask")
            beam_weights[frequency] = fallback
            output[frequency] = x[frequency, 0]
            continue
        values, vectors = np.linalg.eigh(target[frequency])
        if values[-1].real <= np.finfo(float).tiny:
            reasons.append("zero_target_statistic")
            beam_weights[frequency] = fallback
            output[frequency] = x[frequency, 0]
            continue
        steering = vectors[:, int(np.argmax(values.real))]
        if target[frequency, 0, 1] != 0 and np.any(steering == 0):
            raise ValueError("MVDR principal direction loses a nonzero component outside float64 eigensolver support")
        gaps[frequency] = float((values[-1] - values[-2]) / values[-1])
        reason = "mvdr"
        if abs(steering[0]) < 1e-10:
            weight = fallback
            reason = "unusable_reference_component"
        else:
            steering = steering / steering[0]
            noise = interference[frequency]
            scale = float(np.trace(noise).real / 2.0)
            if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
                reasons.append("zero_interference_statistic")
                beam_weights[frequency] = fallback
                output[frequency] = x[frequency, 0]
                continue
            loaded = noise / scale + diagonal_loading * np.eye(2)
            try:
                if diagonal_loading > 1e100:
                    # A common scale cancels in the normalized MVDR weight.
                    loaded.real /= diagonal_loading
                    loaded.imag /= diagonal_loading
                inverse_steering = np.linalg.solve(loaded, steering)
                denominator = np.vdot(steering, inverse_steering)
                weight = (inverse_steering / denominator
                          if np.isfinite(denominator) and denominator.real > 0
                          else fallback)
                if np.isfinite(denominator) and denominator.real > 0:
                    lost = ((inverse_steering.real != 0) & (weight.real == 0)) | ((inverse_steering.imag != 0) & (weight.imag == 0))
                    for m in np.flatnonzero(lost):
                        nr, ni = map(Fraction.from_float, (float(inverse_steering[m].real), float(inverse_steering[m].imag)))
                        dr, di = map(Fraction.from_float, (float(denominator.real), float(denominator.imag)))
                        if ((weight[m].real == 0 and nr*dr+ni*di != 0) or
                                (weight[m].imag == 0 and ni*dr-nr*di != 0)):
                            raise ValueError("MVDR constraint normalization loses a nonzero component outside float64 support")
                if not np.isfinite(denominator) or denominator.real <= 0:
                    reason = "invalid_constraint_denominator"
            except np.linalg.LinAlgError:
                weight = fallback
                reason = "singular_solve"
        reasons.append(reason)
        beam_weights[frequency] = weight
        try:
            with np.errstate(over="raise", invalid="raise"):
                scaled_output = weight.conj() @ normalized[frequency]
                output[frequency] = scaled_output * input_scale[frequency]
            for component in ('real', 'imag'):
                scaled_part, output_part = getattr(scaled_output, component), getattr(output[frequency], component)
                lost = (scaled_part != 0) & (output_part == 0)
                if np.any(lost):
                    if component == 'real':
                        terms = weight.real[:, None]*normalized[frequency].real + weight.imag[:, None]*normalized[frequency].imag
                    else:
                        terms = weight.real[:, None]*normalized[frequency].imag - weight.imag[:, None]*normalized[frequency].real
                    bound = 8*np.finfo(float).eps*np.sum(abs(terms), axis=0)
                    if np.any(lost & (abs(scaled_part) > bound)):
                        raise ValueError("MVDR output restoration loses nonzero components outside float64 support")
                    for t in np.flatnonzero(lost):
                        rounding_positions.append({'frequency_index': frequency, 'frame_index': int(t),
                                                   'component': component,
                                                   'status': 'cannot distinguish exact null from roundoff-scale nonzero residual'})
            # A product can underflow before the common input scale restores
            # it. Only exceptional tiny factors need this exact two-term check.
            if (np.any((np.abs(normalized[frequency]) > 0) &
                       (np.abs(normalized[frequency]) < 1e-150)) or
                    np.any((np.abs(weight) > 0) & (np.abs(weight) < 1e-150))):
                for t in range(x.shape[2]):
                    real = sum((Fraction(float(weight[m].real)) * Fraction(float(x[frequency, m, t].real)) +
                                Fraction(float(weight[m].imag)) * Fraction(float(x[frequency, m, t].imag))) for m in range(2))
                    imag = sum((Fraction(float(weight[m].real)) * Fraction(float(x[frequency, m, t].imag)) -
                                Fraction(float(weight[m].imag)) * Fraction(float(x[frequency, m, t].real))) for m in range(2))
                    if ((real and output[frequency, t].real == 0) or
                            (imag and output[frequency, t].imag == 0)):
                        raise ValueError("MVDR beam sum loses a nonzero component outside float64 support")
        except FloatingPointError as error:
            raise ValueError("MVDR output exceeds the float64 range") from error
        if not np.all(np.isfinite(output[frequency])):
            raise ValueError("MVDR output exceeds the float64 range")
    if return_diagnostics:
        return output, beam_weights, {"routes": reasons, "target_relative_eigenvalue_gap": gaps,
                                      "rounding_underflow_components": len(rounding_positions),
                                      "rounding_underflow_positions": rounding_positions,
                                      "fallback_count": sum(reason != "mvdr" for reason in reasons)}
    return output, beam_weights
