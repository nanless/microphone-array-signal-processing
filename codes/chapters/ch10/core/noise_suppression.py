"""Small power-spectrum subtraction baseline, not an MMSE speech estimator.

The input is a single-channel STFT with axes frequency x time.  A noise-only
prefix supplies a fixed power estimate; this function deliberately does not
implement speech-presence detection or adaptive noise tracking.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_scalar


def _exact_power(coefficient: complex) -> Fraction:
    return Fraction(float(coefficient.real))**2 + Fraction(float(coefficient.imag))**2


def _exact_gain_output(coefficient: complex, ratio: Fraction) -> complex:
    """Round only after the square root and component multiplication."""
    with localcontext() as context:
        context.prec = 100
        scale = (Decimal(ratio.numerator) / Decimal(ratio.denominator)).sqrt()
        components = []
        for component in (float(coefficient.real), float(coefficient.imag)):
            exact_output = Decimal.from_float(component) * scale
            value = float(exact_output)
            if not math.isfinite(value) or (exact_output and value == 0):
                raise ValueError("nonzero enhanced spectrum component is outside float64 support")
            components.append(value)
        return complex(*components)


def power_spectral_subtraction(
    noisy_stft: np.ndarray,
    noise_only_frames: np.ndarray,
    *,
    oversubtraction: float = 1.0,
    floor_ratio: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Return enhanced F×T STFT and F-bin mean noise power.

    For P=|Y|² and noise estimate D, use P_hat=max(P-alpha*D, beta*P),
    then X_hat=sqrt(P_hat/P)*Y where P>0.  A zero observation stays zero:
    its phase is undefined.  beta is relative to *observed* bin power, so
    sqrt(beta) is the minimum positive-bin amplitude gain. A physical mean
    power below float64 range is returned as zero, but exceptional gains retain
    exact squared binary-float inputs. An unrepresentable large mean power or
    nonzero final spectrum component is rejected, rather than called zero.
    """
    raw = np.asarray(noisy_stft)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("noisy_stft must be a numeric F x T array")
    y = np.asarray(raw, dtype=np.complex128)
    if min(y.shape) < 1 or not np.all(np.isfinite(y)):
        raise ValueError("noisy_stft must be nonempty and finite")
    indices = np.asarray(noise_only_frames)
    if indices.ndim != 1 or indices.size < 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("noise_only_frames must be a nonempty one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= y.shape[1]) or np.unique(indices).size != indices.size:
        raise ValueError("noise-only frame indices must be distinct and in range")
    oversubtraction = finite_real_scalar(oversubtraction, "oversubtraction")
    floor_ratio = finite_real_scalar(floor_ratio, "floor_ratio")
    if oversubtraction < 0:
        raise ValueError("oversubtraction must be finite and nonnegative")
    if not 0 <= floor_ratio <= 1:
        raise ValueError("floor_ratio must be finite and in [0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.abs(y) ** 2
        noise_power = np.mean(power[:, indices], axis=1)
        estimated = np.maximum(power - oversubtraction * noise_power[:, None], floor_ratio * power)
    nonzero = (y.real != 0) | (y.imag != 0)
    ordinary = (np.all(np.isfinite(power)) and np.all(np.isfinite(noise_power))
                and np.all(np.isfinite(estimated))
                and np.all((power >= np.finfo(float).tiny) | ~nonzero)
                and np.all((estimated >= np.finfo(float).tiny) | ~nonzero
                           | ((estimated == 0) & (floor_ratio == 0))))
    if ordinary:
        # Preserve the established arithmetic and PCM fixtures in normal range.
        gain = np.zeros_like(power)
        positive = power > 0
        gain[positive] = np.sqrt(estimated[positive] / power[positive])
        enhanced = gain * y
        # A rounded equality |Y|²==alpha*D is not a proof of an exact zero.
        # For example |1+1e-10j|² rounds to 1, hiding a residual amplitude
        # near 1e-10. Prove clipped/equal zeros with exact powers, and recover
        # only positive cancellation residuals. Other ordinary values retain
        # their established operation order.
        cancelled = (estimated == 0) & nonzero if floor_ratio == 0 else np.zeros(y.shape, dtype=bool)
        for f in np.flatnonzero(np.any(cancelled, axis=1)):
            mean = sum((_exact_power(y[f, int(i)]) for i in indices), Fraction(0))/len(indices)
            for t in np.flatnonzero(cancelled[f]):
                observed = _exact_power(y[f, t])
                remaining = observed - Fraction(oversubtraction)*mean
                if remaining > 0:
                    enhanced[f, t] = _exact_gain_output(y[f, t], remaining/observed)
        lost = (gain > 0) & (((y.real != 0) & (enhanced.real == 0))
                            | ((y.imag != 0) & (enhanced.imag == 0)))
        for f in np.flatnonzero(np.any(lost, axis=1)):
            mean = sum((_exact_power(y[f, int(i)]) for i in indices), Fraction(0))/len(indices)
            for t in np.flatnonzero(lost[f]):
                observed = _exact_power(y[f, t])
                ratio = max((observed-Fraction(oversubtraction)*mean)/observed, Fraction(floor_ratio))
                enhanced[f, t] = _exact_gain_output(y[f, t], ratio)
    else:
        # Rare exact-power fallback: log(P)-log(D) can erase adjacent powers
        # even though the remaining amplitude is representable. Keep squares,
        # mean and subtraction exact; round only the final spectrum component.
        enhanced = np.zeros(y.shape, dtype=complex)
        noise_power = np.zeros(y.shape[0])
        alpha, beta = Fraction(oversubtraction), Fraction(floor_ratio)
        for f in range(y.shape[0]):
            powers = [_exact_power(z) for z in y[f]]
            mean = sum((powers[int(i)] for i in indices), Fraction(0)) / len(indices)
            try:
                noise_power[f] = float(mean)
            except OverflowError as error:
                raise ValueError("mean noise power exceeds floating-point range") from error
            if not math.isfinite(noise_power[f]):
                raise ValueError("mean noise power exceeds floating-point range")
            for t, observed in enumerate(powers):
                if observed:
                    ratio = max((observed - alpha * mean) / observed, beta)
                    enhanced[f, t] = _exact_gain_output(y[f, t], ratio)
        return enhanced, noise_power
    if oversubtraction == 0:
        return y.copy(), noise_power
    if not np.all(np.isfinite(enhanced)):
        raise ValueError("enhanced spectrum exceeds floating-point range")
    return enhanced, noise_power
