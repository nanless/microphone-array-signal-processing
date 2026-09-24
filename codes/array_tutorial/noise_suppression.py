"""Small power-spectrum subtraction baseline, not an MMSE speech estimator.

The input is a single-channel STFT with axes frequency x time.  A noise-only
prefix supplies a fixed power estimate; this function deliberately does not
implement speech-presence detection or adaptive noise tracking.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_scalar


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
    sqrt(beta) is the minimum positive-bin amplitude gain.
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
    if not np.all(np.isfinite(power)) or not np.all(np.isfinite(noise_power)) or not np.all(np.isfinite(estimated)):
        raise ValueError("spectral power exceeds floating-point range")
    gain = np.zeros_like(power)
    positive = power > 0
    gain[positive] = np.sqrt(estimated[positive] / power[positive])
    enhanced = gain * y
    if not np.all(np.isfinite(enhanced)):
        raise ValueError("enhanced spectrum exceeds floating-point range")
    return enhanced, noise_power
