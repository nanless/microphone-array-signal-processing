"""Small NumPy cACGMM activity-guided mask and two-mic MVDR teaching chain.

This is an independently written, offline model, not the upstream CuPy/Lhotse
implementation or a CHiME reproduction. The input is a fixed multichannel
STFT and external speaker activity; it does not perform diarization.
"""

from __future__ import annotations

import numpy as np

from .dereverberation import offline_wpe
from .separation import mask_mvdr_2x2, masked_spatial_covariance


def guided_cacgmm_mvdr(
    spectrum: np.ndarray,
    speaker_activity: np.ndarray,
    *,
    iterations: int = 8,
    shape_loading: float = 0.02,
    use_wpe: bool = False,
    target_speaker: int = 0,
) -> dict:
    """Fit per-frequency cACG shapes, form masks/SCMs, then output MVDR STFT.

    ``spectrum`` is (F,2,T); activity is (T,J) with binary values. A final
    always-active background component makes all-speakers-silent frames valid.
    Low-energy STFT points bypass EM and belong to background only. A shape
    matrix with too little posterior mass is reset to identity and counted.
    """

    x = np.asarray(spectrum, dtype=np.complex128)
    activity = np.asarray(speaker_activity)
    if x.ndim != 3 or x.shape[1] != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("spectrum must be a finite nonempty (F,2,T) complex array")
    if activity.ndim != 2 or activity.shape[0] != x.shape[2] or activity.shape[1] < 1 or not np.all(np.isin(activity, [0, 1])):
        raise ValueError("activity must be a binary (T,J) array")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not np.isfinite(shape_loading) or not 0 < shape_loading < 1:
        raise ValueError("shape_loading must be in (0,1)")
    if isinstance(target_speaker, bool) or not isinstance(target_speaker, int) or not 0 <= target_speaker < activity.shape[1]:
        raise ValueError("target_speaker outside the activity columns")
    if not isinstance(use_wpe, bool):
        raise ValueError("use_wpe must be boolean")

    if use_wpe:
        x = offline_wpe(x, taps=3, delay=2, iterations=2)
    frequencies, channels, frames = x.shape
    classes = activity.shape[1] + 1
    allowed = np.concatenate((activity.astype(bool), np.ones((frames, 1), dtype=bool)), axis=1)
    posterior = np.zeros((frequencies, classes, frames), dtype=float)
    shapes = np.empty((frequencies, classes, channels, channels), dtype=np.complex128)
    reset_count = 0
    low_energy_count = 0
    identity = np.eye(channels)

    for f in range(frequencies):
        samples = x[f].T  # (T,2), raw complex STFT amplitudes
        energy = np.linalg.norm(samples, axis=1)
        limit = max(1e-12, 1e-7 * float(np.max(energy)))
        valid = energy > limit
        low_energy_count += int(np.count_nonzero(~valid))
        if not np.any(valid):
            posterior[f, -1] = 1
            shapes[f] = identity
            continue
        z = samples[valid] / energy[valid, None]
        active = allowed[valid]
        masses = np.maximum(active.sum(axis=0).astype(float), 0.05)
        priors = masses / masses.sum()
        b = np.empty((classes, channels, channels), dtype=np.complex128)
        for j in range(classes):
            # Solo activity initializes a speaker; silence initializes the
            # background. If neither exists, the identity is explicit.
            selector = active[:, j].copy()
            if j < classes - 1:
                selector &= active[:, :classes - 1].sum(axis=1) == 1
            else:
                selector = active[:, :classes - 1].sum(axis=1) == 0
            if np.any(selector):
                cov = np.einsum("ti,tj->ij", z[selector], z[selector].conj()) / selector.sum()
                b[j] = _loaded_trace_shape(cov, shape_loading)
            else:
                b[j] = identity

        for _ in range(iterations):
            log_scores = np.full((z.shape[0], classes), -np.inf)
            quadratic = np.empty((z.shape[0], classes), dtype=float)
            for j in range(classes):
                inverse_z = np.linalg.solve(b[j], z.T).T
                q = np.einsum("ti,ti->t", z.conj(), inverse_z).real
                q = np.maximum(q, np.finfo(float).tiny)
                quadratic[:, j] = q
                sign, logdet = np.linalg.slogdet(b[j])
                if sign.real <= 0:
                    raise np.linalg.LinAlgError("cACG shape lost positive definiteness")
                log_scores[:, j] = np.log(priors[j]) - logdet - channels * np.log(q)
            log_scores[~active] = -np.inf
            maximum = np.max(log_scores, axis=1, keepdims=True)
            scores = np.exp(log_scores - maximum)
            gamma = scores / scores.sum(axis=1, keepdims=True)
            priors = np.maximum(gamma.mean(axis=0), 1e-4)
            priors /= priors.sum()
            for j in range(classes):
                mass = float(gamma[:, j].sum())
                if mass < channels:
                    b[j] = identity
                    reset_count += 1
                    continue
                weighted = gamma[:, j] / quadratic[:, j]
                raw = (channels / mass) * np.einsum("t,ti,tj->ij", weighted, z, z.conj())
                b[j] = _loaded_trace_shape(raw, shape_loading)
        posterior[f][:, valid] = gamma.T
        posterior[f, -1, ~valid] = 1
        shapes[f] = b

    target_mask = posterior[:, target_speaker, :]
    other_mask = posterior.sum(axis=1) - target_mask
    target_scm = masked_spatial_covariance(x, target_mask)
    other_scm = masked_spatial_covariance(x, other_mask)
    output, weights = mask_mvdr_2x2(x, target_mask, other_mask)
    return {
        "output": output,
        "posterior": posterior,
        "target_scm": target_scm,
        "other_scm": other_scm,
        "weights": weights,
        "shape_matrices": shapes,
        "wpe_used": use_wpe,
        "iterations": iterations,
        "shape_reset_count": reset_count,
        "low_energy_bins": low_energy_count,
    }


def _loaded_trace_shape(matrix: np.ndarray, loading: float) -> np.ndarray:
    matrix = (matrix + matrix.conj().T) / 2
    trace = float(np.trace(matrix).real)
    if not np.isfinite(trace) or trace <= 0:
        return np.eye(matrix.shape[0], dtype=complex)
    scaled = matrix * (matrix.shape[0] / trace)
    result = (1 - loading) * scaled + loading * np.eye(matrix.shape[0])
    return (result + result.conj().T) / 2
