"""Minimal DOA baselines used by Chapters 2--4.

These are clean-room teaching implementations, not copies of third-party
projects.  They favor explicit validation and formula-to-code correspondence.
"""

from __future__ import annotations

import math

import numpy as np

from .conventions import hermitian_part, validate_cft, validate_frequencies
from .geometry import plane_wave_delays


def mdl_source_count(eigenvalues: np.ndarray, snapshots: int) -> tuple[int, np.ndarray]:
    """Return ``(selected_count, scores_for_k_0_through_M_minus_1)``.

    Wax--Kailath complex Gaussian, spatially white noise MDL (Chapter 4.10).
    Input is a nonempty 1-D vector of finite, strictly positive *real* sample
    covariance eigenvalues, in any order.  Scores use natural logarithms and
    omit the common ``0.5 * log(snapshots)`` penalty; they are not probabilities.
    Snapshots must be an integer >= max(M, 2), representing independent,
    identically distributed, known-zero-mean complex observations.  This
    necessary full-rank sample-size check cannot establish model validity.
    Centered samples need N > M; dependent STFT frames need separate analysis.

    No covariance loading or eigenvalue flooring is applied: singular spectra
    are rejected.  Sorting does not mutate the input; exact ties choose smaller
    k.  Log-domain tail means avoid products and power-scale overflow/underflow.
    Reference: Wax & Kailath, ICASSP 1984, equations (10)--(15),
    doi:10.1109/ICASSP.1984.1172389. This is a teaching baseline, not a detector
    for coherent sources, colored noise, or the real Gaussian model.
    """
    values = np.asarray(eigenvalues)
    if values.ndim != 1 or values.size < 1 or values.dtype.kind not in "iuf":
        raise ValueError("eigenvalues must be a nonempty 1-D real numeric vector")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("eigenvalues must be finite and strictly positive; no loading is applied")
    channels = values.size
    if (isinstance(snapshots, (bool, np.bool_))
            or not isinstance(snapshots, (int, np.integer))
            or snapshots < max(channels, 2)):
        raise ValueError("snapshots must be an integer >= max(number of eigenvalues, 2)")
    try:
        count = float(snapshots)
    except OverflowError as error:
        raise ValueError("snapshots exceeds floating-point range") from error
    if not math.isfinite(count):
        raise ValueError("snapshots exceeds floating-point range")
    logs = np.log(np.sort(values)[::-1])
    scores = np.empty(channels)
    for k in range(channels):
        tail = logs[k:]
        shifted = tail - tail[0]
        # A normalized eigenvalue itself could underflow to zero; its log
        # remains finite here and must still contribute to the geometric mean.
        arithmetic_log = math.log(math.fsum(math.exp(x) for x in shifted) / tail.size)
        geometric_log = math.fsum(shifted) / tail.size
        gap = max(0.0, arithmetic_log - geometric_log)  # AM >= GM; round-off only.
        scores[k] = count * tail.size * gap + 0.5 * k * (2 * channels - k) * math.log(count)
    if not np.all(np.isfinite(scores)):
        raise ValueError("MDL scores exceed floating-point range")
    return int(np.argmin(scores)), scores


def gcc_phat(
    x1: np.ndarray,
    x2: np.ndarray,
    sample_rate: float,
    *,
    max_tau: float | None = None,
    epsilon: float = 1e-12,
    parabolic_interpolation: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate ``tau12 = t1 - t2`` using ``ifft(X1 * conj(X2))``.

    Returns ``(tau_seconds, peak_value, lags_seconds, correlation)``.  The
    FFT padding covers the unweighted linear-correlation support.  PHAT's
    nonlinear spectral normalization does not retain that finite support:
    the result is a sampled periodic inverse transform restricted to physical
    lags, and changing the FFT length can change its values.  Optional
    three-point interpolation is local and does not change the lag search range.
    """
    first = np.asarray(x1, dtype=float)
    second = np.asarray(x2, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.size < 1 or second.size < 1:
        raise ValueError("x1 and x2 must be non-empty one-dimensional signals")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("signals contain NaN or infinity")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    linear_length = first.size + second.size - 1
    n_fft = 1 << (linear_length - 1).bit_length()
    cross = np.fft.fft(first, n_fft) * np.fft.fft(second, n_fft).conj()
    magnitude = np.abs(cross)
    if np.max(magnitude) <= epsilon:
        raise ValueError("GCC-PHAT is undefined for a zero-energy pair")
    cross /= np.maximum(magnitude, epsilon)
    circular = np.fft.ifft(cross).real
    negative_lags = circular[-(second.size - 1) :] if second.size > 1 else circular[:0]
    correlation = np.concatenate((negative_lags, circular[: first.size]))
    lags = np.arange(-(second.size - 1), first.size, dtype=float)
    if max_tau is not None:
        if not np.isfinite(max_tau) or max_tau < 0.0:
            raise ValueError("max_tau must be finite and non-negative")
        keep = np.abs(lags / sample_rate) <= max_tau + np.finfo(float).eps
        correlation = correlation[keep]
        lags = lags[keep]
        if lags.size == 0:
            raise ValueError("max_tau leaves no candidate lag")
    index = int(np.argmax(correlation))
    lag = lags[index]
    peak = correlation[index]
    if parabolic_interpolation and 0 < index < correlation.size - 1:
        left, center, right = correlation[index - 1 : index + 2]
        curvature = left - 2.0 * center + right
        if abs(curvature) > epsilon:
            offset = 0.5 * (left - right) / curvature
            if abs(offset) <= 1.0:
                lag += offset
                peak = center - 0.25 * (left - right) * offset
    return lag / sample_rate, float(peak), lags / sample_rate, correlation


def srp_phat(
    spectra: np.ndarray,
    frequencies_hz: np.ndarray,
    positions: np.ndarray,
    candidate_azimuths_rad: np.ndarray,
    *,
    sound_speed: float = 343.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Scan far-field azimuths by averaging PHAT-normalized microphone pairs."""
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    x = validate_cft(spectra)
    frequencies = validate_frequencies(frequencies_hz)
    candidates = np.atleast_1d(np.asarray(candidate_azimuths_rad, dtype=float))
    if frequencies.size != x.shape[1]:
        raise ValueError("frequencies_hz does not match the STFT frequency axis")
    if candidates.ndim != 1 or not np.all(np.isfinite(candidates)) or candidates.size < 1:
        raise ValueError("candidate_azimuths_rad must be finite and non-empty")
    delays = plane_wave_delays(
        positions, candidates, sound_speed=sound_speed
    )
    if delays.shape[1] != x.shape[0]:
        raise ValueError("positions and spectra have different channel counts")
    score = np.zeros(candidates.size, dtype=float)
    pair_count = 0
    for first in range(x.shape[0]):
        for second in range(first + 1, x.shape[0]):
            cross = x[first] * x[second].conj()
            magnitude = np.abs(cross)
            phat = np.where(magnitude > epsilon, cross / np.maximum(magnitude, epsilon), 0.0)
            mean_cross = np.mean(phat, axis=1)
            predicted = delays[:, first] - delays[:, second]
            phase = np.exp(2.0j * np.pi * predicted[:, None] * frequencies[None, :])
            score += np.real(phase @ mean_cross)
            pair_count += 1
    if pair_count == 0:
        raise ValueError("SRP-PHAT needs at least two microphones")
    return score / (pair_count * frequencies.size)


def _steering_rows(steering: np.ndarray, channels: int) -> np.ndarray:
    candidates = np.asarray(steering, dtype=complex)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    if candidates.ndim != 2 or candidates.shape[1] != channels:
        raise ValueError("steering must have shape candidates x channels")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("steering contains NaN or infinity")
    return candidates


def bartlett_spectrum(covariance: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """Evaluate ``a.H @ R @ a`` for each steering-vector row."""
    matrix = hermitian_part(np.asarray(covariance, dtype=complex))
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a single square matrix")
    candidates = _steering_rows(steering, matrix.shape[0])
    values = np.einsum("km,mn,kn->k", candidates.conj(), matrix, candidates)
    return np.real_if_close(values).real


def capon_spectrum(
    covariance: np.ndarray,
    steering: np.ndarray,
    *,
    relative_diagonal_loading: float = 0.0,
    condition_limit: float = 1e12,
) -> np.ndarray:
    """Evaluate the Capon spectrum using a linear solve, never an explicit inverse."""
    matrix = _load_covariance(covariance, relative_diagonal_loading, condition_limit)
    candidates = _steering_rows(steering, matrix.shape[0])
    solved = np.linalg.solve(matrix, candidates.T)
    denominator = np.einsum("km,mk->k", candidates.conj(), solved)
    # The quadratic form scales inversely with covariance power, so the
    # imaginary round-off tolerance must use its own real-valued scale.
    if (np.any(denominator.real <= 0.0)
            or np.any(np.abs(denominator.imag) > 1e-8 * np.abs(denominator.real))):
        raise np.linalg.LinAlgError("Capon denominator is not positive real")
    return 1.0 / denominator.real


def music_spectrum(
    covariance: np.ndarray,
    steering: np.ndarray,
    *,
    source_count: int,
    denominator_floor: float = 1e-15,
) -> np.ndarray:
    """Evaluate the narrowband MUSIC pseudospectrum using ``numpy.linalg.eigh``."""
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")
    matrix = hermitian_part(np.asarray(covariance, dtype=complex))
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a single square matrix")
    channels = matrix.shape[0]
    if not isinstance(source_count, (int, np.integer)) or not 0 < source_count < channels:
        raise ValueError("source_count must be in [1, channels - 1]")
    candidates = _steering_rows(steering, channels)
    _, eigenvectors = np.linalg.eigh(matrix)
    noise = eigenvectors[:, : channels - source_count]
    projection = candidates.conj() @ noise
    denominator = np.sum(np.abs(projection) ** 2, axis=1)
    return 1.0 / np.maximum(denominator, denominator_floor)


def esprit_ula(
    covariance: np.ndarray,
    *,
    source_count: int,
    spacing_m: float,
    frequency_hz: float,
    sound_speed: float = 343.0,
    alias_tolerance: float = 1e-10,
) -> np.ndarray:
    """Estimate ULA broadside azimuths with least-squares ESPRIT."""
    matrix = hermitian_part(np.asarray(covariance, dtype=complex))
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a single square matrix")
    channels = matrix.shape[0]
    if channels < 2 or not isinstance(source_count, (int, np.integer)):
        raise ValueError("ESPRIT needs at least two channels and an integer source_count")
    if not 0 < source_count < channels:
        raise ValueError("source_count must be in [1, channels - 1]")
    if (not np.all(np.isfinite([spacing_m, frequency_hz, sound_speed]))
            or spacing_m <= 0.0 or frequency_hz <= 0.0 or sound_speed <= 0.0):
        raise ValueError("spacing, frequency, and sound speed must be positive")
    if not np.isfinite(alias_tolerance) or alias_tolerance < 0.0:
        raise ValueError("alias_tolerance must be finite and non-negative")
    _, eigenvectors = np.linalg.eigh(matrix)
    signal = eigenvectors[:, -source_count:]
    first = signal[:-1]
    second = signal[1:]
    transform, *_ = np.linalg.lstsq(first, second, rcond=None)
    modes = np.linalg.eigvals(transform)
    sine = np.angle(modes) * sound_speed / (2.0 * np.pi * frequency_hz * spacing_m)
    if np.any(np.abs(sine) > 1.0 + alias_tolerance):
        raise ValueError("estimated spatial phase has no unaliased physical azimuth")
    return np.sort(np.arcsin(np.clip(sine, -1.0, 1.0)).real)


def _load_covariance(
    covariance: np.ndarray,
    relative_diagonal_loading: float,
    condition_limit: float,
) -> np.ndarray:
    matrix = hermitian_part(np.asarray(covariance, dtype=complex))
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be one finite square matrix")
    if not np.isfinite(condition_limit) or condition_limit < 1.0:
        raise ValueError("condition_limit must be finite and at least one")
    if not np.isfinite(relative_diagonal_loading) or relative_diagonal_loading < 0.0:
        raise ValueError("relative_diagonal_loading must be non-negative")
    scale = np.trace(matrix).real / matrix.shape[0]
    if relative_diagonal_loading:
        if scale <= 0.0:
            raise np.linalg.LinAlgError("non-positive covariance scale cannot be loaded relatively")
        matrix = matrix + relative_diagonal_loading * scale * np.eye(matrix.shape[0])
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalue_scale = float(np.max(np.abs(eigenvalues)))
    if eigenvalue_scale == 0.0:
        raise np.linalg.LinAlgError("zero covariance has no invertible noise model")
    if eigenvalues[0] / eigenvalue_scale < -1e-10:
        raise np.linalg.LinAlgError("covariance must be positive semidefinite")
    condition = np.linalg.cond(matrix)
    if not np.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError(
            "covariance is singular or ill-conditioned; use documented diagonal loading"
        )
    return matrix
