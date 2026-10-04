"""Minimal DOA baselines used by Chapters 2--4.

These are clean-room teaching implementations, not copies of third-party
projects.  They favor explicit validation and formula-to-code correspondence.
"""

from __future__ import annotations

import math

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar, validate_cft, validate_frequencies
from codes.chapters.ch03.core.geometry import plane_wave_delays
from codes.chapters.ch04.core.covariance import _load_covariance, _validate_covariance, _scaled_covariance


def mdl_source_count(eigenvalues: np.ndarray, snapshots: int) -> tuple[int, np.ndarray]:
    """Return ``(selected_count, scores_for_k_0_through_M_minus_1)``.

    Wax--Kailath complex Gaussian, spatially white noise MDL (Chapter 4.9).
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
    channels, count, gaps = _source_count_tail_statistics(eigenvalues, snapshots)
    scores = np.empty(channels)
    for k in range(channels):
        scores[k] = count * (channels-k) * gaps[k] + 0.5 * k * (2 * channels - k) * math.log(count)
    if not np.all(np.isfinite(scores)):
        raise ValueError("MDL scores exceed floating-point range")
    return int(np.argmin(scores)), scores


def _source_count_tail_statistics(eigenvalues, snapshots):
    """Shared real-input validation and stable log AM/GM gaps; no loading."""
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
    gaps = np.empty(channels)
    for k in range(channels):
        tail = logs[k:]
        shifted = tail - tail[0]
        arithmetic_log = math.log(math.fsum(math.exp(x) for x in shifted) / tail.size)
        geometric_log = math.fsum(shifted) / tail.size
        gaps[k] = max(0.0, arithmetic_log - geometric_log)
    return channels, count, gaps


def aic_source_count(eigenvalues: np.ndarray, snapshots: int) -> tuple[int, np.ndarray]:
    """Complex, iid, known-zero-mean white-noise AIC; same domain as MDL.

    AIC(k)=2*N*(M-k)*log(AM/GM)+2*k*(2*M-k), omitting a k-independent
    parameter constant. Scores are not probabilities. See Wax--Kailath,
    ICASSP 1984, equations (10)--(15), doi:10.1109/ICASSP.1984.1172389.
    """
    channels, count, gaps = _source_count_tail_statistics(eigenvalues, snapshots)
    scores = np.array([2*count*(channels-k)*gaps[k]+2*k*(2*channels-k)
                       for k in range(channels)])
    if not np.all(np.isfinite(scores)):
        raise ValueError("AIC scores exceed floating-point range")
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
    FFT padding covers the unweighted linear-correlation support.  Each input
    is scaled by its own peak before the FFT; ``epsilon`` is a fraction of the
    largest resulting cross-spectrum magnitude.  This keeps the PHAT result
    invariant to a positive nonzero overall gain on either channel. A polarity
    reversal changes the signed correlation and is not covered by this invariance.
    PHAT's
    nonlinear spectral normalization does not retain that finite support:
    the result is a sampled periodic inverse transform restricted to physical
    lags, and changing the FFT length can change its values.  Optional
    three-point interpolation is local and does not change the lag search range.
    ``max_tau`` is an inclusive bound on the represented lag times in seconds;
    no absolute floating-point tolerance widens this physical search interval.
    """
    sample_rate = finite_real_scalar(sample_rate, "sample_rate")
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if max_tau is not None:
        max_tau = finite_real_scalar(max_tau, "max_tau")
    first = finite_real_array(x1, "x1")
    second = finite_real_array(x2, "x2")
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
    first_scale = np.max(np.abs(first))
    second_scale = np.max(np.abs(second))
    if first_scale == 0.0 or second_scale == 0.0:
        raise ValueError("GCC-PHAT is undefined for a zero-energy pair")
    cross = np.fft.fft(first / first_scale, n_fft) * np.fft.fft(second / second_scale, n_fft).conj()
    magnitude = np.abs(cross)
    maximum = np.max(magnitude)
    if not np.isfinite(maximum) or maximum == 0.0:
        raise ValueError("GCC-PHAT is undefined for a zero-energy pair")
    protection = epsilon * maximum
    if not np.isfinite(protection) or protection == 0.0:
        raise ValueError("relative PHAT protection is not representable")
    denominator = np.maximum(magnitude, protection)
    if np.min(denominator) >= np.finfo(float).tiny:
        cross /= denominator  # Preserve the ordinary numerical path.
    else:
        cross = cross.real / denominator + 1j * (cross.imag / denominator)
    circular = np.fft.ifft(cross).real
    negative_lags = circular[-(second.size - 1) :] if second.size > 1 else circular[:0]
    correlation = np.concatenate((negative_lags, circular[: first.size]))
    lags = np.arange(-(second.size - 1), first.size, dtype=float)
    if max_tau is not None:
        if not np.isfinite(max_tau) or max_tau < 0.0:
            raise ValueError("max_tau must be finite and non-negative")
        with np.errstate(over="ignore", divide="ignore"):
            keep = np.abs(lags / sample_rate) <= max_tau
        correlation = correlation[keep]
        lags = lags[keep]
        if lags.size == 0:
            raise ValueError("max_tau leaves no candidate lag")
    if not np.all(np.isfinite(correlation)):
        raise ValueError("GCC-PHAT correlation is not finite")
    with np.errstate(over="ignore", divide="ignore"):
        seconds = lags / sample_rate
    if not np.all(np.isfinite(seconds)):
        raise ValueError("lag time axis exceeds floating-point range")
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
    """Scan far-field azimuths by averaging PHAT-normalized microphone pairs.

    Each channel is scaled before forming cross spectra. ``epsilon`` is
    relative to each pair's maximum cross-spectrum magnitude. A recording
    with no usable microphone pair has no direction and raises ``ValueError``.
    """
    epsilon = finite_real_scalar(epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    x = validate_cft(spectra)
    frequencies = validate_frequencies(frequencies_hz)
    candidates = np.atleast_1d(finite_real_array(candidate_azimuths_rad, "candidate_azimuths_rad"))
    if frequencies.size != x.shape[1]:
        raise ValueError("frequencies_hz does not match the STFT frequency axis")
    if candidates.ndim != 1 or not np.all(np.isfinite(candidates)) or candidates.size < 1:
        raise ValueError("candidate_azimuths_rad must be finite and non-empty")
    delays = plane_wave_delays(
        positions, candidates, sound_speed=sound_speed
    )
    if delays.shape[1] != x.shape[0]:
        raise ValueError("positions and spectra have different channel counts")
    # Use componentwise scales: abs(complex) itself can overflow although
    # both components of the input are finite.
    scales = np.max(np.maximum(np.abs(x.real), np.abs(x.imag)), axis=(1, 2))
    normalized = np.zeros_like(x)
    active = scales > 0.0
    normalized[active] = x[active].real / scales[active, None, None] + 1j * (
        x[active].imag / scales[active, None, None]
    )
    score = np.zeros(candidates.size, dtype=float)
    pair_count = 0
    informative_pairs = 0
    for first in range(x.shape[0]):
        for second in range(first + 1, x.shape[0]):
            cross = normalized[first] * normalized[second].conj()
            magnitude = np.abs(cross)
            maximum = np.max(magnitude)
            if maximum == 0.0:
                pair_count += 1
                continue
            if np.any(magnitude[frequencies > 0.0] > epsilon * maximum):
                informative_pairs += 1
            threshold = epsilon * maximum
            if not np.isfinite(threshold) or threshold == 0.0:
                raise ValueError("relative PHAT gate is not representable")
            usable = magnitude > threshold
            phat = np.zeros_like(cross)
            # Do not evaluate a zero or tiny unused denominator.
            if np.min(magnitude[usable], initial=1.0) >= np.finfo(float).tiny:
                phat[usable] = cross[usable] / magnitude[usable]
            else:
                phat[usable] = (cross.real[usable]/magnitude[usable]
                                + 1j*(cross.imag[usable]/magnitude[usable]))
            mean_cross = np.mean(phat, axis=1)
            predicted = delays[:, first] - delays[:, second]
            phase = np.exp(2.0j * np.pi * predicted[:, None] * frequencies[None, :])
            score += np.real(phase @ mean_cross)
            pair_count += 1
    if pair_count == 0:
        raise ValueError("SRP-PHAT needs at least two microphones")
    if informative_pairs == 0:
        raise ValueError("SRP-PHAT has no informative microphone pair")
    if not np.all(np.isfinite(score)):
        raise ValueError("SRP-PHAT scores are not finite for this geometry and frequency grid")
    return score / (pair_count * frequencies.size)


def _steering_rows(steering: np.ndarray, channels: int) -> np.ndarray:
    candidates = np.asarray(steering, dtype=complex)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] != channels:
        raise ValueError("steering must have shape candidates x channels")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("steering contains NaN or infinity")
    if np.any(np.max(np.maximum(abs(candidates.real), abs(candidates.imag)), axis=1) == 0):
        raise ValueError("steering rows must be nonzero")
    return candidates


def bartlett_spectrum(covariance: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """Evaluate ``a.H @ R @ a`` for each steering-vector row."""
    matrix = _validate_covariance(covariance)
    candidates = _steering_rows(steering, matrix.shape[0])
    values = np.einsum("km,mn,kn->k", candidates.conj(), matrix, candidates)
    if not np.all(np.isfinite(values)):
        raise ValueError("Bartlett scores exceed floating-point range")
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
    with np.errstate(over="ignore", invalid="ignore"):
        solved = np.linalg.solve(matrix, candidates.T)
        denominator = np.einsum("km,mk->k", candidates.conj(), solved)
    spectrum_scale = 1.0
    if not np.all(np.isfinite(solved)) or not np.all(np.isfinite(denominator)):
        # A tiny covariance can have an unrepresentable inverse, while the
        # final Capon power is representable. Solve a scaled system instead.
        peak = float(np.max(np.maximum(abs(matrix.real), abs(matrix.imag))))
        with np.errstate(over="ignore", invalid="ignore"):
            solved = np.linalg.solve(_scaled_covariance(matrix, peak), candidates.T)
            denominator = np.einsum("km,mk->k", candidates.conj(), solved)
        spectrum_scale = peak
    if not np.all(np.isfinite(solved)) or not np.all(np.isfinite(denominator)):
        raise ValueError("Capon solve or denominator exceeds floating-point range")
    # The quadratic form scales inversely with covariance power, so the
    # imaginary round-off tolerance must use its own real-valued scale.
    if (np.any(denominator.real <= 0.0)
            or np.any(np.abs(denominator.imag) > 1e-8 * np.abs(denominator.real))):
        raise np.linalg.LinAlgError("Capon denominator is not positive real")
    with np.errstate(over="ignore", invalid="ignore"):
        spectrum = spectrum_scale / denominator.real
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0.0):
        raise ValueError("Capon spectrum exceeds floating-point range")
    return spectrum


def music_spectrum(
    covariance: np.ndarray,
    steering: np.ndarray,
    *,
    source_count: int,
    denominator_floor: float = 1e-15,
) -> np.ndarray:
    """Evaluate the narrowband MUSIC pseudospectrum using ``numpy.linalg.eigh``."""
    denominator_floor = finite_real_scalar(denominator_floor, "denominator_floor")
    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")
    matrix = _validate_covariance(covariance)
    channels = matrix.shape[0]
    if isinstance(source_count, (bool, np.bool_)) or not isinstance(source_count, (int, np.integer)) or not 0 < source_count < channels:
        raise ValueError("source_count must be in [1, channels - 1]")
    scale = float(np.max(np.maximum(np.abs(matrix.real), np.abs(matrix.imag))))
    if scale == 0.0:
        raise ValueError("MUSIC has no signal or noise energy from which to form a subspace")
    candidates = _steering_rows(steering, channels)
    eigenvalues, eigenvectors = np.linalg.eigh(_scaled_covariance(matrix, scale))
    if eigenvalues[0] < -1e-10 * max(1.0, float(np.max(np.abs(eigenvalues)))):
        raise np.linalg.LinAlgError("covariance must be positive semidefinite")
    noise = eigenvectors[:, : channels - source_count]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        projection = candidates.conj() @ noise
        denominator = np.sum(np.abs(projection) ** 2, axis=1)
        spectrum = 1.0 / np.maximum(denominator, denominator_floor)
    if (not np.all(np.isfinite(denominator)) or not np.all(np.isfinite(spectrum))
            or np.any(spectrum <= 0.0)):
        raise ValueError("MUSIC projection or spectrum exceeds floating-point range")
    return spectrum


def esprit_ula(
    covariance: np.ndarray,
    *,
    source_count: int,
    spacing_m: float,
    frequency_hz: float,
    sound_speed: float = 343.0,
    alias_tolerance: float = 1e-10,
) -> np.ndarray:
    """Estimate principal-branch ULA broadside azimuths with LS ESPRIT.

    The selected eigenspace must be numerically separated from its complement,
    and both shifted subarrays must retain its rank. These checks detect some
    degeneracies; they do not establish the true source count or whiten noise.
    For spacing > wavelength/2, physical directions can share a spatial phase:
    the returned principal-branch representative can be aliased; this is not
    disambiguation.
    At half-wavelength spacing the two endfire endpoints also coincide.
    """
    matrix = _validate_covariance(covariance)
    channels = matrix.shape[0]
    if channels < 2 or isinstance(source_count, (bool, np.bool_)) or not isinstance(source_count, (int, np.integer)):
        raise ValueError("ESPRIT needs at least two channels and an integer source_count")
    if not 0 < source_count < channels:
        raise ValueError("source_count must be in [1, channels - 1]")
    spacing_m = finite_real_scalar(spacing_m, "spacing_m")
    frequency_hz = finite_real_scalar(frequency_hz, "frequency_hz")
    sound_speed = finite_real_scalar(sound_speed, "sound_speed")
    alias_tolerance = finite_real_scalar(alias_tolerance, "alias_tolerance")
    if (not np.all(np.isfinite([spacing_m, frequency_hz, sound_speed]))
            or spacing_m <= 0.0 or frequency_hz <= 0.0 or sound_speed <= 0.0):
        raise ValueError("spacing, frequency, and sound speed must be positive")
    if not np.isfinite(alias_tolerance) or alias_tolerance < 0.0:
        raise ValueError("alias_tolerance must be finite and non-negative")
    scale = float(np.max(np.maximum(np.abs(matrix.real), np.abs(matrix.imag))))
    if scale == 0.0:
        raise np.linalg.LinAlgError("zero covariance has no signal subspace")
    eigenvalues, eigenvectors = np.linalg.eigh(_scaled_covariance(matrix, scale))
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -1e-10 * spectral_scale:
        raise np.linalg.LinAlgError("covariance must be positive semidefinite")
    tolerance = 100 * np.finfo(float).eps * channels * spectral_scale
    if (eigenvalues[-source_count] <= tolerance
            or eigenvalues[-source_count] - eigenvalues[-source_count-1] <= tolerance):
        raise np.linalg.LinAlgError("selected signal dimension has no separated positive eigenspace")
    signal = eigenvectors[:, -source_count:]
    first, second = signal[:-1], signal[1:]
    if np.linalg.matrix_rank(first) < source_count or np.linalg.matrix_rank(second) < source_count:
        raise np.linalg.LinAlgError("shifted signal subarray is rank deficient")
    transform, *_ = np.linalg.lstsq(first, second, rcond=None)
    modes = np.linalg.eigvals(transform)
    if np.any(np.abs(modes) <= 100 * np.finfo(float).eps * max(1., np.linalg.norm(transform, 2))):
        raise np.linalg.LinAlgError("zero shift mode has no spatial phase")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sine = np.angle(modes) * ((sound_speed / frequency_hz) / spacing_m) / (2.0 * np.pi)
    if not np.all(np.isfinite(sine)) or np.any(np.abs(sine) > 1.0 + alias_tolerance):
        raise ValueError("principal spatial phase lies outside the physical arcsine domain")
    return np.sort(np.arcsin(np.clip(sine, -1.0, 1.0)).real)
