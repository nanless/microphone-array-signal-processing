"""Small, dependency-free STFT/iSTFT routines for the chapter examples."""

from __future__ import annotations

import numpy as np

from .conventions import validate_cft, validate_waveforms


def periodic_hann(length: int) -> np.ndarray:
    """Return the DFT-periodic Hann window used by the examples."""
    if not isinstance(length, (int, np.integer)) or length < 2:
        raise ValueError("window length must be an integer of at least two")
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def _window_array(window: str | np.ndarray, n_fft: int) -> np.ndarray:
    if isinstance(window, str):
        if window != "hann":
            raise ValueError("only the 'hann' named window is supported")
        result = periodic_hann(n_fft)
    else:
        result = np.asarray(window, dtype=float)
        if result.shape != (n_fft,):
            raise ValueError("window must contain n_fft samples")
    if not np.all(np.isfinite(result)) or not np.any(np.abs(result) > 0.0):
        raise ValueError("window must be finite and non-zero")
    return result


def stft(
    waveforms: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    window: str | np.ndarray = "hann",
    center: bool = True,
) -> np.ndarray:
    """Compute an rFFT STFT with output shape channels x frequency x frames."""
    signals = validate_waveforms(waveforms)
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("n_fft must be an integer of at least two")
    if not isinstance(hop_length, (int, np.integer)) or not 0 < hop_length <= n_fft:
        raise ValueError("hop_length must be in [1, n_fft]")
    analysis_window = _window_array(window, n_fft)
    pad = n_fft // 2 if center else 0
    padded = np.pad(signals, ((0, 0), (pad, pad)))
    if padded.shape[1] < n_fft:
        padded = np.pad(padded, ((0, 0), (0, n_fft - padded.shape[1])))
    remainder = (padded.shape[1] - n_fft) % hop_length
    if remainder:
        padded = np.pad(padded, ((0, 0), (0, hop_length - remainder)))
    frame_count = 1 + (padded.shape[1] - n_fft) // hop_length
    spectra = np.empty((signals.shape[0], n_fft // 2 + 1, frame_count), dtype=complex)
    for frame in range(frame_count):
        start = frame * hop_length
        segment = padded[:, start : start + n_fft] * analysis_window
        spectra[:, :, frame] = np.fft.rfft(segment, n=n_fft, axis=1)
    return spectra


def istft(
    spectra: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    window: str | np.ndarray = "hann",
    center: bool = True,
    length: int | None = None,
    denominator_floor: float = 1e-12,
) -> np.ndarray:
    """Invert :func:`stft` by weighted overlap-add.

    The synthesis divides by the accumulated squared window.  Samples whose
    denominator is zero are rejected instead of silently returning a corrupt
    boundary value.
    """
    coefficients = validate_cft(spectra)
    if coefficients.shape[1] != n_fft // 2 + 1:
        raise ValueError("frequency dimension does not match n_fft")
    if not isinstance(hop_length, (int, np.integer)) or not 0 < hop_length <= n_fft:
        raise ValueError("hop_length must be in [1, n_fft]")
    if length is not None and (not isinstance(length, (int, np.integer)) or length < 0):
        raise ValueError("length must be a non-negative integer")
    synthesis_window = _window_array(window, n_fft)
    output_length = n_fft + hop_length * (coefficients.shape[2] - 1)
    output = np.zeros((coefficients.shape[0], output_length), dtype=float)
    denominator = np.zeros(output_length, dtype=float)
    for frame in range(coefficients.shape[2]):
        start = frame * hop_length
        segment = np.fft.irfft(coefficients[:, :, frame], n=n_fft, axis=1)
        output[:, start : start + n_fft] += segment * synthesis_window
        denominator[start : start + n_fft] += synthesis_window**2
    pad = n_fft // 2 if center else 0
    stop = output_length - pad if center else output_length
    output = output[:, pad:stop]
    denominator = denominator[pad:stop]
    if length is not None:
        if length > output.shape[1]:
            raise ValueError("requested length exceeds the represented signal")
        output = output[:, :length]
        denominator = denominator[:length]
    if denominator.size and np.any(denominator <= denominator_floor):
        raise ValueError("window and hop leave samples without synthesis support")
    return output / denominator[None, :]
