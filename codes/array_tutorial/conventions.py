"""Shared array and shape conventions for the book's NumPy examples.

The implementation follows the forward Fourier transform
``exp(-1j * 2*pi*f*t)``.  Directions point from the array towards the source,
and a multichannel STFT is always arranged as channels x frequency x frames.
"""

from __future__ import annotations

import numpy as np


SPEED_OF_SOUND = 343.0
FOURIER_EXPONENT_SIGN = -1


def validate_waveforms(waveforms: np.ndarray) -> np.ndarray:
    """Return finite real waveforms with shape ``channels x samples``."""
    array = np.asarray(waveforms, dtype=float)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("waveforms must have shape channels x samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("waveforms contain NaN or infinity")
    return array


def validate_cft(spectra: np.ndarray) -> np.ndarray:
    """Return finite complex spectra with shape ``channels x frequency x frames``."""
    array = np.asarray(spectra, dtype=complex)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError("spectra must have shape channels x frequency x frames")
    if not np.all(np.isfinite(array)):
        raise ValueError("spectra contain NaN or infinity")
    return array


def validate_positions(positions: np.ndarray) -> np.ndarray:
    """Validate microphone coordinates with shape ``channels x dimension``."""
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] not in (2, 3):
        raise ValueError("positions must have shape channels x 2 or channels x 3")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions contain NaN or infinity")
    return array


def validate_frequencies(frequencies_hz: np.ndarray) -> np.ndarray:
    """Validate a non-negative one-dimensional physical-frequency grid."""
    array = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    if array.ndim != 1 or array.size < 1:
        raise ValueError("frequencies_hz must be one-dimensional and non-empty")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("frequencies_hz must be finite and non-negative")
    return array


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    """Remove round-off asymmetry from the last two axes of a matrix array."""
    array = np.asarray(matrix, dtype=complex)
    if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
        raise ValueError("matrix must be square on its last two axes")
    return 0.5 * (array + np.swapaxes(array.conj(), -1, -2))
