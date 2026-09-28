"""Small FFT convolution baselines for Appendix A.

Both inputs are finite real one-dimensional arrays. The filter starts at lag
zero. ``fft_overlap_add`` returns the full *linear* convolution, including its
tail; ``blockwise_circular_convolution`` deliberately discards block history
and returns one block per input block. The latter is an error demonstration,
not a usable streaming filter. Neither function promises real-time operation.
"""

from __future__ import annotations

import numpy as np


def _inputs(signal: np.ndarray, taps: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    for name, raw in (("signal", signal), ("taps", taps)):
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
    x = np.asarray(signal, dtype=float)
    h = np.asarray(taps, dtype=float)
    if x.ndim != 1 or h.ndim != 1 or not x.size or not h.size:
        raise ValueError("signal and taps must be non-empty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(h)):
        raise ValueError("signal and taps must be finite")
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    return x, h


def fft_overlap_add(signal: np.ndarray, taps: np.ndarray, block_size: int) -> np.ndarray:
    """Full linear convolution by FFT blocks; each tail overlaps the next.

    The FFT length is ``block_size + len(taps) - 1``. The last partial input
    block is zero-padded, and the returned length is ``len(signal)+len(taps)-1``.
    """
    x, h = _inputs(signal, taps, block_size)
    n_fft = block_size + h.size - 1
    filter_spectrum = np.fft.rfft(h, n_fft)
    out = np.zeros(x.size + h.size - 1, dtype=float)
    for start in range(0, x.size, block_size):
        block = x[start:start + block_size]
        part = np.fft.irfft(np.fft.rfft(block, n_fft) * filter_spectrum, n_fft)
        stop = min(start + n_fft, out.size)
        out[start:stop] += part[:stop - start]
    if not np.all(np.isfinite(out)):
        raise ValueError("convolution exceeded floating-point range")
    return out


def blockwise_circular_convolution(signal: np.ndarray, taps: np.ndarray, block_size: int) -> np.ndarray:
    """Deliberate counterexample: FFT each block circularly, then discard its tail.

    This requires ``len(taps) <= block_size`` and returns ``len(signal)``
    samples. For nonzero delayed taps it can place a response *before* the
    within-block excitation and omit the physically delayed response.
    """
    x, h = _inputs(signal, taps, block_size)
    if h.size > block_size:
        raise ValueError("taps must fit into one circular-convolution block")
    filter_spectrum = np.fft.rfft(h, block_size)
    out = np.empty_like(x)
    for start in range(0, x.size, block_size):
        block = x[start:start + block_size]
        part = np.fft.irfft(np.fft.rfft(block, block_size) * filter_spectrum, block_size)
        out[start:start + block.size] = part[:block.size]
    if not np.all(np.isfinite(out)):
        raise ValueError("convolution exceeded floating-point range")
    return out
