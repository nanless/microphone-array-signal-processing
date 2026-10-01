"""Small FFT convolution baselines for Appendix A.

Both inputs are finite real one-dimensional arrays. The filter starts at lag
zero. ``fft_overlap_add`` returns the full *linear* convolution, including its
tail; ``blockwise_circular_convolution`` deliberately discards block history
and returns one block per input block. The latter is an error demonstration,
not a usable streaming filter. Neither function promises real-time operation.

Float64 FFT support is deliberately finite: each nonzero input's exponent
span is limited to 40 bits, pairwise nonzero products must be normal, and
intermediate FFT values must be finite. This conservative support policy is
not a guarantee of pointwise relative accuracy (cancellation still matters).
Unsupported scales raise ValueError rather than returning lost components.
"""

from __future__ import annotations

import numpy as np


def _inputs(signal: np.ndarray, taps: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    converted = []
    for name, raw in (("signal", signal), ("taps", taps)):
        try:
            values = np.asarray(raw)
            if values.dtype.kind not in 'iuf':
                raise ValueError(f"{name} must contain real numeric values, not bool/string/object")
            converted.append(np.asarray(values, dtype=float))
        except (TypeError, OverflowError) as error:
            raise ValueError(f"{name} cannot be represented as a real float64 vector") from error
    x, h = converted
    if x.ndim != 1 or h.ndim != 1 or not x.size or not h.size:
        raise ValueError("signal and taps must be non-empty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(h)):
        raise ValueError("signal and taps must be finite")
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if int(block_size) + h.size - 1 > np.iinfo(np.intp).max:
        raise ValueError("FFT length exceeds the supported integer range")
    nonzero = [np.abs(v[v != 0]) for v in (x, h)]
    if all(v.size for v in nonzero):
        for values in nonzero:
            exponents = np.frexp(values)[1]
            if int(exponents.max()) - int(exponents.min()) > 40:
                raise ValueError("nonzero input exponent span exceeds float64 FFT support (40 bits)")
        # A subnormal pair product can be lost before FFT accumulation even
        # if several such products have a representable positive final sum.
        minimum_exponent = sum(int(np.frexp(v.min())[1]) for v in nonzero)
        if minimum_exponent < -1020:
            raise ValueError("nonzero products below normal float64 FFT support")
    return x, h


def fft_overlap_add(signal: np.ndarray, taps: np.ndarray, block_size: int) -> np.ndarray:
    """Full linear convolution by FFT blocks; each tail overlaps the next.

    The FFT length is ``block_size + len(taps) - 1``. The last partial input
    block is zero-padded, and the returned length is ``len(signal)+len(taps)-1``.
    """
    x, h = _inputs(signal, taps, block_size)
    block_size = int(block_size)
    n_fft = block_size + h.size - 1
    filter_spectrum = np.fft.rfft(h, n_fft)
    if not np.all(np.isfinite(filter_spectrum)):
        raise ValueError("filter FFT exceeds float64 support")
    out = np.zeros(x.size + h.size - 1, dtype=float)
    for start in range(0, x.size, block_size):
        block = x[start:start + block_size]
        with np.errstate(over='ignore', invalid='ignore'):
            transformed = np.fft.rfft(block, n_fft)
            product = transformed * filter_spectrum
        if not np.all(np.isfinite(transformed)) or not np.all(np.isfinite(product)):
            raise ValueError("FFT multiplication exceeds float64 support")
        part = np.fft.irfft(product, n_fft)
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
    block_size = int(block_size)
    if h.size > block_size:
        raise ValueError("taps must fit into one circular-convolution block")
    filter_spectrum = np.fft.rfft(h, block_size)
    if not np.all(np.isfinite(filter_spectrum)):
        raise ValueError("filter FFT exceeds float64 support")
    out = np.empty_like(x)
    for start in range(0, x.size, block_size):
        block = x[start:start + block_size]
        with np.errstate(over='ignore', invalid='ignore'):
            transformed = np.fft.rfft(block, block_size)
            product = transformed * filter_spectrum
        if not np.all(np.isfinite(transformed)) or not np.all(np.isfinite(product)):
            raise ValueError("FFT multiplication exceeds float64 support")
        part = np.fft.irfft(product, block_size)
        out[start:start + block.size] = part[:block.size]
    if not np.all(np.isfinite(out)):
        raise ValueError("convolution exceeded floating-point range")
    return out
