"""E08-30: two fixed constraint operations, using the book's unique STFT.

The rectangular four-point, hop-two model is a finite algebraic example, not
a trained separator or a general iterative consistency solver.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array
from codes.chapters.ch02.core.spectral import stft, istft


def _spectra(value, name):
    original = np.asarray(value)
    if original.dtype.kind not in "iufc":
        raise ValueError(f"{name} must contain numeric spectra")
    result = np.asarray(original, dtype=complex)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def mixture_projection(estimates, mixture, variances):
    """Positive-weight correction on shared-scale spectra (sources,F,T).

    Mixture is (F,T); variances have exactly the estimates' shape. Weights
    are given, not inferred confidences. This checks no reference-microphone
    identity, source count, real-waveform endpoints, or separation quality.
    """
    y = _spectra(estimates, "estimates")
    x = _spectra(mixture, "mixture")
    v = finite_real_array(variances, "variances")
    if y.ndim != 3 or min(y.shape) < 1 or x.shape != y.shape[1:] or v.shape != y.shape:
        raise ValueError("require estimates/variances (sources,F,T), mixture (F,T)")
    if np.any(v <= 0):
        raise ValueError("all variances must be strictly positive")
    scaled = v/v.max(axis=0)
    alpha = scaled/scaled.sum(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        result = y+alpha*(x-y.sum(axis=0))
    if not np.all(np.isfinite(result)):
        raise ValueError("projected spectra overflowed")
    return result


def rectangular_consistency(spectra):
    """Analysis-after-synthesis for two real four-point frames, hop two.

    Array shape is (sources,3,2); the real rFFT endpoints must be real. This
    deliberately fixes n_fft/window/hop/length/centering to E08-30's model.
    """
    y = _spectra(spectra, "spectra")
    if y.ndim != 3 or y.shape[1:] != (3, 2) or y.shape[0] < 1:
        raise ValueError("require (sources,3,2) two-frame rFFT spectra")
    if np.any(y[:, (0, 2), :].imag != 0):
        raise ValueError("real-waveform DC and Nyquist endpoints must be real")
    waveform = istft(y, n_fft=4, hop_length=2, window=np.ones(4), center=False, length=6)
    return stft(waveform, n_fft=4, hop_length=2, window=np.ones(4), center=False)


def run_experiment():
    x = stft(np.ones(6), n_fft=4, hop_length=2, window=np.ones(4), center=False)[0]
    raw = np.zeros((2, 3, 2), dtype=complex)
    raw[0, 0, 1] = 4  # rFFT of the second frame [1,1,1,1].
    v = np.empty(raw.shape)
    v[0, :, :] = [3, 1]
    v[1, :, :] = [1, 3]
    mix_then_p = rectangular_consistency(mixture_projection(raw, x, v))
    p_then_mix = mixture_projection(rectangular_consistency(raw), x, v)
    def describe(y):
        wave = istft(y, n_fft=4, hop_length=2, window=np.ones(4), center=False, length=6)
        return {'spectra': y, 'frame_samples': np.fft.irfft(y, n=4, axis=1).transpose(0, 2, 1),
                'waveforms': wave, 'sum_spectrum_max_error': float(np.max(abs(y.sum(0)-x))),
                'consistency_max_spectrum_error': float(np.max(abs(rectangular_consistency(y)-y)))}
    equal = np.ones(raw.shape)
    equal_a = rectangular_consistency(mixture_projection(raw, x, equal))
    equal_b = mixture_projection(rectangular_consistency(raw), x, equal)
    return {'mixture_waveform': np.ones(6), 'mixture_spectrum': x,
            'raw': describe(raw), 'positive_variances': v,
            'mix_then_consistency': describe(mix_then_p),
            'consistency_then_mix': describe(p_then_mix),
            'order_max_waveform_difference': float(np.max(abs(describe(mix_then_p)['waveforms']-
                                                                 describe(p_then_mix)['waveforms']))),
            'equal_weight_order_max_spectrum_difference': float(np.max(abs(equal_a-equal_b))),
            'scope': 'given nonzero raw spectra; no blind separation, inference or audio synthesis claim'}
