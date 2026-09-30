"""Finite-window convolution counterexamples, not measured room responses.

The FFT has the NumPy backward normalization; spectra use the negative forward
sign. Analysis and synthesis use the same periodic Hann. Centered boundary
padding and the complete retained output support are specified separately.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array
from codes.chapters.ch02.core.spectral import istft, periodic_hann, stft

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
DELAY_SAMPLES = 320
OUTPUT_SAMPLES = SOURCE_SAMPLES + DELAY_SAMPLES
N_FFT = 512
HOP_LENGTH = 128
FREQUENCIES = (440, 997, 1733, 2819)
BURST_INTERVALS = ((3200, 6400), (11200, 15200), (19200, 23200), (26400, 29600))
CASE_NAMES = ("stft_roundtrip", "full_convolution", "framewise_mtf")


def synthesis_support(length: int, n_fft: int, hop_length: int) -> tuple[np.ndarray, list[int]]:
    """Centered, end-padded frame starts and squared-window sums on retained data."""
    if (isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer))
            or length < 1):
        raise ValueError("length must be a positive integer")
    if (isinstance(n_fft, (bool, np.bool_)) or not isinstance(n_fft, (int, np.integer))
            or n_fft < 2):
        raise ValueError("n_fft must be an integer of at least two")
    if (isinstance(hop_length, (bool, np.bool_))
            or not isinstance(hop_length, (int, np.integer)) or not 0 < hop_length <= n_fft):
        raise ValueError("hop_length must be an integer in [1,n_fft]")
    pad = n_fft // 2
    represented = max(length + 2 * pad, n_fft)
    represented += (-(represented - n_fft)) % hop_length
    window = periodic_hann(n_fft)
    denominator = np.zeros(represented)
    starts = list(range(-pad, represented - n_fft - pad + 1, hop_length))
    for start in starts:
        denominator[start + pad:start + pad + n_fft] += window**2
    return denominator[pad:pad + length], starts


def framewise_filter(signal: np.ndarray, impulse_response: np.ndarray, *,
                     n_fft: int, hop_length: int, remove_circular_folding: bool = False) -> np.ndarray:
    """Multiply each STFT frame by H, then WOLA on the original window support.

    The optional control uses a long enough FFT for the *frame's* full linear
    convolution, but keeps only its first n_fft entries before WOLA. Thus it
    removes circular folding without implementing cross-frame/band filtering.
    Callers pad the input to the full desired output length before analysis;
    this function never claims that retaining only the source length keeps a
    physical filter tail. Only mono finite real inputs and h.size <= n_fft.
    """
    x = finite_real_array(signal, "signal")
    h = finite_real_array(impulse_response, "impulse_response")
    if x.ndim != 1 or not x.size or h.ndim != 1 or not h.size:
        raise ValueError("signal and impulse_response must be nonempty vectors")
    # stft validates n_fft and hop_length before any filter FFT can crop h.
    spectrum = stft(x, n_fft=n_fft, hop_length=hop_length, center=True)
    if h.size > n_fft:
        raise ValueError("filter longer than the frame would be silently cropped")
    if remove_circular_folding:
        frame = np.fft.irfft(spectrum, n=n_fft, axis=1)
        linear_fft_length = 1 << int(n_fft + h.size - 2).bit_length()
        full = np.fft.irfft(np.fft.rfft(frame, n=linear_fft_length, axis=1)
                            * np.fft.rfft(h, n=linear_fft_length)[None, :, None],
                            n=linear_fft_length, axis=1)
        modified = np.fft.rfft(full[:, :n_fft, :], n=n_fft, axis=1)
    else:
        modified = spectrum * np.fft.rfft(h, n=n_fft)[None, :, None]
    return istft(modified, n_fft=n_fft, hop_length=hop_length,
                 center=True, length=x.size)[0]


def finite_window_example() -> dict:
    """E02-16: all six output samples and both distinct approximation errors."""
    source = np.array([1., 2., 3., 4.])
    h = np.array([1., 0., .5])
    padded = np.pad(source, (0, h.size-1))
    window = periodic_hann(4)
    denominator, starts = synthesis_support(padded.size, 4, 2)
    spectrum = stft(padded, n_fft=4, hop_length=2)
    frames = []
    for start in starts:
        z = np.array([padded[q] if 0 <= q < padded.size else 0.
                      for q in range(start, start+4)]) * window
        circular = np.fft.irfft(np.fft.rfft(z)*np.fft.rfft(h, n=4), n=4)
        linear = np.fft.irfft(np.fft.rfft(z, n=8)*np.fft.rfft(h, n=8), n=8)[:6]
        frames.append({"start_sample": start, "windowed_frame": z.tolist(),
                       "circular4": circular.tolist(), "linear8_full_frame_tail": linear.tolist(),
                       "linear8_first4": linear[:4].tolist()})
    exact = np.convolve(source, h, mode="full")
    circ = framewise_filter(padded, h, n_fft=4, hop_length=2)
    linear = framewise_filter(padded, h, n_fft=4, hop_length=2, remove_circular_folding=True)
    return {"source": source.tolist(), "padded_source": padded.tolist(), "rir": h.tolist(),
            "periodic_hann": window.tolist(), "n_fft": 4, "control_fft_length": 8,
            "hop_length": 2, "window_start_samples": starts,
            "window_squared_sum": denominator.tolist(), "frames": frames,
            "full_linear_output": exact.tolist(), "framewise_fft4_output": circ.tolist(),
            "framewise_fft8_cropped_output": linear.tolist(),
            "identity_roundtrip": istft(spectrum, n_fft=4, hop_length=2, length=6)[0].tolist(),
            "circular_folding_difference": (circ-linear).tolist(),
            "remaining_window_model_difference": (linear-exact).tolist(),
            "limits": "The 8-point control removes circular folding, but truncates each frame to its original four window samples. It is not a complete convolution filter bank; remaining error includes discarded frame tails and window/shift coupling. Supported unmodified WOLA reconstructs the padded source."}


def build_convolution_audio() -> dict:
    """One deterministic intermittent source; equal lengths and export gain 1."""
    index = np.arange(SOURCE_SAMPLES)
    source = sum(.06 * np.sin(2*np.pi*frequency*index/SAMPLE_RATE)
                 for frequency in FREQUENCIES)
    envelope = np.zeros(SOURCE_SAMPLES)
    for start, stop in BURST_INTERVALS:
        q = np.arange(stop-start)
        envelope[start:stop] = np.minimum(1., np.minimum(q/320., (stop-start-1-q)/320.))
    source *= envelope
    padded = np.pad(source, (0, DELAY_SAMPLES))
    h = np.zeros(DELAY_SAMPLES+1)
    h[0], h[-1] = 1., .5
    exact = np.convolve(source, h, mode="full")
    spectrum = stft(padded, n_fft=N_FFT, hop_length=HOP_LENGTH)
    outputs = {
        "stft_roundtrip": istft(spectrum, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                length=OUTPUT_SAMPLES)[0],
        "full_convolution": exact,
        "framewise_mtf": framewise_filter(padded, h, n_fft=N_FFT, hop_length=HOP_LENGTH),
    }
    denominator, starts = synthesis_support(OUTPUT_SAMPLES, N_FFT, HOP_LENGTH)
    return {"source": padded, "rir": h, "signals": outputs,
            "synthesis_denominator": denominator, "frame_starts": starts}


def measure_waveform(signal: np.ndarray, reference: np.ndarray) -> dict:
    """Full retained-support errors; no fitted gain, shift, centering or cropping."""
    x, ref = finite_real_array(signal, "signal"), finite_real_array(reference, "reference")
    if x.ndim != 1 or x.size != OUTPUT_SAMPLES or ref.shape != x.shape:
        raise ValueError("Expected two real mono 32320-sample waveforms")
    difference = x-ref
    with np.errstate(over="ignore", invalid="ignore"):
        energy, reference_energy, error_energy = (float(np.sum(v*v)) for v in (x, ref, difference))
    if not np.isfinite([energy, reference_energy, error_energy]).all() or reference_energy <= 0:
        raise ValueError("Finite nonzero reference energy is required")
    return {"interval_samples": [0, OUTPUT_SAMPLES], "denominator_samples": OUTPUT_SAMPLES,
            "duration_s": OUTPUT_SAMPLES/SAMPLE_RATE, "energy": energy,
            "mean_square": energy/OUTPUT_SAMPLES, "reference_energy": reference_energy,
            "error_energy": error_energy, "error_mean_square": error_energy/OUTPUT_SAMPLES,
            "relative_error_energy": error_energy/reference_energy,
            "max_abs_error": float(np.max(np.abs(difference)))}
