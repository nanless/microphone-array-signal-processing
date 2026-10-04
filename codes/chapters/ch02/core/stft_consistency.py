"""E02-20: endpoint-valid real spectra can violate shared-sample consistency.

Only the existing chapter-2 analysis/synthesis implementation is used. This is
one prescribed complex-spectrum projection, not iterative magnitude recovery,
mixture consistency, source separation or an audio-quality benchmark.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.spectral import istft, stft


def consistency_example() -> dict:
    """Three centered rectangular frames of four samples, with hop two.

    All DC/Nyquist coefficients are real. The middle frame demands +1/-1 at
    samples for which its neighboring frame demands zero. WOLA averages these
    incompatible requirements. Full-DFT distances use one-sided [1,2,1]
    weights, not the unweighted Euclidean metric on stored rFFT coefficients.
    """
    n_fft, hop, length = 4, 2, 4
    window = np.ones(n_fft)
    modified = np.zeros((1, 3, 3), dtype=complex)
    modified[0, 1, 1] = 2.
    original = modified.copy()
    inverse_frames = np.fft.irfft(modified, n=n_fft, axis=1)
    waveform = istft(modified, n_fft=n_fft, hop_length=hop,
                     window=window, center=True, length=length)
    projected = stft(waveform, n_fft=n_fft, hop_length=hop, window=window, center=True)
    second_waveform = istft(projected, n_fft=n_fft, hop_length=hop,
                           window=window, center=True, length=length)
    second_projection = stft(second_waveform, n_fft=n_fft, hop_length=hop,
                             window=window, center=True)
    weights = np.array([1., 2., 1.])[None, :, None]
    norm = lambda coefficients: float(np.sum(weights*np.abs(coefficients)**2))
    return {"n_fft": n_fft, "hop_length": hop, "retained_length": length,
            "center": True, "window": window.tolist(), "frame_start_samples": [-2, 0, 2],
            "modified_frames_real": modified[0].T.real.tolist(),
            "modified_frames_imag": modified[0].T.imag.tolist(),
            "inverse_frames": inverse_frames[0].T.tolist(),
            "retained_window_squared_sum": [2., 2., 2., 2.],
            "synthesized_waveform": waveform[0].tolist(),
            "reanalysed_frames_real": projected[0].T.real.tolist(),
            "reanalysed_frames_imag": projected[0].T.imag.tolist(),
            "full_dft_weights": [1, 2, 1], "full_dft_modified_energy": norm(modified),
            "full_dft_projected_energy": norm(projected),
            "full_dft_residual_energy": norm(modified-projected),
            "relative_full_dft_residual_energy": norm(modified-projected)/norm(modified),
            "second_projection_max_abs_difference": float(np.max(np.abs(second_projection-projected))),
            "input_unchanged": bool(np.array_equal(original, modified)),
            "limits": "Known rectangular-window overlap conflict. NOLA and real endpoints do not imply STFT consistency; no source or quality improvement is measured."}
