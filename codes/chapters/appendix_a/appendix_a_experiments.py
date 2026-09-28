"""E12-06..12: reproducible mathematical counterexamples for Appendix A.

All values are constructed teaching inputs, not measurements. The audio case
is mathematical PCM16 synthesis; no file is written by importing or running
this module. Run ``python -m codes.chapters.appendix_a.appendix_a_experiments`` for JSON.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.audio_samples import math_block_case
from codes.array_tutorial.math_foundations import (
    blockwise_circular_convolution, fft_overlap_add,
)


def run_experiments() -> dict:
    """Return stable exercise IDs, inputs, and JSON-safe intermediate results."""
    results = {}

    n, sample_rate = 8, 8000
    signed_frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    cosine = np.cos(2 * np.pi * np.arange(n) / n)
    spectrum = np.fft.fft(cosine)
    results['E12-06'] = {
        'sample_rate_hz': sample_rate, 'fft_length': n,
        'signed_frequency_hz_by_bin': signed_frequencies.tolist(),
        'real_cosine_nonzero_bins': [int(k) for k in np.flatnonzero(np.abs(spectrum) > 1e-12)],
        'real_cosine_nonzero_coefficients': [float(spectrum[k].real) for k in (1, 7)],
        'nyquist_bin': {'index': 4, 'frequency_magnitude_hz': 4000,
                        'one_discrete_bin_for_both_signs': True},
        'scope': 'signed representative of DFT frequencies; k=4 also denotes +4 kHz modulo 8 kHz',
    }

    vector = np.array([1, 1j], dtype=complex)
    results['E12-07'] = {
        'vector_real': vector.real.tolist(), 'vector_imag': vector.imag.tolist(),
        'hermitian_norm_squared': float(np.vdot(vector, vector).real),
        'transpose_square': float(np.dot(vector, vector).real),
        'scope': 'transpose without conjugation is not a complex squared norm',
    }

    x = np.array([1., 2., 3., 4.])
    h = np.array([1., .5])
    audio = math_block_case()
    results['E12-08'] = {
        'input': x.tolist(), 'filter': h.tolist(), 'block_size': 2,
        'linear_output': fft_overlap_add(x, h, 2).tolist(),
        'block_circular_output': blockwise_circular_convolution(x, h, 2).tolist(),
        'audio_case': {
            'block_size': audio['parameters']['block_size'],
            'pulse_positions_samples': audio['parameters']['pulse_positions_samples'],
            'first_correct_echo_sample': audio['parameters']['first_correct_echo_sample'],
            'first_wrong_wrap_sample': audio['parameters']['first_wrong_wrap_sample'],
            'float_analysis': audio['float_analysis'],
            'pcm_analysis': audio['pcm_analysis'],
        },
    }

    x1 = np.array([0., 1., 0.])
    x2 = np.array([1., 0., 0.])
    lags = np.arange(-2, 3)
    r12 = np.correlate(x1, x2, mode='full')
    r21 = np.correlate(x2, x1, mode='full')
    results['E12-09'] = {
        'x1': x1.tolist(), 'x2': x2.tolist(),
        'lags_samples': lags.tolist(), 'r12': r12.tolist(), 'r21': r21.tolist(),
        'r12_peak_lag_samples': int(lags[np.argmax(r12)]),
        'r21_peak_lag_samples': int(lags[np.argmax(r21)]),
        'boundary': 'zero extension outside each three-sample record, not circular correlation',
    }

    snapshots = np.array([[1., 0.], [0., 1.], [0., 0.]])
    mean = snapshots.mean(axis=1, keepdims=True)
    moment = snapshots @ snapshots.T / 2
    centered = (snapshots - mean) @ (snapshots - mean).T / 2
    results['E12-10'] = {
        'snapshots_columns': snapshots.tolist(), 'sample_mean': mean[:, 0].tolist(),
        'uncentered_second_moment': moment.tolist(), 'centered_covariance': centered.tolist(),
        'uncentered_rank': int(np.linalg.matrix_rank(moment)),
        'centered_rank': int(np.linalg.matrix_rank(centered)),
        'centered_eigenvalues': np.linalg.eigvalsh(centered).tolist(),
        'denominator': 2,
    }

    matrix = np.diag([1., 1e-4])
    rhs = np.array([1., 1e-4])
    results['E12-11'] = {
        'matrix': matrix.tolist(), 'example_rhs': rhs.tolist(),
        'condition_A': float(np.linalg.cond(matrix)),
        'condition_normal': float(np.linalg.cond(matrix.T @ matrix)),
        'recovered_x': np.linalg.solve(matrix, rhs).tolist(),
        'scope': '2-norm conditioning of a diagonal example, not a measured solver error',
    }

    target = np.diag([1., 0.])
    reference = np.array([1., 0.])
    singular_noise = np.diag([0., 1.])
    identity_noise = np.eye(2)
    mus = [1., 10., 1e6]
    singular_weights = []
    identity_weights = []
    for mu in mus:
        singular_weights.append(np.linalg.solve(target + mu * singular_noise,
                                                target @ reference).tolist())
        identity_weights.append(np.linalg.solve(target + mu * identity_noise,
                                               target @ reference).tolist())
    results['E12-12'] = {
        'target_covariance': target.tolist(), 'reference': reference.tolist(),
        'singular_noise_covariance': singular_noise.tolist(),
        'identity_noise_covariance': identity_noise.tolist(), 'mus': mus,
        'singular_noise_weights': singular_weights,
        'identity_noise_weights': identity_weights,
        'limiting_statement': 'with target energy in the noise nullspace, increasing mu does not force zero output',
    }
    return results


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
