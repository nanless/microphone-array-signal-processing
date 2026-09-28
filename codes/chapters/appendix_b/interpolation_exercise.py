"""E13-02: fixed fractional-delay amplitude error, with independent PCM readback.

No files, network, randomness or experiments on import. Run as a module.
The ideal output is evaluated from a known continuous signal, not reconstructed.
"""
from __future__ import annotations

import json
import numpy as np

from codes.array_tutorial.audio_samples import interpolation_case, prepare_exports, read_pcm16


def run_exercises() -> dict:
    case = interpolation_case()
    files, groups = prepare_exports({'interpolation': case})
    fs = case['parameters']['sample_rate_hz']
    start, stop = case['parameters']['scoring_interval_samples']
    n = np.arange(start, stop)
    frequencies = np.array(case['parameters']['frequencies_hz'])
    # At alpha=.5, H=e^(-j*w/2)*cos(w/2); the frequencies are below Nyquist.
    expected_one = np.cos(np.pi * frequencies / fs)
    amplitudes = {}
    for name, (blob, _) in files.items():
        _, pcm = read_pcm16(blob)
        amplitudes[name[:-4]] = [float(2 * abs(np.sum(
            pcm[0, start:stop] * np.exp(-2j * np.pi * f * n / fs))) / len(n))
            for f in frequencies]
    measured = {}
    for output, reference in [('linear_half', 'ideal_half'), ('linear_twice', 'ideal_one')]:
        measured[output] = (np.array(amplitudes['interpolation_' + output]) /
                            amplitudes['interpolation_' + reference]).tolist()
    return {'E13-02': {
        'frequencies_hz': frequencies.tolist(), 'sample_rate_hz': fs,
        'scoring_interval_samples': [start, stop],
        'common_export_gain': groups['interpolation']['common_export_gain'],
        'analytic_one_pass_amplitude': expected_one.tolist(),
        'analytic_two_pass_amplitude': (expected_one**2).tolist(),
        'analytic_one_pass_db': (20*np.log10(expected_one)).tolist(),
        'analytic_two_pass_db': (40*np.log10(expected_one)).tolist(),
        'pcm_amplitudes': amplitudes, 'pcm_amplitude_ratios': measured,
        'limits': case['limits'],
    }}


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
