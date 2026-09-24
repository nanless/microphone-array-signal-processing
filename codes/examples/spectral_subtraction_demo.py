"""E10-13: power spectral subtraction, a hand calculation, and listening data.

Run with ``python -m codes.examples.spectral_subtraction_demo``.  No files are
written; WAV exports are generated separately by generate_audio_samples.py.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.audio_samples import build_cases
from codes.array_tutorial.noise_suppression import power_spectral_subtraction


def run_demo() -> dict:
    # The first column is the known noise-only frame.  The second is a
    # noisy target frame.  Both have phase zero in this tiny calculation.
    observed = np.array([[2. + 0j, 3. + 0j, 1. + 0j]])
    soft, noise = power_spectral_subtraction(observed, np.array([0]), floor_ratio=.04)
    zero, _ = power_spectral_subtraction(observed, np.array([0]), floor_ratio=0.)
    case = build_cases()['spectral_subtraction']
    signals = case['signals']
    preamble = slice(0, case['parameters']['noise_only_samples'])
    rms = lambda name: float(np.sqrt(np.mean(signals[name][preamble] ** 2)))
    return {
        'exercise_id': 'E10-13',
        'hand': {'noise_power': float(noise[0]), 'noisy_bin_powers': [9., 1.],
                 'floor_ratio': .04, 'output_powers': [float(abs(soft[0, 1]) ** 2),
                                                       float(abs(soft[0, 2]) ** 2)],
                 'output_amplitudes': [float(abs(soft[0, 1])), float(abs(soft[0, 2]))],
                 'zero_floor_low_bin_amplitude': float(abs(zero[0, 2]))},
        'audio': {'sample_rate_hz': 16000, 'samples': len(signals['spectral_clean']),
                  'noise_only_samples': case['parameters']['noise_only_samples'],
                  'noise_only_frame_count': len(case['parameters']['noise_only_stft_frame_indices']),
                  'noise_only_rms': {name: rms(name) for name in
                                     ('spectral_noisy', 'spectral_floor04', 'spectral_floor00')},
                  'interpretation': 'single deterministic synthetic example, not a speech-quality score'},
    }


if __name__ == '__main__':
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
