"""Exact-size teaching controls for reference gain order and echo tails.

The path and gain are supplied truths, never estimates. No adaptation, audio
file generation, or production double-talk detection is performed here.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch06.core.aec import NLMSState
from codes.chapters.ch06.core.double_talk import LABELS, ncc_activity_states


def reference_gain_order() -> dict:
    """E06-40: compare gain before convolution with gain after prediction.

    Five samples include the complete one-sample echo tail. The physical source
    is zero in that last sample, while the current gain explicitly remains 2.
    """
    source = np.array([1., 1., 1., 1., 0.])
    gain = np.array([1., 1., 2., 2., 2.])
    path = np.array([1., .5])
    late = gain * source
    echo = np.convolve(late, path)[:source.size]
    early_prediction = np.convolve(source, path)[:source.size]
    wrong_prediction = gain * early_prediction
    state = NLMSState(2, initial_weights=path)
    late_residual, late_prediction = state.process(
        late, echo, freeze=np.ones(source.size, dtype=bool))
    terms = np.zeros((source.size, path.size))
    for n in range(source.size):
        for lag in range(path.size):
            if n >= lag:
                terms[n, lag] = path[lag] * (gain[n-lag] - gain[n]) * source[n-lag]
    return {'source': source.tolist(), 'gain': gain.tolist(), 'path': path.tolist(),
            'late_reference': late.tolist(), 'true_echo': echo.tolist(),
            'early_fixed_prediction': early_prediction.tolist(),
            'early_fixed_residual': (echo - early_prediction).tolist(),
            'wrong_after_gain_prediction': wrong_prediction.tolist(),
            'wrong_after_gain_residual': (echo - wrong_prediction).tolist(),
            'gain_difference_terms_current_first': terms.tolist(),
            'late_reference_prediction': late_prediction.tolist(),
            'late_reference_residual': late_residual.tolist(),
            'complete_tail_samples': 1,
            'scope': 'known time-varying gain and fixed FIR; all coefficients frozen; no gain or path estimation'}


def echo_tail_activity() -> dict:
    """E06-41: a silent current reference frame can coexist with pure echo.

    All twelve samples are valid, including the four-sample path tail and one
    further silent frame. The old reference history is retained by the FIR.
    """
    source = np.array([1., -1., 1., -1., 0., 0., 0., 0., 0., 0., 0., 0.])
    path = np.array([0., 0., 0., 0., .5])
    echo = np.convolve(source, path)[:source.size]
    states, ncc = ncc_activity_states(source, echo, frame_size=4,
                                      activity_rms=.1, coherence_threshold=.8)
    state = NLMSState(5, initial_weights=path)
    residual, prediction = state.process(source, echo,
                                         freeze=np.ones(source.size, dtype=bool))
    def rms(values):
        frames = values.reshape(-1, 4)
        centered = frames - frames.mean(axis=1, keepdims=True)
        return np.sqrt(np.mean(centered**2, axis=1)).tolist()
    return {'reference_frames': source.reshape(-1, 4).tolist(),
            'microphone_frames': echo.reshape(-1, 4).tolist(),
            'near_end_truth': np.zeros(source.size).tolist(), 'path': path.tolist(),
            'frame_sample_intervals': [[0, 4], [4, 8], [8, 12]],
            'reference_centered_rms': rms(source), 'microphone_centered_rms': rms(echo),
            'activity_rms': .1, 'coherence_threshold': .8,
            'operational_states': states.tolist(),
            'operational_labels': [LABELS[int(value)] for value in states],
            'absolute_ncc': ncc.tolist(),
            'known_path_prediction': prediction.tolist(),
            'known_path_prior_residual': residual.tolist(),
            'scope': 'physical playback stops; reference is not missing; finite-frame activity labels do not identify near-end components'}
