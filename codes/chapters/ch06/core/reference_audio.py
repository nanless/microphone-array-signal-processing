"""E06-42: known playback gain, causal reference location and pure echo tail.

The true FIR initializes frozen NLMS states. This is an oracle prediction
control, not adaptive training, unknown-path estimation or device audio.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch06.core.aec import NLMSState
from codes.chapters.ch06.core.double_talk import LABELS, ncc_activity_states

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
SAMPLES = 33600
WINDOWS = {'step': (16000, 17600), 'stable': (19200, 28800), 'tail': (32000, 33600)}
FILE_NAMES = {name: 'reference_'+name+'.wav' for name in
              ('early', 'late', 'echo', 'early_residual', 'wrong_gain_residual', 'late_residual')}
LIMITS = ('Original deterministic two-tone mathematical control, not speech, room or device audio. '
          'Known FIR coefficients initialize frozen NLMS; no training, delay estimation or gain fitting. '
          'No near-end signal or noise exists. A current-frame near_only NCC label on a pure echo tail '
          'does not establish near-end speech. Full 1600-sample tail is retained. Float model components '
          'and actual PCM total powers are separate; no clean ERLE or industrial success claim.')


def parameters() -> dict:
    return {'exercise_id': 'E06-42', 'sample_rate_hz': 16000, 'source_samples': 32000,
            'samples_per_channel': 33600, 'convolution_tail_samples': 1600,
            'frequencies_hz': [500.0, 730.0], 'source_amplitudes': [0.08, 0.06],
            'source_model': '0.08*cos(500Hz)+0.06*sin(730Hz); absolute clock, period 1600',
            'fade_samples': 320, 'fade': 'squared sine, endpoints included, common envelope',
            'playback_gain_intervals': [[0, 16000, 1.0], [16000, 33600, 2.0]],
            'early_reference': 'source before time-varying playback gain, zero after 32000',
            'late_reference': 'gain times source before the fixed acoustic FIR, zero after 32000',
            'true_path_delays_samples': [0, 1600], 'true_path_coefficients': [0.7, 0.35],
            'filter_length': 1601, 'initial_weights': 'known true FIR, current sample first',
            'initial_history': '1600 zeros', 'freeze': 'all 33600 samples; history still advances',
            'nlms_step_size': 0.5, 'nlms_epsilon': 1e-8,
            'wrong_gain_prediction': 'current gain times fixed early-reference prediction, tail gain remains 2',
            'common_export_gain': 1.0, 'channel_order': ['mono'],
            'scoring_intervals_samples': {k: list(v) for k, v in WINDOWS.items()},
            'scoring_cycles_per_frequency': {'step': [50, 73], 'stable': [300, 438]},
            'tail_power': 'finite known enveloped samples, not a constant-envelope cycle average',
            'ncc_frame_samples': 160, 'ncc_activity_rms': 0.001, 'ncc_coherence_threshold': 0.85,
            'ncc_input': 'actual decoded late-reference and true-echo PCM',
            'ncc_availability': 'frame end; same-frame use requires a complete frame buffer',
            'alignment': 'same causal sample clock; no fitted delay or gain',
            'near_end': 'none', 'noise': 'none', 'randomness': 'none'}


def generate_signals() -> dict[str, np.ndarray]:
    phase = 2*np.pi*np.arange(1600)/1600
    period = .08*np.cos(50*phase)+.06*np.sin(73*phase)
    source = period[np.arange(SOURCE_SAMPLES) % 1600]
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    source[:320] *= fade
    source[-320:] *= fade[::-1]
    early = np.pad(source, (0, 1600))
    gain = np.ones(SAMPLES)
    gain[16000:] = 2.
    late = gain*early
    # Independent sparse true convolution, including the complete tail.
    echo = .7*late.copy()
    echo[1600:] += .35*late[:-1600]
    path = np.zeros(1601)
    path[[0, 1600]] = [.7, .35]
    predicted = {}
    residual = {}
    for key, reference in [('early', early), ('late', late)]:
        state = NLMSState(1601, step_size=.5, epsilon=1e-8, initial_weights=path)
        residual[key], predicted[key] = state.process(reference, echo, freeze=np.ones(SAMPLES, dtype=bool))
        if not np.array_equal(state.weights, path):
            raise ValueError('known coefficients must remain frozen throughout the tail')
    signals = {'early': early, 'late': late, 'echo': echo,
               'early_residual': residual['early'],
               'wrong_gain_residual': echo-gain*predicted['early'],
               'late_residual': residual['late']}
    if any(not np.isfinite(x).all() or np.max(abs(x)) >= 1 for x in signals.values()):
        raise ValueError('finite unclipped signals at gain 1 required')
    return {k: x[None, :] for k, x in signals.items()}


def analytic_measurements(name: str) -> dict:
    if name not in FILE_NAMES:
        raise ValueError('unknown reference fixture')
    coefficients = {'early': [1., 1.], 'late': [2., 2.], 'echo': [1.75, 2.1],
                    'early_residual': [.7, 1.05], 'wrong_gain_residual': [-.35, 0.],
                    'late_residual': [0., 0.]}
    return {'source_constant_envelope_mean_square': .005,
            'step': {'signed_source_multiplier': coefficients[name][0],
                     'mean_square': .005*coefficients[name][0]**2},
            'stable': {'signed_source_multiplier': coefficients[name][1],
                       'mean_square': .005*coefficients[name][1]**2},
            'tail': {'power_formula': 'sum of squared known enveloped causal samples / 1600',
                     'ideal_zero': name in ('early', 'late', 'wrong_gain_residual', 'late_residual')},
            'interpretation': 'constant-envelope known-component identity; not estimated from output'}


def measure_signal(samples: np.ndarray) -> dict:
    x = np.asarray(samples)
    if x.shape != (1, SAMPLES) or x.dtype.kind not in 'iuf' or not np.isfinite(x).all():
        raise ValueError('finite mono signal of 33600 samples required')
    return {name: {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
                   'channels': 1, 'mean_square': float(np.mean(x[0, start:stop]**2)),
                   'peak': float(np.max(abs(x[0, start:stop])))}
            for name, (start, stop) in WINDOWS.items()}


def measure_pcm_tail_activity(reference: np.ndarray, echo: np.ndarray) -> dict:
    for x in (reference, echo):
        measure_signal(x)
        integers = x*32768
        if (not np.array_equal(integers, np.rint(integers))
                or np.any(integers < -32768) or np.any(integers > 32767)):
            raise ValueError('actual decoded PCM16 required for activity measurement')
    states, ncc = ncc_activity_states(reference[0], echo[0], frame_size=160,
                                     activity_rms=.001, coherence_threshold=.85)
    begin = 32000//160
    ref_frames = reference[0, 32000:].reshape(10, 160)
    echo_frames = echo[0, 32000:].reshape(10, 160)
    defined = ((np.ptp(ref_frames, axis=1) > 0) & (np.ptp(echo_frames, axis=1) > 0)).tolist()
    return {'input': 'actual decoded late-reference and true-echo PCM',
            'frame_samples': 160, 'activity_rms': .001, 'coherence_threshold': .85,
            'tail_interval_samples': [32000, 33600], 'tail_frames': 10,
            'tail_states': states[begin:].tolist(), 'tail_labels': [LABELS[k] for k in states[begin:]],
            'tail_absolute_ncc': ncc[begin:].tolist(), 'tail_ncc_defined': defined,
            'tail_near_end_truth_present': [False]*10,
            'available_after_samples': list(range(32160, 33601, 160)),
            'interpretation': 'zero NCC is an undefined zero-variance placeholder; near_only is activity label, not component truth'}
