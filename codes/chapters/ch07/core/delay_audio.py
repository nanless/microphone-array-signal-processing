"""Known propagation-time leakage, with one real fixed-window regressor.

This is an integer-delay mathematical waveform control, not STFT-WPE, blind
alignment, a measured array or a guarantee that an aligned target is safe.
"""
from __future__ import annotations

import math
import numpy as np
from codes.chapters.ch07.core.prediction_delays import shift_observation, fit_single_real_history

SAMPLE_RATE = 16000
HOP_SAMPLES = 128
PERIOD_SAMPLES = 1536
SOURCE_SAMPLES = 32256
SAMPLES = 33024
SCORING_INTERVAL = (1536, 30720)
FILE_NAMES = {name: 'delay_'+name+'.wav' for name in
              ('reference', 'array', 'common_history', 'aligned_history',
               'common_residual', 'aligned_residual')}
LIMITS = ('Original deterministic 500 Hz pulse control, not speech, STFT-WPE, a room measurement or device audio. '
          'Known integer propagation and history delays; one real regressor, unit prescribed powers, no ridge. '
          'The corrected history has disjoint source-time support only in this declared pulse fixture. '
          'Alignment does not generally prevent predictable-target cancellation. No delay/gain fitting to a clean output, '
          'random independence, noise, human listening test or industrial performance claim. '
          'The common residual is deliberately zero; the aligned residual deliberately duplicates the reference PCM. '
          'Physical propagation tail 384 and maximum history padding 768 are different quantities.')


def parameters() -> dict:
    return {'exercise_id': 'E07-24', 'sample_rate_hz': 16000, 'hop_samples': 128,
            'source_samples': 32256, 'samples_per_channel': 33024,
            'source_period_samples': 1536, 'source_period_frames': 12, 'source_periods': 21,
            'pulse_start_frames_in_period': [0, 4], 'pulse_samples': 128,
            'frequency_hz': 500.0, 'amplitude': 0.1,
            'pulse_phase': 'cosine at pulse-local sample clock, phase zero; four cycles per pulse',
            'pulse_envelope': 'sin(linspace(0,pi,128)) squared, both endpoints included',
            'pulse_repetition': 'repeat the same finite pulse values; source zero outside the two pulse supports',
            'observation_delays_samples': {'reference': 384, 'auxiliary': 0},
            'common_auxiliary_history_delay_samples': 384,
            'aligned_auxiliary_history_delay_samples': 768,
            'reference_protection_delay_samples': 384,
            'source_time_gaps_samples': {'common': 0, 'aligned': 384},
            'physical_propagation_tail_samples': 384, 'maximum_history_padding_samples': 768,
            'negative_time_samples': 'zero', 'complete_retained_support_samples': [0, 33024],
            'channel_order': ['late_reference', 'early_auxiliary'],
            'regression': 'one auxiliary real history, lambda=1; ordinary transpose; no ridge or reference-history column',
            'scoring_interval_samples': [1536, 30720], 'scoring_samples_per_channel': 29184,
            'scoring_source_periods': 19, 'common_export_gain': 1.0,
            'alignment': 'same causal receiver sample clock; no fitted propagation delay or export gain',
            'noise': 'none', 'randomness': 'none'}


def generate_experiment() -> dict:
    pulse_clock = np.arange(HOP_SAMPLES)/SAMPLE_RATE
    pulse = .1*np.cos(2*np.pi*500*pulse_clock)*np.sin(np.linspace(0, np.pi, HOP_SAMPLES))**2
    period = np.zeros(PERIOD_SAMPLES)
    period[:128], period[512:640] = pulse, pulse
    source = np.pad(np.tile(period, 21), (0, 768))
    reference = shift_observation(source, 384)
    auxiliary = shift_observation(source, 0)
    histories = {'common': shift_observation(auxiliary, 384),
                 'aligned': shift_observation(auxiliary, 768)}
    fits = {key: fit_single_real_history(reference, q, score_window=SCORING_INTERVAL)
            for key, q in histories.items()}
    signals = {'reference': reference[None], 'array': np.vstack((reference, auxiliary)),
               'common_history': histories['common'][None], 'aligned_history': histories['aligned'][None],
               'common_residual': fits['common']['residual'][None],
               'aligned_residual': fits['aligned']['residual'][None]}
    for name, value in signals.items():
        channels = 2 if name == 'array' else 1
        if (value.shape != (channels, SAMPLES) or not np.isfinite(value).all()
                or np.max(abs(value)) >= 1):
            raise ValueError('fixed finite unclipped delay fixture required')
    return {'signals': signals, 'regression': {key: {field: fit[field] for field in
            ('coefficient', 'history_energy', 'cross')} for key, fit in fits.items()}}


def analytic_measurements(name: str) -> dict:
    if name not in FILE_NAMES:
        raise ValueError('unknown propagation fixture')
    # Finite trigonometric sum of the declared pulse, independent of waveform fitting.
    pulse_energy = math.fsum((.1*math.cos(2*math.pi*500*n/16000)
                             *math.sin(math.pi*n/127)**2)**2 for n in range(128))
    mean_square = 2*pulse_energy/1536
    channels = 2 if name == 'array' else 1
    relative_errors = ([0.0, 2.0] if name == 'array' else
                       [2.0] if name == 'aligned_history' else
                       [1.0] if name == 'common_residual' else [0.0])
    projection_gains = ([1.0, 0.0] if name == 'array' else
                        [0.0] if name in ('aligned_history', 'common_residual') else [1.0])
    return {'per_pulse_squared_sum': pulse_energy,
            'source_period_mean_square': mean_square,
            'mean_square_per_channel': [0.0 if name == 'common_residual' else mean_square]*channels,
            'reference_error_mean_square_per_channel': [mean_square*v for v in relative_errors],
            'reference_NMSE_per_channel': relative_errors,
            'projection_gain_per_channel': projection_gains,
            'common_history_coefficient': 1.0, 'aligned_history_coefficient': 0.0,
            'scope': 'known repeated finite pulse identity; disjoint support is specific to this fixture'}


def measure_signal(samples: np.ndarray, reference: np.ndarray) -> dict:
    x, r = np.asarray(samples), np.asarray(reference)
    if (x.dtype.kind not in 'iuf' or r.dtype.kind not in 'iuf'
            or x.shape not in ((1, SAMPLES), (2, SAMPLES)) or r.shape != (1, SAMPLES)
            or not np.isfinite(x).all() or not np.isfinite(r).all()):
        raise ValueError('finite real 33024-sample mono/two-channel and mono reference required')
    start, stop = SCORING_INTERVAL
    truth, value = r[0, start:stop], x[:, start:stop]
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            denominator = float(truth@truth)
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('positive finite reference energy required')
            errors = np.sum((value-truth)**2, axis=1)
            powers = np.mean(value**2, axis=1)
            projection = (value@truth)/denominator
    except FloatingPointError as error:
        raise ValueError('delay measurement exceeds float64 support') from error
    return {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
            'channels': len(x), 'reference_squared_sum': denominator,
            'mean_square_per_channel': powers.tolist(),
            'mean_square_all_channels': float(np.mean(powers)),
            'reference_error_squared_sum_per_channel': errors.tolist(),
            'reference_error_mean_square_per_channel': (errors/(stop-start)).tolist(),
            'reference_NMSE_per_channel': (errors/denominator).tolist(),
            'projection_gain_per_channel': projection.tolist(),
            'peak_per_channel': np.max(abs(x), axis=1).tolist()}


def measure_regression(reference: np.ndarray, common: np.ndarray, aligned: np.ndarray) -> dict:
    result = {}
    for name, history in (('common', common), ('aligned', aligned)):
        measure_signal(history, reference)
        fit = fit_single_real_history(reference[0], history[0], score_window=SCORING_INTERVAL)
        result[name] = {field: fit[field] for field in ('coefficient', 'history_energy', 'cross')}
        result[name]['scoring_interval_samples'] = list(SCORING_INTERVAL)
    return result
