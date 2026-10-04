"""Known two-slot polarity/gain stitching control, not a blind CSS separator.

The second block is a prescribed -2 multiple of the same ordered source slots.
The application gain is fitted to observed overlap samples, never to clean truth.
"""
from __future__ import annotations

import math
import numpy as np
from codes.chapters.ch08.core.css import match_two_source_overlap, overlap_application_gain

SAMPLE_RATE = 16000
SAMPLES = 32000
FILE_NAMES = {key: 'css_'+key+'.wav' for key in ('reference', 'naive', 'polarity', 'corrected')}
SCORING_INTERVALS = {'primary': (2400, 29600), 'post_overlap': (19200, 29600)}
LIMITS = ('Original known two-slot sinusoid stitching control, not speech, a blind CSS separator, '
          'STFT separation, speaker recognition or industrial performance. The source-slot order and '
          'sample-clock correspondence are fixed. Absolute correlation assigns slots but does not '
          'repair polarity or gain. A real scalar is fitted only between observed overlapping blocks, '
          'not to clean truth; it cannot repair arbitrary distortion, delay or frequency-dependent phase. '
          'Reference and fully corrected PCM deliberately match. No per-file normalization, fitted '
          'reference delay, noise, random independence or formal listening study. Fixed-window '
          'uncompensated MSE/NMSE measures the gain error; SI-SDR is not used as a gain check.')


def parameters():
    return {'exercise_id': 'E08-32', 'sample_rate_hz': 16000, 'samples_per_channel': 32000,
            'frequencies_hz': [500.0, 1500.0], 'amplitudes': [0.1, 0.1],
            'channel_order': ['fixed_source_slot_1', 'fixed_source_slot_2'],
            'phase': 'sine at the common absolute sample clock, phase zero',
            'envelope': 'sin(linspace(0,pi/2,320)) squared at each end, reversed at end; endpoints zero',
            'fade_samples_each_end': 320, 'first_block_samples': [0, 19200],
            'second_block_samples': [12800, 32000], 'overlap_samples': [12800, 19200],
            'overlap_samples_per_channel': 6400, 'overlap_cycles_per_channel': [200, 600],
            'second_block_multiplier': -2.0,
            'crossfade': 'complementary linear weights: current r=linspace(0,1,6400), previous 1-r, endpoints included',
            'slot_matching': 'absolute centered correlation on observed overlap; apply current indices for previous',
            'application_gain_fit': 'ordinary real raw-sample LS current dot previous / current dot current; no centering',
            'polarity_control': 'sign of the overlap application gain only; magnitude remains 2',
            'scoring_intervals_samples': {k: list(v) for k, v in SCORING_INTERVALS.items()},
            'scoring_samples_per_channel': {'primary': 27200, 'post_overlap': 10400},
            'post_overlap_cycles_per_channel': [325, 975], 'common_export_gain': 1.0,
            'noise': 'none', 'randomness': 'none',
            'alignment': 'same known slot order and receiving sample clock; no delay or clean-reference gain fit'}


def generate_experiment():
    time = np.arange(SAMPLES)/SAMPLE_RATE
    envelope = np.ones(SAMPLES)
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    envelope[:320], envelope[-320:] = fade, fade[::-1]
    envelope[0] = envelope[-1] = 0.0
    reference = np.array([.1*np.sin(2*np.pi*f*time)*envelope for f in (500, 1500)])
    previous = reference[:, :19200].copy()
    current = -2*reference[:, 12800:].copy()
    matching = match_two_source_overlap(previous[:, 12800:], current[:, :6400])
    if matching['status'] != 'matched' or matching['current_indices_for_previous'] != [0, 1]:
        raise ValueError('fixed orthogonal same-clock slots must match')
    current = current[matching['current_indices_for_previous']]
    gains = overlap_application_gain(previous[:, 12800:], current[:, :6400])
    ramp = np.linspace(0, 1, 6400)

    def stitch(application):
        adjusted = current*np.asarray(application)[:, None]
        result = np.empty_like(reference)
        result[:, :12800] = previous[:, :12800]
        result[:, 12800:19200] = previous[:, 12800:]*(1-ramp)+adjusted[:, :6400]*ramp
        result[:, 19200:] = adjusted[:, 6400:]
        return result

    signals = {'reference': reference, 'naive': stitch([1., 1.]),
               'polarity': stitch(np.sign(gains)), 'corrected': stitch(gains)}
    return {'signals': signals, 'overlap_matching': matching,
            'overlap_application_gain': gains.tolist(), 'observed_overlap': {'previous': previous[:, 12800:],
                                                                          'current': current[:, :6400]}}


def analytic_measurements(name):
    """Finite known-waveform sums; no numerical LS or generated array is used."""
    if name not in FILE_NAMES:
        raise ValueError('unknown CSS control')
    def factor(n):
        if name in ('reference', 'corrected') or n < 12800:
            return 1.0
        current = -2.0 if name == 'naive' else 2.0
        return 1+(current-1)*(n-12800)/6399 if n < 19200 else current
    result = {}
    for key, (start, stop) in SCORING_INTERVALS.items():
        powers, errors, references = [], [], []
        for frequency in (500, 1500):
            truth = [.1*math.sin(2*math.pi*frequency*n/16000) for n in range(start, stop)]
            references.append(math.fsum(v*v for v in truth))
            powers.append(math.fsum((v*factor(n))**2 for n, v in zip(range(start, stop), truth))/(stop-start))
            errors.append(math.fsum((v*(factor(n)-1))**2 for n, v in zip(range(start, stop), truth)))
        result[key] = {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
                       'channels': 2, 'reference_squared_sum_per_channel': references,
                       'mean_square_per_channel': powers, 'mean_square_all_channels': sum(powers)/2,
                       'reference_error_squared_sum_per_channel': errors,
                       'reference_error_mean_square_per_channel': [v/(stop-start) for v in errors],
                       'reference_NMSE_per_channel': [e/r for e, r in zip(errors, references)],
                       'scope': 'finite trigonometric sum of the declared crossfade; post-overlap integer-cycle control'}
    return result


def measure_signal(samples, reference):
    x, r = np.asarray(samples), np.asarray(reference)
    if (x.shape != (2, 32000) or r.shape != x.shape or x.dtype.kind not in 'iuf'
            or r.dtype.kind not in 'iuf' or not np.isfinite(x).all() or not np.isfinite(r).all()):
        raise ValueError('finite real two-channel 32000-sample waveform and reference required')
    result = {}
    for key, (start, stop) in SCORING_INTERVALS.items():
        truth, value = r[:, start:stop], x[:, start:stop]
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                denominator = np.sum(truth**2, axis=1)
                error = np.sum((value-truth)**2, axis=1)
                powers = np.mean(value**2, axis=1)
                projection = np.sum(value*truth, axis=1)/denominator
                nmse = error/denominator
        except FloatingPointError as exc:
            raise ValueError('CSS measurement exceeds float64 support') from exc
        if (np.any(denominator <= 0) or not np.isfinite(nmse).all()
                or np.any((error == 0) & np.any(value != truth, axis=1))):
            raise ValueError('CSS reference/error energy exceeds finite support')
        result[key] = {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
                       'channels': 2, 'reference_squared_sum_per_channel': denominator.tolist(),
                       'mean_square_per_channel': powers.tolist(), 'mean_square_all_channels': float(np.mean(powers)),
                       'reference_error_squared_sum_per_channel': error.tolist(),
                       'reference_error_mean_square_per_channel': (error/(stop-start)).tolist(),
                       'reference_NMSE_per_channel': nmse.tolist(), 'projection_gain_per_channel': projection.tolist()}
    return result
