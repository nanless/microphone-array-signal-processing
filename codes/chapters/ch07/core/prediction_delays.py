"""Known source-clock controls for E07-22/23, not a blind WPE implementation.

Integer delays use the caller's declared time unit: frames for the short toy,
samples for the published waveform. The real one-parameter fit has prescribed
unit inverse-power weights, no loading, and no estimated arrival times.
"""
from __future__ import annotations

import math
import numbers
import numpy as np


def _integer(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def source_gap(prediction_delay, arrival_delay, reference_arrival):
    """Reference current source index minus a channel history source index."""
    return (_integer(prediction_delay, 'prediction_delay')
            + _integer(arrival_delay, 'arrival_delay')
            - _integer(reference_arrival, 'reference_arrival'))


def prediction_delays(arrival_delays, reference_index, protection_delay):
    """Return signed required delays; negative results explicitly need future data.

    Positive arrival delay means later arrival. Returning the algebraic result
    does not declare it causal or adapt any existing WPE history interface.
    """
    arrivals = tuple(_integer(v, 'arrival_delay') for v in arrival_delays)
    if not arrivals:
        raise ValueError('arrival_delays must be nonempty')
    reference_index = _integer(reference_index, 'reference_index', minimum=0)
    if reference_index >= len(arrivals):
        raise ValueError('reference_index is outside arrival_delays')
    protection_delay = _integer(protection_delay, 'protection_delay', minimum=0)
    return tuple(protection_delay + arrivals[reference_index] - d for d in arrivals)


def complete_history_interval(frame_count, delays, taps):
    """Half-open current-frame interval with full nonnegative-delay history.

    Delay zero includes the current channel observation. The book's existing
    WPE kernel requires delay >= 1; this function does not change that contract.
    """
    frame_count = _integer(frame_count, 'frame_count', minimum=0)
    taps = _integer(taps, 'taps', minimum=1)
    delays = tuple(_integer(v, 'delay', minimum=0) for v in delays)
    if not delays:
        raise ValueError('delays must be nonempty')
    first = max(delays) + taps - 1
    return min(first, frame_count), frame_count


def _real_vector(signal, name):
    if np.iscomplexobj(signal):
        raise ValueError(f'{name} must be real')
    value = np.asarray(signal, dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite real vector')
    return value


def shift_observation(signal, delay_samples):
    """Causal integer delay in an already declared finite output support."""
    value = _real_vector(signal, 'signal')
    delay_samples = _integer(delay_samples, 'delay_samples', minimum=0)
    result = np.zeros_like(value)
    if delay_samples < len(value):
        result[delay_samples:] = value[:len(value)-delay_samples]
    return result


def fit_single_real_history(target, history, score_window):
    """Fit one real coefficient on a half-open window, then apply it everywhere.

    This bounded float64 teaching control rejects unrepresentable statistics.
    The score window is also the fit window; there is no held-out-score claim.
    """
    target = _real_vector(target, 'target')
    history = _real_vector(history, 'history')
    if target.shape != history.shape:
        raise ValueError('target and history must have the same shape')
    if len(score_window) != 2:
        raise ValueError('score_window needs start and stop')
    start, stop = (_integer(v, 'score_window index', minimum=0) for v in score_window)
    if not start < stop <= len(target):
        raise ValueError('score_window must be nonempty and inside the signal')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            q, y = history[start:stop], target[start:stop]
            energy, cross = float(q @ q), float(q @ y)
            if not math.isfinite(energy) or energy <= 0 or not math.isfinite(cross):
                raise ValueError('history statistics must be finite with positive energy')
            coefficient = cross / energy
            residual = target - coefficient * history
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('fit arithmetic exceeds float64 support') from error
    if not math.isfinite(coefficient) or not np.isfinite(residual).all():
        raise ValueError('fit result must be finite')
    return dict(coefficient=coefficient, history_energy=energy, cross=cross,
                residual=residual)


def source_clock_example():
    """Exact known integer-frame source, one auxiliary regressor, fixed window."""
    source = np.zeros(10)
    source[[0, 4]] = 1
    target = shift_observation(source, 3)
    common = shift_observation(source, 3)
    aligned = shift_observation(source, 6)
    cases = {}
    for name, history in [('common', common), ('aligned', aligned)]:
        fit = fit_single_real_history(target, history, (6, 10))
        cases[name] = {**{k: v for k, v in fit.items() if k != 'residual'},
                       'history': history[6:10].tolist(),
                       'residual': fit['residual'][6:10].tolist()}
    return dict(source=source.tolist(), arrival_delays=[3, 0], reference_index=0,
                protection_delay=3, matched_delays=list(prediction_delays([3, 0], 0, 3)),
                current_frames=[6, 7, 8, 9], target=target[6:10].tolist(), cases=cases,
                scope='known source clock, unit powers, one real auxiliary tap; no STFT, blind TDOA, or general target-preservation guarantee')


def late_power_decay(t60_seconds=.6, hop_seconds=.008, frame_offsets=(1, 10, 25, 75)):
    """Statistical late power free decay; no fresh-source term or PSD estimator.

    The model is distinct from WPE's desired early-speech variance lambda.
    Logs remain meaningful if a finite very small power ratio rounds to zero.
    """
    values = []
    for value, name in [(t60_seconds, 't60_seconds'), (hop_seconds, 'hop_seconds')]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f'{name} must be positive and real')
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
        values.append(value)
    t60_seconds, hop_seconds = values
    offsets = tuple(_integer(v, 'frame_offset', minimum=0) for v in frame_offsets)
    log10_rho = -6 * (hop_seconds / t60_seconds)
    if not math.isfinite(log10_rho) or not math.isfinite(10*log10_rho):
        raise ValueError('decay exponent exceeds float64 support')
    rows = []
    for offset in offsets:
        exponent = log10_rho * offset
        if not math.isfinite(exponent) or not math.isfinite(10 * exponent):
            raise ValueError('decay exponent exceeds float64 support')
        elapsed = offset*hop_seconds
        if not math.isfinite(elapsed):
            raise ValueError('elapsed time exceeds float64 support')
        rows.append(dict(frames=offset, elapsed_seconds=elapsed,
                         power_ratio=10.**exponent, amplitude_ratio=10.**(exponent/2),
                         level_db=10*exponent))
    return dict(t60_seconds=t60_seconds, hop_seconds=hop_seconds,
                power_ratio_per_hop=10.**log10_rho,
                amplitude_ratio_per_hop=10.**(log10_rho/2),
                level_db_per_hop=10*log10_rho, rows=rows,
                scope='given late statistical free-decay model above its mixing-time/frequency boundary; not WPE lambda, a measured room, or the full Habets estimator')
