"""One-slot deterministic lifecycle rules on explicitly timestamped frames.

The rules supplement, rather than replace, a localization or motion filter.
They are not Bernoulli/LMB inference, speaker identification or production
thresholds. Integers in half-sample ticks keep expiry equality exact.
"""
from __future__ import annotations

import math
import numpy as np


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be a true integer')
    value = int(value)
    if value < minimum:
        raise ValueError(name + ' is below its supported minimum')
    return value


def _seconds(tick, denominator):
    try:
        value = tick / denominator
    except OverflowError as error:
        raise ValueError('timestamp exceeds finite float display range') from error
    if not math.isfinite(value) or (tick != 0 and value == 0):
        raise ValueError('timestamp exceeds finite float display range')
    return value


def teaching_lifecycle(frames, *, sample_rate_hz=16000, window_samples=512,
                       confirmation_frames=3, maximum_age_ticks=6400,
                       age_clock='available'):
    """Apply a declared consecutive-hit/expiry policy to a single slot.

    A tick is 1/(2*sample_rate_hz) seconds. The last valid measurement belongs
    to its window centre, not its availability time. Expiry is strictly age
    > maximum_age_ticks; equality is retained. At an incoming frame the old
    track is first aged before a new measurement can refresh it. A candidate
    requires consecutive valid frames and disappears on a miss. A confirmed
    track coasts through misses until expiry. Reacquisition after expiry gets
    a fresh monotonically increasing local ID. Confirmation is published only
    at the completing frame's availability time; no earlier row is relabelled.

    age_clock='state' is a diagnostic alternative using the state epoch; its
    events are still publishable only at availability. Neither mode predicts
    a state to that publication epoch or estimates a permanent speaker ID.
    """
    fs = _integer(sample_rate_hz, 'sample_rate_hz', minimum=1)
    window = _integer(window_samples, 'window_samples', minimum=1)
    needed = _integer(confirmation_frames, 'confirmation_frames', minimum=1)
    maximum = _integer(maximum_age_ticks, 'maximum_age_ticks')
    if type(age_clock) is not str or age_clock not in ('available', 'state'):
        raise ValueError('age_clock must be available or state')
    if age_clock == 'available' and maximum < window+1:
        raise ValueError('maximum age must cover the declared window availability delay')
    required = ('start_sample', 'observation_valid', 'state_time_s', 'available_time_s')
    if type(frames) is not dict or any(key not in frames for key in required):
        raise ValueError('timestamped observation frame fields are required')
    columns = [frames[key] for key in required]
    if any(type(column) not in (list, tuple) for column in columns):
        raise ValueError('frame columns must be ordinary lists or tuples')
    count = len(columns[0])
    if count == 0 or any(len(column) != count for column in columns):
        raise ValueError('nonempty equal-length frame columns required')
    # Validate the whole input before processing; do not silently reinterpret
    # an already-declared timestamp from another clock or frame convention.
    ticks = []
    previous_start = -1
    for start, valid, state_time, available_time in zip(*columns):
        start = _integer(start, 'start_sample')
        if start <= previous_start or type(valid) is not bool:
            raise ValueError('start samples must increase and validity must be bool')
        state_tick, available_tick = 2*start+window-1, 2*(start+window)
        denominator = 2*fs
        expected = (_seconds(state_tick, denominator), _seconds(available_tick, denominator))
        if expected[1] <= expected[0]:
            raise ValueError('float display cannot distinguish state and availability epochs')
        for value, true in zip((state_time, available_time), expected):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating))
                    or not math.isfinite(float(value)) or float(value) != true):
                raise ValueError('declared frame timestamps differ from their actual sample support')
        ticks.append((state_tick, available_tick))
        previous_start = start
    rows, events = [], []
    phase, track_id, next_id, consecutive, last = 'absent', None, 1, 0, None
    for index, ((state_tick, available_tick), valid) in enumerate(zip(ticks, columns[1])):
        now = available_tick if age_clock == 'available' else state_tick
        frame_events = []
        old_age = None if last is None else now-last
        def event(name, identifier, age=None):
            item = {'frame': index, 'event': name, 'track_id': identifier,
                    'state_tick': state_tick, 'available_tick': available_tick,
                    'state_time_s': _seconds(state_tick, 2*fs),
                    'publication_time_s': _seconds(available_tick, 2*fs)}
            if age is not None:
                item.update(age_ticks=age, age_s=_seconds(age, 2*fs))
            events.append(item)
            frame_events.append(name)
        if phase != 'absent' and old_age > maximum:
            event('expired', track_id, old_age)
            phase, track_id, consecutive, last = 'absent', None, 0, None
        if valid:
            if phase == 'absent':
                track_id, next_id = next_id, next_id+1
                phase, consecutive = 'tentative', 0
                event('candidate', track_id)
            consecutive += 1
            last = state_tick
            if phase == 'tentative' and consecutive >= needed:
                phase = 'confirmed'
                event('confirmed', track_id)
        else:
            consecutive = 0
            if phase == 'tentative':
                event('candidate_lost', track_id)
                phase, track_id, last = 'absent', None, None
        age = None if last is None else now-last
        rows.append({'frame': index, 'observation_valid': valid,
            'state_tick': state_tick, 'available_tick': available_tick,
            'state_time_s': _seconds(state_tick, 2*fs),
            'publication_time_s': _seconds(available_tick, 2*fs),
            'phase': phase, 'track_id': track_id,
            'consecutive_valid_frames': consecutive,
            'last_valid_measurement_tick': last,
            'last_valid_measurement_time_s': None if last is None else _seconds(last, 2*fs),
            'age_ticks': age, 'age_s': None if age is None else _seconds(age, 2*fs),
            'publish_confirmed': phase == 'confirmed', 'events': frame_events})
    return {'parameters': {'sample_rate_hz': fs, 'window_samples': window,
            'tick_denominator_hz': 2*fs, 'confirmation_frames': needed,
            'maximum_age_ticks': maximum, 'maximum_age_s': _seconds(maximum, 2*fs),
            'age_clock': age_clock, 'expiry_comparison': 'strictly greater; equality retained',
            'measurement_epoch': 'receiver window centre',
            'event_publication_epoch': 'complete receiver window availability',
            'expiry_order': 'age old candidate/track before applying the incoming observation'},
            'rows': rows, 'events': events,
            'counts': {'frames': count, 'allocated_local_ids': next_id-1,
                'confirmed_publications': sum(row['publish_confirmed'] for row in rows),
                'expired_events': sum(item['event'] == 'expired' for item in events)},
            'scope': 'one already-associated slot; deterministic teaching thresholds only; no Bernoulli/LMB inference, multi-target assignment, permanent speaker identity, motion-filter reset or prediction to publication time'}
