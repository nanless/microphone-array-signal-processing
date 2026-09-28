"""One controlled moving-source GCC/Kalman experiment, not a tracking benchmark.

The source has retarded propagation and an assigned 1/r pressure gain. This
omits directional radiation corrections for moving physical monopoles. The
waveform is a continuous multisine, not speech, and noise is mathematical.
"""
from __future__ import annotations

import io
import wave
import numpy as np

from .audio_samples import pcm16_bytes
from .conventions import finite_real_array, finite_real_scalar
from .doa import gcc_phat
from .moving_source import retarded_emission_times
from .tracking import ConstantVelocityKalman

FS = 16000
SAMPLES = 32000
WINDOW = 512
HOP = 160
SOUND_SPEED = 343.
MICS = np.array([[-.05, 0.], [.05, 0.]])
START = np.array([-.8, 1.5])
VELOCITY = np.array([.8, 0.])


def continuous_source(times, phases):
    """32 equal-amplitude tones; fades and silence are on the emission clock."""
    times = finite_real_array(times, 'emission times')
    phases = finite_real_array(phases, 'phases')
    if phases.shape != (32,):
        raise ValueError('phases must contain 32 values')
    envelope = np.minimum(np.clip(times/.02, 0, 1), np.clip((2-times)/.02, 0, 1))
    envelope *= 1 - np.minimum(np.clip((times-.8)/.02, 0, 1), np.clip((1.1-times)/.02, 0, 1))
    frequencies = np.arange(300, 3401, 100)
    with np.errstate(over='raise', invalid='raise'):
        tones = np.sin(2*np.pi*frequencies[:, None]*times.reshape(1, -1) + phases[:, None])
    return envelope * tones.sum(0).reshape(times.shape) / np.sqrt(32)


def frame_truth(centers):
    """Receiver-center timestamps -> center-retarded direction/event TDOA.

    tau10 uses the *same emission event* at the two microphones. A finite
    moving-signal window need not have one exact constant correlation delay.
    """
    centers = finite_real_array(centers, 'centers')
    emission = retarded_emission_times(centers, np.zeros((2, 2)),
                                      source_start_xy=START, source_velocity_xy=VELOCITY)[0]
    positions = START + emission[:, None]*VELOCITY
    distances = np.linalg.norm(positions[:, None, :] - MICS, axis=2)
    return {'center_emission_time_s': emission.tolist(),
            'truth_angle_deg': np.rad2deg(np.arctan2(positions[:, 0], positions[:, 1])).tolist(),
            'truth_tau10_samples': ((distances[:, 1]-distances[:, 0])/SOUND_SPEED*FS).tolist()}


def analyze_array(waveform, *, export_gain=1.):
    """Read samples only; truth is attached after all estimates are complete.

    Undo the known common export gain before the fixed RMS threshold. This
    is not data-dependent normalization. Frame overlap induces correlated
    observations; R=4 deg² is a teaching setting, not measured calibration.
    """
    waveform = finite_real_array(waveform, 'waveform')
    export_gain = finite_real_scalar(export_gain, 'export_gain')
    if waveform.shape != (2, SAMPLES) or export_gain <= 0:
        raise ValueError('expected two channels, 32000 samples and positive export gain')
    signal = waveform/export_gain
    if not np.all(np.isfinite(signal)):
        raise ValueError('gain correction exceeds floating-point range')
    starts = np.arange(0, SAMPLES-WINDOW+1, HOP)
    centers = (starts+(WINDOW-1)/2)/FS
    fields = {key: [] for key in ('rms_before_export', 'observation_valid', 'observation_tau10_samples',
              'observation_angle_deg', 'filtered_angle_deg', 'velocity_deg_s', 'angle_variance_deg2',
              'state_phase', 'reason')}
    tracker = None
    dt = HOP/FS
    q = 100*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    for start in starts:
        block = signal[:, start:start+WINDOW]
        rms = float(np.sqrt(np.mean(block**2)))
        angle = tau = None
        reason = 'below_rms_gate'
        if rms > .04:
            tau_s, _, _, _ = gcc_phat(block[1]*np.hanning(WINDOW), block[0]*np.hanning(WINDOW),
                                      FS, max_tau=.1/SOUND_SPEED, parabolic_interpolation=True)
            sine = -tau_s*SOUND_SPEED/.1
            if abs(sine) <= 1:
                angle, tau = float(np.rad2deg(np.arcsin(sine))), float(tau_s*FS)
                reason = 'valid'
            else:
                reason = 'outside_physical_delay'
        if tracker is None:
            if angle is not None:
                tracker = ConstantVelocityKalman([angle, 0], np.diag([4., 400.]), q)
                phase = 'initialized_from_observation'
            else:
                phase = 'uninitialized'
        else:
            tracker.predict(dt)
            phase = 'prediction_only'
            if angle is not None:
                tracker.update(angle, 4.)
                phase = 'posterior'
        fields['rms_before_export'].append(rms)
        fields['observation_valid'].append(angle is not None)
        fields['observation_tau10_samples'].append(tau)
        fields['observation_angle_deg'].append(angle)
        for key, value in (('filtered_angle_deg', None if tracker is None else tracker.state[0]),
                           ('velocity_deg_s', None if tracker is None else tracker.state[1]),
                           ('angle_variance_deg2', None if tracker is None else tracker.covariance[0, 0])):
            fields[key].append(None if value is None else float(value))
        fields['state_phase'].append(phase)
        fields['reason'].append(reason)
    fields.update({'start_sample': starts.tolist(), 'state_time_s': centers.tolist(),
                   'available_time_s': ((starts+WINDOW)/FS).tolist(), **frame_truth(centers)})
    raw = np.asarray(fields['observation_angle_deg'], float)
    filtered = np.asarray(fields['filtered_angle_deg'], float)
    truth = np.asarray(fields['truth_angle_deg'])
    valid = np.asarray(fields['observation_valid'])
    missing = ~valid
    initialized = np.isfinite(filtered)
    def rmse(values, mask):
        return float(np.sqrt(np.mean((values[mask]-truth[mask])**2))) if mask.any() else None
    scores = {'frame_count': len(starts), 'valid_observation_count': int(valid.sum()),
              'missing_observation_count': int(missing.sum()),
              'initialized_missing_count': int(np.sum(missing & initialized)),
              'raw_valid_rmse_deg': rmse(raw, valid),
              'filtered_valid_rmse_deg': rmse(filtered, valid & initialized),
              'filtered_missing_rmse_deg': rmse(filtered, missing & initialized)}
    return {'frames': fields, 'scores': scores}


def read_pcm16(data):
    """Decode the actual exported little-endian interleaved PCM samples."""
    with wave.open(io.BytesIO(data), 'rb') as reader:
        if reader.getsampwidth() != 2 or reader.getframerate() != FS:
            raise ValueError('expected 16-bit 16-kHz PCM')
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2')
        return samples.reshape(-1, reader.getnchannels()).T.astype(float)/32768.


def build_fixture():
    """Return PCM byte buffers and reproducible float/PCM analysis metadata."""
    rng = np.random.default_rng(9001)
    phases = rng.uniform(-np.pi, np.pi, 32)
    times = np.arange(SAMPLES)/FS
    emission = retarded_emission_times(times, MICS, source_start_xy=START, source_velocity_xy=VELOCITY)
    distances = np.linalg.norm(START + emission[:, :, None]*VELOCITY-MICS[:, None, :], axis=2)
    source = continuous_source(times, phases)[None]
    array = continuous_source(emission, phases)/distances + .01*rng.standard_normal((2, SAMPLES))
    gain = .7/max(float(np.max(np.abs(source))), float(np.max(np.abs(array))))
    buffers = {'source.wav': pcm16_bytes(source*gain, FS), 'array_noisy.wav': pcm16_bytes(array*gain, FS)}
    pcm_analysis = analyze_array(read_pcm16(buffers['array_noisy.wav']), export_gain=gain)
    metadata = {
        'model': 'continuous multisine with exact retarded time and assigned 1/r gain; independent Gaussian sensor noise; no room, speech or device recording',
        'sample_rate_hz': FS, 'duration_s': 2., 'seed': 9001, 'rng': 'PCG64; 32 phases then channel-major noise samples',
        'frequencies_hz': np.arange(300, 3401, 100).tolist(), 'phases_rad': phases.tolist(),
        'source_amplitude_per_tone': float(1/np.sqrt(32)), 'noise_std_before_export': .01,
        'emission_envelope': 'linear 20-ms onset/end; fade out .8-.82 s, zero .82-1.08 s, fade in 1.08-1.10 s',
        'microphones_xy_m': MICS.tolist(), 'source_start_xy_m': START.tolist(),
        'source_velocity_xy_m_s': VELOCITY.tolist(), 'sound_speed_m_s': SOUND_SPEED,
        'common_export_gain': gain, 'export_peak_ceiling': .7,
        'pcm': '16-bit little endian; array_tutorial.audio_samples.pcm16_bytes; decoded /32768; export gain undone only for fixed RMS gate',
        'analysis_config': {'window_samples': WINDOW, 'hop_samples': HOP, 'window': 'numpy.hanning(512), symmetric',
            'state_timestamp': '(start_sample+255.5)/16000, receiver window center',
            'availability_timestamp': '(start_sample+512)/16000, half-open block complete',
            'lookahead_from_state_ms': 16.03125, 'rms_threshold_before_export': .04,
            'gcc': 'gcc_phat(ch1, ch0), tau10=t1-t0; max_tau=.1/343; parabolic interpolation; no truth input',
            'angle_conversion': 'asin(-343*tau10/.1), far-field approximation; outside [-1,1] -> missing',
            'initial_state': '[first observed angle, 0 deg/s]', 'initial_covariance': [[4., 0.], [0., 400.]],
            'measurement_variance_deg2': 4., 'white_acceleration_density_deg2_s3': 100.,
            'update_order': 'initialize once; thereafter predict each 10 ms then update only with valid observation',
            'truth': 'array-center retarded emission direction; tau10 compares the same emission event at both mics',
            'scope': 'one fixed synthetic sequence; overlapping observations are correlated; assigned R/Q are not calibrated coverage; no identity or beamformer output'},
        'float_analysis': analyze_array(array), 'pcm_analysis': pcm_analysis,
        'retarded_equation_max_residual_s': float(np.max(np.abs(emission+distances/SOUND_SPEED-times))),
    }
    return buffers, metadata
