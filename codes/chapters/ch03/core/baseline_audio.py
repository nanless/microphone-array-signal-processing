"""E03-18 known-direction two-microphone baseline and fixed-offset fixture.

Continuous evaluation of one finite 500 Hz mathematical tone supplies delays.
Known frequency, known directions and the prior |tau| < 1 ms are required.
This is not blind GCC, an unknown-direction calibration or real hardware.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array
from codes.chapters.ch03.core.baseline_calibration import solve_baseline

SAMPLE_RATE = 16000
SAMPLES = 32000
SCORING_INTERVAL = (2400, 29600)
FREQUENCY = 500.0
AMPLITUDE = .2
SOUND_SPEED = 343.0
BASELINE_M = (.04, .03, .02)
OFFSET_S = 20e-6
BASE_DELAY_S = .002
TRAINING_DIRECTIONS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
HELDOUT_DIRECTION = (.6, .8, 0.)
FILE_NAMES = {'source': 'baseline_source.wav', 'train_px': 'baseline_train_px.wav',
              'train_nx': 'baseline_train_nx.wav', 'train_py': 'baseline_train_py.wav',
              'train_pz': 'baseline_train_pz.wav', 'heldout': 'baseline_heldout.wav'}
TRAINING_NAMES = ('train_px', 'train_nx', 'train_py', 'train_pz')
LIMITS = ('Known-direction, synchronized mathematical far-field tone observations with a fixed '
          'channel offset, no speech, room, noise or measured device. The single-tone phase delay '
          'requires the supplied |tau| < 1 ms prior; it is not blind GCC or phase unwrapping. '
          'Cos/sin LS measures a known 500 Hz phasor on an 850-period stable window. The held-out '
          'direction is excluded from fitting. The zero-offset fit is a deliberately misspecified '
          'control, not the calibrated estimate. Readback and playback are not localization accuracy '
          'or a formal listening study. Files use a common gain of 1 and continuous source evaluation, '
          'not a sampled fractional-delay filter. Finite waveform edges are excluded from scoring.')


def baseline_parameters() -> dict:
    return {'exercise_id': 'E03-18', 'sample_rate_hz': SAMPLE_RATE,
            'samples_per_channel': SAMPLES, 'duration_s': SAMPLES / SAMPLE_RATE,
            'frequency_hz': FREQUENCY, 'source_amplitude': AMPLITUDE,
            'source_active_interval_s': [.1, 1.9], 'linear_fade_duration_s': .02,
            'common_base_delay_s': BASE_DELAY_S, 'common_export_gain': 1.0,
            'sound_speed_m_s': SOUND_SPEED, 'baseline_m': list(BASELINE_M),
            'fixed_channel_offset_s': OFFSET_S,
            'training_directions_xyz': [list(u) for u in TRAINING_DIRECTIONS],
            'heldout_direction_xyz': list(HELDOUT_DIRECTION),
            'channel_order': ['reference_microphone_0', 'microphone_1'],
            'direction_convention': 'unit vector from array toward source',
            'delay_model': 'tau=-u dot baseline/c+offset; positive tau means channel 1 arrives later',
            'waveform_model': 'x0=F(t-base), x1=F(t-base-tau); mono source is F(t-base)',
            'delay_implementation': 'continuous source evaluation, no sampled interpolation',
            'scoring_interval_samples': list(SCORING_INTERVAL),
            'scoring_samples_per_channel': SCORING_INTERVAL[1] - SCORING_INTERVAL[0],
            'scoring_cycles': 850, 'phase_delay_prior_abs_upper_bound_s': .001,
            'phasor_convention': 'z=C-jD from cos/sin LS; tau=-arg(z1*conj(z0))/(2*pi*f)',
            'randomness': 'none', 'noise': 'none'}


def _source(time: np.ndarray) -> np.ndarray:
    envelope = np.minimum(np.clip((time-.1)/.02, 0, 1), np.clip((1.9-time)/.02, 0, 1))
    return AMPLITUDE * envelope * np.sin(2*np.pi*FREQUENCY*time)


def truth_delays() -> dict[str, float]:
    return {name: float(-np.dot(u, BASELINE_M)/SOUND_SPEED+OFFSET_S)
            for name, u in zip((*TRAINING_NAMES, 'heldout'),
                               (*TRAINING_DIRECTIONS, HELDOUT_DIRECTION))}


def generate_signals() -> dict[str, np.ndarray]:
    """Return six fixed channels x 32000 observations, without writing files."""
    time = np.arange(SAMPLES)/SAMPLE_RATE
    reference = _source(time-BASE_DELAY_S)
    result = {'source': reference[None, :]}
    for name, tau in truth_delays().items():
        if abs(tau) >= .001:
            raise ValueError('fixture must satisfy the stated unambiguous phase-delay prior')
        result[name] = np.stack((reference, _source(time-BASE_DELAY_S-tau)))
    return result


def _pairs(values: np.ndarray) -> list:
    return np.stack((np.real(values), np.imag(values)), axis=-1).tolist()


def analytic_measurements(name: str) -> dict:
    if name not in FILE_NAMES:
        raise ValueError('unknown baseline fixture')
    delays = [BASE_DELAY_S] if name == 'source' else [BASE_DELAY_S, BASE_DELAY_S+truth_delays()[name]]
    phasors = -1j*AMPLITUDE*np.exp(-1j*2*np.pi*FREQUENCY*np.asarray(delays))
    result = {'channels': len(delays), 'phasor_real_imag': _pairs(phasors),
              'amplitudes': [AMPLITUDE]*len(delays),
              'mean_square_per_channel': [AMPLITUDE**2/2]*len(delays),
              'mean_square_all_channels': AMPLITUDE**2/2}
    if name != 'source':
        tau = truth_delays()[name]
        result.update({'relative_phase_rad': -2*np.pi*FREQUENCY*tau, 'delay_s': tau})
    return result


def measure_signal(samples: np.ndarray) -> dict:
    """Measure known-frequency phase only, requiring positive channel phasors.

    Input is finite real channels x samples, mono source or stereo observation.
    Squared-sum overflow and unrepresentable phasors fail explicitly.
    """
    x = finite_real_array(samples, 'samples')
    if x.shape not in ((1, SAMPLES), (2, SAMPLES)):
        raise ValueError('require finite one- or two-channel 32000-sample input')
    start, stop = SCORING_INTERVAL
    time = np.arange(start, stop)/SAMPLE_RATE
    design = np.column_stack((np.cos(2*np.pi*FREQUENCY*time), np.sin(2*np.pi*FREQUENCY*time)))
    window = x[:, start:stop]
    peak = float(np.max(np.abs(window)))
    if peak == 0:
        raise ValueError('scoring window must have nonzero energy')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        coefficients = np.linalg.lstsq(design, window.T/peak, rcond=None)[0]*peak
        z = coefficients[0]-1j*coefficients[1]
        energies = np.sum(window**2, axis=1)
        residuals = np.sum((window.T-design@coefficients)**2, axis=0)
    if (not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(energies))
            or not np.all(np.isfinite(residuals)) or np.any(energies <= 0)
            or np.any(np.abs(z) == 0)):
        raise ValueError('phasors and squared sums must be positive and representable')
    result = {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
              'channels': len(x), 'phasor_real_imag': _pairs(z), 'amplitudes': abs(z).tolist(),
              'squared_sum_per_channel': energies.tolist(),
              'mean_square_per_channel': (energies/(stop-start)).tolist(),
              'mean_square_all_channels': float(np.mean(energies/(stop-start))),
              'known_tone_fit_residual_squared_sum_per_channel': residuals.tolist(),
              'peak': float(np.max(np.abs(x)))}
    if len(x) == 2:
        # Normalize first: the phase comparison must not overflow at z1*conj(z0).
        phase = float(np.angle((z[1]/abs(z[1]))*np.conj(z[0]/abs(z[0]))))
        tau = -phase/(2*np.pi*FREQUENCY)
        if abs(tau) >= .001:
            raise ValueError('phase boundary violates the strict |tau| < 1 ms prior')
        result.update({'relative_phase_rad': phase, 'delay_s': tau})
    return result


def calibration_results(measurements: dict[str, dict]) -> dict:
    """Fit four training delays, then predict the excluded direction.

    The augmented estimate uses the sole baseline_calibration implementation.
    The separate fixed-zero-offset control uses the ordinary geometric design.
    """
    delays = np.array([measurements[name]['delay_s'] for name in TRAINING_NAMES])
    solution = solve_baseline(TRAINING_DIRECTIONS, delays, SOUND_SPEED)
    baseline = np.asarray(solution['baseline_m'])
    prediction = float(-np.dot(HELDOUT_DIRECTION, baseline)/SOUND_SPEED+solution['offset_s'])
    measured = float(measurements['heldout']['delay_s'])
    truth = truth_delays()['heldout']
    wrong_baseline = np.linalg.lstsq(np.asarray(TRAINING_DIRECTIONS), -SOUND_SPEED*delays, rcond=None)[0]
    wrong_prediction = float(-np.dot(HELDOUT_DIRECTION, wrong_baseline)/SOUND_SPEED)
    return {'training_names': list(TRAINING_NAMES), 'solver': solution,
            'baseline_error_m': (baseline-np.asarray(BASELINE_M)).tolist(),
            'offset_error_s': float(solution['offset_s']-OFFSET_S),
            'heldout': {'direction_xyz': list(HELDOUT_DIRECTION), 'known_truth_delay_s': truth,
                        'measured_delay_s': measured, 'predicted_delay_s': prediction,
                        'prediction_minus_measurement_s': prediction-measured,
                        'prediction_minus_truth_s': prediction-truth},
            'fixed_zero_offset_control': {
                'offset_s': 0.0, 'baseline_m': wrong_baseline.tolist(),
                'baseline_error_m': (wrong_baseline-np.asarray(BASELINE_M)).tolist(),
                'heldout_predicted_delay_s': wrong_prediction,
                'heldout_prediction_minus_measurement_s': wrong_prediction-measured,
                'heldout_prediction_minus_truth_s': wrong_prediction-truth,
                'meaning': 'deliberately omitted offset; not the augmented calibrated estimator'}}
