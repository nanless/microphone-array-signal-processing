"""A known single-frequency model and hard constraints, not a device benchmark.

Weights reuse the Chapter 5 solvers. The actual interference covariance and the
separate 3-D diffuse coherence model are retained as different quantities.
No WAV, estimated DOA, measured latency or robust optimizer is produced.
"""
from __future__ import annotations
import numpy as np
from codes.chapters.ch03.core.geometry import plane_wave_steering
from codes.chapters.ch05.core.beamforming import dsb_weights, mvdr_weights, diffuse_coherence
from codes.chapters.ch11.core.selection import upper_limit_verdict


def _complex_pairs(values):
    return np.stack((np.asarray(values).real, np.asarray(values).imag), axis=-1).tolist()


def beamformer_selection_case():
    """Return four known candidates; all device latency evidence is absent."""
    positions = np.array([[-.04, 0., 0.], [0., 0., 0.], [.04, 0., 0.]])
    frequency, sound_speed = 1000., 343.
    vectors = {key: plane_wave_steering(positions, [frequency], np.deg2rad(angle),
               reference=1, sound_speed=sound_speed)[0]
               for key, angle in (('nominal', 0.), ('interference', 20.), ('actual', 5.))}
    a, b, actual = [vectors[key] for key in ('nominal', 'interference', 'actual')]
    covariance = np.outer(b, b.conj()) + .01*np.eye(3)
    gamma = diffuse_coherence(positions, [frequency], sound_speed=sound_speed)[0]
    candidates = []
    for identifier, label, alpha in [('ds', 'DS', None), ('mvdr_0', 'α=0', 0.),
                                      ('mvdr_0_1', 'α=0.1', .1), ('mvdr_1', 'α=1', 1.)]:
        weight = dsb_weights(a) if alpha is None else mvdr_weights(
            covariance, a, relative_diagonal_loading=alpha)
        nominal_response, actual_response = np.vdot(weight, a), np.vdot(weight, actual)
        norm = np.vdot(weight, weight).real
        noise = np.vdot(weight, covariance@weight).real
        diffuse = np.vdot(weight, gamma@weight).real
        wng = 10*np.log10(abs(nominal_response)**2/norm)
        di = 10*np.log10(abs(nominal_response)**2/diffuse)
        response_error = abs(actual_response-1)
        acoustic = upper_limit_verdict([[-float(wng), -float(wng)],
                                        [float(response_error), float(response_error)]], [0., .03])
        device = upper_limit_verdict([[-float(wng), -float(wng)],
                                      [float(response_error), float(response_error)], None], [0., .03, 150.])
        candidates.append({'candidate_id': identifier, 'label': label,
            'relative_diagonal_loading': alpha,
            'absolute_diagonal_loading': None if alpha is None else float(alpha*np.trace(covariance).real/3),
            'weights_real_imag': _complex_pairs(weight),
            'nominal_response_real_imag': _complex_pairs(nominal_response),
            'actual_response_real_imag': _complex_pairs(actual_response),
            'response_amplitude_error': float(response_error), 'WNG_dB': float(wng), 'DI_dB': float(di),
            'actual_noise_power': float(noise),
            'nominal_NMSE': float(abs(nominal_response-1)**2+noise),
            'actual_NMSE': float(response_error**2+noise), 'acoustic_verdict': acoustic,
            'measured_latency_ms': None, 'device_verdict': device})
    eligible = [row for row in candidates if row['acoustic_verdict'] == 'pass']
    return {'parameters': {'positions_m': positions.tolist(), 'frequency_hz': frequency,
        'sound_speed_m_s': sound_speed, 'reference_channel': 1, 'nominal_target_deg': 0.,
        'interference_deg': 20., 'actual_target_deg': 5., 'target_power': 1.,
        'interference_power': 1., 'independent_sensor_noise_power': .01,
        'azimuth_convention': '+y zero; positive toward +x',
        'WNG_lower_limit_dB': 0., 'response_amplitude_error_upper_limit': .03,
        'latency_upper_limit_ms': 150., 'diffuse_model': 'separate 3-D isotropic sinc coherence'},
        'noise_covariance_real_imag': _complex_pairs(covariance),
        'diffuse_coherence': gamma.real.tolist(),
        'steering_vectors_real_imag': {key: _complex_pairs(value) for key, value in vectors.items()},
        'candidates': candidates, 'acoustically_eligible': [row['candidate_id'] for row in eligible],
        'lowest_actual_NMSE_eligible': min(eligible, key=lambda row: row['actual_NMSE'])['candidate_id'],
        'device_selected': None,
        'scope': 'known single-frequency ensemble covariances; no device measurements, audio or optimizer for worst-case uncertainty'}
