"""Known failed-channel control with fixed real weights and time covariance.

Three deterministic orthogonal tones give the stated covariance only over the
specified integer-period window. This is not a per-frequency SCM estimate,
blind fault detector, physical propagation model or device performance result.
"""
from __future__ import annotations
import numpy as np
from codes.chapters.ch05.core.beamforming import mvdr_weights, apply_beamformer
from codes.chapters.ch10.core.channel_selection import select_mvdr_channels, select_channel_observations

SAMPLE_RATE = 16000
SAMPLES = 32000
SCORING_INTERVAL = (2400, 29600)
FILE_NAMES = {name: 'channel_'+name+'.wav' for name in (
    'reference', 'healthy_array', 'faulty_array', 'healthy_output', 'stale_output', 'recomputed_output')}
LIMITS = ('Known failed-channel, fixed-real-weight time-domain covariance control. '
          'The target and three noise bases are mathematical sinusoids; finite-window orthogonality '
          'is not random independence. The broadband noise covariance is known, not a per-frequency '
          'SCM estimate. Channel zero is prescribed failed and replaced by zero, not diagnosed. '
          'Selection removes both covariance axes and the matching steering and observation entries; '
          'it does not preserve the former noise optimum. No physical propagation, speech, device '
          'recording, blind fault detection, per-file normalization, reference gain/delay fitting '
          'or formal listening study. Equal analytic total NMSE does not imply equal target retention; '
          'the two actual PCM NMSE values need not be exactly equal.')


def parameters():
    return {'exercise_id': 'E10-34', 'sample_rate_hz': 16000, 'samples_per_channel': 32000,
            'target_frequency_hz': 500.0, 'noise_basis_frequencies_hz': [1500.0, 2500.0, 3500.0],
            'amplitude_each': 0.1, 'phase': 'sine, phase zero at the common absolute sample origin',
            'fade_samples_each_end': 320,
            'envelope': 'sin(linspace(0,pi/2,320)) squared; reversed at end; both endpoints zero',
            'common_export_gain': 1.0, 'tail_samples': 0,
            'scoring_interval_samples': [2400, 29600], 'scoring_samples_per_channel': 27200,
            'target_cycles_in_score': 850, 'noise_basis_cycles_in_score': [2550, 4250, 5950],
            'mixing_matrix_B': [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            'steering': [1.0, 1.0, 1.0], 'known_failed_channel': 0, 'selected_channels': [1, 2],
            'fault': 'replace channel zero by zeros before applying the former full weights',
            'noise_covariance': 'known score-window time covariance 0.005*B*B.T; fixed real weights across all tones',
            'solver': 'unique Chapter05 MVDR; selection by Chapter10 channel_selection; no loading',
            'scoring': 'same samples and common reference; no output gain/time matching',
            'randomness': 'none', 'noise_scope': 'three deterministic orthogonal basis tones, not random noise'}


def generate_experiment():
    time = np.arange(SAMPLES) / SAMPLE_RATE
    envelope = np.ones(SAMPLES)
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    envelope[:320], envelope[-320:] = fade, fade[::-1]
    reference = .1*np.sin(2*np.pi*500*time)*envelope
    bases = np.array([.1*np.sin(2*np.pi*f*time)*envelope for f in (1500, 2500, 3500)])
    matrix = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.]])
    covariance = .005*(matrix@matrix.T)
    steering = np.ones(3)
    full_weights = mvdr_weights(covariance, steering)
    selected = select_mvdr_channels(covariance, steering, [1, 2])
    healthy = reference[None]+matrix@bases
    faulty = healthy.copy(); faulty[0] = 0
    # A singleton contraction axis only adapts the existing weight-application
    # API. These samples are time values, not STFT coefficients or frames.
    def apply(values, weights):
        return apply_beamformer(values[:, None, :], weights).real
    signals = {'reference': reference[None], 'healthy_array': healthy, 'faulty_array': faulty,
               'healthy_output': apply(healthy, full_weights),
               'stale_output': apply(faulty, full_weights),
               'recomputed_output': apply(select_channel_observations(faulty, [1, 2]), selected['weights'])}
    return {'signals': signals, 'noise_bases': bases, 'noise_covariance': covariance,
            'full_weights': full_weights.real, 'selected_covariance': selected['covariance'].real,
            'selected_steering': selected['steering'].real, 'selected_weights': selected['weights'].real,
            'selection_matrix': selected['selection_matrix'],
            'stale_remaining_weights': full_weights.real[[1, 2]]}


def analytic_measurements():
    return {name: {'target_gain': gain, 'target_mean_square': .005*gain**2,
                  'noise_mean_square': noise, 'output_mean_square': .005*gain**2+noise,
                  'reference_error_mean_square': .005*(gain-1)**2+noise,
                  'reference_NMSE': (gain-1)**2+noise/.005}
            for name, gain, noise in [('healthy_output', 1.0, .0025),
                                      ('stale_output', 0.0, .0025),
                                      ('recomputed_output', 1.0, .0075)]}


def measure_signal(signal, reference):
    signal, reference = np.asarray(signal), np.asarray(reference)
    if (signal.dtype.kind != 'f' or signal.ndim != 2 or signal.shape[0] not in (1, 3)
            or signal.shape[1] != 32000 or not np.isfinite(signal).all()
            or reference.dtype.kind != 'f' or reference.shape != (1, 32000)
            or not np.isfinite(reference).all()):
        raise ValueError('finite float mono/three-channel waveforms and mono reference required')
    lo, hi = SCORING_INTERVAL
    values, truth = signal[:, lo:hi], reference[:, lo:hi]
    energy = float(np.sum(truth**2))
    if energy <= 0 or not np.isfinite(energy):
        raise ValueError('positive finite reference energy required')
    error = values-truth
    time = np.arange(lo, hi)/16000
    # Report known-frequency diagnostics without changing output gain or delay.
    phasor = 2*np.mean(values*np.exp(-2j*np.pi*500*time), axis=1)
    result = {'scoring_interval_samples': [lo, hi], 'samples_per_channel': hi-lo,
            'mean_square_per_channel': np.mean(values**2, axis=1).tolist(),
            'reference_error_mean_square_per_channel': np.mean(error**2, axis=1).tolist(),
            'reference_NMSE_per_channel': (np.sum(error**2, axis=1)/energy).tolist(),
            'target_frequency_phasor_real_per_channel': phasor.real.tolist(),
            'target_frequency_phasor_imag_per_channel': phasor.imag.tolist(),
            'phasor_convention': '2/N sum x[n] exp(-j*2*pi*500*n/fs); .1 sine -> -.1j; diagnostic only'}
    for key, value in result.items():
        if key.endswith('_per_channel') and not np.isfinite(value).all():
            raise ValueError('measurement arithmetic is outside finite support')
    return result
