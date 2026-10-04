"""E04-24 known coherent direct/reflection mathematical tone fixture.

The single-frequency phasor outer product follows from known deterministic
components. It does not establish random independence or identify direct sound.
The optional 0.1 I covariance is a virtual model, never noise in these WAVs.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
SAMPLES = 32012
FREQUENCY = 4000.0
AMPLITUDE = .1
SCORING_INTERVAL = (2400, 29600)
FILE_NAMES = {'reference': 'reflection_reference.wav', 'direct': 'reflection_direct.wav',
              'reflection': 'reflection_component.wav', 'mixed': 'reflection_mixed.wav'}
LIMITS = ('Original deterministic mathematical 4-kHz tone, not speech, a real room or hardware. '
          'Known direct and reflected components share one source and are coherent. '
          'The reference is the undelayed source; direct delays are 8/8 samples and reflection '
          'delays 12/11 samples. The common extra four samples are a whole tone period. '
          'Known single-tone cos/sin projections are not blind DOA, DPD or CTF estimation. '
          'A phase-only apparent angle does not match the mixed vector amplitude ratio. '
          'Single-vector covariance rank one is an algebraic property of this known model, '
          'not evidence of unknown direct sound or random statistical independence. '
          'The separate virtual covariance vv^H+0.1I contains a model white-noise term; '
          'the actual WAVs contain no added noise. Fades and tail are excluded from scoring. '
          'Playback is not localization accuracy or a formal listening study.')


def parameters() -> dict:
    return {'exercise_id': 'E04-24', 'sample_rate_hz': SAMPLE_RATE,
            'source_samples': SOURCE_SAMPLES, 'samples_per_channel': SAMPLES,
            'frequency_hz': FREQUENCY, 'source_amplitude': AMPLITUDE,
            'sound_speed_m_s': 343.0, 'microphone_spacing_m': .042875,
            'source_direction_convention': 'unit vector from array toward source; tau=-d*sin(theta)/c',
            'direct_angle_deg': 0.0, 'reflection_angle_deg': 30.0,
            'direct_delay_samples': [8, 8], 'reflection_delay_samples': [12, 11],
            'common_export_gain': 1.0, 'fade_samples': 320,
            'fade': 'squared sine, endpoints included, before all delays',
            'waveform_model': 'reference=F; direct=[D8 F,D8 F]; reflection=[D12 F,D11 F]; mixed=direct+reflection',
            'delay_implementation': 'integer causal delay, zero extension, complete 12-sample tail',
            'channel_order': ['microphone_0', 'microphone_1'],
            'scoring_interval_samples': [2400, 29600], 'samples_in_scoring_window': 27200,
            'scoring_cycles': 6800,
            'phasor_convention': 'cos coefficient minus j*sin coefficient; absolute sample clock',
            'normalized_covariance': '(z/0.1)(z/0.1)^H; single known phasor outer product',
            'virtual_covariance_diagonal': .1,
            'virtual_covariance_scope': 'model control only; no noise added to any WAV',
            'randomness': 'none', 'noise': 'none'}


def generate_signals() -> dict[str, np.ndarray]:
    """Return four full-length channel-by-sample signals without writing."""
    n = np.arange(SOURCE_SAMPLES)
    envelope = np.ones(SOURCE_SAMPLES)
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    envelope[:320], envelope[-320:] = fade, fade[::-1]
    # Exact cos(pi*n/2) on this quarter-cycle sample grid.
    source = AMPLITUDE*np.array([1., 0., -1., 0.])[n%4]*envelope
    def delayed(lag):
        return np.pad(source, (lag, SAMPLES-SOURCE_SAMPLES-lag))
    direct = np.stack((delayed(8), delayed(8)))
    reflection = np.stack((delayed(12), delayed(11)))
    return {'reference': delayed(0)[None, :], 'direct': direct,
            'reflection': reflection, 'mixed': direct+reflection}


def _pairs(z) -> list:
    z = np.asarray(z)
    return np.stack((z.real, z.imag), axis=-1).tolist()


def _phasor_results(z) -> dict:
    normalized = z/AMPLITUDE
    covariance = np.outer(normalized, normalized.conj())
    result = {'phasor_real_imag': _pairs(z), 'amplitudes': np.abs(z).tolist(),
              'normalized_phasor_real_imag': _pairs(normalized),
              'normalizing_source_amplitude': AMPLITUDE,
              'normalized_outer_product_real_imag': _pairs(covariance),
              'outer_product_denominator': 1,
              'outer_product_eigenvalues': np.linalg.eigvalsh(covariance).tolist(),
              'outer_product_rank': 1}
    if len(z) == 2:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            ratio = z[1]/z[0]
        if not np.isfinite(ratio) or abs(ratio) == 0:
            raise ValueError("channel phasor ratio must be finite and nonzero")
        phase = float(np.angle(ratio))
        result.update({'channel_1_over_channel_0_real_imag': _pairs(ratio),
                       'relative_phase_rad': phase, 'amplitude_ratio': float(abs(ratio)),
                       'phase_only_apparent_angle_deg': float(np.rad2deg(np.arcsin(phase/np.pi))),
                       'phase_angle_scope': 'principal visible half-wave branch; does not fit amplitude ratio'})
    return result


def analytic_measurements(name: str) -> dict:
    vectors = {'reference': [1], 'direct': [1, 1], 'reflection': [1, 1j], 'mixed': [2, 1+1j]}
    if name not in vectors:
        raise ValueError('unknown reflection fixture')
    z = AMPLITUDE*np.asarray(vectors[name], dtype=complex)
    power = abs(z)**2/2
    return {'channels': len(z), 'mean_square_per_channel': power.tolist(),
            'mean_square_all_channels': float(np.mean(power)), **_phasor_results(z)}


def virtual_covariance_control() -> dict:
    """A separate symbolic covariance control, not a measurement of WAV noise."""
    v = np.array([2, 1+1j])
    matrix = np.outer(v, v.conj())+.1*np.eye(2)
    return {'normalized_mixed_phasor_real_imag': _pairs(v),
            'diagonal_white_noise_variance': .1, 'covariance_real_imag': _pairs(matrix),
            'eigenvalues': np.linalg.eigvalsh(matrix).tolist(),
            'scope': 'virtual vv^H+0.1I model only; actual WAVs have no added noise'}


def measure_signal(samples: np.ndarray) -> dict:
    """Fit one known tone with no delay or gain compensation; stable window only."""
    x = finite_real_array(samples, 'samples')
    if x.shape not in ((1, SAMPLES), (2, SAMPLES)):
        raise ValueError('require finite one- or two-channel 32012-sample input')
    start, stop = SCORING_INTERVAL
    # f/fs=1/4 exactly. Evaluate the four quadrature phases without
    # large-clock trigonometric roundoff; LS still fits both real columns.
    residue = np.arange(start, stop)%4
    design = np.column_stack((np.array([1., 0., -1., 0.])[residue],
                              np.array([0., 1., 0., -1.])[residue]))
    window = x[:, start:stop]
    peak = float(np.max(abs(window)))
    if peak == 0:
        raise ValueError('scoring window must have nonzero energy')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        coefficients = np.linalg.lstsq(design, window.T/peak, rcond=None)[0]*peak
        z = coefficients[0]-1j*coefficients[1]
        energies = np.sum(window**2, axis=1)
        residual = np.sum((window.T-design@coefficients)**2, axis=0)
        normalized_energy = np.sum(abs(z/AMPLITUDE)**2)
    if (not np.all(np.isfinite(z)) or np.any(abs(z) == 0)
            or not np.all(np.isfinite(energies)) or np.any(energies <= 0)
            or not np.all(np.isfinite(residual)) or not np.isfinite(normalized_energy)
            or normalized_energy <= 0):
        raise ValueError('phasor energies and squared sums must be positive and representable')
    result = {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
              'channels': len(x), 'real_ls_design_columns': 2,
              'squared_sum_per_channel': energies.tolist(),
              'mean_square_per_channel': (energies/(stop-start)).tolist(),
              'mean_square_all_channels': float(np.mean(energies/(stop-start))),
              'known_tone_fit_residual_squared_sum_per_channel': residual.tolist(),
              'peak': float(np.max(abs(x))), **_phasor_results(z)}
    return result
