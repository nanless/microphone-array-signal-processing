"""E05-23: known two-tone resynthesis illustrates frequency-dependent phase.

This fixture applies declared complex frequency weights to known components.
It runs no eigensolver, STFT, BAN, noise estimator or speech enhancement chain.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array

SAMPLE_RATE = 16000
SAMPLES = 32000
FREQUENCIES = (500.0, 1500.0)
AMPLITUDE = 0.1
SCORING_INTERVAL = (2400, 29600)
FILE_NAMES = {'reference': 'phase_reference.wav', 'flip': 'phase_flip.wav',
              'quadrature': 'phase_quadrature.wav'}
LIMITS = ('Original known-component mathematical resynthesis, not speech, room or device audio. '
          'The two identical target channels have base weight [0.5,0.5]. '
          'Only the high-frequency weight is multiplied by -1 or +j; conjugate application '
          'makes its output respectively -cos or +sin. No GEV eigensolver, STFT, BAN '
          'or unknown frequency-component extraction is executed. No noise is added. '
          'Equal output energy does not establish reference waveform preservation. '
          'The steady integer-cycle window excludes fades; scores preserve gain and clock. '
          'Cos/sin projections are diagnostics, never applied as gain or delay compensation. '
          'Playback is not a formal listening or intelligibility study.')


def parameters() -> dict:
    return {'exercise_id': 'E05-23', 'sample_rate_hz': SAMPLE_RATE,
            'samples_per_channel': SAMPLES, 'frequencies_hz': list(FREQUENCIES),
            'source_amplitude_per_frequency': AMPLITUDE, 'source_duration_s': 2.0,
            'fade_samples': 320, 'fade': 'squared sine, endpoints included, common envelope',
            'common_export_gain': 1.0, 'channel_order': ['mono_known_component_output'],
            'target_channel_model': 'two identical target channels for each known frequency component',
            'base_weights_real_imag': [[0.5, 0.0], [0.5, 0.0]],
            'high_weight_multiplier_real_imag': {'reference': [1.0, 0.0],
                                                'flip': [-1.0, 0.0], 'quadrature': [0.0, 1.0]},
            'output_model': 'A[n]*0.1*(cos(500Hz)+cos(1500Hz)); high output -cos for flip, +sin for quadrature',
            'application_convention': 'y=w^H*x; output multiplier is conjugate of weight multiplier',
            'generation_model': 'known real frequency components resynthesized before common envelope; no STFT',
            'propagation_delay_samples': 0, 'algorithm_delay_samples': 0,
            'scoring_interval_samples': list(SCORING_INTERVAL), 'samples_in_scoring_window': 27200,
            'scoring_cycles_per_frequency': [850, 2550],
            'phasor_convention': 'cos coefficient minus j*sin coefficient; absolute sample clock',
            'alignment': 'original sample clock and gain; no fitted correction',
            'noise': 'none', 'randomness': 'none'}


def _basis(n: np.ndarray) -> np.ndarray:
    # Both tones occupy exact residues on a 32-point common period. Avoid
    # accumulating absolute-clock trigonometric roundoff over 32000 samples.
    phases = 2*np.pi*np.arange(32)/32
    return np.column_stack((np.cos(phases), np.sin(phases),
                            np.cos(3*phases), np.sin(3*phases)))[n % 32]


def generate_signals() -> dict[str, np.ndarray]:
    n = np.arange(SAMPLES)
    basis = _basis(n)
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    envelope = np.ones(SAMPLES)
    envelope[:320], envelope[-320:] = fade, fade[::-1]
    low, high_cos, high_sin = basis[:, 0], basis[:, 2], basis[:, 3]
    return {name: (AMPLITUDE*envelope*signal)[None, :]
            for name, signal in [('reference', low+high_cos), ('flip', low-high_cos),
                                 ('quadrature', low+high_sin)]}


def analytic_measurements(name: str) -> dict:
    if name not in FILE_NAMES:
        raise ValueError('unknown phase fixture')
    ratios = {'reference': [[1.0, 0.0], [1.0, 0.0]],
              'flip': [[1.0, 0.0], [-1.0, 0.0]],
              'quadrature': [[1.0, 0.0], [0.0, -1.0]]}[name]
    error = {'reference': 0.0, 'flip': 0.02, 'quadrature': 0.01}[name]
    return {'mean_square': 0.01, 'reference_mean_square': 0.01,
            'total_reference_mse': error, 'normalized_squared_reference_error': error/0.01,
            'response_real_imag': ratios,
            'common_pure_delay_possible': name == 'reference',
            'delay_control': 'preserving 500Hz requires 500*tau integer; then 1500*tau is also integer'}


def measure_signal(samples: np.ndarray, reference: np.ndarray) -> dict:
    x, r = finite_real_array(samples, 'samples'), finite_real_array(reference, 'reference')
    if x.shape != (1, SAMPLES) or r.shape != (1, SAMPLES):
        raise ValueError('require mono 32000-sample signals and reference')
    start, stop = SCORING_INTERVAL
    design = _basis(np.arange(start, stop))
    window, truth = x[0, start:stop], r[0, start:stop]
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        coeff = 2*design.T@window/(stop-start)
        ref_coeff = 2*design.T@truth/(stop-start)
        phasor = coeff[::2]-1j*coeff[1::2]
        ref_phasor = ref_coeff[::2]-1j*ref_coeff[1::2]
        energy, ref_energy = float(window@window), float(truth@truth)
        error = float((window-truth)@(window-truth))
        ratio = phasor/ref_phasor
    if (not np.all(np.isfinite(ratio)) or not np.isfinite(energy) or energy <= 0
            or not np.isfinite(ref_energy) or ref_energy <= 0
            or not np.isfinite(error) or np.any(abs(ref_phasor) == 0)):
        raise ValueError('reference/tone energies and scores must be positive and representable')
    pairs = lambda z: np.column_stack((z.real, z.imag)).tolist()
    return {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
            'channels': 1, 'squared_sum': energy, 'reference_squared_sum': ref_energy,
            'error_squared_sum': error, 'mean_square': energy/(stop-start),
            'total_reference_mse': error/(stop-start),
            'normalized_squared_reference_error': error/ref_energy,
            'phasor_real_imag': pairs(phasor), 'reference_phasor_real_imag': pairs(ref_phasor),
            'response_real_imag': pairs(ratio), 'real_projection_columns': 4,
            'projection_scope': 'known integer-cycle cos/sin diagnostic; no correction applied',
            'peak': float(np.max(abs(x)))}
