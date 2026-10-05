"""Run E12-01..18 without writing files, downloading data or importing Acoular.

Usage: python -m codes.chapters.ch12.chapter12_exercises [--exercise E12-06]
The JSON wrapper keeps metadata separate from the stable ID-to-result map.
"""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import numpy as np
from codes.chapters.ch02.core.spectral import stft, periodic_hann
from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch12.core.imaging import (
    two_cell_experiment, spherical_scan_experiment, scan_power, conventional_weights,
    point_spread_function, damas_gauss_seidel, finite_nnls, clean_sc_full_csm,
    hermitian_real_vector, csm_residual, single_source_csm_fit,
    one_sided_csm_density, integrate_psd, region_power,
    damas_csm_objective_experiment, distinct_column_ambiguity_experiment,
)
from codes.chapters.ch12.core.imaging_audio import (
    make_signals, measure_signal, measure_source_pairs, SAMPLE_RATE, parameters,
)

EXERCISE_IDS = tuple('E12-'+str(i).zfill(2) for i in range(1, 19))


def _json(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return np.stack([value.real, value.imag], axis=-1).tolist()
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def run_experiments():
    """Return finite JSON with exercises[E12-01..18]; no file IO or side effects."""
    toy = two_cell_experiment(); a, w, p = toy['A'], toy['W'], toy['P']
    cases = toy['cases']; independent = cases['independent']; coherent = cases['coherent']
    sphere = spherical_scan_experiment()
    signals = make_signals()
    decoded = {key: read_pcm16(pcm16_bytes(array, SAMPLE_RATE))[1] for key, array in signals.items()}
    pcm = {key: measure_signal(array, key, pcm=True) for key, array in decoded.items()}
    source_pairs = measure_source_pairs(decoded)
    quadrature = a@np.array([1., -.5j])
    quadrature_r = np.outer(quadrature, quadrature.conj())
    quadrature_b = scan_power(quadrature_r, w)
    forward = damas_gauss_seidel(p, independent['b'], iterations=2)
    both = damas_gauss_seidel(p, independent['b'], iterations=2, sweep='forward_backward')
    slow_p = np.array([[1., .99], [.99, 1.]])
    slow = damas_gauss_seidel(slow_p, slow_p@toy['q'], iterations=100)
    duplicate_p = np.ones((2, 2)); duplicate_b = np.array([1.25, 1.25])
    off = np.array([1., np.exp(-1j*np.pi/3)])
    off_r = np.outer(off, off.conj()); off_b = scan_power(off_r, w); off_q = np.linalg.solve(p, off_b)
    off_model = a@np.diag(off_q)@a.conj().T
    removed_r = independent['R']-np.diag(np.diag(independent['R']))
    removed_b = 2*scan_power(removed_r, w)
    unequal_a = np.array([1., 2.], complex)
    unequal_w = conventional_weights(unequal_a[:, None])
    unequal_r = np.outer(unequal_a, unequal_a.conj())
    unequal_removed_r = unequal_r-np.diag(np.diag(unequal_r))
    unequal_full_scan = float(scan_power(unequal_r, unequal_w)[0])
    unequal_removed_scan = float(scan_power(unequal_removed_r, unequal_w)[0])
    unequal_energy = float(np.vdot(unequal_a, unequal_a).real)
    unequal_fourth_sum = float(np.sum(abs(unequal_a)**4))
    unequal_compensation = unequal_energy**2/(unequal_energy**2-unequal_fourth_sum)
    microphone_compensation = len(unequal_a)/(len(unequal_a)-1)
    unequal_control = {
        'a_real_imag': unequal_a, 'weights_real_imag': unequal_w[:, 0],
        'full_csm_real_imag': unequal_r, 'diagonal_removed_csm_real_imag': unequal_removed_r,
        'full_scan': unequal_full_scan, 'diagonal_removed_scan': unequal_removed_scan,
        'microphone_count_compensation': microphone_compensation,
        'scan_with_microphone_count_compensation': unequal_removed_scan*microphone_compensation,
        'matched_compensation': unequal_compensation,
        'scan_with_matched_compensation': unequal_removed_scan*unequal_compensation,
    }
    inconsistent_b = np.array([1., 0.])
    valid_v = np.sqrt(4/3)*np.array([1., np.exp(1j*np.pi/3)])
    gains = np.array([.5, 1., 2.])
    uncorrected, corrected = [], []
    for gain in gains:
        v = np.array([1., gain]); r = np.outer(v, v)
        uncorrected.append(float(scan_power(r, w[:, :1])[0]))
        calibration = np.diag([1., 1/gain])
        corrected.append(float(scan_power(calibration@r@calibration, w[:, :1])[0]))
    clean = clean_sc_full_csm(2*np.ones((2, 2)), np.ones((2, 1))/2)
    clean_two = clean_sc_full_csm(independent['R'], w, iterations=2, damping=1.)
    fit_a = np.array([1., 2.]); fit_r = np.diag([1., 10.]).astype(complex); template = np.outer(fit_a, fit_a)
    vf = hermitian_real_vector(template); vu = hermitian_real_vector(template, frobenius=False)
    yf = hermitian_real_vector(fit_r); yu = hermitian_real_vector(fit_r, frobenius=False)
    fit_full = float(vf@yf/(vf@vf)); fit_upper = float(vu@yu/(vu@vu))
    affine = np.linalg.lstsq(np.column_stack([vu, np.ones(len(vu))]), yu, rcond=None)[0]
    n_fft = 24; fs = 24000; n = np.arange(n_fft)
    spectrum_examples = {}
    for name, window, x in [
        ('rectangular_tone', np.ones(n_fft), .2*np.cos(2*np.pi*2*n/n_fft)),
        ('hann_tone', periodic_hann(n_fft), .2*np.cos(2*np.pi*2*n/n_fft)),
        ('dc', np.ones(n_fft), np.full(n_fft, .2)),
        ('nyquist', np.ones(n_fft), .2*(-1.)**n),
    ]:
        spectra = stft(x[None, :], n_fft=n_fft, hop_length=n_fft, window=window, center=False)
        density = one_sided_csm_density(spectra, sample_rate_hz=fs, n_fft=n_fft, window_energy=window@window)[:, 0, 0].real
        spectrum_examples[name] = {'window_energy': float(window@window), 'window_sum': float(window.sum()),
                                   'density': density, 'bin_mean_square': density*(fs/n_fft),
                                   'integrated_mean_square': float(integrate_psd(density, fs/n_fft))}
    all_cells = np.ones(2, dtype=bool)
    joint_r = np.array([[2.5, 4.], [4., 8.5]], complex)
    exercises = {
        'E12-01': {'peak_amplitude': .2, 'mean_square': .2**2/2,
                   'normalization_power_scale': .02, 'CSM_scaling_for_double_amplitude': 4.,
                   'units': 'digital amplitude squared; no acoustic calibration'},
        'E12-02': {'A_real_imag': a, 'W_real_imag': w, 'response_real_imag': w.conj().T@a,
                   'P': p, 'q': toy['q'], 'b': independent['b'], 'P_condition_number': float(np.linalg.cond(p))},
        'E12-03': {'forward': forward, 'forward_backward': both,
                   'nonunit_diagonal': damas_gauss_seidel(2*p, 2*independent['b'], iterations=2),
                   'slow_P': slow_p, 'slow_condition_number': float(np.linalg.cond(slow_p)),
                   'slow_q_after_100': slow['q'], 'slow_error_factor_per_forward_sweep': .99**2},
        'E12-04': {'audio_parameters': parameters(), 'PCM_source_pairs': source_pairs,
                   'quadrature_source_amplitudes_real_imag': np.array([1., -.5j]),
                   'quadrature_R_real_imag': quadrature_r, 'quadrature_b': quadrature_b,
                   'quadrature_inverse_q': np.linalg.solve(p, quadrature_b),
                   'quadrature_magnitude_squared_coherence': 1.,
                   'boundary': 'time-orthogonal same-frequency tones remain spectrally coherent'},
        'E12-05': {'q_true': toy['q'], 'R_real_imag': coherent['R'], 'b': coherent['b'],
                   'q_inverse': coherent['q_inverse'], 'scan_residual': p@coherent['q_inverse']-coherent['b'],
                   'full_csm_residual': coherent['full_csm_residual'],
                   'coherence_gamma': toy['coherence_gamma'], 'coherence_inverse_q': toy['coherence_inverse_q'],
                   'PCM': pcm['array_coherent']},
        'E12-06': {'duplicate_P': duplicate_p, 'duplicate_b': duplicate_b,
                   'two_indistinguishable_q': [[1., .25], [.75, .5]],
                   'duplicate_GS': damas_gauss_seidel(duplicate_p, duplicate_b, iterations=1),
                   'offgrid_b': off_b, 'offgrid_inverse_q': off_q,
                   'offgrid_scan_residual': p@off_q-off_b, 'offgrid_full_csm_residual': csm_residual(off_r, off_model),
                   'spherical_441_grid': {key: sphere[key] for key in (
                       'grid_shape', 'frequency_hz', 'sound_speed_m_s', 'source_positions_m', 'q',
                       'local_peaks', 'dirty_grid_sum', 'source_power_sum', 'scope')},
                   'spherical_source_cell_scan': sphere['b'][sphere['source_grid_indices']],
                   'spherical_source_cell_P': sphere['psf_columns'][sphere['source_grid_indices']]},
        'E12-07': {'diagonal_removed_R_real_imag': removed_r, 'eigenvalues': np.linalg.eigvalsh(removed_r),
                   'normalized_removed_scan': removed_b, 'clipped_scan': np.maximum(removed_b, 0),
                   'changed_P': 2*p-1., 'original_P_is_inapplicable': True,
                   'unequal_amplitude_control': unequal_control},
        'E12-08': {'P': p, 'b': inconsistent_b, 'valid_input_R_real_imag': np.outer(valid_v, valid_v.conj()),
                   'valid_input_scan': scan_power(np.outer(valid_v, valid_v.conj()), w),
                   'GS': damas_gauss_seidel(p, inconsistent_b, iterations=10),
                   'NNLS': finite_nnls(p, inconsistent_b),
                   'unconstrained_inverse_q': np.linalg.solve(p, inconsistent_b),
                   'scope': 'GS fixed point and least-squares NNLS have different objectives for incompatible data'},
        'E12-09': {'second_channel_gains': gains, 'uncorrected_power': uncorrected,
                   'calibrated_power': corrected, 'reference_power': 1.},
        'E12-10': {'point_source_total': float(toy['q'].sum()), 'dirty_two_cell_total': float(independent['b'].sum()),
                   'point_source_total_db_re_1': float(10*np.log10(toy['q'].sum())),
                   'dirty_two_cell_total_db_re_1': float(10*np.log10(independent['b'].sum())),
                   'incorrect_average_source_db': float(np.mean(10*np.log10(toy['q']))),
                   'PCM_measurements': pcm, 'audio_group_is_independent_of_main_109_WAVs': True},
        'E12-11': {'single_source_clean': clean, 'two_source_damping_1': clean_two,
                   'two_source_true_q': toy['q'],
                   'boundary': 'zero full residual need not give the original physical source powers'},
        'E12-12': {'a': fit_a, 'R_real_imag': fit_r, 'full_frobenius_q': fit_full,
                   'upper_unweighted_q': fit_upper, 'upper_real_design': vu, 'upper_real_target': yu,
                   'affine_math_q': float(affine[0]), 'affine_math_intercept': float(affine[1]),
                   'sklearn_executed': False,
                   'full_fit_residual': csm_residual(fit_r, fit_full*template),
                   'upper_fit_residual': csm_residual(fit_r, fit_upper*template)},
        'E12-13': {'sample_rate_hz': fs, 'n_fft': n_fft, 'bin_width_hz': fs/n_fft,
                   'examples': spectrum_examples, 'shared_STFT': 'codes.chapters.ch02.core.spectral.stft',
                   'endpoint_rule': 'factor 1 at DC/even Nyquist; factor 2 at all other rFFT bins'},
        'E12-14': {'original_reference_position_m': [0., 0., 0.], 'original_q': sphere['q'],
                   'new_reference': sphere['reference_change'], 'power_multiplier': 153/73,
                   'boundary': 'reference pressure power changes; sensor CSM does not'},
        'E12-15': {'density_per_hz': [.01, .02], 'frequency_bin_widths_hz': [100., 200.],
                   'integrated_frequency_power': float(integrate_psd([.01, .02], [100., 200.])),
                   'point_quantities': [1., .25], 'point_total': region_power([1., .25], all_cells),
                   'area_density_quantities': [1., .25], 'cell_areas_m2': [.04, .04],
                   'density_area_total': region_power([1., .25], all_cells, cell_area_m2=[.04, .04])},
        'E12-16': {'a': fit_a, 'R_real_imag': joint_r,
                   'joint_fit': single_source_csm_fit(joint_r, fit_a, include_white_noise=True),
                   'one_microphone': single_source_csm_fit([[2.5]], [1.], include_white_noise=True),
                   'one_microphone_identifiable_relation': 'q + sigma_squared = 2.5',
                   'scope': 'known single steering, independent equal white sensor noise; not constrained production CMF'},
        'E12-17': damas_csm_objective_experiment(),
        'E12-18': distinct_column_ambiguity_experiment(),
    }
    return _json({'schema_version': 1, 'scope': 'original finite known-model teaching experiments, no industrial ranking',
                  'complex_encoding': 'last dimension [real,imag] for keys marked real_imag and complex states',
                  'exercises': exercises})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exercise', choices=EXERCISE_IDS)
    args = parser.parse_args(argv)
    result = run_experiments()
    if args.exercise:
        result['exercises'] = {args.exercise: result['exercises'][args.exercise]}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
