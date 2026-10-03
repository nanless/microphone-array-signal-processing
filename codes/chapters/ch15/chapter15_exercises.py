"""Run E15-01..24 with NumPy and in-memory PCM; never write or download.

Usage: python -m codes.chapters.ch15.chapter15_exercises [--exercise E15-06]
Metadata is outside the stable exercise-ID map. Known-model controls are not
the MATLAB DANSE chain, a theorem proof, or a measured distributed system.
"""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch04.core.covariance import _load_covariance
from codes.chapters.ch15.core.distributed import (
    known_models, mwf_weights, mse_components, compressed_mwf, distributed_updates,
    run_covariance_experiments, jacobi_control, step_size_control, tree_sum_control, gevd_control,
)
from codes.chapters.ch15.core.distributed_audio import (
    SAMPLE_RATE, POWER, run_experiment, measure_signal, OUTPUT_REFERENCES,
)

EXERCISE_IDS = tuple('E15-'+str(i).zfill(2) for i in range(1, 25))


def _json(value):
    if isinstance(value, np.ndarray):
        return np.stack([value.real, value.imag], axis=-1).tolist() if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def resource_control():
    fs = 16000; bits = 16; channels = 2; compressed = 1
    network = []
    for nodes in (2, 4, 8):
        total = nodes*channels; d = channels+(nodes-1)*compressed
        network.append({'nodes': nodes, 'channels_per_node': channels, 'compressed_streams_per_node': compressed,
                        'central_upload_PCM16_bits_per_second': total*fs*bits,
                        'compressed_broadcast_PCM16_bits_per_second': nodes*compressed*fs*bits,
                        'compressed_full_mesh_unicast_PCM16_bits_per_second': nodes*(nodes-1)*compressed*fs*bits,
                        'central_matrix_dimension': total, 'node_matrix_dimension': d,
                        'central_full_complex128_covariance_bytes': total**2*16,
                        'node_full_complex128_covariance_bytes': d**2*16,
                        'network_full_complex128_covariance_bytes': nodes*d**2*16,
                        'central_cubic_dimension_proxy': total**3,
                        'network_cubic_dimension_proxy': nodes*d**3})
    return {'sample_rate_hz': fs, 'network': network,
            'two_nodes_PCM16_raw_upload_bps': 4*fs*16,
            'two_nodes_PCM16_scalar_exchange_bps': 2*fs*16,
            'two_nodes_float32_scalar_exchange_bps': 2*fs*32,
            'two_nodes_STFT_complex_float32_exchange_bps': 2*257*64*(fs//128),
            'STFT': {'n_fft': 512, 'hop': 128, 'bins': 257, 'bits_per_complex_value': 64,
                     'new_audio_seconds_per_call': 128/fs},
            'central_second_node_output_return_PCM16_bps': fs*16,
            'scope': 'payload only; excludes framing, timestamps, headers, loss recovery and radio scheduling; cubic dimensions are not measured flops or runtime'}


def run_experiments():
    covariance = run_covariance_experiments(); model = covariance['model']
    rs, rn, a = model['Rs'], model['correlated'], model['a']
    white = covariance['cases']['white']; correlated = covariance['cases']['correlated']
    report, signals = run_experiment()
    decoded = {key: read_pcm16(pcm16_bytes(value, SAMPLE_RATE))[1] for key, value in signals.items()}
    pcm = {key: measure_signal(decoded[key], key, reference=decoded[ref], pcm=True)
           for key, ref in OUTPUT_REFERENCES.items()}
    raw_projection = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 2., -.5]])
    complex_a = np.array([1., .5j, 2., -.5]); complex_rs = np.outer(complex_a, complex_a.conj())
    complex_w = mwf_weights(complex_rs, np.eye(4), 1)
    scale_controls = []
    for scale in (1e-300, 1., 1e300):
        projection = raw_projection.copy(); projection[2] *= scale
        result = compressed_mwf(rs, rn, projection)
        scale_controls.append({'compression_scale': scale, 'effective_weights': result['weights'],
                               'normalized_condition_number': result['condition_number']})
    raw_conditions = []
    for scale in (1e-10, 1., 1e10):
        t = raw_projection.copy(); t[2] *= scale
        raw_conditions.append({'scale': scale, 'unnormalized_covariance_condition': float(np.linalg.cond(t@(rs+rn)@t.T))})
    steps = distributed_updates(rs, rn, model['nodes'], model['references'], [a[:2], a[2:]], max_updates=6, tolerance=0.)
    multi_a = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
    multi_rs = multi_a@multi_a.T; multi_rn = np.eye(4)
    multi_full = np.column_stack([mwf_weights(multi_rs, multi_rn, r) for r in (0, 1)])
    multi_t = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 1.]]
    multi_compressed = [compressed_mwf(multi_rs, multi_rn, multi_t, r) for r in (0, 1)]
    zero = distributed_updates(rs, model['white'], model['nodes'], model['references'],
                               [np.zeros(2), np.zeros(2)], zero_policy='local', max_updates=1, tolerance=0.)
    rejected = {}
    for name, target, noise in [('indefinite_noise', np.eye(2), [[1., 2.], [2., 1.]]),
                                ('ill_conditioned', np.diag([1., 0.]), np.diag([1., 1e-14]))]:
        try:
            mwf_weights(target, noise)
        except (ValueError, np.linalg.LinAlgError) as error:
            rejected[name] = {'rejected': True, 'diagnostic': str(error)}
        else:
            raise AssertionError('expected explicit covariance rejection')
    ill_rs = np.diag([1., 0.]); ill_rn = np.diag([1., 1e-14]); ill_rx = ill_rs+ill_rn
    loaded = _load_covariance(ill_rx, .001, 1e12)
    load = float((loaded-ill_rx)[0, 0].real)
    loaded_w = np.linalg.solve(loaded, ill_rs[:, 0])
    exercises = {
        'E15-01': {'a': a, 'nodes': model['nodes'], 'references': model['references'],
                   'Rs': rs, 'white_Rn': model['white'], 'Rx': rs+model['white'],
                   'cross_covariances': [rs[:, r] for r in model['references']],
                   'desired_reference_powers': [1., 4.]},
        'E15-02': {'weights': white['central_weights'], 'components': white['central_components'],
                   'normal_equation_residual': (rs+model['white'])@white['central_weights']-rs[:, 0]},
        'E15-03': {'local': white['local'], 'central': white['central_components'],
                   'no_observation_zero_weights': mse_components(rs, model['white'], np.zeros(4))},
        'E15-04': {'raw_projection': raw_projection, 'compressed': white['correct'],
                   'raw_compressed_weights': white['correct']['compressed_weights']/np.array([1., 1., np.sqrt(4.25)]),
                   'scope': 'raw weights here apply to [x1,x2,2*x3-.5*x4]; normalized solver coordinates are separately returned'},
        'E15-05': {'Rn': rn, 'noise_eigenvalues': np.linalg.eigvalsh(rn), 'central': correlated['central_components'],
                   'central_weights': correlated['central_weights'], 'stale': correlated['stale'],
                   'lost_constraint_normal': [0., 0., .5, 2.],
                   'optimal_weight_constraint_violation': float(np.array([0., 0., .5, 2.])@correlated['central_weights'].real)},
        'E15-06': {'correct_direction': correlated['central_weights'][2:], 'compressed': correlated['correct']},
        'E15-07': {'a': complex_a, 'reference': 1, 'cross_covariance': complex_rs[:, 1], 'weights': complex_w,
                   'response': np.vdot(complex_w, complex_a), 'desired_response': complex_a[1],
                   'components': mse_components(complex_rs, np.eye(4), complex_w, 1),
                   'node2_weights': correlated['node2_weights'], 'node2_components': correlated['node2_components']},
        'E15-08': {'normalized_scale_controls': scale_controls, 'raw_conditions': raw_conditions,
                   'scope': 'row normalization preserves a nonzero compression subspace; it cannot recover a zero or lost direction'},
        'E15-09': steps,
        'E15-10': covariance['round_robin'],
        'E15-11': {'simultaneous': covariance['simultaneous'], 'simultaneous_half': covariance['simultaneous_half'],
                   'scope': 'finite known-covariance control; relaxation applies only to broadcast local weights, with solved output proposals recorded unchanged'},
        'E15-12': {'A': multi_a, 'full_weights': multi_full, 'remote_weight_rank': int(np.linalg.matrix_rank(multi_full[2:])),
                   'compressed': multi_compressed, 'full_components': [mse_components(multi_rs, multi_rn, multi_full[:, r], r) for r in (0, 1)]},
        'E15-13': {'zero_compression_local_policy': zero, 'default_policy': 'reject zero peer compression',
                   'zero_reference_target_power': mse_components(np.diag([1., 0.]), np.eye(2), np.zeros(2), 1)},
        'E15-14': {'rejections': rejected, 'ill_Rs': ill_rs, 'ill_Rn': ill_rn,
                   'loaded_covariance': loaded, 'additive_load': load, 'loaded_weights': loaded_w,
                   'physical_components': mse_components(ill_rs, ill_rn, loaded_w),
                   'extra_objective_penalty': float(load*np.vdot(loaded_w, loaded_w).real),
                   'scope': 'loading changes the objective; invalid original noise covariance is rejected before loading'},
        'E15-15': resource_control(),
        'E15-16': {'resources': resource_control()['network'],
                   'observed_solve_counts': {key: covariance[key]['total_solve_count'] for key in ('round_robin', 'simultaneous', 'simultaneous_half')},
                   'new_audio_seconds_per_STFT_call': 128/16000,
                   'scope': 'matrix bytes and count proxies; no measured hardware runtime or power'},
        'E15-17': {'parameters': report['parameters']['clock'], 'clock': report['clock'],
                   'synchronous_float': report['float_components']['central_white'],
                   'misaligned_float': report['float_components']['clock_misaligned_white'],
                   'linear_corrected_float': report['float_components']['clock_linear_corrected_white'],
                   'PCM': {key: pcm[key] for key in ('clock_misaligned_white', 'clock_linear_corrected_white')},
                   'warning': 'interpolation attenuates frequencies and changes noise covariance; lower total MSE is not recovery of the original synchronous optimum'},
        'E15-18': {'packet': report['parameters']['packet'],
                   'analytic': {key: report['analytic_expected'][key] for key in ('packet_zerofill_white', 'packet_local_fallback_white')},
                   'float': {key: report['float_components'][key] for key in ('packet_zerofill_white', 'packet_local_fallback_white')},
                   'PCM': {key: pcm[key] for key in ('packet_zerofill_white', 'packet_local_fallback_white')}},
        'E15-19': {'transport': report['transport'], 'parameters': report['parameters']['transport'],
                   'float': report['float_components']['transport_pcm16_white'], 'PCM': pcm['transport_pcm16_white']},
        'E15-20': {'target_steady_power': POWER, 'finite_actual_target_power': report['finite_steady_target_power'],
                   'float_basis_covariance': report['finite_steady_basis_covariance'],
                   'finite_source_noise_cross': report['finite_steady_source_noise_cross'],
                   'PCM': {key: pcm[key] for key in ('central_white', 'central_correlated', 'central_node2_correlated', 'stale_correlated')},
                   'scope': 'all integer scores use the genuine same-reference PCM; no fitted gain, delay or averaged dB'},
        'E15-21': {'unrelaxed': jacobi_control(), 'half': jacobi_control(.5, 40)},
        'E15-22': step_size_control(),
        'E15-23': tree_sum_control(),
        'E15-24': gevd_control(),
    }
    if set(exercises) != set(EXERCISE_IDS):
        raise AssertionError('exercise ID set differs')
    return _json({'schema_version': 1, 'scope': 'original finite known-model teaching controls',
                  'complex_encoding': 'complex scalar -> [real,imag], complex arrays -> trailing [real,imag] axis',
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
