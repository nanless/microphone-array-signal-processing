"""Ten WPE/MINT arithmetic and audio experiments, E07-08 through E07-17.

These are small declared models, not benchmarks. Complex arrays are encoded
as separate real/imag lists for strict JSON. Importing does not run or write.
"""
from __future__ import annotations
import json
import numpy as np
from codes.array_tutorial.audio_samples import wpe_predictable_case


def _complex(value) -> dict:
    a = np.asarray(value, complex)
    return {'real': a.real.tolist(), 'imag': a.imag.tolist()}


def complex_weighted_fit() -> dict:
    # Six actual M=2 frames: delayed first three regressors predict last three.
    observations = np.array([[1, 0, 1, 0, -.5j, 3], [0, 1, 1j, 0, 0, 0]], complex)
    q = observations[:, :3].T
    target = observations[0, 3:]
    weights = np.array([1., 2., 1.])  # prescribed inverse powers, not initial WPE PSD
    R = sum(w * np.outer(v, v.conj()) for w, v in zip(weights, q))
    r = sum(w * v * z.conjugate() for w, v, z in zip(weights, q, target))
    results = {}
    for delta in (0., 1.):
        g = np.linalg.solve(R + delta*np.eye(2), r)
        residual = target - q @ g.conj()
        results[str(int(delta))] = {'coefficients': _complex(g), 'residual': _complex(residual),
                                    'weighted_residual_cost': float(np.sum(weights*abs(residual)**2)),
                                    'weighted_history_residual_cross': _complex(q.T @ (weights*residual.conj()))}
    return {'channels': 2, 'taps': 1, 'delay': 3, 'observations': _complex(observations),
            'valid_frame_indices': [3, 4, 5], 'history_rows': _complex(q),
            'target': _complex(target), 'inverse_power_weights': weights.tolist(),
            'correlation': _complex(R), 'cross': _complex(r), 'absolute_loading_cases': results,
            'scope': 'first output only, prescribed shared powers; delta has correlation units, not a dimensionless loading ratio'}


def power_floor_and_smoothing() -> dict:
    energy = np.array([1., 9.])
    powers = {'unconstrained': energy, 'floor_2': np.maximum(energy, 2.), 'mean_smoothed': np.full(2, energy.mean())}
    return {'residual_energies': energy.tolist(), 'cases': {
        name: {'power': p.tolist(), 'objective_without_constants': float(np.sum(np.log(p)+energy/p))}
        for name, p in powers.items()},
        'zero_energy': {'positive_power_trials': [1., .1, .01],
                        'objective_without_constants': np.log([1., .1, .01]).tolist()},
        'scope': 'fixed residual, independent scalar complex-Gaussian variances; smoothing changes the optimization, and zero energy has no positive unconstrained minimizer'}


def history_permutation() -> dict:
    observations = np.arange(6)[None, :] + 10*np.arange(2)[:, None]
    lag = np.concatenate([observations[:, 5-2-k] for k in range(3)])
    permutation = np.array([0, 2, 4, 1, 3, 5])
    channel = lag[permutation]
    g = np.array([1, 1j, 2, -1j, 0, 1])
    return {'current_frame': 5, 'delay': 2, 'taps': 3, 'observations': observations.tolist(),
            'lag_major_history': lag.tolist(), 'channel_major_history': channel.tolist(),
            'permutation_zero_based': permutation.tolist(), 'lag_major_coefficients': _complex(g),
            'permuted_coefficients': _complex(g[permutation]),
            'prediction_lag_major': _complex(np.vdot(g, lag)),
            'prediction_both_permuted': _complex(np.vdot(g[permutation], channel)),
            'prediction_history_only_permuted': _complex(np.vdot(g, channel)),
            'scope': 'reorder history and filter together; this does not reconcile different upstream time indices'}


def frame_count_and_rank() -> dict:
    target = np.array([1., 2., 1., 2.])
    cases = {'repeated': np.ones((4, 2)), 'independent': np.tile(np.eye(2), (2, 1))}
    result = {}
    for name, q in cases.items():
        R, r = q.T @ q, q.T @ target
        result[name] = {'history_rows': q.tolist(), 'correlation': R.tolist(),
                        'eigenvalues': np.linalg.eigvalsh(R).tolist(), 'rank': int(np.linalg.matrix_rank(R)),
                        'minimum_norm_coefficients': np.linalg.lstsq(q, target, rcond=None)[0].tolist(),
                        'absolute_loading_1_coefficients': np.linalg.solve(R+np.eye(2), r).tolist()}
    return {'dimension': 2, 'valid_frames': 4, 'target': target.tolist(), 'cases': result,
            'scope': 'rank concerns these declared regression snapshots; enough rows is necessary but not sufficient for independent parameters'}


def frame_window_overlap() -> dict:
    cases = []
    for delay in (3, 4):
        history = [-delay*128, 512-delay*128]
        overlap = max(0, min(512, history[1])-max(0, history[0]))
        cases.append({'delay_frames': delay, 'nearest_history_support_samples': history,
                      'overlap_samples': overlap, 'overlap_seconds': overlap/16000,
                      'oldest_frame_start_offset_samples': -(delay+5-1)*128})
    return {'sample_rate_hz': 16000, 'window_samples': 512, 'hop_samples': 128, 'taps': 5,
            'current_support_samples': [0, 512], 'interval_convention': 'half-open unwindowed frame support',
            'cases': cases,
            'scope': 'Hann endpoint zeros do not make overlapping windows independent; history span is not an extra wait for future samples'}


def exponential_statistics() -> dict:
    q, target = np.array([1., 2., 1.]), np.array([.8, 1.6, 1.])
    alpha, R, r = .5, 2., 1.
    records = []
    for x, y in zip(q, target):
        prior_g = r/R
        prior_output = y - prior_g*x
        R, r = alpha*R+x*x, alpha*r+x*y
        records.append({'prior_coefficient': prior_g, 'prior_output': prior_output,
                        'posterior_correlation': R, 'posterior_cross': r, 'posterior_coefficient': r/R})
    weights = alpha**np.arange(2, -1, -1)
    return {'regressors': q.tolist(), 'targets': target.tolist(), 'powers': [1., 1., 1.],
            'alpha': alpha, 'initial_correlation': 2., 'initial_cross': 1., 'steps': records,
            'batch_weights': weights.tolist(), 'batch_correlation': float(alpha**3*2+np.sum(weights*q*q)),
            'batch_cross': float(alpha**3+np.sum(weights*q*target)),
            'two_zero_history_steps_forgetting': {'correlation': alpha**2*R, 'cross': alpha**2*r, 'coefficient': r/R},
            'two_fully_frozen_statistics_steps': {'correlation': R, 'cross': r, 'coefficient': r/R},
            'scope': 'declared powers and direct statistics, both policies still advance frame history; separate locked-upstream silence probe uses its own inverse-matrix update',
            'silence_probe': 'codes/examples/wpe_silence_boundary.py',
            'silence_report': 'codes/reports/chapter07_online_wpe_silence.json'}


def mint_near_common_zero() -> dict:
    result = []
    for a, b in ((.5, -.5), (.5, .49)):
        u = np.array([-b, a])/(a-b)
        response = u[0]*np.array([1., a])+u[1]*np.array([1., b])
        result.append({'a': a, 'b': b, 'constant_inverse_weights': u.tolist(),
                       'summed_impulse_response': response.tolist(),
                       'post_path_independent_unit_noise_variance': float(u @ u),
                       'first_path_second_tap_perturbation': .001,
                       'perturbed_second_response_tap': float(response[1]+.001*u[0])})
    return {'cases': result, 'common_zero_case': {'a': .5, 'b': .5, 'constant_inverse_exists': False},
            'scope': 'two known FIR paths [1,a],[1,b]; independent equal-variance noise enters after each path. Not noise amplification for pre-path common source noise.'}


def wpe_resource_budget() -> dict:
    result = []
    for channels in (8, 16):
        dimension, bins, taps = channels*10, 257, 10
        covariance = bins*dimension**2*16
        predictor = bins*dimension*channels*16
        result.append({'channels': channels, 'taps': taps, 'frequency_bins': bins,
                       'history_dimension': dimension, 'complex128_bytes': 16,
                       'one_correlation_array_bytes': covariance,
                       'one_correlation_array_MiB': covariance/2**20,
                       'one_predictor_array_bytes': predictor, 'one_predictor_array_MiB': predictor/2**20,
                       'per_bin_outer_product_entries': dimension**2,
                       'dense_factorization_cubic_dimension_proxy': dimension**3})
    return {'cases': result, 'scope': 'array storage and dimension-order proxies only; no timing claim, no Hermitian packed storage, and excludes input/history copies, solver workspace and all other state'}


def loaded_wpd_factorization() -> dict:
    A = C = np.diag([2., 1.]); B = np.diag([1., 0.]); delta = 1.
    G = np.linalg.solve(C+delta*np.eye(2), B)
    S = A+delta*np.eye(2)-B.T@G
    Rz = A-B.T@G-G.T@B+G.T@C@G
    correction = delta*np.eye(2)+delta*G.T@G
    covariance = np.block([[A, B.T], [B, C]])+delta*np.eye(4)
    steering = np.array([1., 1., 0., 0.])
    raw = np.linalg.solve(covariance, steering)
    full = raw/(steering@raw)
    raw_wrong = np.linalg.solve(Rz+delta*np.eye(2), np.ones(2))
    wrong = raw_wrong/np.sum(raw_wrong)
    return {'A': A.tolist(), 'B': B.tolist(), 'C': C.tolist(), 'absolute_loading': delta,
            'G': G.tolist(), 'schur': S.tolist(), 'residual_covariance': Rz.tolist(),
            'required_correction': correction.tolist(), 'direct_loaded_filter': full.tolist(),
            'direct_loaded_objective': float(full@covariance@full),
            'omitted_GHG_correction_spatial_filter': wrong.tolist(),
            'omitted_GHG_correction_history_filter': (-G@wrong).tolist(),
            'scope': 'load the full stacked covariance first; the residual spatial problem includes delta I + delta G^H G'}


def predictable_target_audio() -> dict:
    case = wpe_predictable_case()
    return {'files': [name+'.wav' for name in case['signals']], **case['parameters'], 'scope': case['limits']}


def run_experiments() -> dict:
    functions = [complex_weighted_fit, power_floor_and_smoothing, history_permutation,
                 frame_count_and_rank, frame_window_overlap, exponential_statistics,
                 mint_near_common_zero, wpe_resource_budget, loaded_wpd_factorization,
                 predictable_target_audio]
    return {f'E07-{i:02d}': function() for i, function in enumerate(functions, 8)}


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
