"""E05-08--17: exact small models and an actual GSC waveform teaching chain.

Run .venv/bin/python -m codes.chapters.ch05.chapter05_experiments. No files are
written. These are original deterministic examples, not upstream benchmarks.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json
import numpy as np

from codes.chapters.ch00.core.audio_samples import gsc_gate_case
from codes.chapters.ch05.core.beamforming import lcmv_weights, mvdr_weights
from codes.chapters.ch08.core.separation import masked_spatial_covariance


def gsc_lcmv_equivalence() -> dict:
    noise = np.diag([1., 2., 4.])
    target = np.ones(3)
    fixed = target/3
    blocker = np.array([[1., 1.], [-1., 0.], [0., -1.]])
    reference_covariance = blocker.T @ noise @ blocker
    cross = blocker.T @ noise @ fixed
    adaptive = np.linalg.solve(reference_covariance, cross)
    weights = fixed - blocker @ adaptive
    arbitrary_h = np.array([1+2j, -3j])
    return {'noise_covariance': noise, 'target': target, 'fixed_weights': fixed,
            'blocking_matrix': blocker, 'reference_covariance': reference_covariance,
            'reference_desired_cross': cross, 'adaptive_coefficient': adaptive,
            'gsc_weights': weights, 'lcmv_weights': lcmv_weights(noise, target[:, None], [1]),
            'output_noise_power': float(weights @ noise @ weights),
            'arbitrary_h': arbitrary_h,
            'arbitrary_h_target_response': np.vdot(fixed-blocker@arbitrary_h, target)}


def gsc_statistical_boundary() -> dict:
    target = np.ones(2)
    interference = np.exp(1j*np.pi*np.arange(2)*np.sin(np.deg2rad(60)))
    fixed, blocker = target/2, np.array([1., -1.])
    mixture = target + interference
    desired, reference = np.vdot(fixed, mixture), np.vdot(blocker, mixture)
    oracle = np.conj(np.vdot(fixed, interference)/np.vdot(blocker, interference))
    one_snapshot = np.conj(desired/reference)
    correlated = []
    for rho in (0., .5, 1.):
        h = rho/2
        weights = fixed - blocker*h
        correlated.append({'rho': rho, 'h': h, 'weights': weights,
                           'target_response': np.vdot(weights, target),
                           'interference_response': np.vdot(weights, blocker),
                           'joint_output_power': 1-rho*rho})
    return {'sixty_degree_case': {'target': target, 'interference': interference,
            's': 1, 'i': 1, 'd': desired, 'u': reference, 'oracle_h': oracle,
            'single_snapshot_ls_h': one_snapshot,
            'oracle_residual': desired-oracle.conjugate()*reference,
            'single_snapshot_ls_residual': desired-one_snapshot.conjugate()*reference},
            'correlated_model': 'a=[1,1], b=B=[1,-1], d=s,u=2i; E|s|²=E|i|²=1,E[i s*]=rho',
            'correlated_cases': correlated}


def _frost_step(c: np.ndarray, f: np.ndarray, x: np.ndarray) -> dict:
    gram = c.conj().T@c
    fixed = c@np.linalg.solve(gram, f)
    projection = np.eye(c.shape[0])-c@np.linalg.solve(gram, c.conj().T)
    output = np.vdot(fixed, x)
    unprojected = fixed-.1*x*output.conjugate()
    projected = projection@unprojected+fixed
    return {'constraints': c, 'responses': f, 'snapshot': x, 'step_size': .1,
            'initial_weights': fixed, 'projection': projection, 'pre_update_output': output,
            'ordinary_lms_weights': unprojected, 'ordinary_lms_constraint': c.conj().T@unprojected,
            'frost_weights': projected, 'frost_constraint': c.conj().T@projected}


def frost_finite_step() -> dict:
    return {'tap_order': ['x1(n)', 'x2(n)', 'x1(n-1)', 'x2(n-1)'],
            'real_two_mic_two_tap': _frost_step(np.array([[1., 0], [1., 0], [0, 1.], [0, 1.]]),
                                               np.array([1., 0]), np.array([1., 3., 2., 0.])),
            'complex_algebra_check': _frost_step(np.array([[1.], [1j], [0.]]),
                                                np.array([1.]), np.array([1., 2j, 1-1j])),
            'limits': 'The complex subcase is an abstract constraint check, not the real FIR sensor geometry.'}


def zelinski_pair_difference() -> dict:
    x = np.array([[1, 1j, 2], [2j, 1, 0], [-1, 2, 1j]], complex)
    covariance = x@x.conj().T/3
    channels = x.shape[0]
    cross_sum = sum(covariance[i, j].real for i in range(channels) for j in range(i+1, channels))
    estimate = np.trace(covariance).real/channels - 2*cross_sum/(channels*(channels-1))
    pair_powers = [float(np.mean(abs(x[i]-x[j])**2)) for i in range(channels) for j in range(i+1, channels)]
    false_noise = []
    for a in (np.array([1., .8]), np.array([1., 1j])):
        r = np.outer(a, a.conj())
        false_noise.append({'target': a, 'true_noise_power': 0.,
                            'estimated_noise_power': float(np.trace(r).real/2-r[0, 1].real)})
    return {'snapshots': x, 'covariance': covariance, 'estimated_input_noise': float(estimate),
            'pair_difference_powers': pair_powers,
            'pair_expression': sum(pair_powers)/(channels*(channels-1)),
            'noiseless_mismatch_cases': false_noise}


def coherent_noise_pair() -> dict:
    rows = []
    for rho in (.3, .99):
        auto, cross = 4., 3.+rho
        noise = (auto-cross)/(1-rho)
        speech = (cross-rho*auto)/(1-rho)
        biased_noise = (auto-(cross+.01))/(1-rho)
        rows.append({'rho': rho, 'auto': auto, 'cross': cross,
                     'estimated_target_power': speech, 'estimated_noise_power': noise,
                     'cross_perturbation': .01, 'perturbed_noise_estimate': biased_noise,
                     'noise_estimate_change': biased_noise-noise})
    output_noise = (1+.3)/2
    return {'cases': rows, 'dsb_output_noise_at_rho03': output_noise,
            'wiener_gain_at_rho03': 3/(3+output_noise),
            'rho_one': 'auto=cross=speech+noise; two unknown powers cannot be identified',
            'scope': 'Derived real-coherence two-channel model; not a full McCowan implementation.'}


def target_covariance_contamination() -> dict:
    assumed, true = np.ones(2), np.array([1., 1j])
    rows = []
    for power in (0., 1., 10., 100.):
        for name, target in [('matched', assumed), ('mismatched', true)]:
            contaminated = np.eye(2)+power*np.outer(target, target.conj())
            weights = mvdr_weights(contaminated, assumed)
            rows.append({'case': name, 'target_power_in_noise_scm': power, 'weights': weights,
                         'nominal_response': np.vdot(weights, assumed),
                         'actual_target_response': np.vdot(weights, target),
                         'actual_white_noise_power': float(np.vdot(weights, weights).real)})
    return {'noise_covariance': np.eye(2), 'assumed_target': assumed, 'mismatched_target': true,
            'cases': rows, 'scope': 'Exact covariance, fixed wrong constraint, no finite sample estimation.'}


def gev_mwf_scale() -> dict:
    noise, target = np.diag([2., 1.]), np.ones(2)
    speech = 3*np.outer(target, target)
    # Generalized eigenproblem by whitening; canonicalize the otherwise free scale.
    whitener = np.diag([1/np.sqrt(2), 1])
    eigenvalues, vectors = np.linalg.eigh(whitener@speech@whitener)
    gev = whitener@vectors[:, -1]
    gev = gev/gev[-1]
    mvdr = mvdr_weights(noise, target)
    mwf = np.linalg.solve(speech+noise, speech@np.array([1., 0.]))
    rows = []
    for label, w in [('GEV', gev), ('GEV_times_two', 2*gev), ('MVDR', mvdr), ('MWF', mwf)]:
        response = np.vdot(w, target)
        noise_power = float(np.vdot(w, noise@w).real)
        distortion = float(3*abs(response-1)**2)
        rows.append({'method': label, 'weights': w, 'target_response': response,
                     'output_noise_power': noise_power, 'output_snr_linear': float(3*abs(response)**2/noise_power),
                     'target_distortion_power': distortion, 'reference_mse': noise_power+distortion})
    return {'noise_covariance': noise, 'target_covariance': speech,
            'largest_generalized_eigenvalue': float(eigenvalues[-1]), 'cases': rows,
            'scope': 'GEV is explicitly scaled to second component 1; this is not BAN.'}


def first_order_spatial_rank() -> dict:
    equator = np.array([[1., 0, 0], [0, 1., 0], [-1., 0, 0], [0, -1., 0]])
    tetrahedron = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])/np.sqrt(3)
    rows = []
    for name, xyz in [('equator', equator), ('tetrahedron', tetrahedron)]:
        sampling = np.column_stack((np.ones(4), xyz))
        singular = np.linalg.svd(sampling, compute_uv=False)
        rows.append({'geometry': name, 'unit_points': xyz, 'sampling_matrix': sampling,
                     'gram': sampling.T@sampling, 'rank': int(np.linalg.matrix_rank(sampling)),
                     'singular_values': singular,
                     'condition_number': None if singular[-1] == 0 else float(singular[0]/singular[-1])})
    return {'basis': '[1,x,y,z], equivalent to degree <=1 real spherical harmonics; not orthonormalized',
            'cases': rows, 'limits': 'Condition numbers depend on this basis scale. No radial inverse, scattering or hardware performance is modeled.'}


def gsc_audio_results() -> dict:
    return gsc_gate_case()['parameters']


def nonnegative_mask_model() -> dict:
    x = np.eye(2, dtype=complex)[None]
    rejected = False
    try:
        masked_spatial_covariance(x, np.array([[2., -1.]]))
    except ValueError:
        rejected = True
    rows = []
    for label, weights in [('two_frames', [1., 1.]), ('one_frame', [1., 0.]),
                           ('tiny_common_scale', [1e-300, 1e-300])]:
        mask = np.array([weights])
        matrix = masked_spatial_covariance(x, mask)
        normalized = mask[0]/max(mask[0])
        concentration = float(normalized.sum()**2/(normalized@normalized))
        rows.append({'case': label, 'weights': mask[0], 'weight_sum': float(mask.sum()),
                     'positive_support_count': int(np.count_nonzero(mask)),
                     'weight_concentration_count': concentration, 'covariance': matrix[0],
                     'eigenvalues': np.linalg.eigvalsh(matrix[0])})
    return {'spectrum_shape': [1, 2, 2], 'snapshots': x[0], 'negative_mask': [2., -1.],
            'unguarded_signed_weight_result': np.diag([2., -1.]),
            'negative_mask_rejected_by_real_interface': rejected, 'denominator_floor': 1e-12,
            'cases': rows, 'limits': 'Weights need not sum to 1 or lie below 1. Support and concentration are not probability confidence.'}


def _pack(value):
    if isinstance(value, np.ndarray):
        return {'real': value.real.tolist(), 'imag': value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return {'real': float(value.real), 'imag': float(value.imag)}
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_exercises() -> dict:
    functions = [gsc_lcmv_equivalence, gsc_statistical_boundary, frost_finite_step,
                 zelinski_pair_difference, coherent_noise_pair, target_covariance_contamination,
                 gev_mwf_scale, first_order_spatial_rank, gsc_audio_results, nonnegative_mask_model]
    results = {}
    for index, function in enumerate(functions, 8):
        result = function()
        result['metadata'] = {'kind': 'original deterministic teaching experiment',
                              'numpy_version': np.__version__, 'randomness': 'none'}
        results[f'E05-{index:02d}'] = _pack(result)
    return results


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
