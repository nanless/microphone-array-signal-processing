"""E05-08--22: small models and strictly checked published waveform fixtures.

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
import hashlib
import io
import math
from pathlib import Path
import wave
import numpy as np

from codes.chapters.ch00.core.audio_samples import gsc_gate_case
from codes.chapters.ch05.core.beamforming import lcmv_weights, mvdr_weights
from codes.chapters.ch08.core.separation import masked_spatial_covariance

ROOT = Path(__file__).resolve().parents[3]


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


def gsc_audio_results(*, audio_dir: Path | None = None, manifest_path: Path | None = None) -> dict:
    """Keep the floating model, independently read the four published PCM files.

    This never regenerates or repairs an asset. The main manifest's existing
    source set is authoritative; no new generator dependency is invented.
    """
    from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
    from codes.chapters.ch00.core.audio_samples import read_pcm16
    audio_dir = ROOT/'codes/chapters/ch05/audio' if audio_dir is None else Path(audio_dir)
    manifest_path = ROOT/'codes/chapters/ch00/audio/MANIFEST.json' if manifest_path is None else Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in INPUTS}
    if manifest.get('generator_inputs') != sources:
        raise ValueError('published GSC main manifest source set or SHA is stale')
    entries = {row['file']: row for row in manifest['files']}
    if len(entries) != len(manifest['files']):
        raise ValueError('published main manifest has duplicate file names')
    names = ['gsc_reference.wav', 'gsc_array.wav', 'gsc_always_adapt.wav', 'gsc_gate_frozen.wav']
    integer, files = {}, {}
    for name in names:
        blob = (audio_dir/name).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        info = entries.get(name, {})
        channels = 2 if name == 'gsc_array.wav' else 1
        if (info.get('sha256') != digest or info.get('chapter') != 'ch05'
                or info.get('common_export_gain') != 1.0
                or info.get('sample_rate_hz') != 16000 or info.get('channels') != channels
                or info.get('samples') != 32000):
            raise ValueError('published GSC manifest/WAV mismatch: '+name)
        with wave.open(io.BytesIO(blob), 'rb') as reader:
            actual = (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                      reader.getsampwidth(), reader.getcomptype())
            if actual != (16000, channels, 32000, 2, 'NONE'):
                raise ValueError('published GSC PCM format mismatch: '+name)
        rate, decoded = read_pcm16(blob)
        integer[name] = np.rint(decoded*32768).astype(np.int64)
        files[name] = {'sha256': digest, 'sample_rate_hz': rate, 'channels': channels,
                       'samples_per_channel': decoded.shape[1], 'common_export_gain': 1.0}
    start, stop = 16000, 30000
    reference = integer['gsc_reference.wav'][0, start:stop]
    denominator = int(reference@reference)  # bounded PCM16, 14000 points: safe int64
    if denominator <= 0:
        raise ValueError('published GSC reference needs positive PCM energy in the scoring interval')
    scores = {}
    for key in ('always_adapt', 'gate_frozen'):
        y = integer['gsc_'+key+'.wav'][0, start:stop]
        cross, squared_error = int(reference@y), int((y-reference)@(y-reference))
        scores[key] = {'reference_projection_gain': cross/denominator,
                       'normalized_reference_error': math.sqrt(squared_error/denominator),
                       'integer_reference_cross_sum': cross, 'integer_error_squared_sum': squared_error,
                       'integer_reference_squared_sum': denominator,
                       'decoded_reference_squared_sum': denominator/32768**2,
                       'sample_denominator': stop-start}
    result = dict(gsc_gate_case()['parameters'])
    result['published_audio'] = {'files': files, 'source_sha256': sources,
                               'scoring_interval_samples': [start, stop],
                               'pcm_decode_divisor': 32768, 'actual_pcm_measurements': scores,
                               'alignment': 'original sample clock/gain; projection is diagnostic only'}
    return result


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


def derivative_lcmv() -> dict:
    noise = np.diag([1., 2., 4.])
    target, derivative = np.ones(3), np.array([-1j, 0., 1j])
    c = np.column_stack((target, derivative))
    methods = [('single', mvdr_weights(noise, target)),
               ('constrained', lcmv_weights(noise, c, [1., 0.])), ('DSB', target/3)]
    rows = []
    for method, weights in methods:
        responses = []
        for phase in (0., .1, .2, .3):
            a = np.exp(1j*phase*np.array([-1., 0., 1.]))
            b = np.vdot(weights, a)
            responses.append({'phi_rad': phase, 'complex_response': b,
                              'response_magnitude': float(abs(b)), 'response_phase_rad': float(np.angle(b)),
                              'response_error_squared': float(abs(b-1)**2)})
        rows.append({'method': method, 'weights': weights, 'nominal_response': np.vdot(weights, target),
                     'complex_response_derivative_at_zero': np.vdot(weights, derivative),
                     'output_noise_power': float(np.vdot(weights, noise@weights).real),
                     'wng_linear': float(abs(np.vdot(weights, target))**2/np.vdot(weights, weights).real),
                     'responses': responses})
    return {'noise_covariance': noise, 'phase_centre': 'middle microphone',
            'derivative_variable': 'phi=2*pi*f*d*sin(theta)/c, not degrees',
            'constraints': c, 'constraint_responses': [1., 0.], 'cases': rows,
            'limits': 'Fixed exact covariance; derivative constraint is local complex-response flatness and costs freedom/noise power.'}


def norm_ball_robust() -> dict:
    epsilon = .2
    t = 1/(2-math.sqrt(2)*epsilon)
    rows = []
    for method, value in [('nominal_unit_response', .5), ('worst_case_unit_response', t)]:
        w = np.full(2, value)
        norm = float(np.linalg.norm(w))
        perturbation = -epsilon*w/norm
        response = float(w@np.ones(2))
        rows.append({'method': method, 'weights': w, 'nominal_response': response,
                     'weight_norm_squared': norm**2, 'white_noise_output_power': norm**2,
                     'attaining_perturbation': perturbation, 'worst_case_amplitude': response-epsilon*norm,
                     'general_wng_linear': response**2/norm**2,
                     'unit_response_only_shortcut': 1/norm**2})
    return {'nominal_target': [1., 1.], 'noise_covariance': np.eye(2), 'uncertainty_radius': epsilon,
            'infeasible_radius_from': math.sqrt(2), 'cases': rows,
            'limits': 'Exact white-noise norm-ball solution only; nominal response is not forced to 1. No SOCP solver or physical DOA uncertainty distribution is modeled.'}


def souden_reference_channel() -> dict:
    b, noise, power = np.array([2., 1+1j]), np.diag([2., 1.]), 3.
    speech = power*np.outer(b, b.conj())
    product = np.linalg.solve(noise, speech)
    trace = np.trace(product).real
    rows = []
    for channel in (0, 1):
        reference = np.eye(2)[:, channel]
        souden = product@reference/trace
        mwf = np.linalg.solve(speech+noise, speech@reference)
        rtf = b/b[channel]
        row = {'reference_channel_zero_based': channel, 'reference_target_transfer': b[channel],
               'rtf': rtf, 'souden_weights': souden, 'rtf_mvdr_weights': mvdr_weights(noise, rtf)}
        for label, w in [('souden', souden), ('mwf', mwf)]:
            response = np.vdot(w, b)
            pn = float(np.vdot(w, noise@w).real)
            distortion = float(power*abs(response-b[channel])**2)
            row[label+'_measurements'] = {'weights': w, 'target_response': response,
                                          'output_noise_power': pn, 'reference_target_distortion_power': distortion,
                                          'reference_mse': pn+distortion}
        rows.append(row)
    full_target, full_noise, reference = np.diag([2., 1.]), np.eye(2), np.array([1., 0.])
    full_product = np.linalg.solve(full_noise, full_target)
    full_souden = full_product@reference/np.trace(full_product).real
    error = full_souden-reference
    control = {'target_covariance': full_target, 'noise_covariance': full_noise,
               'reference_channel_zero_based': 0, 'trace_normalizer': float(np.trace(full_product).real),
               'souden_weights': full_souden,
               'reference_target_distortion_power': float(error@full_target@error),
               'output_noise_power': float(full_souden@full_noise@full_souden),
               'general_mwf_weights': np.linalg.solve(full_target+full_noise, full_target@reference),
               'rank_one_trace_shortcut_weights': full_target@reference/(1+np.trace(full_target)),
               'limits': 'Full-rank target is legal; no single transfer vector makes this trace expression distortionless. The general MWF coincidence here is specific to this diagonal example.'}
    return {'target_transfer': b, 'noise_covariance': noise, 'target_covariance': speech,
            'source_power': power, 'trace_normalizer': float(trace), 'cases': rows,
            'full_rank_control': control,
            'diagonal_loading': 0., 'trace_denominator_epsilon': 0.,
            'limits': 'Rank-one exact target covariance and explicit complex reference; original analytic fixture, not a TorchAudio package benchmark.'}


def om_lsa_probability_combination() -> dict:
    xi, gamma, g_present, g_absent = 3., 4., .8, .1
    nu = gamma*xi/(1+xi)
    rows = []
    for q in (.5, .2, 0., 1.):
        if q in (0., 1.):
            posterior = 1-q
        else:
            log_odds_absent = math.log(q)-math.log1p(-q)+math.log1p(xi)-nu
            posterior = 1/(1+math.exp(log_odds_absent))
        gain = math.exp(posterior*math.log(g_present)+(1-posterior)*math.log(g_absent))
        rows.append({'prior_absence_probability': q, 'posterior_presence_probability': posterior,
                     'geometric_gain': gain, 'arithmetic_mix_for_comparison': posterior*g_present+(1-posterior)*g_absent})
    return {'prior_snr_linear': xi, 'posterior_snr_linear': gamma, 'nu': nu,
            'given_conditional_gain': g_present, 'absence_gain_floor': g_absent, 'cases': rows,
            'limits': 'Bayes probability and geometric combination only; conditional gain .8 is supplied, not obtained by an LSA integral. No MCRA/IMCRA state or complete OM-LSA is run.'}


def derivative_audio_results() -> dict:
    from codes.chapters.ch05.examples.generate_derivative_audio import check_assets, DEFAULT_OUTPUT
    manifest = check_assets(DEFAULT_OUTPUT)
    return {'parameters': manifest['parameters'], 'files': manifest['files'],
            'source_sha256': manifest['source_sha256'], 'samples': manifest['samples'],
            'float_decomposition': manifest['float_decomposition'], 'limits': manifest['limits']}


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
                 gev_mwf_scale, first_order_spatial_rank, gsc_audio_results, nonnegative_mask_model,
                 derivative_lcmv, norm_ball_robust, souden_reference_channel,
                 om_lsa_probability_combination, derivative_audio_results]
    results = {}
    for index, function in enumerate(functions, 8):
        result = function()
        result['metadata'] = {'kind': 'original deterministic teaching experiment',
                              'numpy_version': np.__version__,
                              'randomness': 'fixed white-noise seed 20260522' if index == 22 else 'none'}
        results[f'E05-{index:02d}'] = _pack(result)
    return results


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
