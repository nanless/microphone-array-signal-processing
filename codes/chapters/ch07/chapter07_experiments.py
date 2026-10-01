"""Fourteen WPE/MINT arithmetic and audio experiments, E07-08 through E07-21.

These are small declared models, not benchmarks. Complex arrays are encoded
as separate real/imag lists for strict JSON. Importing does not run or write.
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
import struct
import wave
from pathlib import Path
from fractions import Fraction
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


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
            'silence_probe': 'codes/chapters/ch07/examples/wpe_silence_boundary.py',
            'silence_report': 'codes/chapters/ch07/reports/chapter07_online_wpe_silence.json'}


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


def _actual_pcm_integers(blob, *, channels, samples):
    try:
        with wave.open(io.BytesIO(blob)) as reader:
            if (reader.getnchannels(), reader.getnframes(), reader.getframerate(), reader.getsampwidth(),
                    reader.getcomptype()) != (channels, samples, 16000, 2, 'NONE'):
                raise ValueError('actual WAV must have the declared PCM16 format')
            raw = reader.readframes(samples)
    except (wave.Error, EOFError) as error:
        raise ValueError('invalid actual WAV') from error
    if len(raw) != 2*channels*samples:
        raise ValueError('truncated actual PCM data')
    return struct.unpack('<'+'h'*(channels*samples), raw)


def predictable_target_audio(repo_root=ROOT) -> dict:
    """Read the four published PCM files; never synthesize or repair them."""
    from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
    root = Path(repo_root)
    manifest_path = root/'codes/chapters/ch00/audio/MANIFEST.json'
    try:
        manifest = json.loads(manifest_path.read_text(),
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite manifest')))
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('main audio manifest is missing or invalid') from error
    if not isinstance(manifest, dict):
        raise ValueError('main audio manifest must be an object')
    sources = manifest.get('generator_inputs', {})
    if set(sources) != set(INPUTS):
        raise ValueError('main audio source set differs from current generator')
    try:
        if any(hashlib.sha256((root/path).read_bytes()).hexdigest() != sha for path, sha in sources.items()):
            raise ValueError('main audio source SHA is stale')
    except OSError as error:
        raise ValueError('main audio source file is missing') from error
    group = manifest.get('groups', {}).get('wpe_predictable', {})
    parameters = group.get('parameters', {})
    if (type(group.get('common_export_gain')) not in (int, float) or group['common_export_gain'] != 1
            or parameters.get('sample_rate_hz') != 16000 or parameters.get('samples') != 32000
            or parameters.get('score_windows_samples') != {'steady_target': [6400, 14400],
                                                          'post_source_tail': [16000, 32000]}):
        raise ValueError('actual predictable-audio parameters differ from E07-17')
    names = ['wpe_predictable_'+part for part in ('target', 'reverberant', 'oracle_inverse', 'output')]
    records = manifest.get('files', [])
    decoded, summary = {}, {}
    for name in names:
        filename = name+'.wav'
        matches = [record for record in records if record.get('file') == filename]
        if len(matches) != 1 or matches[0].get('chapter') != 'ch07' or matches[0].get('group') != 'wpe_predictable':
            raise ValueError('missing or ambiguous main audio record: '+filename)
        path = root/'codes/chapters/ch07/audio'/filename
        if path.is_symlink() or not path.is_file():
            raise ValueError('actual WAV missing or not a regular file: '+filename)
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        if sha != matches[0].get('sha256'):
            raise ValueError('actual WAV SHA mismatch: '+filename)
        decoded[name] = _actual_pcm_integers(blob, channels=1, samples=32000)
        summary[filename] = {'sha256': sha, 'channels': 1, 'samples_per_channel': 32000,
                             'sample_rate_hz': 16000, 'sample_width_bytes': 2}
    truth = decoded[names[0]][6400:14400]
    denominator = sum(v*v for v in truth)
    if denominator <= 0:
        raise ValueError('actual PCM reference power is zero; projection and relative error undefined')
    scores, measurements = {}, {}
    for name in names:
        y = decoded[name]
        numerator = sum((a-b)**2 for a, b in zip(y[6400:14400], truth))
        cross = sum(a*b for a, b in zip(y[6400:14400], truth))
        tail = sum(v*v for v in y[16000:32000])
        scores[name] = {'steady_projection_gain': cross/denominator,
                        'steady_relative_squared_reference_error': numerator/denominator,
                        'tail_mean_square': tail/(16000*32768**2)}
        measurements[name] = {**scores[name], 'integer_reference_squared_sum': denominator,
                               'integer_error_squared_sum': numerator, 'integer_output_reference_cross_sum': cross,
                               'integer_tail_squared_sum': tail, 'steady_sample_denominator': 8000,
                               'tail_sample_denominator': 16000, 'pcm_decode_divisor': 32768}
    before = measurements[names[1]]['integer_tail_squared_sum']
    after = measurements[names[3]]['integer_tail_squared_sum']
    if before <= 0 or after <= 0:
        raise ValueError('declared tail power ratio requires two positive powers')
    ratio = float(10*np.log10(after/before))
    actual_scores = {'files': scores, 'output_to_input_tail_power_ratio_db': ratio}
    recorded = parameters.get('pcm_to_pcm_reference_scores')
    if (recorded != actual_scores
            or type(recorded['output_to_input_tail_power_ratio_db']) not in (int, float)
            or any(type(value) not in (int, float)
                   for record in recorded['files'].values() for value in record.values())):
        raise ValueError('main audio actual PCM scores differ from manifest')
    return {'files': [name+'.wav' for name in names], **parameters,
            'published_audio': {'manifest': 'codes/chapters/ch00/audio/MANIFEST.json',
                                'source_sha256': sources, 'files': summary, 'pcm_measurements': measurements,
                                'tail_ratio_integer_numerator': after, 'tail_ratio_integer_denominator': before,
                                'output_to_input_tail_power_ratio_db': ratio},
            'scope': group.get('limits')}


def real_cepstrum_phase_ambiguity() -> dict:
    h1, h2 = np.array([1., .5]), np.array([.5, 1.])
    H1, H2 = np.fft.fft(h1, 8), np.fft.fft(h2, 8)
    c1, c2 = np.fft.ifft(np.log(abs(H1)**2)).real, np.fft.ifft(np.log(abs(H2)**2)).real
    return {'fft_size': 8, 'h_minimum_phase': h1.tolist(), 'h_maximum_phase': h2.tolist(),
            'spectrum_minimum_phase': _complex(H1), 'spectrum_maximum_phase': _complex(H2),
            'magnitude_squared_minimum_phase': (abs(H1)**2).tolist(),
            'magnitude_squared_maximum_phase': (abs(H2)**2).tolist(),
            'real_cepstrum_minimum_phase': c1.tolist(), 'real_cepstrum_maximum_phase': c2.tolist(),
            'polynomial_zeros': [-.5, -2.],
            'first_eight_causal_inverse_minimum_phase': [(-.5)**n for n in range(8)],
            'first_eight_causal_inverse_maximum_phase': [2*(-2.)**n for n in range(8)],
            'finite_dft_c0_closed_form': float(np.log(255/256)/4), 'continuous_cepstrum_c0': 0.,
            'scope': 'log power real cepstrum loses phase; finite DFT wraps cepstral coefficients. Inverse prefixes are not exact finite-length inverse FIRs.'}


def design_normal_comparison() -> dict:
    from dataclasses import asdict, is_dataclass
    from codes.chapters.ch07.core.dereverberation import solve_prediction_design
    design = np.array([[1., 1.], [0., 1e-9]], complex)
    target = np.array([0., -1e-9], complex)
    cases = {}
    for solver in ('design_lstsq', 'normal'):
        g, diagnostic = solve_prediction_design(design, target, diagonal_loading=0,
                                                solver=solver, return_diagnostics=True)
        if is_dataclass(diagnostic):
            diagnostic = asdict(diagnostic)
        residual = target-design@g
        # Condition infinity is a mathematical rank diagnostic, not JSON NaN.
        diagnostic = dict(diagnostic)
        for key, value in list(diagnostic.items()):
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                diagnostic[key] = None
                diagnostic[key+'_status'] = 'infinite' if np.isinf(value) else 'undefined'
        cases[solver] = {'coefficients': _complex(g), 'residual': _complex(residual),
                          'residual_squared_sum': float(np.vdot(residual, residual).real),
                          'diagnostic': diagnostic}
    return {'design': _complex(design), 'target': _complex(target),
            'exact_coefficients': [1., -1.], 'cases': cases,
            'scope': 'unregularized finite two-column example; forming A^H A squares conditioning and can erase a distinguishing small row'}


def constrained_regularized_mint() -> dict:
    from codes.chapters.ch07.core.mint_teaching import constrained_design
    a, b = Fraction(1, 2), Fraction(49, 100)
    exact = constrained_design(a, b, 0)
    regularized = constrained_design(a, b, Fraction(1, 10000))
    exact['cost_at_regularization_1e_4'] = float(Fraction(4901, 10000))
    return {'a': .5, 'b': .49, 'regularization': .0001,
            'exact_inverse': exact, 'regularized': regularized,
            'scope': 'fixed direct coefficient sum=1; reflection distortion versus weight-norm penalty, not a general MINT optimum'}


def actual_mint_audio(directory=None) -> dict:
    from codes.chapters.ch07.examples.mint_teaching_demo import DEFAULT_OUTPUT, check_assets
    return check_assets(DEFAULT_OUTPUT if directory is None else Path(directory))


def run_experiments(*, repo_root=ROOT, mint_directory=None) -> dict:
    functions = [complex_weighted_fit, power_floor_and_smoothing, history_permutation,
                 frame_count_and_rank, frame_window_overlap, exponential_statistics,
                 mint_near_common_zero, wpe_resource_budget, loaded_wpd_factorization,
                 predictable_target_audio, real_cepstrum_phase_ambiguity,
                 design_normal_comparison, constrained_regularized_mint, actual_mint_audio]
    result = {}
    for i, function in enumerate(functions, 8):
        if function is predictable_target_audio:
            value = function(repo_root)
        elif function is actual_mint_audio:
            value = function(mint_directory)
        else:
            value = function()
        result[f'E07-{i:02d}'] = value
    return result


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
