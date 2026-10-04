"""E04-12..25: deterministic DOA anchors, boundaries and PCM phase checks.

Run with ``python -m codes.chapters.ch04.chapter04_experiments``. No files are
written; no external implementation is imported or run. Local bounds and
single deterministic examples are not measured algorithm performance.
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
from pathlib import Path
import wave
import numpy as np
from codes.chapters.ch04.core.doa import (
    esprit_ula, srp_phat, aic_source_count, mdl_source_count, music_spectrum,
)
from codes.chapters.ch03.core.geometry import plane_wave_steering
from codes.chapters.ch00.core.audio_samples import doa_ambiguity_case, prepare_exports, read_pcm16


ROOT = Path(__file__).resolve().parents[3]


def read_published_audio(group, expected_channels, *, audio_root=None, manifest_path=None):
    """Read existing ch04 PCM; validate actual source SHA, group and format.

    No generation or repair occurs. An outdated asset fails explicitly.
    Generator INPUTS is the existing authoritative source-path collection;
    this reader does not invent new dependencies for the main 109 assets.
    """
    from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
    audio_root = Path(audio_root) if audio_root is not None else ROOT/'codes/chapters/ch04/audio'
    manifest_path = Path(manifest_path) if manifest_path is not None else ROOT/'codes/chapters/ch00/audio/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in INPUTS}
    if manifest.get('generator_inputs') != sources:
        raise ValueError('published main-audio source set or SHA is stale')
    records = [r for r in manifest['files'] if r.get('group') == group]
    if len(records) != len(expected_channels) or {r['file'] for r in records} != set(expected_channels):
        raise ValueError('published audio group file set differs')
    result = {}
    for record in records:
        name = record['file']; blob = (audio_root/name).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        channels = expected_channels[name]
        if (record.get('chapter'), record.get('sample_rate_hz'), record.get('channels'), record.get('samples')) != ('ch04',16000,channels,32000):
            raise ValueError('published manifest format differs: '+name)
        if digest != record['sha256']:
            raise ValueError('published WAV SHA mismatch: '+name)
        with wave.open(io.BytesIO(blob),'rb') as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getnframes(),wav.getsampwidth(),wav.getcomptype()) != (16000,channels,32000,2,'NONE'):
                raise ValueError('published WAV format differs: '+name)
        fs, pcm = read_pcm16(blob)
        result[name] = {'samples': pcm, 'file': name, 'sha256': digest, 'sample_rate_hz': fs,
                        'channels': channels, 'samples_per_channel': pcm.shape[1],
                        'common_export_gain': record['common_export_gain']}
    return result


def nyquist_interpolation():
    """E04-12: original Nyquist is split, and inverse-FFT gain is restored."""
    from scripts.make_figures import gcc_phat_interpolated
    x = np.array([1., 0., 0., 0.])
    lags, values, _ = gcc_phat_interpolated(x, x, 1., interp=2)
    expected = (1 + 2*sum(np.cos(2*np.pi*k*lags/8) for k in range(1, 4))
                + np.cos(np.pi*lags))/8
    return {'nfft': 8, 'interpolation': 2, 'lags_samples': lags.tolist(),
            'correlation': values.tolist(), 'trigonometric_reference': expected.tolist(),
            'max_error': float(np.max(np.abs(values-expected)))}


def srp_frequency_and_grid():
    """E04-13: equal frequency weights and the near-field TDOA objective."""
    positions = np.array([[0., 0.], [.08575, 0.]])
    frequencies = np.array([1000., 2000.])
    candidates = np.deg2rad([-30., 0., 30.])
    x = plane_wave_steering(positions, frequencies, np.deg2rad(30.)).T[..., None]
    values = srp_phat(x, frequencies, positions, candidates)
    mics = np.array([[0., 0.], [4., 0.], [0., 3.], [4., 3.]])
    truth = np.array([5.5, 3.8])
    grid_x, grid_y = np.meshgrid(np.linspace(-.5, 7., 90), np.linspace(-.5, 5.2, 68))
    grid = np.stack((grid_x, grid_y), axis=-1)
    ranges = np.linalg.norm(grid[..., None, :] - mics, axis=-1)
    exact = np.linalg.norm(truth - mics, axis=-1)
    score = np.zeros(grid_x.shape)
    for i in range(4):
        for j in range(i+1, 4):
            error_s = ((ranges[..., i]-ranges[..., j])-(exact[i]-exact[j]))/343
            score += np.exp(-.5*(error_s*6000)**2)/6
    index = np.unravel_index(np.argmax(score), score.shape)
    nearest = np.unravel_index(np.argmin(np.linalg.norm(grid-truth, axis=-1)), score.shape)
    return {'far_field_angles_deg': [-30, 0, 30], 'far_field_scores': values.tolist(),
            'tau12_s': .000125, 'frequency_hz': frequencies.tolist(),
            'near_field_truth_m': truth.tolist(), 'grid_peak_m': grid[index].tolist(),
            'grid_peak_score': float(score[index]), 'grid_error_m': float(np.linalg.norm(grid[index]-truth)),
            'euclidean_nearest_m': grid[nearest].tolist(), 'nearest_score': float(score[nearest]),
            'continuous_truth_score': 1., 'gcc_width_s': 1/6000}


def phat_order_and_floors():
    """E04-14: normalization is nonlinear; three zero protections differ."""
    cross = np.array([1., -9.])
    magnitudes = np.array([1., .01, .001, 0.])
    threshold = .01
    return {'framewise_then_average': float(np.mean(cross/np.abs(cross))),
            'average_then_normalize': float(np.mean(cross)/abs(np.mean(cross))),
            'cross_magnitudes': magnitudes.tolist(), 'threshold': threshold,
            'add': (magnitudes/(magnitudes+threshold)).tolist(),
            'floor': (magnitudes/np.maximum(magnitudes, threshold)).tolist(),
            'gate': np.where(magnitudes > threshold, magnitudes/np.maximum(magnitudes, threshold), 0.).tolist()}


def esprit_basis_and_alias():
    """E04-15: a common basis gives similar shift matrices, not equal entries."""
    a = np.exp(1j*np.pi*np.arange(4)[:, None]*np.array([-.5, .5]))
    t = np.array([[1., 1.], [0., 1.]])
    es = a@t
    psi = np.linalg.lstsq(es[:-1], es[1:], rcond=None)[0]
    covariance = a@a.conj().T + .1*np.eye(4)
    estimates = esprit_ula(covariance, source_count=2, spacing_m=.1715, frequency_hz=1000)
    aliased = np.exp(2j*np.pi*np.arange(4)*np.sin(np.deg2rad(40.)))
    alias = esprit_ula(np.outer(aliased, aliased.conj())+.1*np.eye(4),
                       source_count=1, spacing_m=.343, frequency_hz=1000)
    return {'mixed_basis_shift_real': psi.real.tolist(), 'mixed_basis_shift_imag': psi.imag.tolist(),
            'estimated_angles_deg': np.rad2deg(estimates).tolist(),
            'alias_true_angle_deg': 40., 'spacing_in_wavelengths': 1.,
            'principal_alias_deg': float(np.rad2deg(alias[0])),
            'interpretation': 'Numerical model checks do not certify source count, white noise, or absence of spatial aliasing.'}


def gauss_newton_one_step():
    """E04-16: one unweighted range-difference least-squares step, in metres."""
    mics = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    truth, initial = np.array([.4, .7]), np.array([.5, .5])
    def model(point):
        ranges = np.linalg.norm(point-mics, axis=1)
        return ranges[1:]-ranges[0]
    ranges = np.linalg.norm(initial-mics, axis=1)
    unit = (initial-mics)/ranges[:, None]
    jacobian = unit[1:]-unit[0]
    observation = model(truth)
    residual = observation-model(initial)
    step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
    updated = initial+step
    return {'observation_range_difference_m': observation.tolist(), 'initial_m': initial.tolist(),
            'jacobian': jacobian.tolist(), 'step_m': step.tolist(), 'updated_m': updated.tolist(),
            'residual_before_m': float(np.linalg.norm(residual)),
            'residual_after_m': float(np.linalg.norm(observation-model(updated))),
            'limits': 'One local step, not guaranteed global convergence; independent equal residual weights are a teaching choice.'}


def local_delay_crlb():
    """E04-17: known cosine in real iid Gaussian noise, local delay parameter."""
    frequency, fs, variance = 1000., 8000., .01
    n = np.arange(8)
    derivative = 2*np.pi*frequency*np.sin(2*np.pi*frequency*n/fs)
    information = np.dot(derivative, derivative)/variance
    return {'frequency_hz': frequency, 'sample_rate_hz': fs, 'samples': 8,
            'noise_variance': variance, 'fisher_information_per_s2': float(information),
            'variance_bound_s2': float(1/information), 'std_bound_us': float(1e6/np.sqrt(information)),
            'limits': 'Known amplitude and waveform, independent real noise, local unbiased regular model. '
                      'Not an unknown-signal two-channel GCC bound, achieved error, or global uniqueness guarantee.'}


def audio_phase_ambiguity(*, audio_root=None, manifest_path=None):
    """E04-18: score only known active bins in one complete steady period."""
    case = doa_ambiguity_case()
    files, groups = prepare_exports({'doa_ambiguity': case})
    published = read_published_audio('doa_ambiguity', {'doa_ambiguity_tone.wav':2, 'doa_ambiguity_broadband.wav':2}, audio_root=audio_root, manifest_path=manifest_path)
    rows = []
    for name, (blob, info) in files.items():
        actual = published[name]
        decoded = actual['samples']
        bins = np.array([128]) if 'tone' in name else np.arange(20, 385)
        frequencies = bins*16000/1024
        def score(signal):
            spectrum = np.fft.rfft(signal[:, 8192:9216], axis=1)[:, bins]
            cross = spectrum[0]*spectrum[1].conj()
            phase = cross/np.abs(cross)
            return np.real(np.exp(2j*np.pi*np.array([-6., 2.])[:, None]*frequencies/16000)@phase)/bins.size
        source = case['signals'][name[:-4]]*groups['doa_ambiguity']['common_export_gain']
        if actual['common_export_gain'] != groups['doa_ambiguity']['common_export_gain']:
            raise ValueError('published common gain differs')
        rows.append({key:value for key,value in actual.items() if key != 'samples'})
        rows[-1].update({'float_scores': score(source).tolist(), 'pcm_scores': score(decoded).tolist(),
                        'active_frequency_denominator': int(bins.size), 'scoring_sample_denominator': 1024,
                        'quantization_max_abs_error': float(np.max(abs(decoded-source)))})
    angles = np.rad2deg(np.arcsin(np.array([-6., 2.])*343/(16000*.2)))
    return {'candidate_tau12_samples': [-6, 2], 'candidate_angles_deg': angles.tolist(),
            'physical_max_delay_samples': 16000*.2/343, 'common_export_gain': groups['doa_ambiguity']['common_export_gain'],
            'score_interval_samples': [8192, 9216], 'rows': rows}



def nonunitary_focusing_noise():
    """E04-19: perfect manifold matching does not preserve white noise."""
    a = np.ones(2); q = np.array([1.,-1.]); r = np.ones((2,2))+np.eye(2)
    transform = np.array([[2.,-1.],[-1.,2.]])
    focused = (r+transform@r@transform.T)/2
    noise = (np.eye(2)+transform@transform.T)/2
    pa = np.outer(a,a)/2; pq = np.outer(q,q)/2
    whitening = pa+pq/np.sqrt(5)
    white = whitening@focused@whitening.T
    return {'input_covariance':r.tolist(), 'transform':transform.tolist(),
            'target_after_transform':(transform@a).tolist(), 'pooled_covariance':focused.tolist(),
            'pooled_noise_covariance':noise.tolist(), 'pooled_eigenvalues':np.linalg.eigvalsh(focused).tolist(),
            'whitening':whitening.tolist(), 'whitened_covariance':white.tolist(),
            'whitened_eigenvalues':np.linalg.eigvalsh(white).tolist(),
            'target_rayleigh':float(a@focused@a/2), 'orthogonal_rayleigh':float(q@focused@q/2),
            'frequency_average_denominator':2}


def aic_mdl_comparison():
    """E04-20: one fixed eigenvalue vector; not a sampled detection rate."""
    eigenvalues = np.array([9.,4.,1.2,.8]); snapshots = 100
    aic, aic_scores = aic_source_count(eigenvalues,snapshots)
    mdl, mdl_scores = mdl_source_count(eigenvalues,snapshots)
    candidates = np.arange(4); dimensions = candidates*(8-candidates)
    fit = (aic_scores-2*dimensions)/2
    return {'eigenvalues_descending':eigenvalues.tolist(), 'snapshots':snapshots,
            'fit_terms':fit.tolist(), 'parameter_count_without_common_constant':dimensions.tolist(),
            'aic_penalty':(2*dimensions).tolist(), 'mdl_penalty':(.5*dimensions*np.log(snapshots)).tolist(),
            'aic_scores':aic_scores.tolist(), 'mdl_scores':mdl_scores.tolist(),
            'aic_selected_count':aic, 'mdl_selected_count':mdl}


def correlated_tdoa_gls():
    """E04-21: fixed error at the true centre; OLS vs known-error GLS step."""
    jacobian = -np.sqrt(2)*np.array([[1.,0.],[0.,1.],[1.,1.]])
    residual = np.array([.02,-.01,.03])
    covariance = .0001*(np.eye(3)+np.ones((3,3)))
    precision_j = np.linalg.solve(covariance,jacobian)
    precision_z = np.linalg.solve(covariance,residual)
    ols = np.linalg.lstsq(jacobian,residual,rcond=None)[0]
    gls = np.linalg.solve(jacobian.T@precision_j,jacobian.T@precision_z)
    results = {}
    for name,step in [('ols',ols),('gls',gls)]:
        error = residual-jacobian@step
        results[name] = {'step_m':step.tolist(), 'updated_m':(.5+step).tolist(),
                         'linear_residual_m':error.tolist(), 'linear_sse_m2':float(error@error),
                         'mahalanobis_squared':float(error@np.linalg.solve(covariance,error))}
    return {'jacobian':jacobian.tolist(), 'observation_error_m':residual.tolist(),
            'error_covariance_m2':covariance.tolist(), 'precision_per_m2':np.linalg.inv(covariance).tolist(),
            'normal_matrix_gls':(jacobian.T@precision_j).tolist(), 'methods':results,
            'limits':'A deterministic error at the true initial position; lower GLS cost is not a realized position-accuracy guarantee.'}


def unitary_coherent_focusing():
    """E04-22: exact ideal vectors plus checked independent audio measurements."""
    a_minus=np.array([1.,-1j,-1.,1j]); a_plus=a_minus.conj()
    first=a_minus-1j*a_plus; third=a_plus+1j*a_minus
    focused=third[[0,3,2,1]]
    raw=(np.outer(first,first.conj())+np.outer(third,third.conj()))/2
    pooled=(np.outer(first,first.conj())+np.outer(focused,focused.conj()))/2
    from codes.chapters.ch04.examples.generate_focus_audio import check_assets, DEFAULT_OUTPUT
    manifest=check_assets(DEFAULT_OUTPUT)
    return {'source_coefficients_real_imag':[[[1,0],[0,-1]],[[1,0],[0,1]]],
            'permutation':[0,3,2,1], 'unfocused_eigenvalues':np.linalg.eigvalsh(raw).tolist(),
            'focused_eigenvalues':np.linalg.eigvalsh(pooled).tolist(),
            'source_covariance_after_frequency_average_real':np.eye(2).tolist(),
            'audio_samples':manifest['samples'], 'scoring_interval_samples':[2400,29600],
            'limits':manifest['limits']}


def root_music_polynomial():
    """E04-23 analytic reciprocal roots vs floating repeated-root splitting.

    This fixed three-channel polynomial diagnostic is not a general root-MUSIC
    estimator: no root selection or source-count inference is implemented.
    """
    coefficients=np.array([-1/3,-2/3,2.,-2/3,-1/3])
    roots=np.roots(coefficients[::-1])
    angles=np.angle(roots)
    return {'noise_projector':(np.eye(3)-np.ones((3,3))/3).tolist(),
            'polynomial_coefficients_ascending':coefficients.tolist(),
            'analytic_roots_real':[1.,1.,-2+np.sqrt(3),-2-np.sqrt(3)],
            'computed_roots_real_imag':np.stack((roots.real,roots.imag),axis=-1).tolist(),
            'computed_root_moduli':abs(roots).tolist(), 'computed_root_phases_rad':angles.tolist(),
            'computed_polynomial_residuals':abs(np.polynomial.polynomial.polyval(roots,coefficients)).tolist(),
            'analytic_source_angle_deg':0., 'source_count':1,
            'limits':'Repeated roots split numerically; reciprocal roots and proximity to the unit circle are diagnostics, not automatic correctness evidence.'}


def coherent_reflection_false_peak():
    """E04-24: rank one does not identify the direct path of one emitter.

    The signal covariance is exact and noiseless. The separate 0.1 I control
    is a virtual white-noise covariance, not noise added to the audio assets.
    This is neither a DPD implementation nor a blind room/path estimator.
    """
    frequency, sample_rate, sound_speed = 4000., 16000., 343.
    spacing = sound_speed/(2*frequency)
    direct = np.ones(2, dtype=complex)
    reflection = np.array([1., 1j])
    mixture = direct+reflection
    signal_covariance = np.outer(mixture, mixture.conj())
    power = float(np.vdot(mixture, mixture).real)
    virtual_noise = .1
    covariance = signal_covariance+virtual_noise*np.eye(2)
    relative_response = mixture[1]/mixture[0]
    best_phase = float(np.angle(relative_response))
    best_angle = float(np.arcsin(best_phase*sound_speed/(2*np.pi*frequency*spacing)))
    angles = np.array([0., best_angle, np.pi/6])
    steering = np.exp(2j*np.pi*frequency*spacing/sound_speed
                      * np.sin(angles[:, None])*np.arange(2))
    projector = np.eye(2)-signal_covariance/power
    projection = np.einsum('gi,ij,gj->g', steering.conj(), projector, steering).real
    denominator_minimum = 2-(abs(mixture[0])+abs(mixture[1]))**2/power
    def pairs(value):
        return np.stack((value.real, value.imag), axis=-1).tolist()
    return {
        'frequency_hz':frequency, 'sample_rate_hz':sample_rate,
        'sound_speed_m_s':sound_speed, 'spacing_m':spacing,
        'physical_emitter_count':1, 'propagation_path_count':2,
        'direct_angle_deg':0., 'reflection_angle_deg':30.,
        'direct_vector_real_imag':pairs(direct),
        'reflection_vector_real_imag':pairs(reflection),
        'mixture_vector_real_imag':pairs(mixture),
        'signal_covariance_real_imag':pairs(signal_covariance),
        'signal_eigenvalues':np.linalg.eigvalsh(signal_covariance).tolist(),
        'signal_rank':int(np.linalg.matrix_rank(signal_covariance)),
        'signal_coherence_magnitude':float(abs(signal_covariance[0,1])/np.sqrt(
            signal_covariance[0,0].real*signal_covariance[1,1].real)),
        'relative_response_real_imag':pairs(np.asarray(relative_response)),
        'relative_response_magnitude':float(abs(relative_response)),
        'virtual_white_noise_variance':virtual_noise,
        'virtual_total_eigenvalues':np.linalg.eigvalsh(covariance).tolist(),
        'virtual_eigenvalue_ratio':float(np.linalg.eigvalsh(covariance)[-1]
                                       /np.linalg.eigvalsh(covariance)[0]),
        'noise_projector_real_imag':pairs(projector),
        'best_phase_rad':best_phase, 'false_peak_angle_deg':float(np.rad2deg(best_angle)),
        'minimum_projection_energy':float(denominator_minimum),
        'maximum_music_score':float(1/denominator_minimum),
        'comparison_angles_deg':np.rad2deg(angles).tolist(),
        'comparison_projection_energies':projection.tolist(),
        'comparison_music_scores':music_spectrum(covariance,steering,source_count=1).tolist(),
        'limits':'Exact single-frequency coherent paths from one emitter; unequal channel amplitudes '
                 'miss the unit-amplitude single-plane-wave manifold. The peak is finite. '
                 'Virtual noise is not exported to WAVs; this is not a DPD test or measured localization performance.',
    }


def exact_ctf_cross_relation():
    """E04-25: exact artificial two-tap frame-domain CTF, without STFT audio.

    Cross convolution uses plain coefficients, not conjugated coefficients.
    Source frames and the complete one-frame output tails are retained. Only
    frames 1..3 form the regression; the steady sinusoid is a separate rank
    control, not a replacement of the transient source used for that fit.
    """
    first_path = np.array([1., .5], dtype=complex)
    second_path = np.array([1+1j, .25-.5j])
    source = np.array([1., 0., 0., 1.], dtype=complex)
    first = np.convolve(first_path,source)
    second = np.convolve(second_path,source)
    frames = np.array([1,2,3])
    design = np.column_stack((first[frames],first[frames-1],second[frames-1]))
    coefficients = np.linalg.lstsq(design,second[frames],rcond=None)[0]
    residual = second[frames]-design@coefficients
    cross_residual = np.convolve(first_path,second)-np.convolve(second_path,first)
    singular = np.linalg.svd(design,compute_uv=False)
    omega = np.pi/2
    phase = np.exp(-1j*omega)
    first_response = first_path[0]+first_path[1]*phase
    second_response = second_path[0]+second_path[1]*phase
    full_ratio = second_response/first_response
    # This source exists for all integer frames, unlike the finite transient.
    tone = np.exp(1j*omega*np.arange(4))
    tone_first, tone_second = first_response*tone, second_response*tone
    tone_design = np.column_stack((tone_first[1:],tone_first[:-1],tone_second[:-1]))
    def pairs(value):
        return np.stack((value.real, value.imag),axis=-1).tolist()
    return {
        'ctf_tap_count':2, 'source_frames_real_imag':pairs(source),
        'first_path_real_imag':pairs(first_path), 'second_path_real_imag':pairs(second_path),
        'output_frame_indices':list(range(5)), 'first_output_real_imag':pairs(first),
        'second_output_real_imag':pairs(second), 'regression_frames':frames.tolist(),
        'design_real_imag':pairs(design), 'target_real_imag':pairs(second[frames]),
        'estimated_coefficients_real_imag':pairs(coefficients),
        'design_determinant_real_imag':pairs(np.asarray(np.linalg.det(design))),
        'design_rank':int(np.linalg.matrix_rank(design)), 'design_singular_values':singular.tolist(),
        'design_condition_2':float(singular[0]/singular[-1]),
        'regression_residual_norm':float(np.linalg.norm(residual)),
        'full_cross_convolution_residual_max':float(np.max(abs(cross_residual))),
        'first_coefficient_ratio_real_imag':pairs(np.asarray(coefficients[0])),
        'frame_modulation_frequency_rad':float(omega),
        'whole_ctf_frequency_ratio_real_imag':pairs(np.asarray(full_ratio)),
        'whole_ctf_frequency_ratio_magnitude':float(abs(full_ratio)),
        'whole_ctf_frequency_ratio_phase_deg':float(np.rad2deg(np.angle(full_ratio))),
        'steady_sinusoid_design_real_imag':pairs(tone_design),
        'steady_sinusoid_design_rank':int(np.linalg.matrix_rank(tone_design)),
        'steady_sinusoid_design_singular_values':np.linalg.svd(tone_design,compute_uv=False).tolist(),
        'limits':'Artificial exact frame-domain CTF, not STFT estimation from WAV or a blind DP-RTF '
                 'PSD estimator. Omega is a frame modulation frequency, not an acoustic Hz bin. '
                 'The first CTF coefficient in actual STFT models depends on windows and may contain early reflections.',
    }


def run_exercises():
    functions = [nyquist_interpolation, srp_frequency_and_grid, phat_order_and_floors,
                 esprit_basis_and_alias, gauss_newton_one_step, local_delay_crlb, audio_phase_ambiguity,
                 nonunitary_focusing_noise, aic_mdl_comparison, correlated_tdoa_gls,
                 unitary_coherent_focusing, root_music_polynomial,
                 coherent_reflection_false_peak, exact_ctf_cross_relation]
    return {f'E04-{i:02d}': function() for i, function in enumerate(functions, 12)}


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2))
