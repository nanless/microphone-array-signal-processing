"""E04-12..18: deterministic DOA anchors, boundaries and PCM phase checks.

Run with ``python -m codes.examples.chapter04_experiments``. No files are
written; no external implementation is imported or run. Local bounds and
single deterministic examples are not measured algorithm performance.
"""
from __future__ import annotations
import json
import numpy as np
from codes.array_tutorial.doa import esprit_ula, srp_phat
from codes.array_tutorial.geometry import plane_wave_steering
from codes.array_tutorial.audio_samples import doa_ambiguity_case, prepare_exports, read_pcm16


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


def audio_phase_ambiguity():
    """E04-18: score only known active bins in one complete steady period."""
    case = doa_ambiguity_case()
    files, groups = prepare_exports({'doa_ambiguity': case})
    rows = []
    for name, (blob, info) in files.items():
        _, decoded = read_pcm16(blob)
        bins = np.array([128]) if 'tone' in name else np.arange(20, 385)
        frequencies = bins*16000/1024
        def score(signal):
            spectrum = np.fft.rfft(signal[:, 8192:9216], axis=1)[:, bins]
            cross = spectrum[0]*spectrum[1].conj()
            phase = cross/np.abs(cross)
            return np.real(np.exp(2j*np.pi*np.array([-6., 2.])[:, None]*frequencies/16000)@phase)/bins.size
        source = case['signals'][name[:-4]]*groups['doa_ambiguity']['common_export_gain']
        rows.append({'file': name, 'float_scores': score(source).tolist(), 'pcm_scores': score(decoded).tolist(),
                     'quantization_max_abs_error': info['quantization_max_abs_error']})
    angles = np.rad2deg(np.arcsin(np.array([-6., 2.])*343/(16000*.2)))
    return {'candidate_tau12_samples': [-6, 2], 'candidate_angles_deg': angles.tolist(),
            'physical_max_delay_samples': 16000*.2/343, 'common_export_gain': groups['doa_ambiguity']['common_export_gain'],
            'score_interval_samples': [8192, 9216], 'rows': rows}


def run_exercises():
    functions = [nyquist_interpolation, srp_frequency_and_grid, phat_order_and_floors,
                 esprit_basis_and_alias, gauss_newton_one_step, local_delay_crlb, audio_phase_ambiguity]
    return {f'E04-{i:02d}': function() for i, function in enumerate(functions, 12)}


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2))
