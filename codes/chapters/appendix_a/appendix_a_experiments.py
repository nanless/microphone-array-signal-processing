"""E12-06..20: reproducible mathematical counterexamples for Appendix A.

All values are constructed teaching inputs, not measurements. The audio case
is mathematical PCM16 synthesis; no file is written by importing or running
this module. Run ``python -m codes.chapters.appendix_a.appendix_a_experiments`` for JSON.
"""

from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json

import numpy as np

from codes.chapters.appendix_a.examples.check_main_math_audio import check_main_math_assets, ROOT
from codes.chapters.appendix_a.examples.generate_weighted_audio import check_assets, OUTPUT
from codes.chapters.appendix_a.core.weighted_audio import build_fixture, analyze_fixture, analytic_results
from codes.chapters.appendix_a.core.math_foundations import (
    blockwise_circular_convolution, fft_overlap_add,
)


def correlated_gls_demo() -> dict:
    """E12-20: one fixed known correlated-noise model, without asset IO.

    Both measurements contain the same scalar target. Solve the whitened
    least-squares problem using NumPy's existing implementation; this is not
    a general GLS solver or an estimator of the given noise covariance.
    """
    design = np.ones((2, 1))
    covariance = np.array([[1., 1.5], [1.5, 4.]])
    rhs = np.array([0., 2.])
    factor = np.linalg.cholesky(covariance)
    whitening = np.linalg.solve(factor, np.eye(2))
    whitened_design = whitening @ design
    whitened_rhs = whitening @ rhs
    # The returned row maps whitened observations to the scalar estimate.
    white_row, _, rank, singular = np.linalg.lstsq(
        whitened_design, np.eye(2), rcond=None
    )
    white_weight = white_row.conj().T[:, 0]
    full_weight = whitening.conj().T @ white_weight
    full_solution = np.linalg.lstsq(
        whitened_design, whitened_rhs, rcond=None
    )[0]
    cases = {}
    for name, weight in (
        ('ols', np.array([.5, .5])),
        ('diagonal_only', np.array([.8, .2])),
        ('full_covariance', full_weight),
    ):
        cases[name] = {
            'effective_original_weight': weight.tolist(),
            'target_response': float(np.vdot(weight, design[:, 0]).real),
            'variance_under_given_covariance': float(
                np.vdot(weight, covariance @ weight).real
            ),
            'fixed_observation_estimate': float(np.vdot(weight, rhs).real),
        }
    # Deliberately wrong: whiten the rhs but leave the design unchanged.
    wrong_row = np.linalg.lstsq(design, np.eye(2), rcond=None)[0]
    wrong_weight = whitening.conj().T @ wrong_row.conj().T[:, 0]
    wrong_solution = np.linalg.lstsq(design, whitened_rhs, rcond=None)[0]
    return {
        'exercise_id': 'E12-20', 'numpy_version': np.__version__,
        'design': design.tolist(), 'noise_covariance': covariance.tolist(),
        'determinant': float(np.linalg.det(covariance)),
        'fixed_observation': rhs.tolist(), 'cholesky_factor': factor.tolist(),
        'whitening_matrix': whitening.tolist(),
        'whitened_noise_covariance': (whitening @ covariance @ whitening.conj().T).tolist(),
        'noise_precision': (whitening.conj().T @ whitening).tolist(),
        'whitened_design': whitened_design.tolist(),
        'whitened_observation': whitened_rhs.tolist(),
        'whitened_coordinate_weight': white_weight.tolist(),
        'whitened_lstsq_solution': full_solution.tolist(),
        'whitened_design_rank': int(rank),
        'whitened_design_singular_values': singular.tolist(),
        'cases': cases,
        'wrong_only_observation_whitened': {
            'effective_original_weight': wrong_weight.tolist(),
            'target_response': float(np.vdot(wrong_weight, design[:, 0]).real),
            'fixed_observation_estimate': float(wrong_solution[0]),
        },
        'scope': 'given HPD correlated noise and exact common-target design; '
                 'unbiased population variance, no covariance estimation or audio; '
                 'one fixed observation does not rank errors against unknown truth',
    }


def run_experiments(*, repo_root=ROOT, weighted_directory=OUTPUT) -> dict:
    """Return stable exercise IDs, inputs, and JSON-safe intermediate results."""
    results = {}

    n, sample_rate = 8, 8000
    signed_frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    one_sided_frequencies = np.fft.rfftfreq(n, 1 / sample_rate)
    cosine = np.cos(2 * np.pi * np.arange(n) / n)
    spectrum = np.fft.fft(cosine)
    alternating = (-1.) ** np.arange(n)
    alternating_spectrum = np.fft.fft(alternating)
    results['E12-06'] = {
        'sample_rate_hz': sample_rate, 'fft_length': n,
        'signed_frequency_hz_by_bin': signed_frequencies.tolist(),
        'one_sided_frequency_hz_by_bin': one_sided_frequencies.tolist(),
        'real_cosine_nonzero_bins': [int(k) for k in np.flatnonzero(np.abs(spectrum) > 1e-12)],
        'real_cosine_nonzero_coefficients': [float(spectrum[k].real) for k in (1, 7)],
        'alternating_signal_nonzero_bins': [
            int(k) for k in np.flatnonzero(np.abs(alternating_spectrum) > 1e-12)
        ],
        'alternating_signal_nonzero_coefficients': [float(alternating_spectrum[4].real)],
        'nyquist_bin': {'index': 4, 'frequency_magnitude_hz': 4000,
                        'one_discrete_bin_for_both_signs': True},
        'scope': 'signed representative of DFT frequencies; k=4 also denotes +4 kHz modulo 8 kHz',
    }

    vector = np.array([1, 1j], dtype=complex)
    results['E12-07'] = {
        'vector_real': vector.real.tolist(), 'vector_imag': vector.imag.tolist(),
        'hermitian_norm_squared': float(np.vdot(vector, vector).real),
        'transpose_square': float(np.dot(vector, vector).real),
        'scope': 'transpose without conjugation is not a complex squared norm',
    }

    x = np.array([1., 2., 3., 4.])
    h = np.array([1., .5])
    audio = check_main_math_assets(repo_root)
    results['E12-08'] = {
        'input': x.tolist(), 'filter': h.tolist(), 'block_size': 2,
        'linear_output': fft_overlap_add(x, h, 2).tolist(),
        'block_circular_output': blockwise_circular_convolution(x, h, 2).tolist(),
        'audio_case': {
            'block_size': audio['parameters']['block_size'],
            'pulse_positions_samples': audio['parameters']['pulse_positions_samples'],
            'first_correct_echo_sample': audio['parameters']['first_correct_echo_sample'],
            'first_wrong_wrap_sample': audio['parameters']['first_wrong_wrap_sample'],
            'float_analysis': audio['float_analysis'],
            'pcm_analysis': audio['pcm_analysis'],
            'integer_analysis': audio['integer_analysis'],
            'published_files': audio['published_files'],
            'source_sha256': audio['source_sha256'],
        },
    }

    x1 = np.array([0., 1., 0.])
    x2 = np.array([1., 0., 0.])
    lags = np.arange(-2, 3)
    r12 = np.correlate(x1, x2, mode='full')
    r21 = np.correlate(x2, x1, mode='full')
    results['E12-09'] = {
        'x1': x1.tolist(), 'x2': x2.tolist(),
        'lags_samples': lags.tolist(), 'r12': r12.tolist(), 'r21': r21.tolist(),
        'r12_peak_lag_samples': int(lags[np.argmax(r12)]),
        'r21_peak_lag_samples': int(lags[np.argmax(r21)]),
        'boundary': 'zero extension outside each three-sample record, not circular correlation',
    }

    snapshots = np.array([[1., 0.], [0., 1.], [0., 0.]])
    mean = snapshots.mean(axis=1, keepdims=True)
    moment = snapshots @ snapshots.T / 2
    centered = (snapshots - mean) @ (snapshots - mean).T / 2
    results['E12-10'] = {
        'snapshots_columns': snapshots.tolist(), 'sample_mean': mean[:, 0].tolist(),
        'uncentered_second_moment': moment.tolist(), 'centered_covariance': centered.tolist(),
        'uncentered_rank': int(np.linalg.matrix_rank(moment)),
        'centered_rank': int(np.linalg.matrix_rank(centered)),
        'centered_eigenvalues': np.linalg.eigvalsh(centered).tolist(),
        'denominator': 2,
    }

    matrix = np.diag([1., 1e-4])
    rhs = np.array([1., 1e-4])
    results['E12-11'] = {
        'matrix': matrix.tolist(), 'example_rhs': rhs.tolist(),
        'condition_A': float(np.linalg.cond(matrix)),
        'condition_normal': float(np.linalg.cond(matrix.T @ matrix)),
        'recovered_x': np.linalg.solve(matrix, rhs).tolist(),
        'scope': '2-norm conditioning of a diagonal example, not a measured solver error',
    }

    target = np.diag([1., 0.])
    reference = np.array([1., 0.])
    singular_noise = np.diag([0., 1.])
    identity_noise = np.eye(2)
    mus = [1., 10., 1e6]
    singular_weights = []
    identity_weights = []
    for mu in mus:
        singular_weights.append(np.linalg.solve(target + mu * singular_noise,
                                                target @ reference).tolist())
        identity_weights.append(np.linalg.solve(target + mu * identity_noise,
                                               target @ reference).tolist())
    results['E12-12'] = {
        'target_covariance': target.tolist(), 'reference': reference.tolist(),
        'singular_noise_covariance': singular_noise.tolist(),
        'identity_noise_covariance': identity_noise.tolist(), 'mus': mus,
        'singular_noise_weights': singular_weights,
        'identity_noise_weights': identity_weights,
        'limiting_statement': 'with target energy in the noise nullspace, increasing mu does not force zero output',
    }

    covariance = np.diag([4., 1.])
    steering = np.ones(2)
    absolute_load, relative_coefficient, scale_factor = 1., .4, 10.
    cases = []
    for input_scale in (1., scale_factor):
        scaled_covariance = input_scale * covariance
        for kind in ('absolute', 'relative'):
            diagonal_addition = (absolute_load if kind == 'absolute' else
                                 relative_coefficient * np.trace(scaled_covariance) / 2)
            loaded = scaled_covariance + diagonal_addition * np.eye(2)
            unnormalized = np.linalg.solve(loaded, steering)
            weights = unnormalized / np.vdot(steering, unnormalized)
            cases.append({
                'input_scale': input_scale, 'kind': kind,
                'effective_diagonal_addition': float(diagonal_addition),
                'loaded_covariance': loaded.tolist(),
                'weights': weights.tolist(),
                'condition_2': float(np.linalg.cond(loaded)),
                'target_response': float(np.vdot(weights, steering).real),
            })
    results['E12-13'] = {
        'covariance': covariance.tolist(), 'steering': steering.tolist(),
        'microphones': 2, 'absolute_loading': absolute_load,
        'relative_coefficient': relative_coefficient,
        'scale_factor': scale_factor, 'cases': cases,
        'scope': 'fixed absolute load changes under covariance scaling; trace-relative load scales with the covariance',
    }
    def complex_record(value):
        value = np.asarray(value)
        return {'real':value.real.tolist(), 'imag':value.imag.tolist()}

    design = np.array([[1], [1j]], dtype=complex)
    rhs = np.ones(2, dtype=complex)
    solution = np.linalg.lstsq(design, rhs, rcond=None)[0]
    residual = design@solution-rhs
    results['E12-14'] = {'design':complex_record(design), 'rhs':complex_record(rhs),
        'solution':complex_record(solution), 'residual':complex_record(residual),
        'hermitian_orthogonality':complex_record(design.conj().T@residual),
        'transpose_gram':complex_record(design.T@design),
        'residual_squared_sum':float(np.vdot(residual,residual).real)}

    design = np.ones((2,1))
    rhs = np.array([0.,2.])
    covariance = np.diag([1.,4.])
    precision = np.diag([1.,.25])
    rows = {}
    for name, weight in (('ols',np.eye(2)), ('gls',precision)):
        solution = np.linalg.solve(design.T@weight@design, design.T@weight@rhs)
        residual = design@solution-rhs
        coefficient = np.linalg.solve(design.T@weight@design, design.T@weight)
        rows[name] = {'solution':solution.tolist(),'residual':residual.tolist(),
            'weighted_orthogonality':(design.T@weight@residual).tolist(),
            'ordinary_orthogonality':(design.T@residual).tolist(),
            'weighted_residual_squared_sum':float(residual@weight@residual),
            'common_noise_precision_cost':float(residual@precision@residual),
            'parameter_variance':float((coefficient@covariance@coefficient.T)[0,0])}
    results['E12-15'] = {'design':design.tolist(),'rhs':rhs.tolist(),
        'noise_covariance':covariance.tolist(),'cases':rows,
        'variance_scope':'zero-mean noise with stated covariance; separate from this fixed rhs residual'}

    epsilon = 1e-8
    design = np.diag([1.,epsilon])
    rhs = np.array([1.,epsilon])
    cases = []
    for cutoff in (1e-10,1e-6):
        solution, residual_array, rank, singular = np.linalg.lstsq(design,rhs,rcond=cutoff)
        residual = design@solution-rhs
        cases.append({'rcond':cutoff,'effective_singular_value_threshold':float(cutoff*singular[0]),
            'rank':int(rank),'singular_values':singular.tolist(),
            'solution':solution.tolist(),'returned_residual_array':residual_array.tolist(),
            'actual_residual_norm':float(np.linalg.norm(residual))})
    delta = 1e-16
    ridge = np.linalg.solve(design.T@design+delta*np.eye(2),design.T@rhs)
    results['E12-16'] = {'numpy_version':np.__version__,
        'design':design.tolist(),'rhs':rhs.tolist(),'cases':cases,
        'ridge_delta':delta,'ridge_solution':ridge.tolist(),
        'ridge_residual_squared_sum':float(np.sum((design@ridge-rhs)**2)),
        'ridge_penalty':float(delta*np.sum(ridge**2)),
        'ridge_scope':'ridge minimizes squared residual plus delta times squared parameter norm; not the truncated SVD rule'}

    x1 = np.array([0,1,1j]);x2=np.array([1,1j,0])
    lags = list(range(-2,3))
    proper = np.array([sum(x1[n]*x2[n-q].conjugate() for n in range(3) if 0<=n-q<3) for q in lags])
    exchanged = np.array([sum(x2[n]*x1[n-q].conjugate() for n in range(3) if 0<=n-q<3) for q in lags])
    wrong = np.array([sum(x1[n]*x2[n-q] for n in range(3) if 0<=n-q<3) for q in lags])
    results['E12-17'] = {'x1':complex_record(x1),'x2':complex_record(x2),'lags_samples':lags,
        'r12':complex_record(proper),'r21':complex_record(exchanged),
        'r21_magnitude_peak_lag_samples':lags[int(np.argmax(np.abs(exchanged)))],
        'without_conjugate':complex_record(wrong),
        'magnitude_peak_lag_samples':lags[int(np.argmax(np.abs(proper)))],
        'scope':'zero-extended finite complex correlation; magnitude selects a lag, phase remains complex'}

    asymmetric = np.array([[1.,100.],[0.,2.]])
    imaginary_diagonal = np.diag([1+5j,2-3j])
    cases=[]
    for label, matrix, triangle in (('asymmetric_lower',asymmetric,'L'),
        ('asymmetric_upper',asymmetric,'U'),('imaginary_diagonal',imaginary_diagonal,'L')):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=triangle)
        effective = np.diag(matrix.diagonal().real).astype(complex)
        if triangle=='L':
            effective += np.tril(matrix,-1)+np.tril(matrix,-1).conj().T
        else:
            effective += np.triu(matrix,1)+np.triu(matrix,1).conj().T
        cases.append({'label':label,'input':complex_record(matrix),'UPLO':triangle,
            'effective_hermitian_matrix':complex_record(effective),'eigenvalues':eigenvalues.tolist(),
            'eigenvectors':complex_record(eigenvectors),
            'original_matrix_residual_frobenius':float(np.linalg.norm(matrix@eigenvectors-eigenvectors*eigenvalues)),
            'effective_matrix_residual_frobenius':float(np.linalg.norm(effective@eigenvectors-eigenvectors*eigenvalues))})
    results['E12-18'] = {'cases':cases,'scope':'raw NumPy interface demonstration, not a validated EVD or permission to repair invalid covariance'}

    published = check_assets(weighted_directory)
    fixture = build_fixture()
    results['E12-19'] = {'parameters':fixture['parameters'],'analytic':analytic_results(),
        'floating_point':analyze_fixture(fixture),'pcm_analysis':published['pcm_analysis'],
        'published_files':published['files'],'source_sha256':published['source_sha256']}
    results['E12-20'] = correlated_gls_demo()
    return results


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
