"""Twelve original small experiments E08-12..23, without upstream imports.

Run with python -m codes.chapters.ch08.chapter08_experiments. No files are written.
Substeps and supplied source/slot models are not complete separation systems.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json
import numpy as np

from codes.chapters.ch00.core.audio_samples import css_overlap_case
from codes.chapters.ch08.core.css import match_two_source_overlap
from codes.chapters.ch08.core.gss_teaching import guided_cacgmm_mvdr
from codes.chapters.ch08.core.separation import guided_activity_posterior, masked_spatial_covariance, si_sdr
from codes.chapters.ch00.cross_chapter.enhancement_step_exercises import ip_row
from codes.chapters.ch00.cross_chapter.enhancement_structure_exercises import cacg_relative_density


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {'real': value.real.tolist(), 'imag': value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_experiments() -> dict:
    result = {}
    covariance = np.array([[2, 1j], [-1j, 2]])
    sequential = np.eye(2, dtype=complex)
    for row in range(2):
        sequential[row] = ip_row(sequential, covariance, row).conj()
    simultaneous = np.array([ip_row(np.eye(2), covariance, row).conj() for row in range(2)])
    result['E08-12'] = {'covariance': covariance, 'sequential_rows': sequential,
                        'simultaneous_rows': simultaneous,
                        'sequential_weighted_gram': sequential @ covariance @ sequential.conj().T,
                        'simultaneous_weighted_gram': simultaneous @ covariance @ simultaneous.conj().T}

    y = np.array([[1., 0., 1.], [0., 1., 1.]])
    reference = np.array([1., 2., 3.])
    gram = y @ y.T
    joint = np.linalg.solve(gram, y @ reference)
    scalar = (y @ reference) / np.diag(gram)
    result['E08-13'] = {'outputs': y, 'reference': reference, 'gram': gram,
                        'joint_gains': joint, 'individual_gains': scalar,
                        'joint_sum': joint @ y, 'individual_sum': scalar @ y,
                        'complex_application_gain': np.array([2 + 1j]),
                        'pyroomacoustics_returned_conjugate': np.array([2 - 1j])}

    sources = np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]])
    mixing = np.array([[1., 0.], [1., 1.]])
    demixing = np.linalg.inv(mixing)
    estimates = demixing @ (mixing @ sources)
    zero_image = mixing[0, :, None] * estimates
    try:
        si_sdr(zero_image[1], mixing[0, 1] * sources[1])
    except ValueError:
        status = 'undefined_zero_reference_and_estimate'
    else:
        raise AssertionError('zero source image must not receive a valid SI-SDR')
    result['E08-14'] = {'mixing': mixing, 'demixing': demixing, 'reference0_images': zero_image,
                        'reference1_images': mixing[1, :, None] * estimates, 'second_reference0_score': status}

    mixture = sources.sum(0)
    consistent = np.tile(mixture / 2, (2, 1))
    result['E08-15'] = {'sources': sources, 'mixture': mixture, 'consistent_outputs': consistent,
                        'sum_error': float(np.max(np.abs(consistent.sum(0) - mixture))),
                        'output_rank': int(np.linalg.matrix_rank(consistent)),
                        'fixed_si_sdr_db': [si_sdr(consistent[i], sources[i]) for i in range(2)]}

    original = np.eye(2)
    rescaled = np.diag([1., 3.])
    directions = rescaled.T / np.linalg.norm(rescaled.T, axis=1)[:, None]
    result['E08-16'] = {'directions': directions,
                        'identity_shape_density': cacg_relative_density(directions, np.eye(2)),
                        'original_scm': masked_spatial_covariance(original[None], np.ones((1, 2)))[0],
                        'radially_changed_scm': masked_spatial_covariance(rescaled[None], np.ones((1, 2)))[0]}

    symmetric_x = np.tile(np.eye(2), (1, 6))[None]
    fit = guided_cacgmm_mvdr(symmetric_x, np.ones((12, 2)), iterations=3)
    result['E08-17'] = {'input': symmetric_x, 'posterior': fit['posterior'],
                        'shape_matrices': fit['shape_matrices'], 'resets': fit['shape_reset_count'],
                        'beam_diagnostics': fit['beam_diagnostics']}

    activity = np.array([[1], [0], [0]])
    densities = np.array([[9., 1.], [1., 1.], [1., 1.]])
    prior = np.array([.5, .5])
    posterior = guided_activity_posterior(prior, densities, activity)
    updated = posterior.mean(0)
    before, after = np.log(prior @ densities[0]), np.log(updated @ densities[0])
    result['E08-18'] = {'activity': activity, 'fixed_densities': densities, 'posterior': posterior,
                        'initial_priors': prior, 'mean_posterior_priors': updated,
                        'conditional_log_likelihood_before': before, 'conditional_log_likelihood_after': after,
                        'change': after - before,
                        'scope': 'shapes fixed; inactive-only background frames have conditional density one'}

    power = np.array([[1., 1.], [9., 9.]])
    basis, activation = np.ones((2, 1)), np.ones((1, 2))
    def objective(b, h):
        model = b @ h
        return float(np.sum(power / model + np.log(model)))
    costs = [objective(basis, activation)]
    model = basis @ activation
    basis *= np.sqrt(((power / model ** 2) @ activation.T) / ((1 / model) @ activation.T))
    costs.append(objective(basis, activation))
    model = basis @ activation
    activation *= np.sqrt((basis.T @ (power / model ** 2)) / (basis.T @ (1 / model)))
    costs.append(objective(basis, activation))
    result['E08-19'] = {'observed_power': power, 'updated_basis': basis, 'updated_activation': activation,
                        'updated_power': basis @ activation, 'objectives_initial_basis_activation': costs,
                        'scope': 'positive one-basis Gaussian/IS NMF substeps with demixing fixed'}

    a = np.array([.5 * np.eye(2), .25 * np.eye(2), .25 * np.eye(2)])
    b = np.array([np.diag([.6, .2]), np.diag([.2, .4]), np.diag([.2, .4])])
    observation = np.array([1., 2.])
    result['E08-20'] = {'source_scms_a': a, 'source_scms_b': b,
                        'observation_scm_a': a.sum(0), 'observation_scm_b': b.sum(0),
                        'first_wiener_image_a': a[0] @ np.linalg.solve(a.sum(0), observation),
                        'first_wiener_image_b': b[0] @ np.linalg.solve(b.sum(0), observation)}

    q = np.array([[1., -1.], [0., 1.]])
    g1, g2 = np.array([[3., 2.], [2., 2.]]), np.array([[4., 1.], [1., 1.]])
    diagonal = np.array([q @ g @ q.T for g in (g1, g2)])
    x = np.array([2., 1.])
    z = q @ x
    separated_z = np.array([np.diag(d) / np.diag(diagonal.sum(0)) * z for d in diagonal])
    images = np.array([np.linalg.solve(q, item) for item in separated_z])
    result['E08-21'] = {'Q': q, 'shapes': np.array([g1, g2]), 'congruence_diagonals': diagonal,
                        'QQH': q @ q.T, 'commutator': g1 @ g2 - g2 @ g1,
                        'observation': x, 'transformed_observation': z, 'diagonal_images': separated_z,
                        'microphone_images': images, 'sum_images': images.sum(0)}

    target, other = np.array([1., 1j]), np.array([-.8, 1.])
    observed = target + other
    ideal = target / observed
    bounded = np.clip((target * observed.conj()).real / abs(observed) ** 2, 0, 1)
    result['E08-22'] = {'target': target, 'interference': other, 'mixture': observed,
                        'complex_oracle_mask': ideal, 'best_bounded_real_mask': bounded,
                        'bounded_output': bounded * observed, 'squared_errors': abs(bounded * observed - target) ** 2}

    audio = css_overlap_case()
    same = np.tile([1., -1., 1., -1.], (2, 1))
    result['E08-23'] = {'audio_parameters': audio['parameters'], 'wav_stems': list(audio['signals']),
                        'silence': match_two_source_overlap(np.zeros((2, 4)), np.zeros((2, 4))),
                        'identical_slots': match_two_source_overlap(same, same)}
    return _plain(result)


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
