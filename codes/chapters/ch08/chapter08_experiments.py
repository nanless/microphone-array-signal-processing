"""Eighteen original small experiments E08-12..29, without upstream imports.

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
import hashlib
import math
from pathlib import Path
import struct
import wave
import numpy as np

from codes.chapters.ch00.core.audio_samples import css_overlap_case
from codes.chapters.ch08.core.css import match_two_source_overlap
from codes.chapters.ch08.core.gss_teaching import guided_cacgmm_mvdr
from codes.chapters.ch08.core.separation import guided_activity_posterior, masked_spatial_covariance, si_sdr
from codes.chapters.ch00.cross_chapter.enhancement_step_exercises import ip_row
from codes.chapters.ch00.cross_chapter.enhancement_structure_exercises import cacg_relative_density

ROOT = Path(__file__).resolve().parents[3]
CSS_STEMS = ('css_overlap_reference', 'css_overlap_mixture',
             'css_overlap_naive', 'css_overlap_aligned')


def _integer_centered_si_sdr(estimate, reference):
    """Independent centered PCM score from exact integer second moments.

    Both inputs are decoded signed PCM integers. No waveform gain or delay is
    fitted apart from the scalar projection that defines SI-SDR itself.
    Centered moment numerators retain the common sample denominator exactly.
    This fixture has nonzero projection and residual; silent/perfect cases are
    rejected rather than assigned the teaching score's finite numerical caps.
    """
    n = len(reference)
    if n == 0 or len(estimate) != n:
        raise ValueError('PCM score requires equal nonempty records')
    es = n * sum(x*x for x in reference) - sum(reference)**2
    ee = n * sum(x*x for x in estimate) - sum(estimate)**2
    cross = n * sum(x*y for x, y in zip(estimate, reference)) - sum(estimate)*sum(reference)
    target = cross**2
    residual = es*ee - target
    if es <= 0 or ee <= 0 or target <= 0 or residual <= 0:
        raise ValueError('PCM fixture score is undefined, orthogonal or perfect')
    return {'si_sdr_db': 10*(math.log10(target)-math.log10(residual)),
            'reference_centered_energy_numerator': es,
            'estimate_centered_energy_numerator': ee,
            'centered_cross_numerator': cross,
            'projection_energy_ratio_numerator': target,
            'residual_energy_ratio_denominator': residual}


def css_published_audio(*, manifest_path=None, audio_directory=None):
    """Read/check the four published CSS WAVs and score their actual PCM.

    This function never generates or repairs assets. The main manifest must
    describe all current generator inputs, and each CSS file must have its
    declared SHA, PCM16 format, length, channels and common gain. Statistics
    are independently recomputed from integer samples, not copied from the
    manifest or regenerated/quantized float arrays.
    """
    from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
    manifest_path = Path(manifest_path) if manifest_path is not None else ROOT/'codes/chapters/ch00/audio/MANIFEST.json'
    directory = Path(audio_directory) if audio_directory is not None else ROOT/'codes/chapters/ch08/audio'
    manifest = json.loads(manifest_path.read_text(),
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValueError('main audio manifest contains a nonfinite JSON value')))
    sources = manifest.get('generator_inputs', {})
    if set(sources) != set(INPUTS):
        raise ValueError('main audio generator input set differs')
    for name, digest in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'main audio source SHA differs: {name}')
    group = manifest['groups']['css_overlap']
    parameters = group['parameters']
    if group['common_export_gain'] != 1 or parameters['common_export_gain'] != 1:
        raise ValueError('CSS common export gain differs')
    if parameters['score_interval_samples'] != [0, 32000]:
        raise ValueError('CSS score interval differs')
    records = [item for item in manifest['files'] if item['group'] == 'css_overlap']
    if len(records) != 4 or {item['file'] for item in records} != {s+'.wav' for s in CSS_STEMS}:
        raise ValueError('CSS manifest file set differs')
    decoded, measurements = {}, {}
    for item in records:
        stem = Path(item['file']).stem
        path = directory/item['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_channels = 1 if stem == 'css_overlap_mixture' else 2
        if (digest != item['sha256'] or item['chapter'] != 'ch08'
                or item['common_export_gain'] != 1 or item['sample_rate_hz'] != 16000
                or item['samples'] != 32000 or item['channels'] != expected_channels):
            raise ValueError(f'CSS asset metadata or SHA differs: {stem}')
        with wave.open(str(path), 'rb') as handle:
            if (handle.getsampwidth() != 2 or handle.getcomptype() != 'NONE'
                    or handle.getframerate() != 16000 or handle.getnframes() != 32000
                    or handle.getnchannels() != expected_channels):
                raise ValueError(f'CSS PCM format differs: {stem}')
            blob = handle.readframes(32000)
        if len(blob) != 2*32000*expected_channels:
            raise ValueError(f'CSS PCM data is truncated: {stem}')
        integers = struct.unpack('<'+'h'*(32000*expected_channels), blob)
        decoded[stem] = [list(integers[channel::expected_channels]) for channel in range(expected_channels)]
        measurements[stem] = {'sha256': digest, 'channels': expected_channels,
                              'samples_per_channel': 32000}
    reference = decoded['css_overlap_reference']
    for stem in CSS_STEMS[1:]:
        channels = decoded[stem]
        scores = [_integer_centered_si_sdr(channels[min(i, len(channels)-1)], reference[i]) for i in range(2)]
        measurements[stem]['fixed_identity_scores'] = scores
        recorded = parameters['pcm_si_sdr_db'][stem]
        if (not isinstance(recorded, list) or len(recorded) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in recorded)):
            raise ValueError(f'CSS recorded PCM scores require two finite numeric values: {stem}')
        if any(abs(score['si_sdr_db']-expected) > 1e-10 for score, expected in zip(scores, recorded)):
            raise ValueError(f'CSS recorded PCM score differs: {stem}')
    return {'manifest_path': str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
            'source_sha256': sources, 'common_export_gain': 1,
            'score_interval_samples': [0, 32000], 'samples_per_channel': 32000,
            'centering': 'per record; exact integer centered second moments',
            'fixed_identity_no_pit_no_delay_fit': True,
            'actual_pcm_measurements': measurements}


def whitening_minicase():
    """Four equiprobable non-Gaussian points; whitening is not independence."""
    sources = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]).T
    mixing = np.array([[1., .5], [.5, 1.]])
    mixture = mixing @ sources
    whitener = np.linalg.inv(mixing)  # this positive symmetric A gives a valid whitening
    whitened = whitener @ mixture
    rotation = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2)
    rotated = rotation @ whitened
    # Event identities are derived from the exact integer source coordinates,
    # rather than treating a floating-point tolerance as a probability model.
    zero1 = sources[0]+sources[1] == 0
    zero2 = -sources[0]+sources[1] == 0
    return {'source_points': sources, 'mixing': mixing, 'mixture_points': mixture,
            'mixture_covariance': mixture @ mixture.T/4,
            'whitener': whitener, 'whitened_points': whitened,
            'rotation': rotation, 'rotated_points': rotated,
            'rotated_covariance': rotated @ rotated.T/4,
            'probability_first_zero': np.mean(zero1), 'probability_second_zero': np.mean(zero2),
            'probability_both_zero': np.mean(zero1 & zero2),
            'independent_product_probability': np.mean(zero1)*np.mean(zero2),
            'rank_one_covariance': np.ones((2, 2))}


def congruence_model_boundary():
    """With the first shape I, the whitened remaining shapes must commute."""
    shapes = np.array([np.eye(2), np.diag([1., 2.]), [[2., 1.], [1., 2.]]])
    commutator = shapes[1] @ shapes[2]-shapes[2] @ shapes[1]
    return {'shapes': shapes, 'shape_eigenvalues': np.linalg.eigvalsh(shapes),
            'whitened_remaining_commutator': commutator,
            'joint_congruence_exists_for_all_three': False,
            'scope': 'positive definite first shape I; pairwise existence does not imply joint existence'}


def fastmnmf_scale_objective():
    """A fixed model's physical likelihood is invariant under Q/g rescaling."""
    q = np.array([[1., -1.], [0., 1.]])
    shapes = np.array([[[3., 2.], [2., 2.]], [[4., 1.], [1., 1.]]])
    diagonal = np.array([np.diag(q @ shape @ q.T) for shape in shapes])
    observation = np.array([2., 1.])
    cases = []
    for multiplier in (1., 2.):
        transform = multiplier*q
        d = multiplier**2*diagonal.sum(0)
        z = transform @ observation
        quadratic = float(np.sum(abs(z)**2/d))
        log_variance = float(np.log(d).sum())
        jacobian = float(-2*np.linalg.slogdet(transform)[1])
        physical_covariance = np.linalg.solve(transform, np.diag(d)) @ np.linalg.inv(transform).T
        cases.append({'Q': transform, 'diagonal_variance': d, 'transformed_observation': z,
                      'quadratic_term': quadratic, 'log_variance_term': log_variance,
                      'jacobian_term': jacobian, 'full_objective': quadratic+log_variance+jacobian,
                      'objective_without_jacobian': quadratic+log_variance,
                      'physical_covariance': physical_covariance})
    return {'cases': cases, 'frames': 1, 'frequencies': 1,
            'scope': 'proper complex likelihood, fixed parameters; not an estimated FastMNMF model'}


def mixture_consistency_minicase():
    """Fixed positive-weight KKT projection; sum consistency is not separation."""
    estimate, mixture, variance = np.array([1., 1.]), 4., np.array([1., 3.])
    correction = mixture-estimate.sum()
    projected = estimate+variance/variance.sum()*correction
    equal = estimate+correction/len(estimate)
    return {'mixture': mixture, 'estimates': estimate, 'variances': variance,
            'mixture_residual': correction, 'weighted_projection': projected,
            'equal_projection': equal,
            'weighted_cost_at_weighted_projection': float(.5*np.sum((projected-estimate)**2/variance)),
            'weighted_cost_at_equal_projection': float(.5*np.sum((equal-estimate)**2/variance)),
            'kkt_correction_over_variance': (projected-estimate)/variance,
            'scope': 'one shared reference and scale; positive fixed weights, not learned confidence'}


def css_polarity_minicase():
    """Absolute-correlation matching fixes permutation, not polarity or gain."""
    previous = np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]])
    cases = []
    for multiplier in (-1., -2.):
        current = multiplier*previous
        matching = match_two_source_overlap(previous, current)
        aligned = current[matching['current_indices_for_previous']]
        gain = np.sum(previous*aligned, axis=1)/np.sum(aligned**2, axis=1)
        signed = previous @ current.T/2/abs(multiplier)
        cases.append({'current_multiplier': multiplier, 'current': current,
                      'matching': matching, 'signed_centered_correlation': signed,
                      'naive_equal_overlap': (previous+aligned)/2,
                      'polarity_only_overlap': (previous+np.sign(multiplier)*aligned)/2,
                      'direct_application_gain': gain,
                      'corrected_equal_overlap': (previous+gain[:, None]*aligned)/2})
    return {'previous': previous, 'samples_per_overlap': 4, 'cases': cases,
            'scope': 'same sample times, nonzero clean overlap, known scalar relation; no universal identity/delay repair'}


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
                        'float_matching': audio['parameters']['matching'],
                        'floating_si_sdr_db': audio['parameters']['floating_si_sdr_db'],
                        'published_audio': css_published_audio(),
                        'silence': match_two_source_overlap(np.zeros((2, 4)), np.zeros((2, 4))),
                        'identical_slots': match_two_source_overlap(same, same)}
    result['E08-24'] = whitening_minicase()
    result['E08-25'] = congruence_model_boundary()
    result['E08-26'] = fastmnmf_scale_objective()
    result['E08-27'] = mixture_consistency_minicase()
    from codes.chapters.ch08.core.mask_representation import run_experiment
    from codes.chapters.ch08.examples.mask_representation_demo import check_assets
    mask_report, _ = run_experiment()
    result['E08-28'] = {'floating_point_experiment': mask_report,
                        'published_audio': check_assets()}
    result['E08-29'] = css_polarity_minicase()
    return _plain(result)


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
