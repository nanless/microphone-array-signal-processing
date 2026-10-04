"""Independent-scope AEC arithmetic experiments E06-22..33 and E06-40..41.

All values are dimensionless unless units are explicitly recorded. These
small cases isolate identifiability, state timing and score definitions;
they are not speech-quality, device, or convergence benchmarks.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json
import numpy as np

from codes.chapters.ch06.core.aec import NLMSState
from codes.chapters.ch06.core.aec_kalman_matrix import KalmanAECState
from codes.chapters.ch06.core.aec_rls import RLSState
from codes.chapters.ch06.core.double_talk import ncc_activity_states
from codes.chapters.ch06.aec_kalman_scalar_demo import scalar_kalman_step
from codes.chapters.ch06.aec_controlled_doubletalk import increment_metrics
from codes.chapters.ch06.core.reference_timing import reference_gain_order, echo_tail_activity


def shared_reference_normalization() -> dict:
    x = np.ones(2)
    records = {}
    for label, denominator in [('joint', float(x @ x)), ('separate', 1.)]:
        w = np.zeros(2)
        errors, paths = [], []
        for _ in range(3):
            error = 1. - float(w @ x)
            w += 1.5 * error * x / denominator
            errors.append(error)
            paths.append(w.tolist())
        records[label] = {'prior_errors': errors, 'posterior_weights': paths,
                          'next_error': float(1. - w @ x)}
    return {'reference': x.tolist(), 'microphone': 1., 'step_size': 1.5,
            'initial_weights': [0., 0.], 'normalization': records,
            'scope': 'one joint two-coordinate update, not two independent echo targets'}


def reference_identifiability() -> dict:
    training = np.array([[1., 1.], [-1., -1.]])  # rows are time samples
    truth, alternative = np.array([1., 2.]), np.array([1.5, 1.5])
    holdout = np.array([1., -1.])
    covariance = training.T @ training / len(training)
    return {'training_rows': training.tolist(), 'true_weights': truth.tolist(),
            'alternative_weights': alternative.tolist(), 'correlation': covariance.tolist(),
            'correlation_eigenvalues': np.linalg.eigvalsh(covariance).tolist(),
            'true_training_echo': (training @ truth).tolist(),
            'alternative_training_echo': (training @ alternative).tolist(),
            'holdout': holdout.tolist(), 'true_holdout_echo': float(holdout @ truth),
            'alternative_holdout_echo': float(holdout @ alternative),
            'scope': 'equal programs leave a null direction; no holdout adaptation'}


def causal_delay_support() -> dict:
    physical_delay, taps = 3, 3
    cases = []
    for reference_delay in (1, 3, 4):
        remaining = physical_delay - reference_delay
        cases.append({'reference_delay_samples': reference_delay,
                      'remaining_path_delay_samples': remaining,
                      'representable_by_causal_fir': bool(0 <= remaining < taps)})
    return {'physical_delay_samples': physical_delay, 'filter_taps': taps,
            'support_samples': [0, taps - 1], 'cases': cases,
            'scope': 'pure integer delay; delaying the algorithm reference too far requires prediction'}


def clock_drift_phase() -> dict:
    fs, frequency, rho = 16000., 1000., 250e-6
    seconds = np.array([0., 1., 2.])
    delay = rho * fs * seconds
    phase = 2 * np.pi * frequency * delay / fs
    return {'sample_rate_hz': fs, 'frequency_hz': frequency, 'drift_ppm': rho * 1e6,
            'times_seconds': seconds.tolist(), 'uncompensated_delay_samples': delay.tolist(),
            'phase_error_radians': phase.tolist(),
            'residual_to_echo_power_ratio': (2 - 2 * np.cos(phase)).tolist(),
            'scope': 'declared linearly increasing required delay; instantaneous single-frequency equal-amplitude phasor score, not whole-record ERLE'}


def missing_reference_block() -> dict:
    x, h = np.arange(1., 7.), np.array([1., .5])
    available = x.copy()
    available[2:4] = 0.
    echo = np.convolve(x, h)[:x.size]
    state = NLMSState(2, initial_weights=h)
    residual, predicted = state.process(available, echo, freeze=np.ones(x.size, bool))
    return {'physical_reference': x.tolist(), 'available_reference': available.tolist(),
            'path': h.tolist(), 'echo': echo.tolist(), 'prediction': predicted.tolist(),
            'prior_residual': residual.tolist(), 'missing_interval_samples': [2, 4],
            'first_restored_reference_sample': 4, 'first_fully_recovered_output_sample': 5,
            'scope': 'known fixed path and frozen coefficients; reference history still advances'}


def ncc_gate_boundaries() -> dict:
    alternating = np.array([1., -1., 1., -1.])
    orthogonal = np.array([1., 1., -1., -1.])
    zero = np.zeros(4)
    x = np.concatenate((zero, alternating, zero, alternating, alternating, alternating))
    d = np.concatenate((zero, .5 * alternating, orthogonal, orthogonal, zero, -alternating))
    states, ncc = ncc_activity_states(x, d, frame_size=4, activity_rms=.1,
                                      coherence_threshold=.8)
    return {'frame_labels': ['silence', 'far_only', 'near_only', 'double_talk',
                             'far_active_mic_quiet', 'inverted_polarity_far'],
            'reference_frames': x.reshape(-1, 4).tolist(),
            'microphone_frames': d.reshape(-1, 4).tolist(),
            'states': states.tolist(), 'absolute_ncc': ncc.tolist(),
            'activity_rms': .1, 'coherence_threshold': .8,
            'buffering': {'example_frame_samples': 160, 'sample_rate_hz': 16000,
                          'frame_duration_seconds': .01,
                          'first_sample_wait_seconds': 159 / 16000,
                          'decision_at_last_sample_index': 159},
            'scope': 'same-frame gating needs buffering; absolute activity threshold remains scale-dependent'}


def prior_posterior_scoring() -> dict:
    result = scalar_kalman_step(previous_weight=0., previous_variance=1., transition=1.,
                               process_variance=0., reference=1., observation=2.,
                               observation_variance=1.)
    prior = float(result['prior_error'].real)
    posterior = float(result['posterior_error'].real)
    return {'inputs': {'prior_weight': 0, 'prior_variance': 1, 'reference': 1,
                       'observation': 2, 'observation_variance': 1},
            'gain': float(result['gain'].real), 'posterior_weight': float(result['posterior_weight'].real),
            'prior_error': prior, 'posterior_error': posterior,
            'same_observation_fit_power_difference_db': float(20 * np.log10(abs(prior / posterior))),
            'scope': 'posterior fits the scored observation; not a causal cancellation improvement'}


def freeze_state_clocks() -> dict:
    state = KalmanAECState(1, transition=.5, process_covariance=[[1.]],
                           observation_variance=1., initial_covariance=[[3.]], initial_weights=[2.])
    steps = [state.step(1., 9., freeze=True) for _ in range(2)]
    rls = RLSState(1, forgetting_factor=.5, initial_regularization=1 / 3,
                   initial_weights=[2.])
    rls.process([1., 1.], [9., 9.], freeze=[True, True])
    wrong = KalmanAECState(1, transition=1., process_covariance=[[0.]],
                           observation_variance=1., initial_covariance=[[0.]])
    wrong_step = wrong.step(1., 2.)
    return {'kalman': {'initial_weight': 2., 'initial_variance': 3., 'transition': .5,
                       'process_variance': 1., 'weights': [float(s['posterior_weights'][0]) for s in steps],
                       'variances': [float(s['posterior_covariance'][0, 0]) for s in steps]},
            'rls': {'forgetting_factor': .5, 'weight': float(rls.weights[0]),
                    'inverse_reference_correlation': float(rls.inverse_covariance[0, 0])},
            'zero_uncertainty_wrong_prior': {'true_weight': 2., 'gain': float(wrong_step['gain'][0]),
                                            'posterior_weight': float(wrong.weights[0]),
                                            'prior_error': wrong_step['prior_error']},
            'scope': 'Kalman prediction clock continues under observation freeze; RLS freezes forgetting too; P and Q zero cannot repair a wrong prior'}


def covariance_model_boundaries() -> dict:
    candidates = {'singular_psd': [[1., 1.], [1., 1.]],
                  'indefinite': [[1., 2.], [2., 1.]],
                  'small_asymmetric': [[1e-20, 1e-20], [0., 1e-20]],
                  'negative_variance': [[-1e-11, 0.], [0., 1.]]}
    results = {}
    for name, value in candidates.items():
        try:
            state = KalmanAECState(2, transition=1., process_covariance=np.zeros((2, 2)),
                                   observation_variance=1., initial_covariance=value)
        except ValueError:
            results[name] = {'accepted': False}
        else:
            results[name] = {'accepted': True, 'covariance': state.covariance.tolist()}
    return {'inputs': candidates, 'outcomes': results,
            'scope': 'covariance may be singular PSD, but negative variance or material asymmetry is not roundoff'}


def block_db_aggregation() -> dict:
    before, after = np.array([1., 100.]), np.array([.01, 100.])
    scores = 10 * np.log10(before / after)
    return {'equal_length_block_input_energies': before.tolist(),
            'block_output_energies': after.tolist(), 'block_power_ratios_db': scores.tolist(),
            'arithmetic_mean_block_db': float(np.mean(scores)),
            'combined_energy_ratio_db': float(10 * np.log10(before.sum() / after.sum())),
            'scope': 'different defined summaries; no double talk and no discarded/failed blocks'}


def normalized_update_expectation() -> dict:
    x, s = np.array([1., -1., 2., -2.]), np.array([1., -1., -.5, .5])
    h = .8
    d = h * x + s
    return {'reference_states': x.tolist(), 'near_states': s.tolist(),
            'state_probabilities': [.25] * 4, 'true_path': h,
            'cross_second_moment': float(np.mean(x * s)),
            'mean_near_over_reference': float(np.mean(s / x)),
            'batch_least_squares_path': float(x @ d / (x @ x)),
            'expected_nlms_equilibrium': float(h + np.mean(s / x)),
            'scope': 'IID equal-probability state draws, epsilon=0, scalar constant-step NLMS, current draw independent of prior weight; not a four-sample online trajectory'}


def increment_orthogonal_decomposition() -> dict:
    target = np.array([1., -1., 1., -1.])
    orthogonal = .3 * np.array([1., 1., -1., -1.])
    increment = .8 * target + orthogonal
    metrics = increment_metrics(np.zeros(4), increment, target, 0, 4, sample_unit='dimensionless')
    gain = metrics['increment_projection_gain_no_delay_fit']
    remainder = increment - gain * target
    return {'target': target.tolist(), 'increment': increment.tolist(),
            'projection_gain': gain, 'orthogonal_remainder': remainder.tolist(),
            'target_remainder_inner_product': float(target @ remainder),
            'gain_error_term': float((gain - 1)**2),
            'orthogonal_error_term': float(remainder @ remainder / (target @ target)),
            'total_relative_squared_error': metrics['fixed_sample_relative_squared_error'],
            'scope': 'output increment may contain adaptive-state changes; decomposition is algebra, not isolated near-end recovery'}


def run_experiments() -> dict:
    functions = [shared_reference_normalization, reference_identifiability, causal_delay_support,
                 clock_drift_phase, missing_reference_block, ncc_gate_boundaries,
                 prior_posterior_scoring, freeze_state_clocks, covariance_model_boundaries,
                 block_db_aggregation, normalized_update_expectation,
                 increment_orthogonal_decomposition]
    cases = {f'E06-{index:02d}': function() for index, function in enumerate(functions, 22)}
    cases['E06-40'] = reference_gain_order()
    cases['E06-41'] = echo_tail_activity()
    return cases


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
