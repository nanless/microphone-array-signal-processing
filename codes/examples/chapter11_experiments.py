"""E11-10..19: computed selection examples, not product benchmark scores.

Run ``python -m codes.examples.chapter11_experiments``. No network, model or
recording is used. All score tables are constructed except the actual FIR
processing and PCM round trip in E11-19; those are mathematical tones.
"""
from __future__ import annotations
import json
import math
import numpy as np
from codes.array_tutorial.selection import (upper_limit_verdict, pareto_minima,
                                            small_slot_word_errors)
from codes.array_tutorial.audio_samples import selection_tradeoff_case
from codes.examples.tracking_time_exercises import zero_failure_upper_bound


def pareto_anchor():
    names = ['A', 'B', 'C', 'D']
    latency, wer, power = np.array([80., 100., 90., 160.]), np.array([.12, .10, .14, .08]), np.array([.8, .8, 1., .7])
    feasible = np.flatnonzero((latency <= 150) & (power <= 1))
    indices = pareto_minima(np.column_stack([latency[feasible], wer[feasible], power[feasible]]))
    latency_difference = (latency[1] - latency[0]) / 150
    wer_difference = (wer[0] - wer[1]) / .20
    rankings = {}
    for weight in (.5, .8):
        loss = weight * wer[feasible] / .2 + (1-weight) * latency[feasible] / 150
        rankings[str(weight)] = {'losses': loss.tolist(), 'winner': names[feasible[np.argmin(loss)]]}
    return {'names': names, 'latency_ms': latency.tolist(), 'wer': wer.tolist(), 'power_w': power.tolist(),
            'feasible': [names[i] for i in feasible], 'pareto': [names[feasible[i]] for i in indices],
            'wer_scale': .2, 'latency_scale_ms': 150., 'weight_crossing': latency_difference / (latency_difference + wer_difference),
            'weighted_rankings': rankings}


def composition_anchor():
    errors = np.array([[90, 30], [8, 252]])
    words = np.array([[900, 100], [100, 900]])
    weights = np.array([.5, .5])
    rates = errors / words
    return {'errors': errors.tolist(), 'words': words.tolist(), 'scene_weights': weights.tolist(),
            'scene_rates': rates.tolist(), 'pooled_rates': (errors.sum(axis=1) / words.sum(axis=1)).tolist(),
            'fixed_weight_rates': (rates @ weights).tolist()}


def missing_score_anchor():
    total, successful, score_sum, bounds, competitor = 10, 8, 30.4, [1., 5.], 3.6
    interval = [(score_sum + (total-successful)*bound) / total for bound in bounds]
    return {'total': total, 'successful': successful, 'sum_successful_scores': score_sum,
            'physical_score_bounds': bounds, 'successful_mean': score_sum / successful,
            'complete_mean_bounds': interval, 'competitor_complete_mean': competitor,
            'ranking_determined': interval[0] > competitor or interval[1] < competitor,
            'is_confidence_interval': False}


def simultaneous_risk_anchor():
    scenarios, trials, alpha, target = 3, 100, .05, .01
    per_alpha = alpha / scenarios
    minimum = math.ceil(math.log(per_alpha) / math.log1p(-target))
    return {'scenarios': scenarios, 'zero_failures_each': 0, 'trials_each': trials,
            'joint_alpha': alpha, 'allocated_alpha_each': per_alpha,
            'separate_95_upper': zero_failure_upper_bound(trials),
            'simultaneous_upper_each': zero_failure_upper_bound(trials, 1-per_alpha),
            'target_upper': target, 'minimum_zero_failure_trials_each': minimum,
            'previous_n_upper': zero_failure_upper_bound(minimum-1, 1-per_alpha),
            'minimum_n_upper': zero_failure_upper_bound(minimum, 1-per_alpha)}


def interval_anchor():
    limits = [150., 1.]
    candidates = {'A': [[142, 148], [.85, .95]], 'B': [[147, 153], [.85, .95]],
                  'C': [[151, 155], [.8, .9]], 'D': [[140, 145], None]}
    return {'limits_latency_ms_power_w': limits, 'intervals': candidates,
            'verdicts': {name: upper_limit_verdict(bounds, limits) for name, bounds in candidates.items()}}


def interaction_anchor():
    errors, words = {'none': 20, 'A': 22, 'B': 23, 'AB': 10}, 100
    rates = {k: v / words for k, v in errors.items()}
    difference = errors['AB'] - errors['A'] - errors['B'] + errors['none']
    return {'errors': errors, 'reference_words_each': words, 'wer': rates,
            'interaction_percentage_points': 100 * difference / words,
            'best_configuration': min(errors, key=errors.get),
            'constructed_scores_not_asr_run': True}


def slots_anchor():
    utterances, speakers, hypotheses = [['a'], ['b'], ['c']], [0, 1, 2], [['a', 'c'], ['b']]
    return {'reference_utterances': utterances, 'speaker_labels': speakers, 'hypothesis_slots': hypotheses,
            **small_slot_word_errors(utterances, speakers, hypotheses),
            'scorer': 'bounded independent token enumeration; no upstream production kernel'}


def threshold_anchor():
    positives, negative_events = np.array([.9, .8, .55, .45]), np.array([.7, .4, .3, .1])
    exposure_hours = 2.
    rows = []
    for threshold in (.5, .75):
        misses, false_events = int(np.sum(positives < threshold)), int(np.sum(negative_events >= threshold))
        rows.append({'threshold': threshold, 'misses': misses, 'false_events': false_events,
                     'miss_rate': misses / len(positives), 'false_events_per_hour': false_events / exposure_hours,
                     'cost_2miss_5false': 2*misses + 5*false_events,
                     'cost_10miss_1false': 10*misses + false_events,
                     'passes_miss_rate_limit': misses / len(positives) <= .25})
    return {'positive_scores': positives.tolist(), 'negative_event_scores': negative_events.tolist(),
            'negative_exposure_hours': exposure_hours, 'positive_trials': len(positives),
            'activation_rule': 'score >= threshold', 'miss_rate_limit': .25, 'rows': rows}


def exact_sro_anchor():
    epsilon, times = 80e-6, np.array([1., 10., 300.])
    initial_bound, budget, upper = 20e-6, 100e-6, 90e-6
    return {'epsilon': epsilon, 'reference_times_s': times.tolist(),
            'first_order_ms': (1000 * epsilon * times).tolist(),
            'same_index_exact_ms': (1000 * epsilon * times / (1+epsilon)).tolist(),
            'exact_interval_without_initial_residual_s': budget*(1+epsilon)/epsilon,
            'first_order_interval_without_initial_residual_s': budget/epsilon,
            'residual_alignment_bound_s': initial_bound, 'epsilon_bounds': [70e-6, upper],
            'absolute_error_budget_s': budget,
            'robust_interval_s': (budget-initial_bound)*(1+upper)/upper}


def audio_anchor():
    case = selection_tradeoff_case()
    return {key: value for key, value in case.items() if key != 'signals'}


def run_experiments():
    functions = [pareto_anchor, composition_anchor, missing_score_anchor, simultaneous_risk_anchor,
                 interval_anchor, interaction_anchor, slots_anchor, threshold_anchor, exact_sro_anchor, audio_anchor]
    return {f'E11-{i}': function() for i, function in enumerate(functions, 10)}


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
