"""E11-10..25: computed selection examples, not product benchmark scores.

Run ``python -m codes.chapters.ch11.chapter11_experiments``. No network, model or
recording is used. Tables are constructed; E11-19/25 strictly check and read
published mathematical-tone WAVs. Missing or stale assets are never regenerated.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))
import json
import math
from fractions import Fraction
import numpy as np
from codes.chapters.ch11.core.selection import (upper_limit_verdict, pareto_minima,
                                            small_slot_word_errors)
from codes.chapters.ch00.cross_chapter.tracking_time_exercises import zero_failure_upper_bound


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
    from codes.chapters.ch11.examples.generate_selection_audio import check_main_selection_assets
    return check_main_selection_assets()


def unsupported_weight_anchor():
    # Common factor 5 preserves exact ordering without float point conversion.
    names, numerators = ['A', 'B', 'C'], [[0, 5], [3, 3], [5, 0]]
    points = [[Fraction(v, 5) for v in row] for row in numerators]
    a, b, c = points
    slope = lambda row: row[0] - row[1]
    # B has slope zero; solve B <= A and B <= C independently.
    upper = (a[1] - b[1]) / (slope(b) - slope(a))
    lower = (b[1] - c[1]) / (slope(c) - slope(b))
    limit = Fraction(3, 5)
    feasible = [i for i, row in enumerate(points) if row[1] <= limit]
    winner = min(feasible, key=lambda i: points[i][0])
    return {'names': names, 'normalized_points': [[float(v) for v in row] for row in points],
            'pareto': [names[i] for i in pareto_minima(numerators)],
            'B_minimum_weight': float(lower), 'B_maximum_weight': float(upper),
            'B_selectable_by_linear_weight': lower <= upper,
            'second_metric_upper_limit': float(limit),
            'constraint_feasible': [names[i] for i in feasible], 'constraint_winner': names[winner]}


def interval_dominance_anchor():
    from codes.chapters.ch11.core.selection import interval_dominates
    candidates = {'A': [[80, 90], [.10, .12]], 'B': [[100, 110], [.14, .16]],
                  'C': [[85, 95], [.11, .13]]}
    examples = [{'A': [90, .12], 'C': [85, .11]}, {'A': [80, .10], 'C': [95, .13]}]
    for example in examples:
        # Singletons are valid interval bounds on these exact selected points.
        av = [[v, v] for v in example['A']]
        cv = [[v, v] for v in example['C']]
        example['A_dominates_C'] = interval_dominates(av, cv)
        example['C_dominates_A'] = interval_dominates(cv, av)
    return {'intervals_latency_ms_wer': candidates, 'bounds_are_deterministic_rectangles': True,
            'A_dominates_B': interval_dominates(candidates['A'], candidates['B']),
            'A_dominates_C': interval_dominates(candidates['A'], candidates['C']),
            'C_dominates_A': interval_dominates(candidates['C'], candidates['A']),
            'point_counterexamples': examples, 'latency_upper_limit_ms': 92,
            'latency_verdicts': {name: upper_limit_verdict([rows[0]], [92])
                                 for name, rows in candidates.items()}}


def poisson_exposure_anchor():
    from codes.chapters.ch11.core.selection import zero_event_poisson_upper_bound
    exposure, confidence, target = 2., .95, .1
    alpha = 1 - confidence
    minimum = -math.log(alpha) / target
    return {'exposure_hours': exposure, 'observed_events': 0, 'confidence': confidence,
            'poisson_rate_upper_per_hour': zero_event_poisson_upper_bound(exposure, confidence),
            'target_rate_per_hour': target, 'minimum_fixed_exposure_hours': minimum,
            'zero_probability_at_upper': math.exp(-zero_event_poisson_upper_bound(exposure, confidence)*exposure),
            'scope': 'fixed exposure, homogeneous Poisson process; no KWS execution'}


def utterance_boundary_anchor():
    hypotheses = [['a'], ['b']]
    cases = {}
    for label, utterances in (('whole', [['a', 'b']]), ('split', [['a'], ['b']])):
        speakers = [0] * len(utterances)
        cases[label] = {'reference_utterances': utterances, 'speaker_labels': speakers,
                        **small_slot_word_errors(utterances, speakers, hypotheses)}
    return {'hypothesis_slots': hypotheses, 'cases': cases,
            'scope': 'same words; only reference utterance boundaries change; not ASR improvement'}


def time_edit_anchor():
    from codes.chapters.ch11.core.selection import token_edit_distance, tiny_time_constrained_edit_distance
    reference, hypothesis = ['a'], ['a']
    reference_times = [[0., 1.]]
    cases = []
    for label, times, collar in (('late', [[10., 11.]], 0.),
                                ('overlapping', [[.2, .8]], 0.),
                                ('wide_collar', [[10., 11.]], 10.)):
        error = tiny_time_constrained_edit_distance(reference, hypothesis, reference_times, times, collar)
        cases.append({'name': label, 'hypothesis_intervals_s': times, 'collar_s': collar,
                      'time_constrained_errors': error, 'time_constrained_wer': error / len(reference)})
    return {'reference_tokens': reference, 'hypothesis_tokens': hypothesis,
            'reference_intervals_s': reference_times,
            'ordinary_errors': token_edit_distance(reference, hypothesis), 'reference_words': len(reference),
            'cases': cases, 'scope': 'tiny unit-cost time edit, not complete tcpWER or original upstream run'}


def scenario_audio_anchor(directory=None):
    from codes.chapters.ch11.core.selection_audio import build_fixture, analytic_results, analyze_fixture
    from codes.chapters.ch11.examples.generate_selection_audio import check_assets
    fixture = build_fixture()
    manifest = check_assets() if directory is None else check_assets(directory)
    return {'parameters': fixture['parameters'], 'analytic': analytic_results(fixture),
            'floating_point': analyze_fixture(fixture), 'published_audio': manifest}


def run_experiments(*, scenario_directory=None):
    functions = [pareto_anchor, composition_anchor, missing_score_anchor, simultaneous_risk_anchor,
                 interval_anchor, interaction_anchor, slots_anchor, threshold_anchor, exact_sro_anchor, audio_anchor,
                 unsupported_weight_anchor, interval_dominance_anchor, poisson_exposure_anchor,
                 utterance_boundary_anchor, time_edit_anchor,
                 lambda: scenario_audio_anchor(scenario_directory)]
    return {f'E11-{i}': function() for i, function in enumerate(functions, 10)}


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
