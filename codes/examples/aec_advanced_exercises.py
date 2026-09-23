"""Exact, dimensionless answer key for E06-11..E06-18.

The cases are deliberately tiny. They verify algebra and model boundaries,
not speech quality, detector performance, or industrial AEC throughput.
Importing this module performs no experiment or file access.
"""
from __future__ import annotations

from fractions import Fraction as F
import json
from math import sqrt


def run_exercises() -> dict:
    # E06-11: x=[1,1|0,0], a one-sample true path. The high input band is zero.
    high_input = [0.0, 0.0]
    high_output = [-1 / sqrt(2), 1 / sqrt(2)]

    # E06-12: x=[1,0,0,0]. The diagonal-only *system model* is not AEC output.
    true_echo = [0.0, 1.0, 0.0, 0.0]
    diagonal_only = [0.0, 0.5, 0.0, 0.5]
    missing_cross_terms = [a - b for a, b in zip(true_echo, diagonal_only)]

    # E06-13/14: equation (6-4), keeping both denominator floors.
    old = [F(4, 5), F(1, 5)]
    prior_error = F(1, 5)
    gain = [F(9, 20), F(3, 10)]  # kappa=0, epsilon_g=2
    denominator = sum(gain) + F(1, 4)
    delta = [prior_error * g / denominator for g in gain]
    uniform_gain = [F(1, 2), F(1, 2)]
    uniform_delta = [prior_error * g / (sum(uniform_gain) + F(1, 4))
                     for g in uniform_gain]

    # E06-15: exponential forgetting measures UPDATE count, not physical seconds.
    old_weight = 0.99 ** 100
    # E06-16: the second observation includes known near-end contribution 2.
    prior_weight = F(10, 11)
    prior_covariance = F(1, 11)
    second_innovation = F(3) - prior_weight
    low_gain = prior_covariance / (prior_covariance + F(1, 10))
    high_gain = prior_covariance / (prior_covariance + F(10))
    low_weight = prior_weight + low_gain * second_innovation
    high_weight = prior_weight + high_gain * second_innovation

    # E06-17: independently solve Rw=p after two RLS samples.
    # R=[[7/4,1],[1,5/4]], p=[5/2,2].
    determinant = F(7, 4) * F(5, 4) - F(1)
    batch_weights = [((F(5, 4) * F(5, 2) - F(2)) / determinant),
                     ((-F(5, 2) + F(7, 4) * F(2)) / determinant)]

    # E06-18: second innovation variance with and without cross covariance.
    complete_covariance = [[F(5, 6), F(-2, 3)],
                           [F(-2, 3), F(4, 3)]]
    second_reference = [F(-1), F(1)]
    projected = [sum(complete_covariance[i][j] * second_reference[j]
                     for j in range(2)) for i in range(2)]
    complete_s = sum(second_reference[i] * projected[i] for i in range(2)) + 1
    diagonal_s = complete_covariance[0][0] + complete_covariance[1][1] + 1

    return {
        "E06-11": {"input_high_band": high_input, "echo_high_band": high_output,
                   "scope": "two-band Haar, one-sample delay; not a room measurement"},
        "E06-12": {"true_echo": true_echo, "diagonal_only_model_output": diagonal_only,
                   "missing_cross_terms": missing_cross_terms,
                   "scope": "fixed exact model terms, not trained AEC residual"},
        "E06-13": {"gain": [float(x) for x in gain],
                   "gain_sum": float(sum(gain)), "denominator": float(denominator),
                   "update": [float(x) for x in delta],
                   "uniform_update": [float(x) for x in uniform_delta],
                   "scope": "equation (6-4) with explicit nonzero floors"},
        "E06-14": {"pure_proportionate_zero_start_update": [0.0, 0.0],
                   "reason": "all g_l vanish at kappa=1 and zero weights"},
        "E06-15": {"old_observation_weight_after_100_updates": old_weight,
                   "elapsed_seconds_if_16khz_sample_updates": 100 / 16000,
                   "elapsed_seconds_if_10ms_block_updates": 100 * .01},
        "E06-16": {"prior_weight": float(prior_weight),
                   "second_innovation": float(second_innovation),
                   "fixed_low_variance_gain": float(low_gain),
                   "fixed_low_variance_weight": float(low_weight),
                   "oracle_high_variance_gain": float(high_gain),
                   "oracle_high_variance_weight": float(high_weight),
                   "oracle_freeze_weight": float(prior_weight),
                   "scope": "known near-end injection, not a double-talk detector"},
        "E06-17": {"batch_weights": [float(x) for x in batch_weights],
                   "normal_determinant": float(determinant)},
        "E06-18": {"complete_innovation_variance": float(complete_s),
                   "diagonal_only_innovation_variance": float(diagonal_s)},
    }


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
