"""E10-16/17 and E11-09: event boundaries, timestamp scope, paired evidence.

Deterministic teaching inputs; no device, model, network or file writes.
Run with ``python -m codes.examples.engineering_boundary_exercises``.
"""

from __future__ import annotations

import json
import math
from numbers import Integral

import numpy as np

from codes.array_tutorial.engineering import _finite_1d, simulate_deadline_queue


def callback_timing_ns(input_adc_ns, callback_ns, output_dac_ns):
    """Differences of three ordered timestamps in one known clock domain.

    Inputs are constructed integer nanoseconds, not a claim about device
    accuracy. The first-sample latency requires matching input/output samples;
    the helper cannot establish that correspondence or clock calibration.
    """
    values = (input_adc_ns, callback_ns, output_dac_ns)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in values):
        raise ValueError("timestamps must be integer nanoseconds")
    adc, callback, dac = map(int, values)
    if not adc <= callback <= dac:
        raise ValueError("this example requires adc <= callback <= dac")
    try:
        result = {
            "input_age_ms": (callback - adc) / 1_000_000,
            "output_lead_ms": (dac - callback) / 1_000_000,
            "matched_first_sample_latency_ms": (dac - adc) / 1_000_000,
        }
    except OverflowError as error:
        raise ValueError("timestamp differences exceed finite float64 range") from error
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("timestamp differences exceed finite float64 range")
    return result


def paired_sign_test_lower_is_better(baseline, candidate):
    """One-sided exact sign test conditional on non-ties, H0: P(win)=1/2.

    Independent paired observations, a prespecified direction, and the stated
    null are prerequisites. Magnitudes and pooled WER are not tested. All ties
    return no p-value; ties remain in the reported total. Equality is exact on
    the supplied observations, not determined by an implicit tolerance.
    """
    a = _finite_1d(baseline, "baseline")
    b = _finite_1d(candidate, "candidate")
    if a.shape != b.shape or not a.size:
        raise ValueError("paired arrays must have equal non-zero length")
    wins = int(np.count_nonzero(b < a))
    losses = int(np.count_nonzero(b > a))
    informative = wins + losses
    # Compare observations without subtracting: opposite large finite values
    # must not overflow merely to determine a sign.
    tail = (sum(math.comb(informative, k) for k in range(wins, informative + 1))
            if informative else None)
    denominator = (1 << informative) if informative else None
    return {
        "pairs": int(a.size), "wins": wins, "losses": losses,
        "ties": int(a.size) - informative, "informative_pairs": informative,
        "one_sided_pvalue": tail / denominator if informative else None,
        "status": "valid" if informative else "no_non_tied_pairs",
    }


def run_exercises():
    period = 0.1
    adc, callback, dac = 5_000_000_000, 5_015_000_000, 5_045_000_000
    a = np.array([12, 8, 20, 5, 15, 10])
    b = np.array([10, 9, 14, 5, 12, 9])
    return {
        "E10-16": {
            "period_ms": period, "frames": 30,
            "equal_duration": simulate_deadline_queue(np.full(30, period), period, 1),
            "longer_duration": simulate_deadline_queue(np.full(30, 0.11), period, 1),
        },
        "E10-17": {
            "sample_rate_hz": 16000, "frame_samples": 160,
            "input_adc_ns": adc, "callback_ns": callback, "output_dac_ns": dac,
            **callback_timing_ns(adc, callback, dac),
            "compute_ms": 3.0,
            "added_output_block_ms": 10.0,
            "latency_after_one_extra_output_block_ms": 55.0,
        },
        "E11-09": {
            "reference_words_per_session": [100] * 6,
            "baseline_errors": a.tolist(), "candidate_errors": b.tolist(),
            "baseline_pooled_wer": int(a.sum()) / 600,
            "candidate_pooled_wer": int(b.sum()) / 600,
            "reduction_percentage_points": int(a.sum() - b.sum()) / 6,
            "relative_error_reduction": int(a.sum() - b.sum()) / int(a.sum()),
            **paired_sign_test_lower_is_better(a, b),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
