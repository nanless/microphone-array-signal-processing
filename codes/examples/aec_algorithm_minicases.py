"""Four dimensionless AEC hand cases; no recording or device benchmark.

Run ``python -m codes.examples.aec_algorithm_minicases`` from the repository
root. The FFT case illustrates partitioned *convolution*, not the adaptive
coefficient update of a complete PBFDAF. IPNLMS follows the parameterization
in Benesty and Huang, EUSIPCO 2004, Table 1:
https://www.eurasip.org/Proceedings/Eusipco/Eusipco2004/defevent/papers/cr1026.pdf
"""

from __future__ import annotations

import json

import numpy as np


def _circular_convolution(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Use the DFT product, with NumPy's inverse-DFT normalization."""

    return np.fft.ifft(np.fft.fft(signal) * np.fft.fft(taps)).real


def overlap_save_two_partitions() -> dict:
    """Demonstrate the valid half and one-block delay with N=2, P=2."""

    reference = np.array([1., 2., 3., 4.])
    partitions = [np.array([1., 2., 0., 0.]),
                  np.array([3., 4., 0., 0.])]
    # Each length-four FFT input contains the previous two and new two samples.
    reference_blocks = [np.array([0., 0., 1., 2.]),
                        np.array([1., 2., 3., 4.])]
    circular_terms = []
    block_outputs = []
    for block_index in range(2):
        terms = []
        for partition_index, taps in enumerate(partitions):
            old_index = block_index - partition_index
            old_block = (reference_blocks[old_index] if old_index >= 0
                         else np.zeros(4))
            terms.append(_circular_convolution(old_block, taps))
        circular_terms.append(terms)
        block_outputs.append(np.sum(terms, axis=0)[2:])
    return {
        "model": "dimensionless fixed FIR, zero input before sample 0; not adaptive PBFDAF",
        "block_length": 2,
        "fft_length": 4,
        "reference": reference.tolist(),
        "path_partitions": [[1, 2], [3, 4]],
        "fft_reference_blocks": [block.tolist() for block in reference_blocks],
        "circular_terms_by_block_and_partition": [
            [np.round(term, 12).tolist() for term in terms]
            for terms in circular_terms
        ],
        "discarded_first_halves": [
            np.round(np.sum(terms, axis=0)[:2], 12).tolist()
            for terms in circular_terms
        ],
        "valid_output": np.round(np.concatenate(block_outputs), 12).tolist(),
    }


def ipnlms_two_tap_case() -> dict:
    """Show per-tap allocation; one update does not rank convergence speed."""

    weights = np.array([.8, .2])
    regressor = np.array([1., 1.])
    microphone = 1.2
    prior_echo = float(weights @ regressor)
    # Round only the displayed hand-case scalar; 1.2 - 1.0 is represented
    # in binary floating point as 0.19999999999999996.
    error = round(microphone - prior_echo, 12)
    updates = {}
    for kappa in (-1, 0, 1):
        # Exact hand-case convention: both regularizers are zero because
        # ||weights||_1 and x^T G x are strictly positive here.
        gain = ((1 - kappa) / (2 * weights.size)
                + (1 + kappa) * np.abs(weights) / (2 * np.abs(weights).sum()))
        denominator = float(regressor @ (gain * regressor))
        delta = gain * regressor * error / denominator
        updates[str(kappa)] = {
            "tap_allocation": np.round(gain, 12).tolist(),
            "normalization": denominator,
            "delta_weights": np.round(delta, 12).tolist(),
            "updated_weights": np.round(weights + delta, 12).tolist(),
        }
    # With kappa=1 and exactly zero weights, the pure-proportionate formula
    # has all-zero per-tap gains. A positive denominator floor only avoids
    # division by zero; it does not make any coefficient move.
    return {
        "model": "dimensionless two-tap IPNLMS one-step illustration, not a speed test",
        "initial_weights": weights.tolist(),
        "regressor_current_first": regressor.tolist(),
        "microphone": microphone,
        "prior_echo": prior_echo,
        "prior_residual": error,
        "step_size": 1.,
        "kappa_cases": updates,
        "pure_proportionate_zero_start": {
            "initial_weights": [0., 0.],
            "tap_allocation_without_a_floor": [0., 0.],
            "update_possible_without_a_tap_floor": False,
            "remedy": "use kappa < 1 or a positive per-tap allocation floor",
        },
    }


def geigel_two_boundaries() -> dict:
    """Two hand cases in which a bare peak ratio is misleading/undefined."""

    reference_history = np.array([1., 1.])
    path = np.array([.4, .4])
    far_end_only_mic = float(path @ reference_history)
    reference_peak = float(np.max(np.abs(reference_history)))
    ratio = abs(far_end_only_mic) / reference_peak
    return {
        "model": "dimensionless two-tap acoustic path, no near-end speech or noise",
        "reference_history_current_first": reference_history.tolist(),
        "path": path.tolist(),
        "microphone": far_end_only_mic,
        "threshold": .5,
        "peak_ratio": ratio,
        "bare_rule_says_double_talk": bool(ratio > .5),
        "ground_truth_double_talk": False,
        "zero_reference_case": {
            "reference_history": [0., 0.],
            "microphone": 1.,
            "peak_ratio": None,
            "decision": None,
            "reason": "far-end inactive; denominator zero, so do not evaluate ratio",
        },
    }


def signed_delay_polarity_case() -> dict:
    """Check r_xd[k] = sum_n x[n] d[n+k] on k=0,...,3."""

    reference = np.array([1., 0., 0., 0.])
    microphone = np.array([0., 0., -1., 0.])
    correlations = np.array([
        float(reference[:reference.size - lag] @ microphone[lag:])
        for lag in range(reference.size)
    ])
    return {
        "model": "dimensionless negative-polarity two-sample delayed impulse; not a robust delay estimator",
        "reference": reference.tolist(),
        "microphone": microphone.tolist(),
        "lag_convention": "r_xd[k] = sum_n x[n] d[n+k], k >= 0",
        "candidate_lags": [0, 1, 2, 3],
        "signed_correlations": correlations.tolist(),
        "largest_signed_peak_lag": int(np.argmax(correlations)),
        "largest_absolute_peak_lag": int(np.argmax(np.abs(correlations))),
        "true_delay_samples": 2,
        "limitation": "absolute peaks can also be ambiguous for periodic or multipath signals",
    }


def run_demo() -> dict:
    """Return all fixed cases; importing this module has no side effects."""

    return {
        "overlap_save": overlap_save_two_partitions(),
        "ipnlms": ipnlms_two_tap_case(),
        "geigel": geigel_two_boundaries(),
        "delay_polarity": signed_delay_polarity_case(),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
