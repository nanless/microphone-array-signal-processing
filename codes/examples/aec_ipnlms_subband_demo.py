"""Small exact arithmetic cases for IPNLMS and two-channel subband AEC.

Run ``python -m codes.examples.aec_ipnlms_subband_demo``. Inputs are
dimensionless; no result here is a real-recording ERLE or speed claim.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec_ipnlms import IPNLMSState
from codes.array_tutorial.aec_subband import (
    HaarDiagonalSubbandNLMSState,
    haar_analyze,
    haar_synthesize,
    two_tap_crossband_matrices,
    two_tap_subband_outputs,
)


def run_demo() -> dict:
    initial = [0.8, 0.2]
    ip = IPNLMSState(2, step_size=1.0, kappa=0.0,
                     denominator_floor=0.25, gain_floor=2.0,
                     initial_weights=initial, initial_history=[1.0])
    gains = ip.tap_gains()
    residual, echo = ip.process([1.0], [1.2], freeze=[False])
    equal = IPNLMSState(2, step_size=1.0, kappa=-1.0,
                        denominator_floor=0.25, gain_floor=2.0,
                        initial_weights=initial, initial_history=[1.0])
    equal_gain = equal.tap_gains()
    equal.process([1.0], [1.2])

    x = np.array([1.0, 1.0, 0.0, 0.0])
    h = np.array([0.0, 1.0])
    y = np.convolve(x, h)[:x.size]
    exact, diagonal, reconstructed = two_tap_subband_outputs(x, h)
    a0, a1 = two_tap_crossband_matrices(h)
    diagonal_state = HaarDiagonalSubbandNLMSState(2, step_size=0.5)
    diagonal_residual, diagonal_prior_echo = diagonal_state.process(x, y)

    return {
        "scope": "dimensionless arithmetic; no measured recording or convergence ranking",
        "ipnlms": {
            "initial_weights": initial,
            "reference_regressor": [1.0, 1.0],
            "observation": 1.2,
            "kappa": 0.0,
            "gain_floor": 2.0,
            "denominator_floor": 0.25,
            "gains": gains.tolist(),
            "gain_sum": float(sum(gains)),
            "prior_echo": float(echo[0]),
            "prior_residual": float(residual[0]),
            "posterior_weights": ip.weights.tolist(),
            "equal_gain_kappa_minus_one": equal_gain.tolist(),
            "equal_gain_posterior_weights": equal.weights.tolist(),
        },
        "haar": {
            "reference": x.tolist(),
            "physical_path": h.tolist(),
            "physical_echo": y.tolist(),
            "reference_bands": haar_analyze(x).tolist(),
            "echo_bands": haar_analyze(y).tolist(),
            "current_crossband_matrix": a0.tolist(),
            "previous_crossband_matrix": a1.tolist(),
            "exact_crossband_prediction": exact.tolist(),
            "diagonal_only_prediction": diagonal.tolist(),
            "reconstructed_echo": reconstructed.tolist(),
            "perfect_reconstruction_reference": haar_synthesize(haar_analyze(x)).tolist(),
            "diagonal_adaptive_prior_echo": diagonal_prior_echo.tolist(),
            "diagonal_adaptive_prior_residual": diagonal_residual.tolist(),
            "diagonal_adaptive_weights_after_two_blocks": diagonal_state.weights.tolist(),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
