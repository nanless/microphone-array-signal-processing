"""Two-tap, two-sample RLS arithmetic example for chapter 6.

Run ``python -m codes.examples.aec_rls_demo`` from the repository root.
The numbers are dimensionless teaching inputs, not audio measurements or
evidence that full-matrix RLS is suitable for a long real-time echo path.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec_rls import RLSState


def _rounded(values: np.ndarray) -> list:
    return np.round(np.asarray(values), 12).tolist()


def run_demo() -> dict:
    """Return the prior outputs and independent batch weighted-LS comparison."""

    state = RLSState(2, forgetting_factor=0.5, initial_regularization=1.0)
    first_error, first_echo = state.process([1.0], [1.0])
    first_weights = state.weights
    first_inverse = state.inverse_covariance
    second_error, second_echo = state.process([1.0], [2.0])

    # Batch construction is independent of the recursive inverse update.
    regressors = np.array([[1.0, 0.0], [1.0, 1.0]])
    observations = np.array([1.0, 2.0])
    weighted_normal = (0.5 ** 2 * np.eye(2)
                       + 0.5 * np.outer(regressors[0], regressors[0])
                       + np.outer(regressors[1], regressors[1]))
    weighted_cross = (0.5 * regressors[0] * observations[0]
                      + regressors[1] * observations[1])
    batch_weights = np.linalg.solve(weighted_normal, weighted_cross)

    return {
        "scope": "dimensionless two-tap RLS arithmetic, not measured AEC",
        "convention": "regressor is [current reference, previous reference]; output uses prior taps",
        "forgetting_factor": 0.5,
        "initial_regularization": 1.0,
        "initial_R": [[1.0, 0.0], [0.0, 1.0]],
        "first": {
            "regressor": [1.0, 0.0], "observation": 1.0,
            "prior_echo": _rounded(first_echo)[0],
            "prior_residual": _rounded(first_error)[0],
            "posterior_weights": _rounded(first_weights),
            "posterior_inverse_covariance": _rounded(first_inverse),
        },
        "second": {
            "regressor": [1.0, 1.0], "observation": 2.0,
            "prior_echo": _rounded(second_echo)[0],
            "prior_residual": _rounded(second_error)[0],
            "posterior_weights": _rounded(state.weights),
            "posterior_inverse_covariance": _rounded(state.inverse_covariance),
        },
        "batch_weighted_normal": _rounded(weighted_normal),
        "batch_weighted_cross": _rounded(weighted_cross),
        "batch_weights": _rounded(batch_weights),
        "max_recursive_vs_batch_weight_difference": float(
            np.max(np.abs(state.weights - batch_weights))),
        "exact_hand_values": {
            "first_weights": ["2/3", "0"],
            "second_gain": ["4/19", "12/19"],
            "second_prior_residual": "4/3",
            "second_weights": ["18/19", "16/19"],
            "second_inverse_covariance": [["20/19", "-16/19"],
                                          ["-16/19", "28/19"]],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
