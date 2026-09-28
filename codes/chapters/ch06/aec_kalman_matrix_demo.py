"""Run a two-tap full-covariance Kalman AEC hand example.

This is real, time-domain, and uses known Q/R.  It illustrates why discarding
cross-tap covariance changes the next gain; it is not FDKF/PBFDKF.

Run ``python -m codes.examples.aec_kalman_matrix_demo`` from the repo root.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec_kalman_matrix import KalmanAECState


def _plain(step: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in step.items()
    }


def run_demo() -> dict:
    """Return two observations and an explicit diagonal-approximation contrast."""
    state = KalmanAECState(
        2,
        transition=1.0,
        process_covariance=np.zeros((2, 2)),
        observation_variance=1.0,
        initial_covariance=np.diag([1.0, 4.0]),
        initial_weights=np.zeros(2),
        initial_history=[1.0],
    )
    first = state.step(1.0, 3.0)   # taps [1, 1]
    second = state.step(-1.0, 0.0)  # taps [-1, 1]
    diagonal_second_s = float(
        second["taps"] @ (np.diag(np.diag(second["prior_covariance"])) @ second["taps"])
        + state.observation_variance
    )
    return {
        "scope": "real two-tap full-covariance known-Q/R Kalman; not FDKF/PBFDKF or measured double talk",
        "model": "h[n]=a*h[n-1]+q[n], d[n]=taps[n]@h[n]+v[n]",
        "initial": {
            "weights": [0.0, 0.0],
            "covariance": [[1.0, 0.0], [0.0, 4.0]],
            "history_newest_first": [1.0],
            "a": 1.0,
            "Q": [[0.0, 0.0], [0.0, 0.0]],
            "R": 1.0,
        },
        "first": _plain(first),
        "second": _plain(second),
        "second_innovation_variance_if_cross_covariance_discarded": diagonal_second_s,
        "hand_values": {
            "first_S": "6",
            "first_K": ["1/6", "2/3"],
            "first_weights": ["1/2", "2"],
            "first_covariance": [["5/6", "-2/3"], ["-2/3", "4/3"]],
            "second_full_S": "9/2",
            "second_diagonal_S": "19/6",
            "second_full_K": ["-1/3", "4/9"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
