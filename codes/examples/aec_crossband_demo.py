"""Two-band full-crossband Haar AEC: hand step and held-out model check.

Run ``python -m codes.examples.aec_crossband_demo``. All inputs are synthetic,
dimensionless arrays. The scores do not represent a device or speech result.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec_subband import (
    HaarCrossbandNLMSState,
    HaarDiagonalSubbandNLMSState,
    fir_crossband_matrices,
)


def run_demo() -> dict:
    hand = HaarCrossbandNLMSState(2, step_size=1, epsilon=1)
    hand_x = np.array([1., 1., 0., 0.])
    hand_d = np.array([0., 1., 1., 0.])
    hand_error, hand_echo = hand.process(hand_x, hand_d)

    rng = np.random.default_rng(20260923)
    block_values = rng.normal(size=2000)
    x = np.repeat(block_values, 2)
    d = np.convolve(x, [0., 1.])[:x.size]
    train_samples = 3000
    frozen = np.ones((x.size - train_samples) // 2, dtype=bool)
    cross = HaarCrossbandNLMSState(2, step_size=.5, epsilon=.05)
    diagonal = HaarDiagonalSubbandNLMSState(2, step_size=.5, epsilon=.05)
    cross.process(x[:train_samples], d[:train_samples])
    diagonal.process(x[:train_samples], d[:train_samples])
    cross_error, _ = cross.process(x[train_samples:], d[train_samples:], freeze=frozen)
    diagonal_error, _ = diagonal.process(x[train_samples:], d[train_samples:], freeze=frozen)
    physical_power = float(np.mean(d[train_samples:] ** 2))

    return {
        "scope": "synthetic dimensionless one-sample delay; not measured device or NSAF",
        "hand_step": {
            "reference": hand_x.tolist(),
            "microphone": hand_d.tolist(),
            "block_taps": 2,
            "step_size": 1.,
            "epsilon": 1.,
            "prior_prediction": hand_echo.tolist(),
            "prior_residual": hand_error.tolist(),
            "weights_after_two_blocks": hand.weights.tolist(),
            "known_path_matrices": fir_crossband_matrices([0., 1.]).tolist(),
        },
        "held_out": {
            "seed": 20260923,
            "reference": "2000 independent N(0,1) block values, each repeated twice",
            "physical_path": [0., 1.],
            "train_blocks": train_samples // 2,
            "frozen_test_blocks": frozen.size,
            "step_size": .5,
            "epsilon": .05,
            "physical_echo_mean_square": physical_power,
            "crossband_residual_mean_square": float(np.mean(cross_error ** 2)),
            "diagonal_residual_mean_square": float(np.mean(diagonal_error ** 2)),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
