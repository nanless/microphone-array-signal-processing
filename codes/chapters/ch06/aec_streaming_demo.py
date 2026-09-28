"""Chapter 6 block-boundary example with a known two-tap echo path.

Run ``python -m codes.examples.aec_streaming_demo`` from the repository root.
The numbers are a mathematical fixture, not a recording or device benchmark.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec import NLMSState, nlms


def run_demo() -> dict:
    """Show why carrying coefficients without reference history is insufficient."""

    reference = np.array([1., 2., 3., 4.])
    path = np.array([.5, .25])
    # Independent hand convolution: [.5, 1.25, 2, 2.75].
    microphone = np.array([.5, 1.25, 2., 2.75])
    split = 2

    state = NLMSState(2, step_size=0., initial_weights=path)
    first_residual, first_echo = state.process(reference[:split], microphone[:split])
    carried_history = state.history.tolist()
    second_residual, second_echo = state.process(reference[split:], microphone[split:])

    # This intentionally restarts the reference history for the second block.
    restarted_residual = np.r_[
        nlms(reference[:split], microphone[:split], 2,
             step_size=0., initial_weights=path)[0],
        nlms(reference[split:], microphone[split:], 2,
             step_size=0., initial_weights=path)[0],
    ]
    return {
        "reference": reference.tolist(),
        "microphone": microphone.tolist(),
        "true_path": path.tolist(),
        "split_after_samples": split,
        "history_at_split_oldest_first": carried_history,
        "streaming_echo_hat": np.r_[first_echo, second_echo].tolist(),
        "streaming_residual": np.r_[first_residual, second_residual].tolist(),
        "restarted_residual": restarted_residual.tolist(),
        "model": "dimensionless known two-tap linear FIR, no near-end speech or noise",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
