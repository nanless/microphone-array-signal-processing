"""Reproduce a two-block constrained PBFDAF update with hand-checkable values.

Run ``python -m codes.examples.aec_partitioned_demo`` at the repository root.
This is a dimensionless algorithm fixture, not an acoustic recording or a
measured convergence/latency result.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec_partitioned import PartitionedFDAFState


def run_demo() -> dict:
    """Return the independent N=2, P=2 example used in the regression test."""

    state = PartitionedFDAFState(4, 2, step_size=.5, epsilon=1.,
                                 constrain_gradient=True)
    reference = np.array([1., 0., 0., 0.])
    microphone = np.array([1., 0., 2., 0.])
    residual_first, echo_first = state.process(reference[:2], microphone[:2])
    first_weights = state.partition_time_coefficients
    residual_second, echo_second = state.process(reference[2:], microphone[2:])
    candidate_second = state.last_candidate_partition_time_coefficients
    return {
        "model": "dimensionless two-block real FIR, zero prior reference and zero initial weights",
        "scope": "teaching Eq. (6-3) instantaneous-power PBFDAF; not Speex AUMDF, WebRTC AEC3, a double-talk detector or real-time audio",
        "block_length": state.block_length,
        "fft_length": state.fft_length,
        "partitions": state.partitions,
        "step_size": state.step_size,
        "epsilon": state.epsilon,
        "reference": reference.tolist(),
        "microphone": microphone.tolist(),
        "prior_echo_hat": np.r_[echo_first, echo_second].tolist(),
        "residual": np.r_[residual_first, residual_second].tolist(),
        "weights_after_first_block": np.round(first_weights, 12).tolist(),
        "candidate_second_block": np.round(candidate_second, 12).tolist(),
        "weights_after_second_block": np.round(state.partition_time_coefficients, 12).tolist(),
        "fir_weights_after_second_block": np.round(state.fir_weights, 12).tolist(),
        "candidate_note": "first partition's lag-2 candidate is removed by time-domain gradient projection",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
