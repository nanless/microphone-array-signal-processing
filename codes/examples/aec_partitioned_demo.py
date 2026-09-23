"""Reproduce a hand update and a fixed-seed PBFDAF identifiability test.

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


def run_identifiability_demo() -> dict:
    """Compare white versus tonal training on an independently convolved FIR.

    Training and holdout references are separate. The same broadband holdout
    is used for both conditions, with all holdout weight updates frozen.
    This synthetic, noiseless test is not an acoustic or runtime benchmark.
    """

    seed = 2094
    train_samples = 4096
    holdout_samples = 1024
    block_length = 16
    path = np.array([0., 0., .7, -.2, .08, 0., 0., 0., .05])
    rng = np.random.default_rng(seed)
    white_training = rng.normal(size=train_samples)
    holdout = rng.normal(size=holdout_samples)
    tonal_training = np.sin(2 * np.pi * np.arange(train_samples) / 8)
    conditions = {}

    for name, training in (("white_training", white_training),
                           ("tone_training", tonal_training)):
        reference = np.r_[training, holdout]
        # The oracle is direct time-domain convolution, not the tested FDAF.
        microphone = np.convolve(reference, path)[:reference.size]
        state = PartitionedFDAFState(path.size, block_length,
                                     step_size=.3, epsilon=.05)
        train_residual, _ = state.process(reference[:train_samples],
                                          microphone[:train_samples])
        estimated_path = state.fir_weights
        holdout_residual, _ = state.process(reference[train_samples:],
                                            microphone[train_samples:],
                                            freeze=np.ones(holdout_samples // block_length,
                                                           dtype=bool))
        conditions[name] = {
            "training_tail_residual_mse": float(np.mean(train_residual[-1024:] ** 2)),
            "relative_path_error": float(np.linalg.norm(estimated_path - path)
                                         / np.linalg.norm(path)),
            "frozen_broadband_holdout_residual_mse": float(
                np.mean(holdout_residual ** 2)),
        }

    return {
        "scope": "dimensionless noiseless synthetic FIR; no real recording, ERLE, or speed claim",
        "seed": seed,
        "training_samples": train_samples,
        "holdout_samples": holdout_samples,
        "block_length": block_length,
        "filter_length": path.size,
        "step_size": .3,
        "epsilon": .05,
        "true_path": path.tolist(),
        "tone_period_samples": 8,
        "conditions": conditions,
    }


if __name__ == "__main__":
    print(json.dumps({"hand_update": run_demo(),
                      "identifiability": run_identifiability_demo()},
                     ensure_ascii=False, indent=2, allow_nan=False))
