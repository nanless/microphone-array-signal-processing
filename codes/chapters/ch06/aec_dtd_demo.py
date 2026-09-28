"""Controlled four-state NCC detector and its NLMS freeze consequence.

All waveforms and truth labels are generated here; no recorded speech is used.
Truth labels are withheld from the detector and used only for evaluation.
"""

from __future__ import annotations

import json
import numpy as np

from codes.array_tutorial.aec import nlms
from codes.array_tutorial.double_talk import LABELS, confusion_counts, ncc_activity_states


def run_experiment(seed: int = 20260924) -> dict:
    fs, frame_size, frames_per_phase = 16000, 160, 20
    frames = frames_per_phase * 5
    rng = np.random.default_rng(seed)
    far = rng.standard_normal(frames * frame_size)
    near = rng.standard_normal(frames * frame_size)
    truth = np.repeat([0, 1, 2, 3, 1], frames_per_phase).astype(np.int8)
    far_active = np.isin(truth, [1, 3]).repeat(frame_size)
    near_active = np.isin(truth, [2, 3]).repeat(frame_size)
    reference = 0.3 * far * far_active
    injected_near = 0.22 * near * near_active
    path = np.array([0.6, 0.25])
    echo = np.convolve(reference, path, mode="full")[:reference.size]
    microphone = echo + injected_near

    predicted, ncc = ncc_activity_states(
        reference, microphone, frame_size=frame_size,
        activity_rms=0.02, coherence_threshold=0.85,
    )
    counts = confusion_counts(truth, predicted)
    detector_freeze = np.repeat(predicted == 3, frame_size)
    oracle_freeze = np.repeat(truth == 3, frame_size)
    results = {}
    for name, freeze in (("none", np.zeros(reference.size, bool)),
                         ("detector", detector_freeze), ("oracle", oracle_freeze)):
        residual, echo_hat, final_weights = nlms(reference, microphone, 2, step_size=0.15,
                                                  epsilon=1e-4, freeze=freeze)
        # Only the synthesized echo component is scored; the near-end injection
        # is known and excluded algebraically, not estimated from residual.
        double = np.repeat(truth == 3, frame_size)
        results[name] = {
            "final_path_error_l2": float(np.linalg.norm(final_weights - path)),
            "double_echo_residual_rms": float(np.sqrt(np.mean((echo[double] - echo_hat[double]) ** 2))),
            "freeze_samples": int(np.count_nonzero(freeze)),
        }

    # A separate far-only multipath counterexample. NCC of a single aligned
    # reference cannot distinguish changed echo shape from near-end activity.
    stress_x = 0.3 * rng.standard_normal(20 * frame_size)
    changed_echo = np.convolve(stress_x, [0.1, 0.75], mode="full")[:stress_x.size]
    stress_state, stress_ncc = ncc_activity_states(
        stress_x, changed_echo, frame_size=frame_size,
        activity_rms=0.02, coherence_threshold=0.85,
    )
    return {
        "signal": "synthetic Gaussian far/near sources, exact two-tap echo, no room or device",
        "sample_rate_hz": fs,
        "frame_size": frame_size,
        "seed": seed,
        "path": path.tolist(),
        "thresholds": {"activity_rms": 0.02, "absolute_ncc": 0.85},
        "class_order": list(LABELS),
        "confusion_rows_true_columns_predicted": counts.tolist(),
        "ncc_mean_by_true_class": {LABELS[i]: float(np.mean(ncc[truth == i])) for i in range(4)},
        "nlms": results,
        "changed_path_far_only_false_double_frames": int(np.count_nonzero(stress_state == 3)),
        "changed_path_far_only_frames": int(stress_state.size),
        "changed_path_mean_ncc": float(np.mean(stress_ncc)),
        "limits": "single aligned reference, fixed sample thresholds, no independent calibration, no speech or device validation; oracle freeze is diagnostic only",
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))
