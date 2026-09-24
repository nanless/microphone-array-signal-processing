"""Show exact E-step consequences of missing and false speaker activity labels.

The spatial likelihoods are fixed numbers. This script does not estimate cACG
shape matrices, run EM, form spatial covariances, beamform, or score speech.
"""

from __future__ import annotations

import json
import numpy as np

from codes.array_tutorial.separation import guided_activity_posterior


def run_experiment() -> dict:
    weights = np.array([0.4, 0.4, 0.2])  # speaker 1, speaker 2, background
    likelihoods = np.array([[8.0, 1.0, 1.0],
                            [1.0, 8.0, 1.0],
                            [1.0, 1.0, 1.0]])
    correct = np.array([[1, 0], [0, 1], [0, 0]])
    missing_first = correct.copy()
    missing_first[0, 0] = 0
    false_on_silence = correct.copy()
    false_on_silence[2, 0] = 1
    posteriors = {
        "correct": guided_activity_posterior(weights, likelihoods, correct),
        "missed_speaker_1_first_frame": guided_activity_posterior(weights, likelihoods, missing_first),
        "false_speaker_1_on_silent_frame": guided_activity_posterior(weights, likelihoods, false_on_silence),
    }
    return {
        "scope": "one frequency, fixed likelihoods and priors; E step only, no fitted cACGMM or audio separation",
        "class_order": ["speaker_1", "speaker_2", "always_active_background"],
        "frame_truth": ["speaker_1_only", "speaker_2_only", "all_speakers_silent"],
        "mixture_weights": weights.tolist(),
        "relative_spatial_likelihoods": likelihoods.tolist(),
        "speaker_activity": {"correct": correct.tolist(),
                             "missed_speaker_1_first_frame": missing_first.tolist(),
                             "false_speaker_1_on_silent_frame": false_on_silence.tolist()},
        "posteriors": {name: value.tolist() for name, value in posteriors.items()},
        "speaker_1_first_frame_posterior_correct": float(posteriors["correct"][0, 0]),
        "speaker_1_first_frame_posterior_when_missed": float(posteriors["missed_speaker_1_first_frame"][0, 0]),
        "speaker_1_silent_frame_posterior_correct": float(posteriors["correct"][2, 0]),
        "speaker_1_silent_frame_posterior_when_falsely_active": float(posteriors["false_speaker_1_on_silent_frame"][2, 0]),
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
