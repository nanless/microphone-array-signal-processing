"""Offline PBFDAF on pinned real playback-loopback/microphone pairs.

The Microsoft AEC Challenge recordings are read from the ignored upstream
cache.  This script neither downloads nor writes audio.  There is no isolated
echo or near-end ground truth: every reported dB value is a total digital
input/output power change, not ERLE or near-end preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from codes.array_tutorial.aec_partitioned import PartitionedFDAFState
from codes.examples.aec_doubletalk_experiment import load_doubletalk
from codes.examples.aec_real_pair_experiment import (
    FRAME, PAIR_DIR, RATE, SOURCE_COMMIT, load_pinned_pair,
)


# Speex's pinned mdf.c rounds an API request of 4096 taps up to 26 complete
# 160-sample partitions.  Matching that *capacity* does not match the update
# rule, preprocessing, delay control, or output sampling point.
SPEEX_REQUESTED_TAPS = 4096
FILTER_LENGTH = 26 * FRAME
STEP_SIZE = 0.3
EPSILON = 0.05
SCORE_START = 3 * RATE
DOUBLE_TALK_SCORE_STOP = 8 * RATE


def power_change_db(microphone: np.ndarray, residual: np.ndarray) -> float:
    """Return 10 log10(sum(mic**2) / sum(residual**2)) on equal samples."""

    before = np.asarray(microphone, dtype=np.float64)
    after = np.asarray(residual, dtype=np.float64)
    if before.ndim != 1 or before.shape != after.shape or before.size == 0:
        raise ValueError("power comparison needs nonempty, equal-length mono arrays")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("power comparison needs finite values")
    with np.errstate(over="raise", invalid="raise"):
        before_energy = float(np.sum(before * before))
        after_energy = float(np.sum(after * after))
    if before_energy <= 0 or after_energy <= 0:
        raise ValueError("finite nonzero input and output energies are required")
    return float(10 * np.log10(before_energy / after_energy))


def reference_conditions(reference: np.ndarray, *, rate: int) -> dict[str, np.ndarray]:
    """Make paired, zero, and deliberately one-second-late reference controls."""

    if reference.ndim != 1 or rate <= 0 or rate >= reference.size:
        raise ValueError("reference must be 1-D and longer than one second")
    late = np.zeros_like(reference)
    late[rate:] = reference[:-rate]
    return {
        "paired": reference,
        "zero_reference": np.zeros_like(reference),
        "late_reference_1s": late,
    }


def evaluate_arrays(
    reference_pcm: np.ndarray,
    microphone_pcm: np.ndarray,
    *,
    score_start: int,
    score_stop: int | None,
    rate: int = RATE,
    block_length: int = FRAME,
    filter_length: int = FILTER_LENGTH,
    step_size: float = STEP_SIZE,
    epsilon: float = EPSILON,
) -> dict:
    """Evaluate each condition from a fresh PBFDAF state, without audio writes.

    Explicit settings permit tiny independent test fixtures.  ``run`` below
    always uses the pinned production settings; no score-based tuning occurs.
    """

    x_pcm = np.asarray(reference_pcm)
    d_pcm = np.asarray(microphone_pcm)
    if (x_pcm.dtype != np.int16 or d_pcm.dtype != np.int16
            or x_pcm.ndim != 1 or d_pcm.ndim != 1):
        raise ValueError("inputs must be mono PCM16 arrays")
    if (not isinstance(rate, int) or not isinstance(block_length, int)
            or rate <= 0 or block_length <= 0 or rate % block_length):
        raise ValueError("positive integer rate must be divisible by block length")
    usable = min(x_pcm.size, d_pcm.size) // block_length * block_length
    if score_stop is None:
        score_stop = usable
    if not (0 <= score_start < score_stop <= usable):
        raise ValueError("score interval must fit the common complete blocks")
    x = x_pcm[:usable].astype(np.float64) / 32768.0
    d = d_pcm[:usable].astype(np.float64) / 32768.0
    controls = reference_conditions(x, rate=rate)
    score = slice(score_start, score_stop)
    results = {}
    for name, control in controls.items():
        state = PartitionedFDAFState(
            filter_length, block_length, step_size=step_size, epsilon=epsilon,
            constrain_gradient=True,
        )
        residual, _ = state.process(control, d)
        windows = []
        for start in range(score_start, score_stop, rate):
            stop = min(start + rate, score_stop)
            windows.append({
                "sample_interval_half_open": [start, stop],
                "samples": stop - start,
                "input_output_total_power_change_db": power_change_db(
                    d[start:stop], residual[start:stop]),
            })
        output_score = residual[score]
        results[name] = {
            "input_output_total_power_change_db": power_change_db(d[score], output_score),
            "score_output_rms_digital": float(np.sqrt(np.mean(output_score ** 2))),
            "whole_output_peak_abs_digital": float(np.max(np.abs(residual))),
            "whole_output_samples_over_pcm16_range": int(np.count_nonzero(
                (residual > 32767 / 32768) | (residual < -1))),
            "whole_output_float64_le_sha256": hashlib.sha256(
                np.asarray(residual, dtype="<f8").tobytes()).hexdigest(),
            "score_windows": windows,
        }
    return {
        "pcm_conversion": "signed PCM16 / 32768 -> float64; no resampling, gain, clipping, or time shift",
        "frame_samples": block_length,
        "sample_rate_hz": rate,
        "common_complete_block_samples": usable,
        "discarded_tail_samples": {
            "lpb": int(x_pcm.size - usable), "mic": int(d_pcm.size - usable),
        },
        "convergence_interval_half_open": [0, score_start],
        "score_interval_half_open": [score_start, score_stop],
        "score_input_rms_digital": float(np.sqrt(np.mean(d[score] ** 2))),
        "controls": results,
        "paired_output_relative_to_zero_reference_db": float(
            results["paired"]["input_output_total_power_change_db"]
            - results["zero_reference"]["input_output_total_power_change_db"]
        ),
        "metric_warning": (
            "All dB values compare total microphone and float64 residual power on identical "
            "samples. They are not clean-component ERLE, near-end preservation, speech quality, "
            "or a ranking of SpeexDSP/WebRTC versus this teaching PBFDAF. Consecutive score "
            "windows are not independent trials. No clipping was applied to float64 output."
        ),
    }


def run(pair: str = "both", directory: Path = PAIR_DIR) -> dict:
    """Read only the pinned official pair(s), then evaluate fixed settings."""

    if pair not in ("farend_singletalk", "doubletalk", "both"):
        raise ValueError("pair must be farend_singletalk, doubletalk, or both")
    requested = ("farend_singletalk", "doubletalk") if pair == "both" else (pair,)
    results = {}
    for kind in requested:
        loader = load_pinned_pair if kind == "farend_singletalk" else load_doubletalk
        reference, microphone, files = loader(directory)
        stop = None if kind == "farend_singletalk" else DOUBLE_TALK_SCORE_STOP
        data = evaluate_arrays(reference, microphone, score_start=SCORE_START,
                               score_stop=stop)
        data["files"] = files
        results[kind] = data
    return {
        "dataset": "Microsoft AEC Challenge real playback-loopback/microphone recordings",
        "source_commit": SOURCE_COMMIT,
        "license_note": (
            "Crowd-recording redistribution terms are not established; raw WAVs remain "
            "in the Git-ignored upstream cache. This script creates no audio files."
        ),
        "algorithm": "original teaching PartitionedFDAFState; instantaneous-bin normalization",
        "algorithm_source": "codes/array_tutorial/aec_partitioned.py",
        "parameters": {
            "block_samples": FRAME,
            "filter_samples": FILTER_LENGTH,
            "partitions": FILTER_LENGTH // FRAME,
            "fft_samples": 2 * FRAME,
            "step_size": STEP_SIZE,
            "epsilon": EPSILON,
            "gradient_projection_every_block": True,
            "double_talk_freeze": False,
            "residual_postfilter": False,
            "parameter_selection": (
                "step_size=0.3 and epsilon=0.05 reuse the existing synthetic teaching "
                "demo; neither value was selected for these real score windows"
            ),
            "speex_comparison_boundary": (
                "Pinned Speex mdf.c receives 4096 requested taps but rounds to "
                "ceil(4096/160)=26 partitions, or 4160 coefficient positions; "
                "matching partition capacity does not match algorithms."
            ),
        },
        "reference_conditions": {
            "paired": "original lpb at the same sample index as mic; no alignment shift",
            "zero_reference": "all-zero reference; fresh filter state",
            "late_reference_1s": "prepend exactly 16000 zeros, then prior lpb samples; fresh filter state",
        },
        "pairs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=("farend_singletalk", "doubletalk", "both"),
                        default="both")
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.pair, args.pair_dir), ensure_ascii=False, indent=2,
                     allow_nan=False))


if __name__ == "__main__":
    main()
