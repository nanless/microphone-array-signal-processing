"""Check online WPE chunk continuity and future-frame influence.

The input is fixed-seed complex STFT data, not recorded speech. Outer chunks
only group calls to ``step_frame``; they are not ``OnlineWPE.step_block`` input
windows. The optional upstream check requires the pinned nara-wpe 0.0.11.

Run from the repository root:
    .venv/bin/python -m codes.examples.wpe_temporal_contract
"""

from __future__ import annotations

import importlib.metadata
import json

import numpy as np

from codes.examples.compare_online_wpe_reference import (
    NARA_WPE_VERSION,
    NumpyOnlineWPE011,
    _load_locked_upstream,
)


SEED = 20260924
FRAME_COUNT = 48
PERTURBED_FRAME = 30
PERTURBATION = 5.0 + 2.0j
CHUNK_PARTITIONS = ((48,), (1,) * 48, (7, 13, 5, 23), (30, 18))


def fixed_frames() -> tuple[np.ndarray, np.ndarray]:
    """Return two arrays differing only at one declared future frame."""

    generator = np.random.default_rng(SEED)
    shape = (FRAME_COUNT, 2, 1)  # frame, frequency, channel
    original = generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    changed = original.copy()
    changed[PERTURBED_FRAME] += PERTURBATION
    return original, changed


def process_chunks(factory, frames: np.ndarray, partition: tuple[int, ...]) -> np.ndarray:
    """Retain one online object while grouping frame calls in outer chunks."""

    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("frames must have shape (positive time, frequency, channel)")
    if not partition or any(type(size) is not int or size <= 0 for size in partition):
        raise ValueError("partition must contain positive integer chunk lengths")
    if sum(partition) != frames.shape[0]:
        raise ValueError("partition does not cover the frame sequence exactly")
    state = factory()
    output = []
    offset = 0
    for size in partition:
        output.extend(state.step_frame(frame) for frame in frames[offset:offset + size])
        offset += size
    return np.stack(output)


def _numpy_factory():
    return NumpyOnlineWPE011(
        taps=2, delay=2, alpha=0.95, frequency_bins=2, channels=1
    )


def temporal_summary(factory) -> dict:
    """Measure prefix invariance, future effect, and four chunk partitions."""

    original, changed = fixed_frames()
    original_output = process_chunks(factory, original, CHUNK_PARTITIONS[0])
    changed_output = process_chunks(factory, changed, CHUNK_PARTITIONS[0])
    chunk_differences = []
    for partition in CHUNK_PARTITIONS[1:]:
        chunked = process_chunks(factory, original, partition)
        chunk_differences.append({
            "partition": list(partition),
            "bitwise_equal_to_whole": bool(np.array_equal(chunked, original_output)),
            "max_abs_difference": float(np.max(np.abs(chunked - original_output))),
        })
    difference = np.abs(changed_output - original_output)
    return {
        "seed": SEED,
        "input_shape_frame_frequency_channel": list(original.shape),
        "taps": 2,
        "delay_argument": 2,
        "alpha": 0.95,
        "perturbed_zero_based_frame": PERTURBED_FRAME,
        "complex_perturbation": [PERTURBATION.real, PERTURBATION.imag],
        "changed_input_samples": int(np.count_nonzero(changed != original)),
        "earlier_online_output_max_abs_difference": float(
            np.max(difference[:PERTURBED_FRAME])
        ),
        "at_perturbation_online_output_max_abs_difference": float(
            np.max(difference[PERTURBED_FRAME])
        ),
        "later_online_output_max_abs_difference": float(
            np.max(difference[PERTURBED_FRAME + 1:])
        ),
        "outer_chunk_checks": chunk_differences,
    }


def report() -> dict:
    """Run the adapted baseline and, when installed, the locked upstream."""

    result = {
        "scope": "Fixed complex-STFT interface experiment; no room, speech, audio quality, or real-time claim.",
        "outer_chunk_meaning": "Each chunk only groups step_frame calls; one state object spans all chunks.",
        "adapted_numpy_reference": temporal_summary(_numpy_factory),
    }
    try:
        OnlineWPE, _, _, source_sha256 = _load_locked_upstream()
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        result["locked_upstream"] = {"status": "not installed", "reason": str(error)}
        return result

    def upstream_factory():
        return OnlineWPE(
            taps=2, delay=2, alpha=0.95, frequency_bins=2, channel=1
        )

    from nara_wpe.wpe import wpe_v6

    original, changed = fixed_frames()
    original_offline = wpe_v6(
        original.transpose(1, 2, 0), taps=2, delay=2, iterations=1,
        statistics_mode="full",
    )
    changed_offline = wpe_v6(
        changed.transpose(1, 2, 0), taps=2, delay=2, iterations=1,
        statistics_mode="full",
    )
    result["locked_upstream"] = {
        "status": "checked",
        "version": NARA_WPE_VERSION,
        "source_module_sha256": source_sha256,
        "online": temporal_summary(upstream_factory),
        "offline_control": {
            "function": "wpe_v6, one full-record iteration, statistics_mode=full",
            "earlier_output_max_abs_difference": float(np.max(np.abs(
                original_offline[..., :PERTURBED_FRAME]
                - changed_offline[..., :PERTURBED_FRAME]
            ))),
            "meaning": "A future input frame can change an earlier output when a full-record filter is estimated.",
        },
    }
    return result


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2, allow_nan=False))
