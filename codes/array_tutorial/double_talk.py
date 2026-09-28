"""Frame-wise NCC activity gate for a controlled, aligned one-reference AEC.

The four labels are silence, far-end only, near-end only, and double talk.
The detector sees only playback and microphone signals, never oracle labels.
This is a teaching baseline; multipath and path changes can imitate double talk.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


LABELS = ("silence", "far_only", "near_only", "double_talk")


def ncc_activity_states(
    reference: np.ndarray,
    microphone: np.ndarray,
    *,
    frame_size: int,
    activity_rms: float,
    coherence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer states and absolute NCC for non-overlapping frames.

    The reference and microphone must already be aligned. A quiet reference
    disables the double-talk decision: microphone activity then means near-only.
    A live reference with a quiet microphone is still far-end-only activity.
    A frame is decided only once all its samples have arrived; applying its
    decision to that same frame requires a full-frame buffer.
    Thresholds are amplitude in input units and dimensionless NCC respectively.
    """

    x = finite_real_array(reference, "reference")
    d = finite_real_array(microphone, "microphone")
    if x.ndim != 1 or d.shape != x.shape or x.size == 0:
        raise ValueError("reference and microphone must be equal-length nonempty 1-D arrays")
    if (isinstance(frame_size, (bool, np.bool_))
            or not isinstance(frame_size, (int, np.integer))
            or frame_size <= 1 or x.size % frame_size):
        raise ValueError("frame_size must be >1 and divide the signal length")
    activity_rms = finite_real_scalar(activity_rms, "activity_rms")
    coherence_threshold = finite_real_scalar(coherence_threshold, "coherence_threshold")
    if activity_rms <= 0:
        raise ValueError("activity_rms must be positive")
    if not 0 < coherence_threshold < 1:
        raise ValueError("coherence_threshold must be strictly between 0 and 1")

    frames_x = x.reshape(-1, frame_size)
    frames_d = d.reshape(-1, frame_size)

    def centered_unit_frames(frames):
        # Scale BEFORE centering: even a finite frame's raw sum can overflow.
        scale = np.max(np.abs(frames), axis=1, keepdims=True)
        unit = np.divide(frames, scale, out=np.zeros_like(frames), where=scale > 0)
        centered = unit - unit.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
        normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
        # Standard deviation <= peak original magnitude. Compare in input
        # units, retaining the absolute activity threshold (not scale-free).
        rms = scale[:, 0] * np.minimum(norm[:, 0] / np.sqrt(frame_size), 1.0)
        return normalized, rms >= activity_rms

    centered_x, far = centered_unit_frames(frames_x)
    centered_d, mic = centered_unit_frames(frames_d)
    coherence = np.clip(np.abs(np.sum(centered_x * centered_d, axis=1)), 0., 1.)
    states = np.zeros(frames_x.shape[0], dtype=np.int8)
    states[far] = 1
    states[~far & mic] = 2
    states[far & mic & (coherence < coherence_threshold)] = 3
    return states, coherence


def confusion_counts(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Rows are true classes, columns predicted classes in ``LABELS`` order."""

    actual = np.asarray(truth)
    found = np.asarray(predicted)
    if actual.shape != found.shape or actual.ndim != 1 or actual.dtype.kind not in "iu" or found.dtype.kind not in "iu":
        raise ValueError("truth and predicted must be equal-length integer vectors")
    if np.any((actual < 0) | (actual > 3) | (found < 0) | (found > 3)):
        raise ValueError("states must be integers in [0,3]")
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (actual, found), 1)
    return counts
