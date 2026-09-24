"""Frame-wise NCC activity gate for a controlled, aligned one-reference AEC.

The four labels are silence, far-end only, near-end only, and double talk.
The detector sees only playback and microphone signals, never oracle labels.
This is a teaching baseline; multipath and path changes can imitate double talk.
"""

from __future__ import annotations

import numpy as np


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
    Thresholds are amplitude in input units and dimensionless NCC respectively.
    """

    x = np.asarray(reference, dtype=float)
    d = np.asarray(microphone, dtype=float)
    if x.ndim != 1 or d.shape != x.shape or x.size == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(d)):
        raise ValueError("reference and microphone must be equal-length finite 1-D arrays")
    if isinstance(frame_size, bool) or not isinstance(frame_size, int) or frame_size <= 1 or x.size % frame_size:
        raise ValueError("frame_size must be >1 and divide the signal length")
    if not np.isfinite(activity_rms) or activity_rms <= 0:
        raise ValueError("activity_rms must be finite and positive")
    if not np.isfinite(coherence_threshold) or not 0 < coherence_threshold < 1:
        raise ValueError("coherence_threshold must be strictly between 0 and 1")

    frames_x = x.reshape(-1, frame_size)
    frames_d = d.reshape(-1, frame_size)
    # Centering removes a DC offset; the input experiment has zero-mean frames.
    centered_x = frames_x - frames_x.mean(axis=1, keepdims=True)
    centered_d = frames_d - frames_d.mean(axis=1, keepdims=True)
    x_rms = np.sqrt(np.mean(centered_x ** 2, axis=1))
    d_rms = np.sqrt(np.mean(centered_d ** 2, axis=1))
    numerator = np.abs(np.sum(centered_x * centered_d, axis=1))
    denominator = np.sqrt(np.sum(centered_x ** 2, axis=1) * np.sum(centered_d ** 2, axis=1))
    coherence = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    far = x_rms >= activity_rms
    mic = d_rms >= activity_rms
    states = np.zeros(frames_x.shape[0], dtype=np.int8)
    states[far & mic] = 1
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
