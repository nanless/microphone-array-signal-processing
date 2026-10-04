"""Two-stream overlap association only; no separation or speaker recognition.

Both arrays must describe the SAME sample times. Matching does not correct
latency, polarity, gain or waveform distortion. Ambiguity returns no mapping.
The separate scalar helper requires a previously matched overlap and does not
turn a correlation decision into a speaker identity or a delay estimate.
"""
from __future__ import annotations

import math
import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


def overlap_application_gain(previous, current, *, minimum_rms=1e-8):
    """Return the real LS scalar applied directly to each matched current row.

    Inputs are real (2,L), L>=2, at identical sample times, already associated.
    The objective uses original samples, without centering: ||p-g*c||^2.
    Current RMS <= minimum_rms is rejected; a nonzero but contaminated overlap
    is not detected by this limited helper. There is no gain limit, latency
    correction, cross-block policy, or reference-speaker information.
    """
    old = finite_real_array(previous, "previous")
    new = finite_real_array(current, "current")
    if old.ndim != 2 or old.shape[0] != 2 or old.shape[1] < 2 or new.shape != old.shape:
        raise ValueError("overlaps must have the same (2,L) shape with L >= 2")
    threshold = finite_real_scalar(minimum_rms, "minimum_rms")
    if threshold < 0:
        raise ValueError("minimum_rms must be nonnegative")
    gains = []
    for p, c in zip(old, new):
        cp = float(np.max(np.abs(c)))
        if cp == 0:
            raise ValueError("current overlap has zero energy")
        cn = c/cp
        rms = float(np.linalg.norm(cn)/np.sqrt(c.size))
        with np.errstate(over="ignore", under="ignore"):
            too_small = rms <= threshold/np.float64(cp)
        if too_small:
            raise ValueError("current overlap is below the RMS policy threshold")
        pp = float(np.max(np.abs(p)))
        if pp == 0:
            gains.append(0.)
            continue
        numerator = float(cn @ (p/pp))
        denominator = float(cn @ cn)
        # Keep scale exponents separate: neither pp/cp nor coefficient*pp
        # needs to be representable when their final product is finite.
        pm, pe = math.frexp(pp)
        cm, ce = math.frexp(cp)
        try:
            gain = math.ldexp((numerator/denominator)*(pm/cm), pe-ce)
        except OverflowError as exc:
            raise ValueError("application gain is not representable as float64") from exc
        if not np.isfinite(gain):
            raise ValueError("application gain is not representable as float64")
        gains.append(float(gain))
    return np.array(gains)


def match_two_source_overlap(previous, current, *, minimum_rms=1e-8,
                             minimum_correlation=0.5, minimum_margin=0.05):
    """Associate two current slots with two previous slots without references.

    Inputs are real (2, overlap_samples), at least two samples. Scores are
    absolute centered correlations. ``current_indices_for_previous`` directly
    indexes current rows into previous order. Thresholds are explicit policy,
    not probabilities of correct identity; RMS is centered, in input units.
    """
    old = finite_real_array(previous, "previous")
    new = finite_real_array(current, "current")
    if old.ndim != 2 or old.shape[0] != 2 or old.shape[1] < 2 or new.shape != old.shape:
        raise ValueError("overlaps must have the same (2,L) shape with L >= 2")
    minimum_rms = finite_real_scalar(minimum_rms, "minimum_rms")
    minimum_correlation = finite_real_scalar(minimum_correlation, "minimum_correlation")
    minimum_margin = finite_real_scalar(minimum_margin, "minimum_margin")
    if minimum_rms < 0 or not 0 <= minimum_correlation <= 1 or not 0 <= minimum_margin <= 1:
        raise ValueError("RMS threshold must be nonnegative and correlation/margin in [0,1]")
    unit = []
    low_energy = False
    for row in np.concatenate((old, new)):
        peak = float(np.max(np.abs(row)))
        if peak == 0:
            low_energy = True
            unit.append(np.zeros_like(row))
            continue
        centered = row / peak
        centered -= centered.mean()
        norm = float(np.linalg.norm(centered))
        rms_scaled = norm / np.sqrt(row.size)
        # Compare without squaring the signal or multiplying enormous scales.
        with np.errstate(over="ignore", under="ignore"):
            below = rms_scaled <= minimum_rms / np.float64(peak)
        if norm == 0 or below:
            low_energy = True
            unit.append(np.zeros_like(row))
        else:
            unit.append(centered / norm)
    unit = np.asarray(unit)
    correlation = np.clip(np.abs(unit[:2] @ unit[2:].T), 0, 1)
    scores = np.array([np.trace(correlation), correlation[0, 1] + correlation[1, 0]]) / 2
    best = int(np.argmax(scores))
    margin = float(scores[best] - scores[1 - best])
    mapping = [0, 1] if best == 0 else [1, 0]
    if low_energy:
        reason = "low_energy_overlap"
    elif margin <= minimum_margin:
        reason = "ambiguous_assignment"
    elif min(correlation[i, mapping[i]] for i in range(2)) < minimum_correlation:
        reason = "weak_correlation"
    else:
        reason = "matched"
    return {"status": "matched" if reason == "matched" else "ambiguous",
            "reason": reason, "current_indices_for_previous": mapping if reason == "matched" else None,
            "absolute_centered_correlation": correlation.tolist(),
            "candidate_mean_scores_identity_swap": scores.tolist(), "margin": margin,
            "thresholds": {"minimum_centered_rms": minimum_rms,
                           "minimum_correlation": minimum_correlation, "minimum_margin": minimum_margin}}
