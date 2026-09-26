"""Deterministic E09-07/08, E10-15 and E11-08; no device measurements."""
from __future__ import annotations

from collections import deque
import json
import math

import numpy as np

from codes.array_tutorial.conventions import finite_real_array, finite_real_scalar
from codes.array_tutorial.engineering import HysteresisVAD
from codes.array_tutorial.tracking import CircularParticleFilter, ConstantVelocityKalman


def predict_timestamped_direction(state, covariance, measurement_time_s,
                                  consumption_time_s, valid_for_s=0.1,
                                  acceleration_density=2.0):
    """Forward-predict a state *already estimated at measurement time*.

    This does not fuse an out-of-sequence raw observation into a current filter.
    q is continuous white angular-acceleration density in deg^2/s^3.
    """
    measurement_time_s = finite_real_scalar(measurement_time_s, "measurement_time_s")
    consumption_time_s = finite_real_scalar(consumption_time_s, "consumption_time_s")
    valid_for_s = finite_real_scalar(valid_for_s, "valid_for_s")
    acceleration_density = finite_real_scalar(acceleration_density, "acceleration_density")
    if valid_for_s < 0 or acceleration_density < 0:
        raise ValueError("lifetime and density must be nonnegative")
    dt = float(consumption_time_s - measurement_time_s)
    if dt < 0:
        raise ValueError("state timestamp is in the future")
    # Absolute clock origins change subtraction roundoff. Allow only two ULPs
    # at the input timestamp scale, not a fixed application-level grace period.
    time_tolerance = 2 * max(math.ulp(measurement_time_s), math.ulp(consumption_time_s),
                             math.ulp(valid_for_s))
    if not math.isfinite(dt) or dt > valid_for_s + time_tolerance:
        raise ValueError("state has expired")
    q = acceleration_density * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    tracker = ConstantVelocityKalman(state, covariance, q)
    if dt:
        tracker.predict(dt)
    return tracker.state.copy(), tracker.covariance.copy()


def vad_preroll_segments(energies):
    """Fixed 16-kHz/160-sample teaching stream, two pre-roll/hangover frames.

    Constant-valued waveforms have the specified mean-square energies. They
    test event/buffer semantics, not speech recognition or realistic audio.
    Sample intervals are half open. An open final segment is explicitly marked.
    """
    energies = finite_real_array(energies, "energies")
    if energies.ndim != 1 or not np.all(np.isfinite(energies)) or np.any(energies < 0):
        raise ValueError("energies must be a finite nonnegative vector")
    vad = HysteresisVAD(0.08, 0.03, 2)
    history = deque(maxlen=2)
    segments, active_flags = [], []
    segment = None
    for index, energy in enumerate(energies):
        # update() consumes samples, not the scalar energy used to design them.
        active = vad.update(np.full(160, np.sqrt(energy)))
        active_flags.append(active)
        if active:
            if segment is None:
                segment = {"trigger_frame": index, "frame_indices": list(history),
                           "trigger_available_ms": (index + 1) * 10}
                history.clear()
            segment["frame_indices"].append(index)
        else:
            if segment is not None:
                segment["end_event_available_ms"] = (index + 1) * 10
                segments.append(segment)
                segment = None
            history.append(index)
    if segment is not None:
        segment["end_event_available_ms"] = None  # end-of-input is not VAD-off
        segments.append(segment)
    for item in segments:
        indices = item["frame_indices"]
        item["sample_interval"] = [indices[0] * 160, (indices[-1] + 1) * 160]
        item["sample_count"] = len(indices) * 160
    return {"active_flags": active_flags, "segments": segments}


def zero_failure_upper_bound(trials, confidence=0.95):
    """Exact one-sided binomial upper confidence bound after zero failures.

    Trials must be independent Bernoulli trials with a common failure rate.
    This is a frequentist confidence bound, not a posterior probability.
    """
    if isinstance(trials, (bool, np.bool_)) or not isinstance(trials, (int, np.integer)) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    confidence = finite_real_scalar(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    return -math.expm1(math.log1p(-confidence) / trials)


def run_exercises():
    particles = CircularParticleFilter(np.array([0.0, 10.0]))
    particles.update(0.0, 10.0, clutter_probability=0.0)
    first = particles.weights.copy()
    particles.update(0.0, 10.0, clutter_probability=0.0)
    state, covariance = predict_timestamped_direction(
        [30.0, 30.0], [[4.0, 1.0], [1.0, 9.0]], 1.0, 1.08)
    vad = vad_preroll_segments([0.0, 0.01, 0.04, 0.09, 0.04, 0, 0, 0, 0, 0])
    return {
        "E09-07": {"first_weights": first.tolist(), "second_weights": particles.weights.tolist(),
                   "incorrect_reset_weights": first.tolist()},
        "E09-08": {"measurement_time_s": 1.0, "publication_time_s": 1.05,
                   "consumption_time_s": 1.08, "elapsed_s": 0.08,
                   "state": state.tolist(), "covariance": covariance.tolist(),
                   "stale_angle_lag_deg": 2.4},
        "E10-15": vad,
        "E11-08": {"confidence": 0.95,
                   "upper_bounds": {str(n): zero_failure_upper_bound(n) for n in (20, 100, 1000)},
                   "target_failure_rate": 0.01,
                   "minimum_zero_failure_trials": math.ceil(math.log(0.05) / math.log1p(-0.01))},
    }


if __name__ == "__main__":
    print(json.dumps(run_exercises(), indent=2, ensure_ascii=False))
