"""Deterministic angle-tracking counterexample for Chapter 9.

This is a sequence of synthetic angle detections, not audio or device data.
The minimum-cost one-to-one assignment is a deliberately small two-track
teaching rule, not JPDA, MHT, or a complete track-lifecycle implementation.
Truth labels are used only by the oracle diagnostic and evaluation, never by
the blind assignment.
"""

from __future__ import annotations

from itertools import combinations, permutations
import json

import numpy as np

from codes.array_tutorial.tracking import ConstantVelocityKalman, wrap_angle


TRACKS = ("A", "B")
TRUTH_DEG = (
    {"A": 20.0, "B": 60.0},
    {"A": 30.0, "B": 50.0},
    {"A": 41.0, "B": 39.0},
    {"A": 50.0, "B": 30.0},
    {"A": 60.0, "B": 20.0},
)
DETECTIONS = (
    (),
    (("A", 30.0), ("B", 50.0)),
    (("B", 39.0), ("A", 41.0)),  # Deliberate arrival order at the tie.
    (("B", 30.0),),              # A is silent, not absent from the room.
    (("A", 60.0), ("B", 20.0)),
)
PROCESS_NOISE = np.diag([0.1, 0.01])
MEASUREMENT_VARIANCE_DEG2 = 1.0
GATE_DEG = 15.0


def _minimum_cost_assignment(
    predicted_deg: tuple[float, float], measured_deg: tuple[float, ...]
) -> dict[int, int]:
    """Match as many gated detections as possible, then minimize squared error.

    At exactly equal costs, preserve track and detection order. This explicit
    tie rule exposes why angle alone cannot identify two crossing sources.
    """
    if len(measured_deg) > 2:
        raise ValueError("this teaching example supports at most two detections")
    if not measured_deg:
        return {}
    for count in range(len(measured_deg), 0, -1):
        candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
        for detection_indices in combinations(range(len(measured_deg)), count):
            for track_indices in permutations(range(2), count):
                pairs = tuple(zip(track_indices, detection_indices))
                squared = []
                for track_index, detection_index in pairs:
                    residual = float(wrap_angle(
                        measured_deg[detection_index] - predicted_deg[track_index]
                    ))
                    if abs(residual) > GATE_DEG:
                        break
                    squared.append(residual**2)
                else:
                    candidates.append((sum(squared), pairs))
        if candidates:
            return dict(min(candidates)[1])
    return {}


def _new_filters() -> dict[str, ConstantVelocityKalman]:
    return {
        "A": ConstantVelocityKalman([20.0, 10.0], np.eye(2), PROCESS_NOISE),
        "B": ConstantVelocityKalman([60.0, -10.0], np.eye(2), PROCESS_NOISE),
    }


def _run_tracker(*, oracle: bool) -> list[dict]:
    filters = _new_filters()
    rows = []
    for frame in range(1, len(TRUTH_DEG)):
        for name in TRACKS:
            filters[name].predict(1.0)
        observations = DETECTIONS[frame]
        if oracle:
            assigned = {name: (name, angle) for name, angle in observations}
        else:
            indices = _minimum_cost_assignment(
                tuple(float(filters[name].state[0]) for name in TRACKS),
                tuple(angle for _, angle in observations),
            )
            assigned = {TRACKS[track_index]: observations[detection_index]
                        for track_index, detection_index in indices.items()}
        for name, (_, angle) in assigned.items():
            filters[name].update(angle, MEASUREMENT_VARIANCE_DEG2)
        rows.append({
            "frame": frame,
            "measurement_time_s": float(frame),
            "observations": [{"truth_label": label, "angle_deg": angle}
                             for label, angle in observations],
            "assigned_detection_label": {name: assigned[name][0] if name in assigned else None
                                         for name in TRACKS},
            "state_angle_deg": {name: float(filters[name].state[0]) for name in TRACKS},
            "state_velocity_deg_s": {name: float(filters[name].state[1]) for name in TRACKS},
            "angle_variance_deg2": {name: float(filters[name].covariance[0, 0]) for name in TRACKS},
            "absolute_angle_error_deg": {
                name: abs(float(wrap_angle(filters[name].state[0] - TRUTH_DEG[frame][name])))
                for name in TRACKS
            },
            "wrong_detection_labels": sum(name != label for name, (label, _) in assigned.items()),
            "missing_tracks": [name for name in TRACKS if name not in assigned],
        })
    return rows


def _limited_step(current_deg: float, desired_deg: float, max_speed_deg_s: float, dt_s: float) -> float:
    max_step = max_speed_deg_s * dt_s
    residual = float(wrap_angle(desired_deg - current_deg))
    return float(wrap_angle(current_deg + np.clip(residual, -max_step, max_step)))


def _dropout_and_control() -> list[dict]:
    """Two missing-observation frames with a slower beam steering actuator."""
    tracker = ConstantVelocityKalman([20.0, 10.0], np.eye(2), PROCESS_NOISE)
    steer_angle = 20.0
    rows = []
    for frame in (1, 2):
        tracker.predict(1.0)
        steer_angle = _limited_step(steer_angle, float(tracker.state[0]), 5.0, 1.0)
        rows.append({
            "frame": frame,
            "observation": None,
            "predicted_angle_deg": float(tracker.state[0]),
            "angle_variance_deg2": float(tracker.covariance[0, 0]),
            "steer_angle_deg": steer_angle,
            "control_lag_deg": abs(float(wrap_angle(tracker.state[0] - steer_angle))),
        })
    return rows


def run_experiment() -> dict:
    return {
        "model": {
            "input": "deterministic synthetic angle detections, not waveforms or recordings",
            "frame_interval_s": 1.0,
            "initial_state_A": [20.0, 10.0],
            "initial_state_B": [60.0, -10.0],
            "initial_covariance": [[1.0, 0.0], [0.0, 1.0]],
            "process_noise_per_prediction": PROCESS_NOISE.tolist(),
            "measurement_variance_deg2": MEASUREMENT_VARIANCE_DEG2,
            "association_gate_deg": GATE_DEG,
            "control_max_speed_deg_s": 5.0,
            "randomness": "none",
            "aggregation": "one fixed sequence; no uncertainty interval",
        },
        "truth_deg": list(TRUTH_DEG),
        "minimum_cost_assignment": _run_tracker(oracle=False),
        "truth_association_diagnostic": _run_tracker(oracle=True),
        "dropout_control": _dropout_and_control(),
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
