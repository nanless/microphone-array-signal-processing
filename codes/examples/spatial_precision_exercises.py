"""E02-07, E04-10 and E05-06: deterministic algebra, not audio benchmarks.

Run ``.venv/bin/python codes/examples/spatial_precision_exercises.py``.
No files are written. TDOA units are microseconds; covariance uses their square.
"""

from __future__ import annotations

import json

import numpy as np


def forgetting_time() -> dict:
    """Relate the decay per actual update to elapsed physical time."""
    alpha = 0.98
    target_seconds = 0.4
    rows = []
    for hop_seconds in (0.008, 0.016):
        rows.append({
            "update_interval_seconds": hop_seconds,
            "fixed_alpha": alpha,
            "time_constant_seconds": float(-hop_seconds / np.log(alpha)),
            "alpha_for_0_4_seconds": float(np.exp(-hop_seconds / target_seconds)),
            "old_contribution_after_0_4_seconds": float(
                np.exp(-hop_seconds / target_seconds) ** (target_seconds / hop_seconds)
            ),
        })
    return {"cases": rows, "weight_concentration_limit": (1 + alpha) / (1 - alpha)}


def tdoa_consistency() -> dict:
    """Project one equal-variance cycle; propagate shared-reference TOA errors."""
    observed = np.array([30.0, 20.0, -40.0])
    residual = float(observed.sum())
    corrected = observed - residual / 3
    # Rows correspond to t2-t1 and t3-t1. This is an assumed TOA error model,
    # not a claim that separately estimated GCC errors have this covariance.
    difference = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    covariance = difference @ (100 * np.eye(3)) @ difference.T
    grid = np.arange(0.0, 4.1, 2.0)
    truths = np.array([2.0, 2.5, 3.0])
    grid_error = np.min(np.abs(truths[:, None] - grid), axis=1)
    return {
        "observed_cycle_us": observed.tolist(),
        "closure_residual_us": residual,
        "equal_weight_projection_us": corrected.tolist(),
        "projected_residual_us": float(corrected.sum()),
        "shared_reference_covariance_us_squared": covariance.tolist(),
        "shared_reference_correlation": float(covariance[0, 1] / covariance[0, 0]),
        "grid_step_degrees": 2.0,
        "truth_degrees": truths.tolist(),
        "ideal_nearest_grid_error_degrees": grid_error.tolist(),
    }


def mismatch_bound() -> dict:
    """A complex norm-ball bound is sharp, but need not describe physical DOA error."""
    epsilon = 0.02
    nominal = np.ones(2)
    rows = []
    for name, weights in (("equal_average", np.array([0.5, 0.5])),
                          ("differential_example", np.array([2.0, -1.0]))):
        norm = float(np.linalg.norm(weights))
        error = -epsilon * weights / norm
        actual_response = float(weights @ (nominal + error))
        rows.append({
            "name": name,
            "weights": weights.tolist(),
            "nominal_response": float(weights @ nominal),
            "weight_norm": norm,
            "white_noise_gain_linear": 1 / norm**2,
            "white_noise_gain_db": float(-20 * np.log10(norm)),
            "response_error_bound": epsilon * norm,
            "attaining_error_vector": error.tolist(),
            "attaining_response": actual_response,
        })
    return {"uncertainty_radius": epsilon, "cases": rows}


def run_exercises() -> dict:
    results = {
        "E02-07": forgetting_time(),
        "E04-10": tdoa_consistency(),
        "E05-06": mismatch_bound(),
    }
    for result in results.values():
        result["metadata"] = {"kind": "deterministic mathematical example",
                              "numpy_version": np.__version__, "randomness": "none"}
    return results


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
