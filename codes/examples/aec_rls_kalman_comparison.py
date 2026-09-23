"""Controlled RLS/Kalman AEC comparison with known, synthetic components.

This is a two-tap arithmetic experiment, not a speech recording, FDKF, a
double-talk detector, or an industrial performance benchmark. All five runs
receive exactly the same aligned render and microphone samples. The two
oracle controls use the *known* near-end interval, unavailable to a real AEC.

Run from the repository root:
    .venv/bin/python -m codes.examples.aec_rls_kalman_comparison
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.aec_kalman_matrix import KalmanAECState
from codes.array_tutorial.aec_rls import RLSState


SEED = 20260923
SAMPLE_RATE = 8000  # A nominal time axis only; these are not audio assets.
LENGTH = 512
SEGMENTS = {
    "initial_far_end_only": (0, 128),
    "known_near_end_injection": (128, 256),
    "far_end_recovery": (256, 384),
    "changed_echo_path": (384, 512),
}
PATH_A = np.array([0.70, -0.25])
PATH_B = np.array([-0.20, 0.60])


def make_signals() -> dict[str, np.ndarray]:
    """Make one fixed-seed reference, piecewise FIR echo and injected near end.

    The microphone is the sum of the three returned components. ``path[n]``
    applies to both render taps at sample ``n``; no extra render delay exists.
    The changing path is an artificial discontinuity, not a measured RIR.
    """

    rng = np.random.default_rng(SEED)
    render = rng.normal(scale=0.5, size=LENGTH)
    path = np.tile(PATH_A, (LENGTH, 1))
    path[SEGMENTS["changed_echo_path"][0]:] = PATH_B
    previous = np.r_[0.0, render[:-1]]
    echo = path[:, 0] * render + path[:, 1] * previous
    near_end = np.zeros(LENGTH)
    begin, end = SEGMENTS["known_near_end_injection"]
    near_end[begin:end] = 0.8 * np.sin(2 * np.pi * np.arange(end - begin) / 17)
    noise = rng.normal(scale=0.01, size=LENGTH)
    microphone = echo + near_end + noise
    return {
        "render": render,
        "microphone": microphone,
        "echo": echo,
        "near_end": near_end,
        "noise": noise,
        "path": path,
    }


def _run_rls(signals: dict[str, np.ndarray], *, oracle_freeze: bool) -> dict[str, np.ndarray]:
    state = RLSState(2, forgetting_factor=0.98, initial_regularization=0.5)
    prior_echo = np.empty(LENGTH)
    prior_residual = np.empty(LENGTH)
    posterior_path = np.empty((LENGTH, 2))
    inverse_trace = np.empty(LENGTH)
    near_begin, near_end = SEGMENTS["known_near_end_injection"]
    for n, (x, d) in enumerate(zip(signals["render"], signals["microphone"])):
        frozen = oracle_freeze and near_begin <= n < near_end
        error, estimate = state.process([x], [d], freeze=[frozen])
        prior_echo[n] = estimate[0]
        prior_residual[n] = error[0]
        posterior_path[n] = state.weights
        inverse_trace[n] = np.trace(state.inverse_covariance)
    return {
        "prior_echo": prior_echo,
        "prior_residual": prior_residual,
        "posterior_path": posterior_path,
        "inverse_correlation_trace": inverse_trace,
    }


def _run_kalman(
    signals: dict[str, np.ndarray], *, near_end_control: str
) -> dict[str, np.ndarray]:
    state = KalmanAECState(
        2, transition=1.0, process_covariance=np.eye(2) * 1e-4,
        observation_variance=1e-4, initial_covariance=np.eye(2) * 2.0,
    )
    prior_echo = np.empty(LENGTH)
    prior_residual = np.empty(LENGTH)
    posterior_path = np.empty((LENGTH, 2))
    covariance_trace = np.empty(LENGTH)
    gain_norm = np.empty(LENGTH)
    near_begin, near_end = SEGMENTS["known_near_end_injection"]
    for n, (x, d) in enumerate(zip(signals["render"], signals["microphone"])):
        in_near_end = near_begin <= n < near_end
        # The high-variance and freeze cases use the experiment's known truth.
        state.observation_variance = (1.0 if near_end_control == "oracle_variance"
                                      and in_near_end else 1e-4)
        step = state.step(
            x, d, freeze=near_end_control == "oracle_freeze" and in_near_end
        )
        prior_echo[n] = step["prior_echo"]
        prior_residual[n] = step["prior_error"]
        posterior_path[n] = step["posterior_weights"]
        covariance_trace[n] = np.trace(step["posterior_covariance"])
        gain_norm[n] = np.linalg.norm(step["gain"])
    return {
        "prior_echo": prior_echo,
        "prior_residual": prior_residual,
        "posterior_path": posterior_path,
        "path_covariance_trace": covariance_trace,
        "gain_l2_norm": gain_norm,
    }


def run_experiment() -> dict:
    """Return raw traces and segment summaries; make no performance ranking.

    ``echo_error_mse`` compares the known synthetic echo with the *prior*
    linear echo estimate. ``posterior_path_rmse`` compares the posterior
    coefficients with the specified instantaneous path. Both exist only
    because the synthetic generator exposes ground truth. Neither is ERLE,
    perceptual quality, a near-end preservation score, or product latency.
    """

    signals = make_signals()
    runs = {
        "rls_continuous": _run_rls(signals, oracle_freeze=False),
        "rls_oracle_freeze": _run_rls(signals, oracle_freeze=True),
        "kalman_fixed_variance": _run_kalman(signals, near_end_control="fixed"),
        "kalman_oracle_variance": _run_kalman(signals, near_end_control="oracle_variance"),
        "kalman_oracle_freeze": _run_kalman(signals, near_end_control="oracle_freeze"),
    }
    summary = {}
    for name, trace in runs.items():
        summary[name] = {}
        for label, (begin, end) in SEGMENTS.items():
            echo_error = signals["echo"][begin:end] - trace["prior_echo"][begin:end]
            path_error = signals["path"][begin:end] - trace["posterior_path"][begin:end]
            row = {
                "echo_error_mse": float(np.mean(echo_error ** 2)),
                "posterior_path_rmse": float(np.sqrt(np.mean(path_error ** 2))),
            }
            if "gain_l2_norm" in trace:
                row["mean_gain_l2_norm"] = float(
                    np.mean(trace["gain_l2_norm"][begin:end])
                )
            summary[name][label] = row
    return {
        "conditions": {
            "seed": SEED,
            "nominal_sample_rate_hz": SAMPLE_RATE,
            "length_samples": LENGTH,
            "segments_half_open": SEGMENTS,
            "path_a": PATH_A.tolist(),
            "path_b": PATH_B.tolist(),
            "render_distribution": "independent Gaussian, standard deviation 0.5",
            "near_end": "known 0.8-amplitude, 17-sample-period sine in [128,256)",
            "background_noise": "independent Gaussian, standard deviation 0.01",
            "rls": {"forgetting_factor_per_sample": 0.98,
                    "initial_regularization": 0.5},
            "kalman": {"transition": 1.0, "process_covariance_diagonal": 1e-4,
                       "initial_covariance_diagonal": 2.0,
                       "far_end_observation_variance": 1e-4,
                       "oracle_near_end_observation_variance": 1.0},
            "metrics": {
                "echo_error_mse": "mean((known echo - prior linear echo estimate)^2)",
                "posterior_path_rmse": "sqrt(mean((known path - posterior taps)^2))",
                "mean_gain_l2_norm": "mean Euclidean norm of Kalman gain",
            },
            "scope": "dimensionless synthetic two-tap arithmetic; oracle controls "
                     "use ground truth, not a detector; no audio, ERLE, runtime "
                     "benchmark or industrial FDKF/PBFDKF claim",
        },
        "signals": signals,
        "runs": runs,
        "summary": summary,
    }


def run_demo() -> dict:
    """Return the JSON-serializable conditions and metric table only."""

    experiment = run_experiment()
    return {key: experiment[key] for key in ("conditions", "summary")}


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, allow_nan=False, indent=2))
