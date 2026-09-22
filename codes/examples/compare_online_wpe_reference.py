"""Check the time indices and state lifetime of nara-wpe 0.0.11 online WPE.

Run from the repository root with:
    .venv/bin/python codes/examples/compare_online_wpe_reference.py

The NumPy reference and the hand-calculated index probe have no optional
dependencies.  When the locked nara-wpe 0.0.11 is installed, the script also
checks its unmodified implementation.  Nothing is downloaded or installed.

This is an interface and state experiment on synthetic complex STFT frames.
It is not a speech-quality, latency, or public-dataset evaluation.

NumpyOnlineWPE011 adapts the buffer ordering, update equations and positive
inverse safeguard of nara_wpe/wpe.py (OnlineWPE and _stable_positive_inverse),
version 0.0.11, revision a166779cca2088817e330481bd20af1a2c598555.
Copyright (c) 2018 Communications Engineering Group, Paderborn University.
The upstream MIT notice is retained in ../licenses/nara_wpe_MIT.txt.
Changes: a compact NumPy-only comparison class, explicit input-shape check,
fixed time-index probes, deterministic fixtures, and comparison reporting.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import numpy as np


NARA_WPE_VERSION = "0.0.11"
NARA_WPE_MODULE_SHA256 = "385a6f1c67071ba3e243c8a24fe4a041484c55280ad4ed0dbadefa4679a606d2"


def fixed_index_hand_calculation() -> dict:
    """Return a one-tap hand calculation for three 0.0.11 interfaces.

    At current frame t=3, the four scalar observations are 10, 20, 30, 40,
    ``delay=1``, and the fixed prediction tap is one.  The three interfaces
    select different regressors, so their residuals differ before any update.
    """

    frames = np.array([10.0, 20.0, 30.0, 40.0])
    current_index = 3
    delay = 1
    taps = 1
    current = float(frames[current_index])

    offline_index = current_index - delay
    offline_regressor = float(frames[offline_index])

    stateless_buffer_start = current_index - (taps + delay)
    stateless_buffer = frames[stateless_buffer_start:current_index + 1]
    stateless_regressor = float(stateless_buffer[:-delay - 1][::-1][0])
    stateless_index = stateless_buffer_start

    stateful_buffer_start = current_index - (taps + delay + 1)
    stateful_old_buffer = frames[stateful_buffer_start:current_index]
    stateful_regressor = float(stateful_old_buffer[:-delay - 1][0])
    stateful_index = stateful_buffer_start

    return {
        "frames": frames.tolist(),
        "current_zero_based_index": current_index,
        "delay_argument": delay,
        "taps": taps,
        "fixed_prediction_tap": 1.0,
        "offline_build_y_tilde": {
            "regressor_index": offline_index,
            "regressor": offline_regressor,
            "residual": current - offline_regressor,
        },
        "stateless_online_wpe_step": {
            "regressor_index": stateless_index,
            "regressor": stateless_regressor,
            "residual": current - stateless_regressor,
        },
        "stateful_OnlineWPE_step_frame": {
            "regressor_index": stateful_index,
            "regressor": stateful_regressor,
            "residual": current - stateful_regressor,
        },
    }


class NumpyOnlineWPE011:
    """Small adapted NumPy reference for the locked stateful 0.0.11 behavior.

    The class deliberately preserves the locked implementation's buffer and
    delay convention.  It is only a reference for this regression experiment,
    not a new public online-WPE API for the tutorial.
    """

    def __init__(
        self,
        taps: int,
        delay: int,
        alpha: float,
        frequency_bins: int,
        channels: int,
    ) -> None:
        self.taps = taps
        self.delay = delay
        self.alpha = alpha
        size = taps * channels
        self.inverse_covariance = np.broadcast_to(
            np.eye(size, dtype=np.complex128),
            (frequency_bins, size, size),
        ).copy()
        self.filter_taps = np.zeros(
            (frequency_bins, size, channels), dtype=np.complex128
        )
        self.buffer = np.zeros(
            (taps + delay + 1, frequency_bins, channels),
            dtype=np.complex128,
        )

    def step_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one ``(frequency, channel)`` frame and retain all state."""

        observation = np.asarray(frame, dtype=np.complex128)
        if observation.shape != self.buffer.shape[1:]:
            raise ValueError(
                f"frame shape {observation.shape} != {self.buffer.shape[1:]}"
            )

        frequency_bins, channels = observation.shape
        window = self.buffer[:-self.delay - 1]
        window = window.transpose(1, 2, 0).reshape(
            frequency_bins, self.taps * channels
        )
        prediction = observation - np.einsum(
            "fid,fi->fd", np.conjugate(self.filter_taps), window
        )

        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = observation
        power = np.mean(np.abs(self.buffer) ** 2, axis=(0, 2))

        projected = np.einsum("fij,fj->fi", self.inverse_covariance, window)
        denominator = self.alpha * power + np.einsum(
            "fi,fi->f", np.conjugate(window), projected
        ).real
        floor = 1e-10 * float(np.max(denominator))
        if floor == 0.0:
            inverse_denominator = np.ones_like(denominator)
        else:
            inverse_denominator = 1.0 / np.maximum(denominator, floor)
        kalman_gain = projected * inverse_denominator[:, None]

        for frequency in range(frequency_bins):
            correction = np.outer(
                kalman_gain[frequency], np.conjugate(projected[frequency])
            )
            self.inverse_covariance[frequency] = (
                self.inverse_covariance[frequency] - correction
            ) / self.alpha
            self.filter_taps[frequency] += np.outer(
                kalman_gain[frequency], np.conjugate(prediction[frequency])
            )
        return prediction


def deterministic_frames() -> np.ndarray:
    """Return the fixed complex frame sequence used by the state experiment."""

    rng = np.random.default_rng(20260922)
    shape = (48, 2, 1)  # (frame, frequency, channel)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _new_numpy_state() -> NumpyOnlineWPE011:
    return NumpyOnlineWPE011(
        taps=2, delay=2, alpha=0.95, frequency_bins=2, channels=1
    )


def _process(state, frames: np.ndarray) -> np.ndarray:
    return np.stack([state.step_frame(frame) for frame in frames], axis=0)


def state_lifetime_experiment(factory=_new_numpy_state) -> dict:
    """Compare continuous processing, outer chunking, and a state reset."""

    frames = deterministic_frames()
    split = 24

    continuous = _process(factory(), frames)

    chunked_state = factory()
    chunked = np.concatenate(
        [
            _process(chunked_state, frames[:split]),
            _process(chunked_state, frames[split:]),
        ],
        axis=0,
    )

    reset = np.concatenate(
        [_process(factory(), frames[:split]), _process(factory(), frames[split:])],
        axis=0,
    )
    reset_error_by_frame = np.max(np.abs(reset - continuous), axis=(1, 2))
    changed = np.flatnonzero(reset_error_by_frame > 1e-12)

    return {
        "seed": 20260922,
        "shape_TFD": list(frames.shape),
        "taps": 2,
        "delay_argument": 2,
        "alpha": 0.95,
        "split_zero_based_frame": split,
        "continuous_vs_same_state_chunks_max_abs": float(
            np.max(np.abs(chunked - continuous))
        ),
        "continuous_vs_reset_max_abs": float(np.max(reset_error_by_frame)),
        "first_changed_zero_based_frame_after_reset": (
            int(changed[0]) if changed.size else None
        ),
        "continuous_output": continuous,
    }


def _load_locked_upstream():
    version = importlib.metadata.version("nara-wpe")
    if version != NARA_WPE_VERSION:
        raise RuntimeError(
            f"Expected optional nara-wpe {NARA_WPE_VERSION}; found {version}."
        )
    from nara_wpe.wpe import OnlineWPE, build_y_tilde, online_wpe_step

    source_path = Path(inspect.getfile(build_y_tilde))
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != NARA_WPE_MODULE_SHA256:
        raise RuntimeError(
            "Installed wpe.py differs from the locked nara-wpe 0.0.11 source."
        )
    return OnlineWPE, build_y_tilde, online_wpe_step, source_sha256


def compare_locked_upstream() -> dict:
    """Compare hand and NumPy expectations with unmodified nara-wpe 0.0.11."""

    OnlineWPE, build_y_tilde, online_wpe_step, source_sha256 = _load_locked_upstream()
    hand = fixed_index_hand_calculation()
    frames = np.asarray(hand["frames"], dtype=np.complex128)

    offline_regressor = build_y_tilde(frames[None, :], taps=1, delay=1)[0, 3]
    stateless_prediction, _, _ = online_wpe_step(
        frames[1:, None, None],
        power_estimate=np.ones(1),
        inv_cov=np.ones((1, 1, 1)),
        filter_taps=np.ones((1, 1, 1)),
        alpha=0.95,
        taps=1,
        delay=1,
    )
    stateful_probe = OnlineWPE(
        taps=1, delay=1, alpha=0.95, frequency_bins=1, channel=1
    )
    stateful_probe.buffer[:, 0, 0] = frames[:3]
    stateful_probe.filter_taps[:] = 1.0
    stateful_prediction = stateful_probe.step_frame(frames[3:].reshape(1, 1))

    np.testing.assert_allclose(
        offline_regressor, hand["offline_build_y_tilde"]["regressor"]
    )
    np.testing.assert_allclose(
        stateless_prediction[0, 0], hand["stateless_online_wpe_step"]["residual"]
    )
    np.testing.assert_allclose(
        stateful_prediction[0, 0],
        hand["stateful_OnlineWPE_step_frame"]["residual"],
    )

    def upstream_factory():
        return OnlineWPE(
            taps=2, delay=2, alpha=0.95, frequency_bins=2, channel=1
        )

    numpy_result = state_lifetime_experiment()
    upstream_result = state_lifetime_experiment(upstream_factory)
    upstream_continuous = upstream_result.pop("continuous_output")
    numpy_continuous = numpy_result.pop("continuous_output")
    adapted_max_abs = float(
        np.max(np.abs(upstream_continuous - numpy_continuous))
    )
    np.testing.assert_allclose(upstream_continuous, numpy_continuous, atol=1e-12)
    if upstream_result["continuous_vs_same_state_chunks_max_abs"] != 0.0:
        raise AssertionError("Keeping the same object across chunks changed its output.")
    if upstream_result["first_changed_zero_based_frame_after_reset"] != 24:
        raise AssertionError("Reset experiment did not first diverge at the reset frame.")

    return {
        "status": "checked",
        "reference": f"nara-wpe {NARA_WPE_VERSION}",
        "reference_module_sha256": source_sha256,
        "fixed_index_observed": {
            "offline_regressor": float(np.real(offline_regressor)),
            "stateless_residual": float(np.real(stateless_prediction[0, 0])),
            "stateful_residual": float(np.real(stateful_prediction[0, 0])),
        },
        "adapted_numpy_vs_upstream_max_abs": adapted_max_abs,
        "state_lifetime": upstream_result,
    }


def report() -> dict:
    hand = fixed_index_hand_calculation()
    numpy_state = state_lifetime_experiment()
    numpy_state.pop("continuous_output")
    result = {
        "scope": (
            "Synthetic complex-STFT index and state checks; not speech quality, "
            "latency, or a public-dataset result."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "fixed_index_hand_calculation": hand,
        "numpy_state_lifetime": numpy_state,
    }
    try:
        result["optional_locked_upstream"] = compare_locked_upstream()
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        result["optional_locked_upstream"] = {
            "status": "not installed",
            "reason": str(error),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2))
