"""E02-08/E04-11/E05-07/E12-05: exact models and their implementation boundaries.

Run ``.venv/bin/python -m codes.examples.spatial_model_exercises``.
The examples write no files and use no audio, random snapshots or measured data.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.beamforming import mvdr_weights
from codes.array_tutorial.doa import music_spectrum


def _complex_record(array: np.ndarray) -> dict:
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def rfft_mean_square(signal: np.ndarray, nfft: int | None = None) -> float:
    """Mean square over the ORIGINAL real samples, with optional zero padding.

    No window or detrending is applied. NumPy's default unscaled forward DFT
    is used. The odd-length final bin is paired; the even Nyquist bin is not.
    """
    raw = np.asarray(signal)
    if np.iscomplexobj(raw):
        raise ValueError("signal must be real")
    x = np.asarray(raw, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be a non-empty finite vector")
    if nfft is None:
        nfft = x.size
    if (isinstance(nfft, (bool, np.bool_))
            or not isinstance(nfft, (int, np.integer)) or nfft < x.size):
        raise ValueError("nfft must be an integer no smaller than the input")
    spectrum = np.fft.rfft(x, n=nfft)
    multiplicity = np.full(spectrum.size, 2.0)
    multiplicity[0] = 1.0
    if nfft % 2 == 0:
        multiplicity[-1] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(multiplicity * np.abs(spectrum)**2) / (nfft * x.size))
    if not np.isfinite(result):
        raise ValueError("spectral energy exceeds floating-point range")
    return result


def parseval_power() -> dict:
    rows = []
    for name, x, nfft in (
        ("interior_bin", [1, 0, -1, 0], 4),
        ("dc", [1, 1, 1, 1], 4),
        ("nyquist", [1, -1, 1, -1], 4),
        ("zero_padded", [1, 0, -1, 0], 8),
        ("odd_length", [1, -1, 0], 3),
    ):
        x = np.asarray(x, dtype=float)
        rows.append({"case": name, "signal": x.tolist(), "nfft": nfft,
                     "rfft": _complex_record(np.fft.rfft(x, n=nfft)),
                     "time_mean_square": float(np.mean(x*x)),
                     "rfft_mean_square": rfft_mean_square(x, nfft)})
    return {"cases": rows, "normalization": "weighted |rfft|^2 / (nfft * original_length)",
            "window": "none", "detrending": "none"}


def colored_noise_music() -> dict:
    """Compare three pipelines for one target and EXACT known colored noise.

    Half-wavelength ULA; a_m(theta)=exp(j*pi*m*sin(theta)); K=1 target.
    The rank-one disturbance at 30 degrees belongs to the known noise model.
    All dictionary rows keep their natural scale; no row normalization.
    """
    grid = np.arange(-800, 801) / 10
    dictionary = np.exp(1j * np.pi * np.sin(np.deg2rad(grid[:, None])) * np.arange(3))
    target = np.ones(3, dtype=complex)
    disturbance = np.array([1, 1j, -1])
    noise = np.eye(3) + 9 * np.outer(disturbance, disturbance.conj())
    covariance = noise + np.outer(target, target.conj())
    values, vectors = np.linalg.eigh(noise)
    whitener = (vectors * (1 / np.sqrt(values))) @ vectors.conj().T
    whitened_covariance = whitener @ covariance @ whitener.conj().T
    # Rows store column steering vectors transposed, NOT conjugate-transposed.
    transformed_dictionary = dictionary @ whitener.T
    rows = []
    for name, matrix, steering in (
        ("ordinary_music", covariance, dictionary),
        ("covariance_only", whitened_covariance, dictionary),
        ("covariance_and_steering", whitened_covariance, transformed_dictionary),
    ):
        spectrum = music_spectrum(matrix, steering, source_count=1)
        index = int(np.argmax(spectrum))
        _, eigenvectors = np.linalg.eigh(matrix)
        noise_basis = eigenvectors[:, :2]
        projection = np.sum(np.abs(steering.conj() @ noise_basis)**2, axis=1)
        rows.append({"pipeline": name, "peak_degrees": float(grid[index]),
                     "minimum_projection_energy": float(projection[index])})
    return {"target_degrees": 0.0, "disturbance_degrees": 30.0,
            "grid_degrees": [-80.0, 80.0], "grid_step_degrees": 0.1,
            "noise_covariance": _complex_record(noise),
            "whitener": _complex_record(whitener),
            "noise_eigenvalues": values.tolist(),
            "whitened_total_eigenvalues": np.linalg.eigvalsh(whitened_covariance).tolist(),
            "whitening_identity_max_error": float(np.max(np.abs(
                whitener @ noise @ whitener.conj().T - np.eye(3)))),
            "cases": rows}


def loading_from_wng() -> dict:
    """Choose an absolute load for this two-channel covariance, not all arrays."""
    covariance = np.array([[11, -10j], [10j, 11]])
    target = np.ones(2)
    interference = np.array([1, 1j])
    desired_wng = 1.6
    load = max(0.0, np.sqrt(50 / (1 / desired_wng - 0.5)) - 11)
    rows = []
    for absolute_load in (0.0, 8.0, float(load), 20.0):
        relative_load = absolute_load / 11
        weights = mvdr_weights(covariance, target, relative_diagonal_loading=relative_load)
        norm_squared = float(np.vdot(weights, weights).real)
        response = np.vdot(weights, interference)
        rows.append({"absolute_load": absolute_load, "relative_load": relative_load,
                     "weights": _complex_record(weights),
                     "nominal_target_response": _complex_record(np.asarray(np.vdot(weights, target))),
                     "wng_linear": 1 / norm_squared,
                     "interference_power_gain": float(abs(response)**2),
                     "original_covariance_output_power": float(np.vdot(weights, covariance @ weights).real)})
    return {"required_wng_linear": desired_wng, "minimum_absolute_load": float(load),
            "maximum_wng_limit": 2.0, "cases": rows}


def singular_mvdr_counterexample() -> dict:
    """Replacing R^-1 by R^+ fails for THIS R and a: the nullspace is useful."""
    covariance = np.diag([0.0, 1.0])
    steering = np.ones(2)
    candidate = np.linalg.pinv(covariance) @ steering
    candidate /= np.vdot(steering, candidate)
    optimum = np.array([1.0, 0.0])
    rows = []
    for load in (1.0, 0.1, 0.01):
        weights = mvdr_weights(covariance + load * np.eye(2), steering)
        rows.append({"absolute_load": load, "weights": weights.real.tolist(),
                     "original_covariance_output_power": float(np.vdot(weights, covariance @ weights).real)})
    return {"covariance": covariance.tolist(), "steering": steering.tolist(),
            "pseudoinverse_candidate": candidate.tolist(),
            "candidate_constraint": float(np.vdot(candidate, steering)),
            "candidate_output_power": float(candidate @ covariance @ candidate),
            "optimal_weights": optimum.tolist(), "optimal_output_power": 0.0,
            "loaded_cases": rows}


def run_exercises() -> dict:
    results = {"E02-08": parseval_power(), "E04-11": colored_noise_music(),
               "E05-07": loading_from_wng(), "E12-05": singular_mvdr_counterexample()}
    for result in results.values():
        result["metadata"] = {"kind": "deterministic mathematical example",
                              "numpy_version": np.__version__, "randomness": "none"}
    return results


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
