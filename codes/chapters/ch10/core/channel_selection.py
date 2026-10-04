"""Known channel selection for E10-34; reuse the Chapter 5 MVDR solver.

Channel indices retain their supplied order and their original physical
identity. Steering is selected, never divided by a new reference entry.
This adapter does not detect faults, estimate a covariance, or reset a stream.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np

from codes.chapters.ch04.core.covariance import _validate_covariance
from codes.chapters.ch05.core.beamforming import mvdr_weights


def _indices(indices, channel_count: int) -> list[int]:
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError("indices must be an explicit ordered sequence")
    if isinstance(indices, np.ndarray) and indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    result = []
    for index in indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise ValueError("channel indices must be integers, not bool or float")
        index = int(index)
        if not 0 <= index < channel_count:
            raise ValueError("channel index outside the original channel axis")
        result.append(index)
    if not result or len(set(result)) != len(result):
        raise ValueError("selection must be nonempty without repeated channels")
    return result


def select_channel_observations(values, indices) -> np.ndarray:
    """Copy selected channels from a finite numeric channels-first array.

Accept, for example, time samples ``(M, N)`` or spectra ``(M, F, T)``.
No time sample, spectrum, gain, or channel reference is changed.
"""
    array = np.asarray(values)
    if (array.dtype.kind not in "iufc" or array.ndim < 1
            or min(array.shape) < 1 or not np.all(np.isfinite(array))):
        raise ValueError("observations must be finite numeric channels-first data")
    selected = _indices(indices, array.shape[0])
    return array[selected].copy()


def select_mvdr_channels(covariance, steering, indices, *,
                         relative_diagonal_loading: float = 0.0,
                         condition_limit: float = 1e12) -> dict:
    """Select data-model axes consistently, then solve the selected MVDR.

    Input shapes are ``R: (M,M), a: (M,)`` or ``R: (F,M,M), a: (F,M)``.
    Returns the ordered indices, real selection matrix ``(K,M)``, selected
    covariance/steering, and Chapter 5 weights. Covariances keep their input
    units. The complete input must be Hermitian PSD under Chapter 4's relative
    1e-10 checks, including omitted channels; this is a given valid model,
    not an estimator allowed to hide corrupt entries by dropping a channel.

    Rank-deficient inputs may have a usable selected principal submatrix.
    The matrix actually solved must satisfy Chapter 5's positive-definite
    and condition limits, or a caller-declared relative load must make it so.
    A positive scalar normalization of each selected covariance avoids
    scale-dependent overflow in the existing solver; the selected covariance
    returned to the caller remains in the original units. This normalization
    preserves MVDR and its relative load. No automatic loading, pseudoinverse,
    or beamforming fallback is supplied.
    """
    raw = np.asarray(covariance)
    vector = np.asarray(steering)
    single = raw.ndim == 2
    if (raw.dtype.kind not in "iufc" or vector.dtype.kind not in "iufc"
            or raw.ndim not in (2, 3) or min(raw.shape) < 1
            or raw.shape[-2] != raw.shape[-1]
            or not np.all(np.isfinite(vector))):
        raise ValueError("covariance and steering must be finite matching numeric arrays")
    expected = (raw.shape[-1],) if single else (raw.shape[0], raw.shape[-1])
    if vector.shape != expected:
        raise ValueError("steering must match the frequency and channel axes")
    selected = _indices(indices, raw.shape[-1])
    # Shared validation precedes selection; it also returns fresh arrays.
    matrices = (_validate_covariance(raw) if single else
                np.stack([_validate_covariance(matrix) for matrix in raw]))
    reduced = matrices[..., selected, :][..., selected].copy()
    reduced_vector = vector[..., selected].astype(complex, copy=True)
    peak = np.max(np.maximum(abs(reduced.real), abs(reduced.imag)), axis=(-2, -1))
    # Keep zero matrices zero so the unique solver explicitly rejects them.
    scale = np.where(peak > 0, peak, 1.0)
    scale_axes = scale[..., None, None]
    normalized = reduced.real/scale_axes + 1j*(reduced.imag/scale_axes)
    weights = mvdr_weights(normalized, reduced_vector,
                           relative_diagonal_loading=relative_diagonal_loading,
                           condition_limit=condition_limit)
    selection = np.zeros((len(selected), raw.shape[-1]))
    selection[np.arange(len(selected)), selected] = 1.0
    return {"selected_indices": selected, "selection_matrix": selection,
            "covariance": reduced, "steering": reduced_vector,
            "weights": weights, "solver_covariance_scale": scale.copy()}


def teaching_channel_selection() -> dict:
    """Return the exact E10-34 given-statistics control, without any audio IO."""
    factor = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.]])
    covariance = factor @ factor.T
    steering = np.ones(3)
    healthy = select_mvdr_channels(covariance, steering, [0, 1, 2])
    recomputed = select_mvdr_channels(covariance, steering, [1, 2])
    old = healthy["weights"][recomputed["selected_indices"]]
    rows = {}
    for name, weights, model, direction in (
            ("healthy", healthy["weights"], covariance, steering),
            ("stale", old, recomputed["covariance"], recomputed["steering"]),
            ("recomputed", recomputed["weights"], recomputed["covariance"],
             recomputed["steering"])):
        response = np.vdot(weights, direction)
        noise = float(np.vdot(weights, model @ weights).real)
        damage = float(abs(response - 1.0)**2)
        rows[name] = {"weights": weights.real.tolist(),
                      "target_response_real_imag": [float(response.real), float(response.imag)],
                      "normalized_noise_power": noise,
                      "normalized_target_damage": damage,
                      "normalized_total_error": damage + noise}
    return {"noise_factor_B": factor.tolist(), "noise_covariance_R": covariance.tolist(),
            "steering": steering.tolist(), "known_failed_channel": 0,
            "selected_indices": recomputed["selected_indices"],
            "selection_matrix": recomputed["selection_matrix"].tolist(),
            "selected_covariance": recomputed["covariance"].real.tolist(),
            "selected_steering": recomputed["steering"].real.tolist(),
            "controls": rows,
            "scope": "known failure and given positive-definite statistics; no fault detector or PCM inference"}
