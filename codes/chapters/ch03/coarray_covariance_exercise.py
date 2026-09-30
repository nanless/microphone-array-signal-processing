"""E03-07: reconstruct small co-array statistics from physical covariances.

Run with ``.venv/bin/python -m codes.chapters.ch03.coarray_covariance_exercise``.
The numbers are exact narrowband teaching inputs, not recorded microphone data.
The finite-snapshot example shows why a lag-filled Toeplitz matrix must be
checked before it is called a usable covariance matrix.
"""

from __future__ import annotations

import json

import numpy as np


POSITIONS = np.array([0, 1, 3], dtype=int)  # half-wavelength grid units


def _is_hermitian(matrix: np.ndarray) -> bool:
    """Compare at a common safe scale; zero is exactly Hermitian."""
    scale = max(float(np.max(abs(matrix.real))), float(np.max(abs(matrix.imag))))
    if scale == 0:
        return True
    scaled = matrix.real / scale + 1j * (matrix.imag / scale)
    return bool(np.allclose(scaled, scaled.conj().T, atol=1e-12, rtol=1e-12))


def average_ordered_lags(covariance: np.ndarray, positions: np.ndarray = POSITIONS) -> dict[int, complex]:
    """Average finite Hermitian R[i,j] over signed integer p[i]-p[j].

    This is a reordering tool, not a PSD validator. The caller must establish
    that an input described as a physical covariance has the required model.
    Python integers preserve differences exceeding a fixed-width integer dtype.
    """
    matrix = np.asarray(covariance, dtype=complex)
    coordinates = np.asarray(positions)
    if (matrix.shape != (coordinates.size, coordinates.size)
            or coordinates.ndim != 1 or coordinates.size == 0
            or np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(coordinates))
            or np.any(coordinates != np.rint(coordinates))):
        raise ValueError("require finite square covariance and integer one-dimensional positions")
    if not _is_hermitian(matrix):
        raise ValueError("input statistics must be Hermitian")
    coordinates = [int(p) for p in coordinates]
    grouped: dict[int, list[complex]] = {}
    for i, pi in enumerate(coordinates):
        for j, pj in enumerate(coordinates):
            grouped.setdefault(pi-pj, []).append(matrix[i, j])
    result = {}
    for lag, values in sorted(grouped.items()):
        # Preserve ordinary finite means. Only an overflowing accumulation
        # needs division before summing; a finite average can still exist.
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.mean(values)
        if not np.isfinite(mean):
            mean = sum(value / len(values) for value in values)
        if not np.isfinite(mean):
            raise ValueError("lag average exceeds supported floating-point range")
        result[lag] = complex(mean)
    return result


def virtual_toeplitz(lags: dict[int, complex], size: int) -> np.ndarray:
    """Fill a contiguous virtual ULA; this operation alone does not ensure PSD."""
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    required = set(range(1-size, size))
    if not required.issubset(lags):
        raise ValueError("all signed lags in the requested contiguous segment are required")
    result = np.array([[lags[i-j] for j in range(size)] for i in range(size)], dtype=complex)
    if not np.all(np.isfinite(result)):
        raise ValueError("lag values must be finite")
    if not _is_hermitian(result):
        raise ValueError("lag values must satisfy conjugate symmetry")
    return result


def run_exercise() -> dict:
    """Return an ideal PSD case and a finite-snapshot indefinite counterexample."""
    # Broadside z=1 and +30-degree z=i: a_m = z**p_m under this book's
    # positive-angle, half-wavelength ULA convention. Each has unit power.
    a_broadside = np.ones(3, dtype=complex)
    a_30 = (1j) ** POSITIONS
    physical = (np.outer(a_broadside, a_broadside.conj())
                + np.outer(a_30, a_30.conj()) + 0.1*np.eye(3))
    lags = average_ordered_lags(physical)
    virtual = virtual_toeplitz(lags, 4)

    # A single finite sample is a valid rank-one physical covariance.
    # Its lag-filled virtual Toeplitz matrix has a negative eigenvalue.
    sample = np.array([1, 0, 1], dtype=complex)
    sample_physical = np.outer(sample, sample.conj())
    sample_lags = average_ordered_lags(sample_physical)
    sample_virtual = virtual_toeplitz(sample_lags, 4)

    encode = lambda values: {str(k): [float(v.real), float(v.imag)] for k, v in values.items()}
    return {
        "physical_positions_half_wavelength": POSITIONS.tolist(),
        "physical_channels": 3,
        "virtual_positions_half_wavelength": [0, 1, 2, 3],
        "source_directions_deg": [0, 30],
        "source_powers": [1.0, 1.0],
        "independent_channel_noise_power": 0.1,
        "ideal_lags_real_imag": encode(lags),
        "ideal_virtual_eigenvalues": np.linalg.eigvalsh(virtual).tolist(),
        "sample_snapshot_real_imag": [[float(z.real), float(z.imag)] for z in sample],
        "sample_lags_real_imag": encode(sample_lags),
        "sample_virtual_eigenvalues": np.linalg.eigvalsh(sample_virtual).tolist(),
        "sample_physical_eigenvalues": np.linalg.eigvalsh(sample_physical).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_exercise(), ensure_ascii=False, indent=2))
