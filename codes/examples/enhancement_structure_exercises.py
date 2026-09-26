"""Independent arithmetic for E07-07, E08-11 and E09-09.

These small functions expose structural identities. They are not full WPD,
cACGMM or IMM pipelines and do not process or generate audio.
"""

from __future__ import annotations

import json
from functools import wraps

import numpy as np


def _finite_arithmetic(function):
    """Reject unrepresentable arithmetic rather than returning NaN or inf."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = function(*args, **kwargs)
        except FloatingPointError as exc:
            raise ValueError("intermediate arithmetic exceeds float64 range") from exc

        def check(value):
            if isinstance(value, dict):
                return all(check(v) for v in value.values())
            if isinstance(value, tuple):
                return all(check(v) for v in value)
            return bool(np.all(np.isfinite(value)))

        if not check(result):
            raise ValueError("result is not finite")
        return result
    return wrapped


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return np.asarray(value, dtype=float)


def _hermitian(value, name, *, positive=True):
    matrix = np.asarray(value, dtype=np.complex128)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix))):
        raise ValueError(f"{name} must be a finite nonempty square matrix")
    scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(scale):
        raise ValueError(f"{name} magnitude exceeds float64 range")
    scaled = matrix / scale if scale > 0 else matrix
    if not np.allclose(scaled, scaled.conj().T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be Hermitian")
    if positive:
        try:
            np.linalg.cholesky(scaled)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
    return matrix


@_finite_arithmetic
def wpd_factorization(covariance, steering):
    """Factor an unregularized positive-definite WPD quadratic form.

    The first len(steering) entries are current channels, the remainder are
    delayed history. Every block must come from the same weighted statistics.
    """
    r = _hermitian(covariance, "covariance")
    r_scale = float(np.max(np.abs(r)))
    r = r / r_scale
    v = np.asarray(steering, dtype=np.complex128)
    if (v.ndim != 1 or not 0 < v.size < r.shape[0]
            or not np.all(np.isfinite(v)) or np.max(np.abs(v)) == 0):
        raise ValueError("steering must be finite, nonzero and shorter than covariance")
    v_scale = float(np.max(np.abs(v)))
    v = v / v_scale
    m = v.size
    a, b, c = r[:m, :m], r[:m, m:], r[m:, m:]
    prediction = np.linalg.solve(c, b.conj().T)
    schur = a - b @ prediction
    direction = np.linalg.solve(schur, v)
    current = (direction / np.vdot(v, direction)) / v_scale
    history = -prediction @ current
    return {"prediction": prediction, "schur": schur * r_scale, "current": current,
            "history": history, "full": np.r_[current, history]}


@_finite_arithmetic
def cacg_shape_step(directions, weights, old_shape):
    """One fixed-point shape step, followed by trace normalization only.

    directions has (frames, channels), with unit-norm rows. A singular result
    is returned, not silently loaded; it cannot be reused as an invertible
    density shape without an explicit regularization policy.
    """
    b = _hermitian(old_shape, "old_shape")
    b_scale = float(np.max(np.abs(b)))
    b = b / b_scale
    z = np.asarray(directions, dtype=np.complex128)
    gamma = _real(weights, "weights")
    if (z.ndim != 2 or z.shape[1] != b.shape[0] or z.shape[0] == 0
            or not np.all(np.isfinite(z))
            or not np.allclose(np.sum(abs(z) ** 2, axis=1), 1, rtol=1e-12, atol=1e-12)):
        raise ValueError("directions must contain finite unit-norm rows")
    if (gamma.shape != (z.shape[0],) or not np.all(np.isfinite(gamma))
            or np.any(gamma < 0) or np.max(gamma) <= 0):
        raise ValueError("weights must be finite, nonnegative and have positive mass")
    gamma = gamma / np.max(gamma)
    quadratic = np.einsum("ti,it->t", z.conj(), np.linalg.solve(b, z.T)).real
    raw = b.shape[0] * np.einsum("t,ti,tj->ij", gamma / quadratic, z, z.conj()) / gamma.sum()
    normalized = raw * b.shape[0] / np.trace(raw).real
    return raw * b_scale, normalized


@_finite_arithmetic
def cacg_relative_density(directions, shape):
    """Return cACG density without the common sphere normalization constant."""
    b = _hermitian(shape, "shape")
    b = b / np.max(np.abs(b))
    z = np.asarray(directions, dtype=np.complex128)
    if (z.ndim != 2 or z.shape[1] != b.shape[0] or z.shape[0] == 0
            or not np.all(np.isfinite(z))
            or not np.allclose(np.sum(abs(z) ** 2, axis=1), 1, rtol=1e-12, atol=1e-12)):
        raise ValueError("directions must contain finite unit-norm rows")
    quadratic = np.einsum("ti,it->t", z.conj(), np.linalg.solve(b, z.T)).real
    _, logdet = np.linalg.slogdet(b)
    return np.exp(-logdet - b.shape[0] * np.log(quadratic))


@_finite_arithmetic
def imm_mix(probabilities, transition, means, covariances):
    """IMM interaction only, for same-dimensional real Euclidean states.

    transition[i,j] is P(new mode j | old mode i), so rows sum to one.
    Unreachable destination modes are rejected instead of dividing by zero.
    Angular states must already be unwrapped in a common local chart.
    """
    mu = _real(probabilities, "probabilities")
    p = _real(transition, "transition")
    x = _real(means, "means")
    cov = _real(covariances, "covariances")
    if (mu.ndim != 1 or mu.size == 0 or not np.all(np.isfinite(mu))
            or np.any(mu < 0) or not np.isclose(mu.sum(), 1, rtol=1e-12, atol=1e-12)):
        raise ValueError("probabilities must be a finite probability vector")
    n = mu.size
    if (p.shape != (n, n) or not np.all(np.isfinite(p)) or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1, rtol=1e-12, atol=1e-12)):
        raise ValueError("transition must be row-stochastic")
    if (x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0
            or not np.all(np.isfinite(x)) or cov.shape != (n, x.shape[1], x.shape[1])
            or not np.all(np.isfinite(cov))):
        raise ValueError("means and covariances must have compatible finite shapes")
    for matrix in cov:
        _hermitian(matrix, "covariance", positive=False)
        scale = float(np.max(np.abs(matrix)))
        if np.linalg.eigvalsh(matrix / scale if scale else matrix).min() < 0:
            raise ValueError("covariances must be positive semidefinite")
    prior = mu @ p
    if np.any(prior <= 0):
        raise ValueError("each destination mode must have positive prior mass")
    mixing = mu[:, None] * p / prior[None, :]
    mixed_mean = mixing.T @ x
    mixed_cov = np.zeros_like(cov)
    for j in range(n):
        for i in range(n):
            difference = x[i] - mixed_mean[j]
            mixed_cov[j] += mixing[i, j] * (cov[i] + np.outer(difference, difference))
    return {"prior": prior, "mixing": mixing,
            "means": mixed_mean, "covariances": mixed_cov}


def run_exercises():
    r = np.array([[2, 0, 1, 0], [0, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 1.]])
    wpd = wpd_factorization(r, [1, 1])
    raw, shape = cacg_shape_step(np.eye(2), [.75, .25], np.eye(2))
    second_raw, second_shape = cacg_shape_step(np.eye(2), [.75, .25], shape)
    imm = imm_mix([.75, .25], [[.9, .1], [.2, .8]], [[0], [10]], [[[1]], [[4]]])
    likelihood = np.array([.2, .8])
    posterior = imm["prior"] * likelihood
    posterior /= posterior.sum()
    return {
        "E07-07": {key: value.real.tolist() for key, value in wpd.items()},
        "E08-11": {"raw": raw.real.tolist(), "shape": shape.real.tolist(),
                   "relative_density": cacg_relative_density(np.eye(2), shape).tolist(),
                   "twice_shape_density": cacg_relative_density(np.eye(2), 2 * shape).tolist(),
                   "second_raw": second_raw.real.tolist(), "second_shape": second_shape.real.tolist()},
        "E09-09": {**{key: value.tolist() for key, value in imm.items()},
                   "likelihood": likelihood.tolist(), "posterior": posterior.tolist()},
    }


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2))
