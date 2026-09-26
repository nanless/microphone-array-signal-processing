"""Five independent arithmetic exercises, E06-21/E07-06/E08-08..10.

No audio, model weights, training, file writes or network access. Arrays use the
chapter's column-vector convention; an AuxIVA matrix stores conjugate rows.
Run from the repository root with ``python -m codes.examples.enhancement_step_exercises``.
"""
from __future__ import annotations

import json
import numpy as np

from codes.array_tutorial.aec import NLMSState


def ip_row(matrix: np.ndarray, covariance: np.ndarray, source: int) -> np.ndarray:
    """Return one normalized AuxIVA IP column; do not mutate the input matrix.

    W,V are finite square complex matrices of equal shape. V must be Hermitian
    positive definite, W invertible. This is an IP substep, not full AuxIVA.
    Write ``result.conj()`` to W[source] before updating the next source.
    """
    w = np.asarray(matrix, dtype=np.complex128)
    v = np.asarray(covariance, dtype=np.complex128)
    if (w.ndim != 2 or w.shape[0] == 0 or w.shape[0] != w.shape[1]
            or v.shape != w.shape or not np.isfinite(w).all() or not np.isfinite(v).all()):
        raise ValueError("W and V must be equal-size finite nonempty square matrices")
    if isinstance(source, (bool, np.bool_)) or not isinstance(source, (int, np.integer)) or not 0 <= source < len(w):
        raise ValueError("source must be a valid integer row index")
    scale = np.max(np.abs(v))
    if scale == 0 or not np.allclose(v / scale, v.conj().T / scale, rtol=1e-12, atol=1e-14):
        raise ValueError("V must be Hermitian positive definite")
    try:
        np.linalg.cholesky(v / scale)
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            u = np.linalg.solve(w @ v, np.eye(len(w), dtype=complex)[:, source])
            quadratic = np.vdot(u, v @ u)
            if not np.isfinite(quadratic) or quadratic.real <= 0:
                raise ValueError("IP quadratic form must be finite and positive")
            result = u / np.sqrt(quadratic.real)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("IP system is singular or not representable") from error
    if not np.isfinite(result).all():
        raise ValueError("IP row is not representable")
    return result


def run_exercises() -> dict:
    x = np.array([1., 1., -1., -1.])
    s = np.array([.5, -.5, .5, -.5])
    d = .8 * x + s
    state = NLMSState(1, step_size=.5, epsilon=0, initial_weights=np.array([.8]))
    residual, path = [], []
    for index in range(len(x)):
        e, _ = state.process(x[index:index+1], d[index:index+1])
        residual.append(float(e[0]))
        path.append(float(state.weights[0]))

    errors = np.array([1+1j, 2])
    v = np.array([[2., 1.], [1., 2.]])
    ip = ip_row(np.eye(2), v, 0)
    basis = np.array([[1., .2], [.5, 1.]])
    activation = np.array([[2., .5], [.1, 3.]])
    rescaled_basis, rescaled_activation = basis.copy(), activation.copy()
    rescaled_basis[:, 0] *= 2
    rescaled_activation[0] /= 2

    snapshots = np.array([[1., 1.], [1., -1.]])  # columns are snapshots
    target = (snapshots * np.array([.9, .1])) @ snapshots.T
    interference = (snapshots * np.array([.1, .9])) @ snapshots.T
    steering = np.ones(2)
    inverse_action = np.linalg.solve(interference, steering)
    beam = inverse_action / (steering @ inverse_action)
    return {
        'E06-21': {'cross_sum': float(x @ s), 'batch_path': float(x @ d / (x @ x)),
                   'prior_residuals': residual, 'online_paths': path,
                   'correlated_batch_path': float(x @ (.8*x+.25*x) / (x@x))},
        'E07-06': {'channel_powers': (np.abs(errors)**2).tolist(),
                   'shared_power': float(np.mean(np.abs(errors)**2)),
                   'one_channel_scaled': float(np.mean(np.abs(errors*np.array([1., 2.]))**2)),
                   'all_channels_scaled': float(np.mean(np.abs(2*errors)**2)),
                   'zero_raw_power': 0.0},
        'E08-08': {'column_real': ip.real.tolist(),
                   'weighted_norm': float(np.vdot(ip, v@ip).real),
                   'euclidean_norm_squared': float(np.vdot(ip, ip).real)},
        'E08-09': {'power': (basis@activation).tolist(),
                   'rescaled_power': (rescaled_basis@rescaled_activation).tolist(),
                   'one_basis_power': np.outer(basis[:, 0], activation[0]).tolist()},
        'E08-10': {'target_scm': target.tolist(), 'interference_scm': interference.tolist(),
                   'weights': beam.tolist(), 'snapshot_outputs': (beam@snapshots).tolist(),
                   'target_output_power': float(beam@target@beam),
                   'interference_output_power': float(beam@interference@beam),
                   'gev_eigenvalues': sorted(np.linalg.eigvals(np.linalg.solve(interference, target)).real.tolist())},
    }


if __name__ == '__main__':
    print(json.dumps(run_exercises(), indent=2, allow_nan=False))
