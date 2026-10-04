"""E08-31: PCA loss and a supplied OverIVA background constraint.

There is no iterative OverIVA fit. The target row is supplied, and statistical
orthogonality does not assert independent sources or target identification.
"""
from __future__ import annotations

import numpy as np


def background_constraint(demixing, covariance):
    """Construct U=[J,-I] from supplied target rows W and positive-definite C.

    W is (target_count,channels), 0<K<M, with y=W*x (rows are w^H).
    C is (channels,channels), Hermitian positive definite. The first K columns
    define this coordinate chart; singular E1*C*W^H is rejected rather than
    silently choosing another chart, loading, or a pseudoinverse.
    """
    for value in (demixing, covariance):
        if np.asarray(value).dtype.kind not in "iufc":
            raise ValueError("demixing and covariance must be numeric")
    w, c = np.asarray(demixing, dtype=complex), np.asarray(covariance, dtype=complex)
    if (w.ndim != 2 or not 0 < w.shape[0] < w.shape[1] or
            c.shape != (w.shape[1], w.shape[1])):
        raise ValueError("require W (K,M), 0<K<M, and C (M,M)")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(c)):
        raise ValueError("demixing and covariance must be finite")
    if not np.allclose(c, c.conj().T, rtol=1e-12, atol=0):
        raise ValueError("covariance must be Hermitian")
    np.linalg.cholesky(c)
    k = w.shape[0]
    with np.errstate(over="ignore", invalid="ignore"):
        cw = c @ w.conj().T
    if not np.all(np.isfinite(cw)):
        raise ValueError("cross moment exceeds float64 support")
    # Right-side system J*C1=C2 becomes C1.T*J.T=C2.T. This ordinary
    # transpose implements a right solve; W^H was already used in cw.
    j = np.linalg.solve(cw[:k].T, cw[k:].T).T
    if not np.all(np.isfinite(j)):
        raise ValueError("background coefficients exceed float64 support")
    u = np.concatenate((j, -np.eye(w.shape[1]-k)), axis=1)
    extended = np.concatenate((w, u), axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        statistics = {'target_background_cross_second_moment': w @ c @ u.conj().T,
                      'target_second_moment': w @ c @ w.conj().T,
                      'background_second_moment': u @ c @ u.conj().T}
    if any(not np.all(np.isfinite(value)) for value in statistics.values()):
        raise ValueError("output second moments exceed float64 support")
    return {'J': j, 'background_rows': u, 'extended_demixing': extended, **statistics}


def run_experiment():
    c = np.diag([9., 4., 1.])
    pca = np.array([[1., 0., 0.], [0., 1., 0.]])
    weak_target = np.array([0., 0., 1.])
    control = background_constraint([[1., .5]], [[2., 1.], [1., 2.]])
    return {'observed_second_moment': c, 'descending_eigenvalues': np.array([9., 4., 1.]),
            'background_direction': np.array([1., 0., 0.]),
            'target_directions': np.array([[0., 1., 0.], weak_target]),
            'retained_top_power_pca_rows': pca,
            'weak_target_after_projection': pca @ weak_target,
            'isotropic_loading_one_eigenvalues': np.array([10., 5., 2.]),
            'supplied_target_rows': np.array([[1., .5]]),
            'two_channel_control_covariance': np.array([[2., 1.], [1., 2.]]),
            'background_constraint_control': control,
            'scope': 'known coordinate roles and supplied target row; not a fitted OverIVA separator'}
