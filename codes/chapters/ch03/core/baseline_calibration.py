"""Known-direction calibration of one 3-D baseline and one fixed delay.

The observation is tau[q] = -u[q] dot b / c + delta_t. Directions and sound
speed are external inputs, not estimated source locations or a blind clock
synchronizer. With ell_t = c * delta_t the design [-U, 1] is dimensionless
and all four fitted parameters have units metres.
"""

from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


def solve_baseline(directions, delays_s, sound_speed=343.0) -> dict:
    """Fit a baseline in metres and a scene-invariant relative delay in seconds.

    ``directions`` is Q x 3, Q >= 4, with real unit rows pointing toward
    externally known far-field sources. Row lengths must differ from one by
    no more than 1e-10; they are checked, never silently normalized.
    ``delays_s`` is a real length-Q vector of signed channel-1 minus
    channel-0 delays. Sound speed is a known positive scalar in m/s.

    SVD rank uses Q * float64 epsilon * the largest singular value. A design
    with rank below four is rejected rather than assigned a pseudoinverse
    representative. The reported condition number belongs to [-U, 1], with
    all unknowns expressed in metres. It is not an acoustic error bound.
    Least squares is unweighted. The result retains predicted and residual
    delays (observed minus predicted); no input is modified.

    The RHS is scaled before solving to support very large or small finite
    length data. Nonrepresentable length conversion or outputs are rejected.
    Scaling cannot restore information already rounded out of the input.
    """
    unit = finite_real_array(directions, "directions")
    delays = finite_real_array(delays_s, "delays_s")
    speed = finite_real_scalar(sound_speed, "sound_speed")
    if (unit.ndim != 2 or unit.shape[1] != 3 or unit.shape[0] < 4
            or delays.shape != (unit.shape[0],)):
        raise ValueError("require Q x 3 directions and Q delays, with Q >= 4")
    if speed <= 0:
        raise ValueError("sound_speed must be positive")
    # Unit vectors have bounded components, so reject huge rows before their
    # squared norm can overflow. No array is modified during validation.
    if np.any(np.abs(unit) > 1.0 + 1e-10):
        raise ValueError("directions must be unit vectors")
    lengths = np.sqrt(np.sum(unit * unit, axis=1))
    if np.any(np.abs(lengths - 1.0) > 1e-10):
        raise ValueError("directions must be unit vectors")
    design = np.column_stack((-unit, np.ones(unit.shape[0])))
    singular = np.linalg.svd(design, compute_uv=False)
    cutoff = max(design.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    if rank != 4:
        raise ValueError("known-direction augmented design is rank deficient")
    condition = float(singular[0] / singular[-1])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rhs = speed * delays
    if (not np.all(np.isfinite(rhs))
            or np.any((delays != 0) & (rhs == 0))):
        raise ValueError("delay-to-length conversion exceeds supported range")
    scale = float(np.max(np.abs(rhs)))
    normalized = rhs / scale if scale else rhs
    solution_scaled, _, _, _ = np.linalg.lstsq(
        design, normalized, rcond=max(design.shape) * np.finfo(float).eps)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        solution = solution_scaled * scale
        offset = solution[3] / speed
        # Evaluate at the solve scale, avoiding a large unscaled dot product.
        predicted = (design @ solution_scaled) * (scale / speed)
        residual = delays - predicted
    if (not np.all(np.isfinite(solution)) or not np.isfinite(offset)
            or not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(residual))
            or (solution[3] != 0 and offset == 0)):
        raise ValueError("baseline, offset or fitted delays exceed supported range")
    return {"baseline_m": solution[:3].tolist(), "offset_s": float(offset),
            "rank": rank, "condition_number": condition,
            "predicted_delays_s": predicted.tolist(),
            "residual_delays_s": residual.tolist()}
