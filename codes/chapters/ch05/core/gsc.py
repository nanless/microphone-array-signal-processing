"""Stateful scalar-reference GSC NLMS teaching loop, not an array front end.

The caller supplies d=wq.H x and u=B.H x. A single complex coefficient h
produces e=d-conj(h)*u BEFORE adaptation. This module estimates no direction,
SPP, blocking matrix or room transfer function. Epsilon has units of |u|².
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_scalar


class ScalarGSCNLMS:
    """One-reference NLMS with explicit block continuity, freeze and reset."""

    def __init__(self, *, step_size: float = 0.1, epsilon: float = 1e-8):
        self.step_size = finite_real_scalar(step_size, "step_size")
        self.epsilon = finite_real_scalar(epsilon, "epsilon")
        if not 0 < self.step_size < 2 or self.epsilon <= 0:
            raise ValueError("require 0 < step_size < 2 and epsilon > 0")
        self.reset()

    def reset(self) -> None:
        """Reset the coefficient and processed-sample count to zero."""
        self.coefficient = 0j
        self.samples_processed = 0

    def process(self, desired: np.ndarray, reference: np.ndarray, *,
                update: bool | np.ndarray = True) -> np.ndarray:
        """Process one block; false update entries freeze h, not the output.

        Empty blocks are valid. Invalid input or unrepresentable arithmetic
        leaves the state unchanged. No output gain or delay fitting is applied.
        """
        d, u = np.asarray(desired, complex), np.asarray(reference, complex)
        if d.ndim != 1 or u.shape != d.shape or not np.all(np.isfinite(d)) or not np.all(np.isfinite(u)):
            raise ValueError("desired and reference must be matching finite vectors")
        gate = np.asarray(update)
        if gate.dtype.kind != 'b' or gate.shape not in ((), d.shape):
            raise ValueError("update must be boolean or a matching boolean vector")
        gate = np.broadcast_to(gate, d.shape)
        h = complex(self.coefficient)
        out = np.empty_like(d)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            for index in range(d.size):
                error = d[index] - h.conjugate() * u[index]
                if not np.isfinite(error):
                    raise ValueError("GSC output exceeds floating-point range")
                out[index] = error
                if gate[index]:
                    denominator = u[index].real**2 + u[index].imag**2 + self.epsilon
                    candidate = h + self.step_size * u[index] * error.conjugate() / denominator
                    if not np.isfinite(denominator) or not np.isfinite(candidate):
                        raise ValueError("GSC update exceeds floating-point range")
                    h = complex(candidate)
        self.coefficient = h
        self.samples_processed += d.size
        return out
