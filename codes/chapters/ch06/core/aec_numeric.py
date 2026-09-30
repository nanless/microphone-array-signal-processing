"""Exceptional-range arithmetic for the short AEC teaching filters.

Ordinary inputs keep their original NumPy arithmetic. Only exceptional scales
use exact ratios of the supplied binary floats, with one final float rounding.
This slow fallback is a numerical teaching safeguard, not a real-time kernel.
"""
from __future__ import annotations

from fractions import Fraction
import numpy as np


def exceptional_scale(*values: object) -> bool:
    """Conservative guard for compound products (including Joseph products).

The ordinary interval leaves ample exponent headroom for several products.
Zeros are legitimate and do not trigger the fallback.
"""
    for value in values:
        array = np.asarray(value)
        # Inspect complex components, not a modulus that could overflow.
        components = (array.real, array.imag) if np.iscomplexobj(array) else (array,)
        for component in components:
            absolute = np.abs(component)
            nonzero = absolute[absolute != 0]
            if nonzero.size and (np.any(nonzero < 2.**-200)
                                 or np.any(nonzero > 2.**200)):
                return True
    return False


def ratio(value: float) -> Fraction:
    """Exact rational value of one already validated binary float."""
    return Fraction.from_float(float(value))


def rounded(value: Fraction, name: str, *, positive_variance: bool = False) -> float:
    """Round once; reject unrepresentable finite results and positive variance 0.

True zero variances remain legal. We never invent a positive variance floor.
"""
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds float64 range") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} exceeds float64 range")
    if positive_variance and value > 0 and result == 0:
        raise ValueError(f"{name}: positive variance is below float64 range")
    return result


def exact_dot(left: object, right: object) -> Fraction:
    """Accumulate products before rounding, retaining cancellation and small sums."""
    return sum((ratio(a) * ratio(b) for a, b in zip(left, right)), Fraction(0))


def real_dot(left: np.ndarray, right: np.ndarray) -> float:
    """Preserve the original NumPy dot except at exceptional scales."""
    if exceptional_scale(left, right):
        return rounded(exact_dot(left, right), "AEC prediction")
    return float(left @ right)


def nlms_weights(weights, taps, observation, step, epsilon) -> np.ndarray:
    """Exact compound NLMS update for exceptional scales, before final rounding."""
    u, w = list(map(ratio, taps)), list(map(ratio, weights))
    error = ratio(observation)-sum((a*b for a, b in zip(u, w)), Fraction(0))
    denominator = sum((a*a for a in u), ratio(epsilon))
    mu = ratio(step)
    return np.array([rounded(wi+mu*error*ui/denominator, "NLMS weights")
                     for wi, ui in zip(w, u)])


def ipnlms_weights(weights, taps, observation, step, kappa, floor, gain_floor) -> np.ndarray:
    """Keep gain allocation, denominator and update products until final rounding."""
    u, w = list(map(ratio, taps)), list(map(ratio, weights))
    k, mu = ratio(kappa), ratio(step)
    uniform = (1-k)/(2*len(w))
    gain_denominator = 2*sum(map(abs, w), Fraction(0))+ratio(gain_floor)
    gains = [uniform+(1+k)*abs(wi)/gain_denominator for wi in w]
    denominator = sum((g*ui*ui for g, ui in zip(gains, u)), ratio(floor))
    # Preserve this interface's existing finite-normalization support boundary.
    # The fallback fixes removable update underflow within that boundary.
    rounded(denominator, "IPNLMS normalization", positive_variance=True)
    error = ratio(observation)-sum((ui*wi for ui, wi in zip(u, w)), Fraction(0))
    return np.array([rounded(wi+mu*error*g*ui/denominator, "IPNLMS weights")
                     for wi, g, ui in zip(w, gains, u)])


def rls_update(weights, inverse, taps, observation, forgetting) -> tuple[np.ndarray, np.ndarray]:
    """Exceptional compound RLS gain and inverse update, without storing P*u first."""
    length = len(weights)
    u, w = list(map(ratio, taps)), list(map(ratio, weights))
    p = [[ratio(item) for item in row] for row in inverse]
    q = [sum((p[i][j]*u[j] for j in range(length)), Fraction(0)) for i in range(length)]
    lam = ratio(forgetting)
    denominator = sum((ui*qi for ui, qi in zip(u, q)), lam)
    if denominator <= 0:
        raise ValueError("RLS innovation denominator is not positive")
    error = ratio(observation)-sum((ui*wi for ui, wi in zip(u, w)), Fraction(0))
    new_w = np.array([rounded(wi+qi*error/denominator, "RLS weights")
                      for wi, qi in zip(w, q)])
    new_p = np.array([[rounded((p[i][j]-q[i]*q[j]/denominator)/lam,
                               "RLS inverse covariance", positive_variance=(i == j))
                       for j in range(length)] for i in range(length)])
    return new_w, new_p


def complex_ratio(value: complex) -> tuple[Fraction, Fraction]:
    return ratio(value.real), ratio(value.imag)


def complex_product(left: tuple, right: tuple) -> tuple:
    ar, ai = left
    br, bi = right
    return ar*br-ai*bi, ar*bi+ai*br


def rounded_complex(value: tuple, name: str) -> complex:
    return complex(rounded(value[0], name), rounded(value[1], name))
