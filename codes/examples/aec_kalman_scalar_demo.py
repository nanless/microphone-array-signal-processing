"""Run the single-bin scalar Kalman hand case in chapter 6, Eq. (6-9).

This deliberately excludes FFT block formation, partition coupling, variance
estimation, double-talk detection, nonlinear echo and device timing. It is
neither a complete FDKF/PBFDKF nor a double-talk recording experiment.

Original frequency-domain AEC context: Enzner and Vary, Signal Processing 86
(2006), https://doi.org/10.1016/j.sigpro.2005.09.013 .
Run ``python -m codes.examples.aec_kalman_scalar_demo`` from the repo root.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.conventions import finite_real_scalar


def _finite_complex_scalar(value: object, name: str) -> complex:
    """Accept one real/complex numeric scalar without converting text or bool."""

    original = np.asarray(value)
    if original.ndim != 0 or original.dtype.kind not in "iufc":
        raise ValueError(f"{name} must be a finite numeric scalar")
    converted = complex(original)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite and representable as complex128")
    return converted


def scalar_kalman_step(
    *,
    previous_weight: complex,
    previous_variance: float,
    transition: complex,
    process_variance: float,
    reference: complex,
    observation: complex,
    observation_variance: float,
) -> dict:
    """Return prior/innovation/gain/posterior for one known-variance bin.

    ``observation = reference * true_weight + disturbance`` uses no conjugate
    on the prediction; the gain therefore conjugates ``reference``. The model
    assumes non-negative prior/process variances and strictly positive
    observation variance. Supplied variances are model inputs, not estimates
    learned from this one residual.
    """

    w = _finite_complex_scalar(previous_weight, "previous_weight")
    a = _finite_complex_scalar(transition, "transition")
    x = _finite_complex_scalar(reference, "reference")
    d = _finite_complex_scalar(observation, "observation")
    p = finite_real_scalar(previous_variance, "previous_variance")
    phi = finite_real_scalar(process_variance, "process_variance")
    psi = finite_real_scalar(observation_variance, "observation_variance")
    if p < 0.0 or phi < 0.0 or psi <= 0.0:
        raise ValueError("variances require P >= 0, Phi >= 0, Psi > 0")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise",
                         under="ignore"):
            prior_weight = a * w
            prior_variance = abs(a) ** 2 * p + phi
            prior_error = d - x * prior_weight
            denominator = abs(x) ** 2 * prior_variance + psi
            gain = prior_variance * x.conjugate() / denominator
            posterior_weight = prior_weight + gain * prior_error
            # Algebraically (1-KX)P^-; this positive form avoids subtracting
            # nearly equal floating-point numbers when observation noise is low.
            posterior_variance = prior_variance * psi / denominator
            posterior_error = d - x * posterior_weight
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("scalar Kalman intermediate exceeds float64 range") from error
    if not np.all(np.isfinite((prior_weight, prior_variance, prior_error,
                               denominator, gain, posterior_weight,
                               posterior_variance, posterior_error))):
        raise ValueError("scalar Kalman intermediate exceeds float64 range")
    return {
        "prior_weight": prior_weight,
        "prior_variance": float(prior_variance),
        "prior_error": prior_error,
        "innovation_variance": float(denominator),
        "gain": gain,
        "posterior_weight": posterior_weight,
        "posterior_variance": float(posterior_variance),
        "posterior_error": posterior_error,
    }


def _real_report(step: dict) -> dict:
    """Emit 12-decimal JSON values only for the chosen real hand case."""

    result = {}
    for key, value in step.items():
        if abs(complex(value).imag) > 1e-12:
            raise ValueError("real hand case unexpectedly acquired imaginary part")
        result[key] = round(float(complex(value).real), 12)
    return result


def _complex_pair(value: complex) -> dict:
    return {"real": round(float(complex(value).real), 12),
            "imag": round(float(complex(value).imag), 12)}


def run_demo() -> dict:
    """Return chapter hand values and two controlled algebraic contrasts."""

    common = dict(previous_weight=.2, previous_variance=.5, transition=1.,
                  process_variance=.1, reference=2., observation=1.4)
    baseline = scalar_kalman_step(**common, observation_variance=.4)
    larger_psi = scalar_kalman_step(**common, observation_variance=4.)
    complex_reference = scalar_kalman_step(
        previous_weight=0., previous_variance=1., transition=1.,
        process_variance=0., reference=1j, observation=1.,
        observation_variance=1.)
    return {
        "model": "one known-variance scalar frequency bin; equation (6-9), chapter 6",
        "scope": "not complete FDKF/PBFDKF and not measured double talk",
        "display_decimal_places": 12,
        "chapter_inputs": {**common, "observation_variance": .4},
        "chapter_step": _real_report(baseline),
        "larger_observation_variance": {
            "observation_variance": 4.,
            "step": _real_report(larger_psi),
            "interpretation": "same prior and observation; larger assumed disturbance variance reduces gain, not a double-talk decision",
        },
        "complex_reference_check": {
            "inputs": {"previous_weight": 0., "previous_variance": 1.,
                       "transition": 1., "process_variance": 0.,
                       "reference": _complex_pair(1j), "observation": _complex_pair(1.),
                       "observation_variance": 1.},
            "gain": _complex_pair(complex_reference["gain"]),
            "posterior_weight": _complex_pair(complex_reference["posterior_weight"]),
            "posterior_variance": complex_reference["posterior_variance"],
            "posterior_error": _complex_pair(complex_reference["posterior_error"]),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
