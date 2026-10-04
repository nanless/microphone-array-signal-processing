"""Known-excitation digital FIR identification for E02-19.

Real mono vectors use the book's unnormalized forward rFFT. The inverse solves
an unconstrained n_fft-point *circular* parameter problem. It does not impose a
short causal FIR support or implement Farina's harmonic-response separation.
Imports do not generate assets or run experiments. Regularization is absolute
and has the units of squared excitation-spectrum amplitude.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar

MAX_FFT_LENGTH = 1 << 20
SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
RESPONSE_SAMPLES = 32160
FFT_LENGTH = 65536
FADE_SAMPLES = 160
DELAY_SAMPLES = 160
NOISE_SEED = 2026100402
REGULARIZATION_RATIOS = (0., 1e-8, 1e-6, 1e-4)


def _vector(value, name):
    result = finite_real_array(value, name)
    if result.ndim != 1 or not result.size:
        raise ValueError(f"{name} must be a nonempty real mono vector")
    return result


def regularized_inverse(source, response, *, n_fft: int, regularization: float) -> dict:
    """Return a full n_fft-point digital IR, without delay/gain fitting or crop.

    The objective at each full-DFT bin is |U H-Y|**2 + epsilon*|H|**2.
    Zero epsilon requires every excitation coefficient to be nonzero; a zero
    squared magnitude caused by underflow is not treated as a coefficient zero.
    Positive epsilon yields the zero minimizer for zero excitation, explicitly
    marked no_excitation. Intermediate range failures raise ValueError rather
    than silently fabricating a recoverable response. The resource budget is
    MAX_FFT_LENGTH, separate from mathematical invertibility.
    """
    u, y = _vector(source, "source"), _vector(response, "response")
    if (isinstance(n_fft, (bool, np.bool_)) or not isinstance(n_fft, (int, np.integer))
            or not max(2, u.size, y.size) <= n_fft <= MAX_FFT_LENGTH):
        raise ValueError("n_fft must represent both full records and fit the teaching FFT budget")
    epsilon = finite_real_scalar(regularization, "regularization")
    if epsilon < 0.:
        raise ValueError("regularization must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        U, Y = np.fft.rfft(u, n=n_fft), np.fft.rfft(y, n=n_fft)
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(Y)):
        raise ValueError("record FFT exceeds floating-point range")
    zero = U == 0
    if epsilon == 0. and np.any(zero):
        raise ValueError("unregularized per-bin division has a zero excitation coefficient")
    magnitude = np.abs(U)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        power = magnitude**2
        if epsilon == 0.:
            estimated = Y / U
        else:
            # The usual expression preserves the published ordinary-scale case.
            numerator = U.conj()*Y
            raw_denominator = power+epsilon
            estimated = numerator/raw_denominator
            numerator_underflow = (numerator == 0.) & ~zero & (Y != 0.)
            exceptional = (~np.isfinite(power) | ((power == 0.) & ~zero)
                           | ~np.isfinite(raw_denominator)
                           | numerator_underflow | ~np.isfinite(estimated))
            if np.any(exceptional):
                # Normalize each exceptional bin before its squared magnitude.
                scale = np.maximum(magnitude[exceptional], np.sqrt(epsilon))
                a = U[exceptional]/scale
                b = Y[exceptional]/scale
                denominator = np.abs(a)**2+(np.sqrt(epsilon)/scale)**2
                estimated[exceptional] = a.conj()*b/denominator
    if not np.all(np.isfinite(estimated)):
        raise ValueError("inverse estimate exceeds the supported floating-point range")
    impulse = np.fft.irfft(estimated, n=n_fft)
    if not np.all(np.isfinite(impulse)):
        raise ValueError("inverse waveform exceeds floating-point range")
    representable_power = bool(np.all(np.isfinite(power)))
    return {"impulse_response": impulse, "input_spectrum": U,
            "response_spectrum": Y, "estimated_spectrum": estimated,
            "excitation_power": power if representable_power else None,
            "excitation_power_representable": representable_power,
            "excitation_power_underflow": bool(np.any((power == 0.) & ~zero)),
            "zero_excitation_bins": np.flatnonzero(zero).tolist(),
            "no_excitation": bool(np.all(zero)), "n_fft": int(n_fft),
            "regularization": epsilon,
            "parameter_space": "unconstrained n_fft-point circular IR; no short-support constraint"}


def score_impulse_response(estimate, reference, *, support_length: int) -> dict:
    """Fixed-reference full-IR error and an explicitly diagnostic support split.

    The shorter reference is zero-extended to the estimate's length. Nothing is
    fitted or discarded. Powers/energies must be representable in float64; this
    scoring interface rejects a zero, underflowed or overflowing denominator.
    The support_length is known truth for a diagnostic, not an estimated gate.
    """
    estimate, reference = _vector(estimate, "estimate"), _vector(reference, "reference")
    if estimate.size < reference.size:
        raise ValueError("estimate cannot truncate the complete reference")
    if (isinstance(support_length, (bool, np.bool_))
            or not isinstance(support_length, (int, np.integer))
            or not 1 <= support_length <= estimate.size):
        raise ValueError("support_length must be an integer within the full estimate")
    if reference.size > support_length and np.any(reference[support_length:] != 0.):
        raise ValueError("declared reference support omits nonzero samples")
    ref = np.pad(reference, (0, estimate.size-reference.size))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        error = estimate-ref
        reference_energy = float(np.dot(ref, ref))
        error_energy = float(np.dot(error, error))
        support_error_energy = float(np.dot(error[:support_length], error[:support_length]))
        outside_energy = float(np.dot(estimate[support_length:], estimate[support_length:]))
    if (not np.all(np.isfinite([reference_energy, error_energy, support_error_energy, outside_energy]))
            or reference_energy <= 0.):
        raise ValueError("IR scoring needs finite energies and a representable positive reference energy")
    full_ratio = error_energy/reference_energy
    support_ratio = support_error_energy/reference_energy
    outside_ratio = outside_energy/reference_energy
    if not np.all(np.isfinite([full_ratio, support_ratio, outside_ratio])):
        raise ValueError("IR relative error exceeds the supported floating-point range")
    for energy, ratio, samples in ((error_energy, full_ratio, error),
                                  (support_error_energy, support_ratio, error[:support_length]),
                                  (outside_energy, outside_ratio, estimate[support_length:])):
        if energy == 0. and np.any(samples != 0.):
            raise ValueError("nonzero IR error energy underflows the scoring representation")
        if energy > 0. and ratio == 0.:
            raise ValueError("positive IR relative error underflows the scoring representation")
    return {"full_ir_relative_error_energy": full_ratio,
            "true_support_relative_error_energy": support_ratio,
            "outside_support_energy": outside_energy, "reference_energy": reference_energy,
            "error_energy": error_energy, "support_error_energy": support_error_energy,
            "max_ir_error": float(np.max(np.abs(error))),
            "full_interval_samples": [0, int(estimate.size)],
            "known_support_interval_samples": [0, int(support_length)],
            "outside_support_interval_samples": [int(support_length), int(estimate.size)],
            "delay_fit": False, "gain_fit": False}


def build_sweep_cases() -> dict:
    """One finite digital exponential sweep and three known-FIR response cases.

    Source length is 32000; each response has all 160 extra tail samples. Noise
    is added after convolution. The cut condition deliberately replaces the
    complete output's last 160 samples by zero; it does not shorten the array.
    Float outputs are filtered before each WAV's independent PCM quantization.
    """
    q = np.arange(SOURCE_SAMPLES)
    duration = SOURCE_SAMPLES/SAMPLE_RATE
    ratio_log = np.log(6000./100.)
    phase = 2*np.pi*100.*duration/ratio_log*np.expm1(q/SAMPLE_RATE*ratio_log/duration)
    envelope = np.minimum(1., np.minimum(q/FADE_SAMPLES, (SOURCE_SAMPLES-1-q)/FADE_SAMPLES))
    source = .2*np.sin(phase)*envelope
    h = np.zeros(DELAY_SAMPLES+1)
    h[0], h[-1] = .8, .25
    complete = np.convolve(source, h, mode="full")
    noise = .003*np.random.default_rng(NOISE_SEED).standard_normal(RESPONSE_SAMPLES)
    noisy = complete+noise
    cut = complete.copy()
    cut[SOURCE_SAMPLES:] = 0.
    return {"source": source, "signals": {"sweep_complete": complete,
            "sweep_noisy": noisy, "sweep_cut": cut}, "impulse_response": h, "noise": noise,
            "parameters": {"sample_rate_hz": SAMPLE_RATE, "source_samples": SOURCE_SAMPLES,
             "response_samples": RESPONSE_SAMPLES, "fft_length": FFT_LENGTH,
             "source_amplitude": .2, "start_frequency_hz": 100., "end_frequency_hz": 6000.,
             "phase_duration_s": duration,
             "last_instantaneous_frequency_hz": float(100*np.exp((SOURCE_SAMPLES-1)/SAMPLE_RATE*ratio_log/duration)),
             "fade_samples": FADE_SAMPLES, "direct_gain": .8, "reflection_gain": .25,
             "reflection_delay_samples": DELAY_SAMPLES, "noise_seed": NOISE_SEED,
             "noise_standard_deviation": .003, "common_export_gain": 1.,
             "regularization_ratios": list(REGULARIZATION_RATIOS),
             "noise_position": "after known digital FIR convolution",
             "scope": "mathematical digital system identification; no real room or nonlinear-response measurement"}}
