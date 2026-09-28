"""Chapter 3: a ULA phase-gauge counterexample and *anchored* alternating fit.

Run from the repository root:
    .venv/bin/python -m codes.chapters.ch03.examples.self_calibration_demo

This creates no files.  The generated complex spectra are mathematical inputs,
not audio, measured microphones, or a reproduction of a third-party system.
"""

from __future__ import annotations

import json

import numpy as np

from codes.chapters.ch03.core.calibration import fit_anchored_ula, ula_prediction


def run_demo() -> dict:
    spacing, frequency, speed = 0.04, 1000.0, 343.0
    kd = 2.0 * np.pi * frequency * spacing / speed
    angles = np.array([-40.0, 0.0, 35.0])
    gains = np.array([1.0, 1.08, 0.93 * np.exp(0.12j),
                      1.03 * np.exp(-0.08j)])
    seed, frames, noise_std = 307, 48, 0.005
    rng = np.random.default_rng(seed)
    sources = (rng.standard_normal((angles.size, frames))
               + 1j * rng.standard_normal((angles.size, frames))) / np.sqrt(2.0)
    clean = ula_prediction(angles, gains, sources, spacing_m=spacing,
                           frequency_hz=frequency, sound_speed=speed)

    # Any common shift in sin(theta) can be hidden in a channel-phase slope.
    # Both alternatives keep g[0]=1 and explain every noiseless sample exactly.
    sine_shift = 0.15
    shifted_angles = np.rad2deg(np.arcsin(np.sin(np.deg2rad(angles)) + sine_shift))
    shifted_gains = gains * np.exp(-1j * kd * np.arange(gains.size) * sine_shift)
    same_clean = ula_prediction(shifted_angles, shifted_gains, sources,
                                spacing_m=spacing, frequency_hz=frequency,
                                sound_speed=speed)

    noise = noise_std * (rng.standard_normal(clean.shape)
                         + 1j * rng.standard_normal(clean.shape)) / np.sqrt(2.0)
    observed = clean + noise
    fit = fit_anchored_ula(
        observed, spacing_m=spacing, frequency_hz=frequency,
        sound_speed=speed, measured_relative_phase_rad=0.0)

    # Even noiseless data cannot expose a wrong external phase anchor by
    # residual alone: another common slope produces an exact fit.
    wrong_phase = np.deg2rad(5.0)
    wrong_fit = fit_anchored_ula(
        clean, spacing_m=spacing, frequency_hz=frequency,
        sound_speed=speed, measured_relative_phase_rad=wrong_phase)
    wrong_theory = np.rad2deg(np.arcsin(
        np.sin(np.deg2rad(angles)) - wrong_phase / kd))

    return {
        "model": "one free-field narrowband source per scene; synchronized four-mic ULA",
        "parameters": {
            "sound_speed_m_s": speed, "frequency_hz": frequency,
            "spacing_m": spacing, "seed": seed, "frames_per_scene": frames,
            "complex_noise_rms": noise_std, "angle_interval_deg": [-60.0, 60.0],
            "true_angles_deg_for_scoring_only": angles.tolist(),
            "true_relative_gains_for_scoring_only": [[float(v.real), float(v.imag)] for v in gains],
        },
        "unanchored_gauge": {
            "common_sine_shift": sine_shift,
            "alternative_angles_deg": shifted_angles.tolist(),
            "alternative_channel_1_phase_deg": float(np.rad2deg(np.angle(shifted_gains[1]))),
            "max_same_sample_prediction_difference": float(np.max(np.abs(clean - same_clean))),
            "meaning": "g0=1 alone leaves an exact common phase-slope ambiguity",
        },
        "correct_external_anchor": {
            "phase_deg": 0.0, "status": fit["status"],
            "iterations": fit["iterations"],
            "estimated_angles_deg": fit["angles_deg"].tolist(),
            "estimated_relative_gains": [[float(v.real), float(v.imag)]
                                         for v in fit["relative_gains"]],
            "angle_errors_deg": (fit["angles_deg"] - angles).tolist(),
            "first_objective": fit["objective_history"][0],
            "last_objective": fit["objective_history"][-1],
            "objective_history": fit["objective_history"],
        },
        "wrong_external_anchor_on_noiseless_data": {
            "assumed_phase_deg": 5.0, "status": wrong_fit["status"],
            "estimated_angles_deg": wrong_fit["angles_deg"].tolist(),
            "theoretical_biased_angles_deg": wrong_theory.tolist(),
            "middle_scene_bias_deg": float(wrong_fit["angles_deg"][1]),
            "last_objective": wrong_fit["objective_history"][-1],
            "meaning": "a tiny residual cannot certify that the external phase anchor is correct",
        },
        "limits": "Single-seed synthetic result; true values are used only for scoring. "
                  "The measured phase is assumed independent and exact in the main fit. "
                  "No room, multiple simultaneous sources, frequency-varying gain, clock error, "
                  "global-optimum guarantee or device accuracy claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
