"""One controlled DSB/MVDR/LCMV comparison for Chapter 5.

All numbers use an exact, single-frequency covariance and deterministic
steering vectors. They are not speech scores, sample estimates, or an
industrial beamformer benchmark.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.beamforming import dsb_weights, lcmv_weights, mvdr_weights
from codes.array_tutorial.geometry import plane_wave_steering


def run_experiment() -> dict:
    sound_speed_m_s = 343.0
    frequency_hz = 2000.0
    microphone_spacing_m = 0.04
    interferer_power = 10.0
    white_noise_power_per_mic = 1.0
    target_power = 1.0
    design_target_deg = 0.0
    true_target_deg = 10.0
    interferer_deg = 40.0
    relative_loading = 1.0

    positions_m = np.column_stack(
        (np.arange(4, dtype=float) * microphone_spacing_m, np.zeros(4))
    )

    def steering(angle_deg: float) -> np.ndarray:
        return plane_wave_steering(
            positions_m, [frequency_hz], np.deg2rad(angle_deg),
            sound_speed=sound_speed_m_s,
        )[0]

    design_target = steering(design_target_deg)
    true_target = steering(true_target_deg)
    interferer = steering(interferer_deg)
    interference_plus_noise = (
        interferer_power * np.outer(interferer, interferer.conj())
        + white_noise_power_per_mic * np.eye(4)
    )
    methods = {
        "DSB": dsb_weights(design_target),
        "MVDR": mvdr_weights(interference_plus_noise, design_target),
        "MVDR_loaded": mvdr_weights(
            interference_plus_noise, design_target,
            relative_diagonal_loading=relative_loading,
        ),
        "LCMV": lcmv_weights(
            interference_plus_noise,
            np.column_stack((design_target, interferer)),
            np.array([1.0, 0.0]),
        ),
    }
    rows = []
    for name, weights in methods.items():
        nominal_response = np.vdot(weights, design_target)
        true_response = np.vdot(weights, true_target)
        interferer_response = np.vdot(weights, interferer)
        noise_power = float(np.vdot(weights, interference_plus_noise @ weights).real)
        norm_squared = float(np.vdot(weights, weights).real)
        rows.append({
            "method": name,
            "nominal_target_response_abs": float(abs(nominal_response)),
            "true_target_response_abs": float(abs(true_response)),
            "true_target_gain_db": float(20 * np.log10(abs(true_response))),
            "interferer_response_abs": float(abs(interferer_response)),
            "white_noise_gain_db": float(
                10 * np.log10(1.0 / norm_squared)
            ),
            "interference_plus_noise_output_power": noise_power,
            "true_target_output_sinr_db": float(
                10 * np.log10(target_power * abs(true_response) ** 2 / noise_power)
            ),
        })

    return {
        "model": {
            "geometry": "four-element ULA, x=0,0.04,0.08,0.12 m; broadside +y",
            "frequency_hz": frequency_hz,
            "sound_speed_m_s": sound_speed_m_s,
            "design_target_deg": design_target_deg,
            "true_target_deg": true_target_deg,
            "interferer_deg": interferer_deg,
            "target_power": target_power,
            "interferer_power": interferer_power,
            "white_noise_power_per_mic": white_noise_power_per_mic,
            "relative_diagonal_loading": relative_loading,
            "absolute_diagonal_loading": relative_loading
            * float(np.trace(interference_plus_noise).real / 4),
            "covariance_source": "exact interference plus white noise, no target leakage",
            "statistics": "deterministic analytic powers; no random trials or confidence interval",
        },
        "reference_mic_sinr_db": float(
            10 * np.log10(target_power / (interferer_power + white_noise_power_per_mic))
        ),
        "results": rows,
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
