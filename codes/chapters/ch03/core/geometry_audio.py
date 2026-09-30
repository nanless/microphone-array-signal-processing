"""E03-17: two frequencies, one source and two six-channel plane-wave models.

This original mathematical fixture is not speech, a room recording, a DOA
estimator or a sampled fractional-delay filter. Continuous source evaluation
preserves propagation envelopes. Eight-kHz phase equivalence is asserted only
inside the steady scoring window; finite envelopes need not be equivalent.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array
from codes.chapters.ch03.core.geometry import plane_wave_delays, plane_wave_steering

SAMPLE_RATE = 32000
SAMPLES = 64000
SCORING_INTERVAL = (4800, 59200)
FREQUENCIES = (4000.0, 8000.0)
FILE_NAMES = {"reference": "geometry_reference.wav", "direction_u": "geometry_u.wav",
              "direction_v": "geometry_v.wav"}


def geometry_parameters() -> dict:
    radius, speed = .0375, 343.0
    b = speed / (np.sqrt(3) * radius * 8000)
    a = np.sqrt(1-b*b)
    directions = np.array([[a, b], [a, -b]])
    azimuths = np.arctan2(directions[:, 0], directions[:, 1])
    angles = np.arange(6)*np.pi/3
    positions = radius*np.column_stack((np.cos(angles), np.sin(angles)))
    return {"exercise_id": "E03-17", "sample_rate_hz": SAMPLE_RATE,
            "samples_per_channel": SAMPLES, "duration_s": SAMPLES/SAMPLE_RATE,
            "sound_speed_m_s": speed, "positions_m": positions.tolist(),
            "channel_order": list(range(6)), "reference_channel_zero_based": 0,
            "directions_xy": directions.tolist(), "azimuths_deg": np.rad2deg(azimuths).tolist(),
            "frequencies_hz": list(FREQUENCIES), "tone_amplitudes": [.08, .08],
            "source_active_interval_s": [.1, 1.9], "linear_fade_duration_s": .02,
            "common_base_delay_s": .002, "common_export_gain": 1.0,
            "reference_model": "F(t-b), on the array-centre time axis, not microphone 0",
            "observation_model": "x_m(t)=F(t-b+r_m dot u/c)",
            "delay_implementation": "continuous source evaluation, not an estimated delay or sampled interpolation filter",
            "scoring_interval_samples": list(SCORING_INTERVAL),
            "scoring_samples_per_channel": SCORING_INTERVAL[1]-SCORING_INTERVAL[0],
            "phase_convention": "cos coefficient - j*sin coefficient, under exp(+j*2*pi*f*t) synthesis",
            "profile_definition": "min_alpha sum_m |z_m-alpha*a_m|^2 / sum_m |z_m|^2, independently per frequency",
            "randomness": "none", "noise": "none"}


def _source(time: np.ndarray) -> np.ndarray:
    envelope = np.minimum(np.clip((time-.1)/.02, 0, 1), np.clip((1.9-time)/.02, 0, 1))
    return .08*envelope*(np.sin(2*np.pi*4000*time)+np.sin(2*np.pi*8000*time))


def generate_signals() -> dict[str, np.ndarray]:
    """Return centre reference (1xN) and two raw observations (6xN)."""
    parameters = geometry_parameters()
    positions = np.asarray(parameters["positions_m"])
    directions = np.asarray(parameters["directions_xy"])
    angles = np.deg2rad(parameters["azimuths_deg"])
    time = np.arange(SAMPLES)/SAMPLE_RATE
    result = {"reference": _source(time-.002)[None, :]}
    for name, angle, direction in zip(("direction_u", "direction_v"), angles, directions):
        # Shared geometry returns relative-to-mic-0 delays. Add the mic-0
        # projection to express every arrival on the common array-centre axis.
        delays = plane_wave_delays(positions, angle) - positions[0]@direction/343
        result[name] = np.array([_source(time-.002-delay) for delay in delays])
    return result


def _complex_pairs(values: np.ndarray) -> list:
    values = np.asarray(values)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def measure_signal(samples: np.ndarray) -> dict:
    """Fit both tones with four real LS columns, then profile unknown source.

    A six-channel input uses only its six phasors and the two geometric
    candidates. The centre reference is not a clean microphone-0 regressor
    and is never used for matching. All energy denominators are recorded.
    """
    x = finite_real_array(samples, "samples")
    if x.shape not in ((1, SAMPLES), (6, SAMPLES)) or not np.all(np.isfinite(x)):
        raise ValueError("require finite one- or six-channel 64000-sample input")
    start, stop = SCORING_INTERVAL
    time = np.arange(start, stop)/SAMPLE_RATE
    design = np.column_stack([function(2*np.pi*f*time)
                              for f in FREQUENCIES for function in (np.cos, np.sin)])
    coefficients = np.linalg.lstsq(design, x[:, start:stop].T, rcond=None)[0]
    fit = design@coefficients
    params = geometry_parameters()
    positions = np.asarray(params["positions_m"])
    angles = np.deg2rad(params["azimuths_deg"])
    candidates = plane_wave_steering(positions, FREQUENCIES, angles)
    tones = []
    for index, frequency in enumerate(FREQUENCIES):
        z = coefficients[2*index] - 1j*coefficients[2*index+1]
        energy = float(np.vdot(z, z).real)
        if not np.isfinite(energy) or energy <= 0 or abs(z[0]) == 0:
            raise ValueError("each tone needs positive representable reference and phasor energy")
        tone = {"frequency_hz": frequency, "phasor_real_imag": _complex_pairs(z),
                "amplitudes": abs(z).tolist(),
                "relative_steering_real_imag": _complex_pairs(z/z[0]),
                "observed_phasor_squared_sum": energy}
        if len(x) == 6:
            profiles = {}
            for name, candidate in zip(("direction_u", "direction_v"), candidates[:, index, :]):
                alpha = np.vdot(candidate, z)/np.vdot(candidate, candidate).real
                numerator = float(np.vdot(z-alpha*candidate, z-alpha*candidate).real)
                profiles[name] = {"unknown_complex_alpha_real_imag": _complex_pairs(alpha),
                                  "residual_squared_sum": numerator,
                                  "observed_phasor_squared_sum": energy,
                                  "relative_residual": numerator/energy,
                                  "candidate_squared_norm": float(np.vdot(candidate, candidate).real)}
            tone["candidate_profiles"] = profiles
        tones.append(tone)
    return {"scoring_interval_samples": [start, stop], "samples_per_channel": stop-start,
            "channels": len(x), "tones": tones,
            "time_domain_squared_sum": float(np.sum(x[:, start:stop]**2)),
            "two_tone_fit_residual_squared_sum": float(np.sum((x[:, start:stop].T-fit)**2)),
            "peak": float(np.max(abs(x)))}


LIMITS = ("Mathematical synchronized, omnidirectional, equal-amplitude far-field model, no room, "
          "speech, noise or physical device. Each frequency has a separate unknown complex source "
          "coefficient in matching; the mono source reference is not used to fit direction. "
          "The 8 kHz equality concerns steady relative phase, not full-record finite envelopes. "
          "Six-channel playback may be downmixed or reordered by a browser/device and is not DOA evidence. "
          "No DOA estimator, sampled fractional-delay algorithm or formal listening study is run.")
