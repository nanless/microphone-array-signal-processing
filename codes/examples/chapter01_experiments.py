"""Chapter 1: finite records, alignment error and two distinct head models.

Run ``.venv/bin/python -m codes.examples.chapter01_experiments``. E01-04--06
use deterministic mathematical inputs, not device or listener measurements.
No files are written. Powers are dimensionless digital mean squares, without
subtracting sample means; dB values are relative levels, never dB SPL.
"""
from __future__ import annotations

import json
import numpy as np


def finite_record_cross_terms() -> dict:
    """E01-04: expand the square before interpreting a short-record gain."""
    noise = np.array([[1., 1., -1., -1.], [1., -1., 1., -1.]])
    rows = []
    for count in (4, 3):
        first, second = noise[:, :count]
        mean = (first + second) / 2
        p1, p2 = float(np.mean(first**2)), float(np.mean(second**2))
        cross = float(np.mean(first * second))
        power = float(np.mean(mean**2))
        rows.append({
            "samples": count, "noise_1": first.tolist(), "noise_2": second.tolist(),
            "products": (first * second).tolist(), "average": mean.tolist(),
            "average_squared": (mean**2).tolist(),
            "input_mean_squares": [p1, p2], "cross_second_moment": cross,
            "diagonal_contribution": (p1 + p2) / 4,
            "cross_contribution": cross / 2, "output_mean_square": power,
            "gain_relative_to_channel_1_db": float(10 * np.log10(p1 / power)),
        })
    return {"cases": rows,
            "measurement": "Uncentered finite-record mean square; no sample-mean removal",
            "limits": "Fixed sequences demonstrate the identity, not statistical independence. "
                      "Cropping shares samples, so these are not independent trials."}


def residual_sample_delay() -> dict:
    """E01-05: target-only loss when two equal channels differ by one sample.

    Compare sinusoidal amplitudes, not pointwise error against an unmatched
    time reference. Integer-period projection is independent of the analytic
    cosine identity. After alignment both channels equal s[n-1].
    """
    fs = 16000
    index = np.arange(16000)
    rows = []
    for frequency in (1000, 4000):
        phase = 2 * np.pi * frequency * index / fs
        first = np.sin(phase)
        second = np.sin(phase - 2 * np.pi * frequency / fs)
        averaged = (first + second) / 2
        amplitude = float(2 * abs(np.mean(averaged * np.exp(-1j * phase))))
        ratio = float(np.cos(np.pi * frequency / fs))
        rows.append({"frequency_hz": frequency,
                     "interchannel_phase_rad": float(2 * np.pi * frequency / fs),
                     "target_amplitude_ratio": ratio,
                     "projected_target_amplitude_ratio": amplitude,
                     "target_power_ratio": ratio**2,
                     "target_level_change_db": float(20 * np.log10(ratio)),
                     "aligned_target_amplitude_ratio": 1.0})
    return {"sample_rate_hz": fs, "residual_delay_samples": 1,
            "residual_delay_seconds": 1 / fs, "cases": rows,
            "unaligned_common_delay_samples": .5,
            "aligned_common_delay_samples": 1,
            "limits": "No noise is present: these are target level changes, not SNR gains. "
                      "Exact alignment is known, not estimated."}


def woodworth_comparison() -> dict:
    """E01-06: compare a ray model around a sphere with unobstructed points.

    Woodworth: Aaronson & Hartmann (2014), JASA 135, 817--823,
    DOI 10.1121/1.4861243. Antipodal ears, distant source, frontal hemisphere;
    the signed formula extends the unsigned ray length by left/right symmetry.
    It is a high-frequency approximation, not a listener threshold or HRTF.
    """
    radius, speed, fs = .0875, 343., 16000
    rows = []
    for angle in (-90, -30, -1, 0, 1, 30, 90):
        theta = np.deg2rad(angle)
        sphere = float(radius * (theta + np.sin(theta)) / speed)
        free = float(2 * radius * np.sin(theta) / speed)
        rows.append({"azimuth_deg": angle, "azimuth_rad": float(theta),
                     "sphere_itd_us": sphere * 1e6,
                     "free_point_itd_us": free * 1e6,
                     "sphere_itd_samples": sphere * fs,
                     "free_point_itd_samples": free * fs})
    return {"head_radius_m": radius, "sound_speed_m_s": speed,
            "sample_rate_hz": fs, "cases": rows,
            "angle_convention": "Zero is front; positive angles point right",
            "itd_convention": "left arrival minus right arrival; positive means right arrives first",
            "limits": "The sphere and free points are different physical models; "
                      "neither predicts a universal human discrimination threshold."}


def run_exercises() -> dict:
    """Stable exercise IDs; metadata lives inside each individual result."""
    return {"E01-04": finite_record_cross_terms(),
            "E01-05": residual_sample_delay(), "E01-06": woodworth_comparison()}


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
