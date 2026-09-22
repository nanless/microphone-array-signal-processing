"""Twelve deterministic engineering/mathematics exercises, Chapters 10–13.

Run with ``python -m codes.examples.exercises_engineering``. No hardware,
network, playback, or file writes are performed. Values are teaching inputs,
not measured product performance. Audio samples have a separate generator.
"""

from __future__ import annotations

import json

import numpy as np

from codes.array_tutorial.covariance import spatial_covariance
from codes.array_tutorial.engineering import (
    HysteresisVAD, PeakProtectAGC, RingBuffer, estimate_sro_ppm,
    q15_quantize, simulate_deadline_queue,
)


def run_exercises() -> dict:
    """Return stable exercise IDs with JSON-safe intermediate and final values."""
    results = {}
    times = np.array([0., 10., 20.])
    delays = np.array([.002, .003, .004])
    ppm, offset = estimate_sro_ppm(times, delays)
    results["E10-01"] = {
        "times_s": times.tolist(), "delays_s": delays.tolist(),
        "first_order_ppm": ppm, "offset_ms": offset * 1000,
        "output_input_rate_ratio": 1 / (1 + ppm * 1e-6),
        "drift_600s_ms": ppm * 1e-6 * 600 * 1000,
    }
    vad = HysteresisVAD(.08, .03, 2)
    energies = [0., .09, .04, 0., 0., 0.]
    results["E10-02"] = {
        "energies": energies,
        "active": [bool(vad.update(np.full(160, np.sqrt(e)))) for e in energies],
    }
    ring = RingBuffer(4)
    ring.write(np.array([1., 2., 3.]))
    ring.write(np.array([4., 5.]))
    results["E10-03"] = {"retained": ring.read(4).tolist(), "dropped": ring.dropped}
    costs = np.array([2., 18., 18., 2., 2., 2.])
    results["E10-04"] = {
        "offered_work_rtf": float(costs.sum() / (len(costs) * 10)),
        **simulate_deadline_queue(costs, 10., 2),
    }
    quantizer_input = np.array([-1., -.5/32768, .5/32768, 1.5/32768, 1.])
    results["E10-05"] = {"input": quantizer_input.tolist(),
                            "q15": q15_quantize(quantizer_input).tolist()}
    agc = PeakProtectAGC(.8, 4., .5, .5)
    gains, peaks = [], []
    for amplitude in (.2, 2.):
        output, gain = agc.process(np.array([-amplitude, amplitude]))
        gains.append(float(gain))
        peaks.append(float(np.max(np.abs(output))))
    results["E10-06"] = {"gains": gains, "output_peaks": peaks}
    base_latency = sum([16, 8, 22, 5, 70])
    results["E11-01"] = {
        "alias_boundary_hz": 343 / (.04 * 2),
        "drift_300s_ms": 80e-6 * 300 * 1000,
        "maximum_alignment_interval_s": .0001 / 80e-6,
        "base_latency_ms": base_latency,
        "with_separator_ms": base_latency + 48 + 18,
    }
    average_power = .2 * .1 + 1.2 * .9
    results["E11-02"] = {
        "scored_files": 8, "expected_files": 10, "coverage": 8 / 10,
        "average_power_w": average_power,
        "nominal_energy_wh": 3.7 * 2,
        "estimated_runtime_h": 3.7 * 2 * .8 / average_power,
    }
    x, h = np.array([1., 2.]), np.array([1., .5])
    results["E12-01"] = {
        "linear": np.convolve(x, h).tolist(),
        "fft_length_2": np.fft.irfft(np.fft.rfft(x, 2) * np.fft.rfft(h, 2), 2).tolist(),
        "fft_length_3": np.fft.irfft(np.fft.rfft(x, 3) * np.fft.rfft(h, 3), 3).tolist(),
    }
    spectra = np.array([[1, 1], [1j, -1j]], dtype=complex)[:, None, :]
    moment = spatial_covariance(spectra)[0]
    covariance = spatial_covariance(spectra, demean=True)[0]
    results["E12-02"] = {
        "second_moment_real": moment.real.tolist(),
        "second_moment_imag": moment.imag.tolist(),
        "centered_covariance_real": covariance.real.tolist(),
        "centered_covariance_imag": covariance.imag.tolist(),
    }
    a, b = np.array([[1., 1.], [2., 2.]]), np.array([2., 4.])
    solution, _, rank, singular_values = np.linalg.lstsq(a, b, rcond=None)
    results["E12-03"] = {
        "solution": solution.tolist(), "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "residual_norm": float(np.linalg.norm(a @ solution - b)),
    }
    first, second = np.array([0., .4, -.4, 0.]), np.array([0., .2, -.2, 0.])
    common_gain = .8 / max(np.max(np.abs(first)), np.max(np.abs(second)))
    results["E13-01"] = {
        "common_gain": float(common_gain),
        "common_peaks": [float(np.max(np.abs(v * common_gain))) for v in (first, second)],
        "amplitude_difference_db": float(20 * np.log10(2.)),
        "separately_normalized_difference": float(np.max(np.abs(
            first / np.max(np.abs(first)) - second / np.max(np.abs(second))))),
    }
    return results


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
