"""Twenty-five deterministic engineering/mathematics exercises, Chapters 10–13.

Run with ``python -m codes.examples.exercises_engineering``. No hardware,
network, playback, or file writes are performed. Values are teaching inputs,
not measured product performance. Audio samples have a separate generator.
"""

from __future__ import annotations

import json
import math

import numpy as np

from codes.array_tutorial.covariance import spatial_covariance
from codes.array_tutorial.engineering import (
    HysteresisVAD, PeakProtectAGC, RingBuffer, estimate_sro_ppm,
    q15_quantize, simulate_deadline_queue,
)


def block_consumption(input_block: int, consumer_block: int, callbacks: int) -> dict:
    """Track half-open sample ranges; this models buffering, not CPU timing.

    Callback 1 delivers [0, input_block) at input_block / sample_rate seconds.
    Every complete consumer block is consumed immediately; the remainder stays.
    """
    for value in (input_block, consumer_block, callbacks):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("block sizes and callback count must be positive integers")
    consumed, events, leftovers = 0, [], []
    for callback in range(1, callbacks + 1):
        available = callback * input_block
        while available - consumed >= consumer_block:
            events.append({"callback": callback, "start": consumed,
                           "stop": consumed + consumer_block})
            consumed += consumer_block
        leftovers.append(available - consumed)
    return {"events": events, "leftovers": leftovers, "consumed": consumed}


def coupled_resource_budget() -> dict:
    """E10-14: serial work, in-flight storage, and repeated-schedule energy."""
    costs = [4., 4., 25., 4., 4., 4.]
    period = 10.
    finish = 0.
    timeline = []
    for index, duration in enumerate(costs):
        arrival = index * period
        start = max(arrival, finish)
        finish = start + duration
        timeline.append({"arrival_ms": arrival, "start_ms": start,
                         "finish_ms": finish, "response_ms": finish-arrival})
    queue = simulate_deadline_queue(np.asarray(costs), period, 3)
    frame_kib = 4 * 160 * 4 / 1024
    state_kib = 400 + 200 + 60
    active_fraction = sum(costs) / (period * len(costs))
    average_power = 1.2 * active_fraction + .2 * (1-active_fraction)
    return {"frame_period_ms": period, "processing_ms": costs, "timeline": timeline,
            "offered_work_rtf": active_fraction, **queue,
            "frame_kib": frame_kib, "state_kib": state_kib,
            "peak_memory_kib": state_kib + 3 * frame_kib,
            "memory_limit_kib": 700., "sample_max_response_ms": max(row["response_ms"] for row in timeline),
            "active_power_w": 1.2, "idle_power_w": .2,
            "average_power_w_if_schedule_repeats": average_power,
            "estimated_runtime_h_if_schedule_repeats": 3.7 * 2 * .8 / average_power,
            "model_scope": "constructed six-frame serial schedule; no device timing or stable p99"}


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
    results["E10-07"] = {
        "scalar_samples": 160 * 4, "bytes_per_callback": 160 * 4 * 2,
        "duration_ms": 160 / 16000 * 1000, "bytes_per_second": 16000 * 4 * 2,
    }
    results["E10-08"] = {
        "input_block": 160, "sample_rate_hz": 16000,
        "common_multiple_samples": math.lcm(160, 240, 512),
        "consumers": {str(size): block_consumption(160, size, 48)
                      for size in (240, 512)},
    }
    correct_gain, reversed_gain = (1 + .9) / 2, (1 - .9) / 2
    results["E10-09"] = {
        "correct_target_gain": correct_gain, "reversed_target_gain": reversed_gain,
        "relative_target_level_db": 20 * math.log10(reversed_gain / correct_gain),
    }
    smoothing = {}
    for interval_ms, steps in ((10, 32), (32, 10)):
        coefficient = -math.expm1(-interval_ms / 100)
        remaining = 1.
        for _ in range(steps):
            remaining *= 1 - coefficient
        smoothing[str(interval_ms)] = {"coefficient": coefficient, "steps": steps,
                                      "remaining_error_fraction": remaining}
    results["E10-10"] = {"time_constant_ms": 100, "duration_ms": 320,
                         "updates": smoothing}
    direction = np.array([1., 0., 0.])
    displacements = np.array([[.0005, 0., 0.], [0., .0005, 0.]])
    projected = displacements @ direction
    results["E10-11"] = {
        "projected_displacements_m": projected.tolist(),
        "phase_errors_deg": (360 * 4000 * projected / 343).tolist(),
    }
    clock_times = np.arange(6.)
    known_step = np.where(clock_times >= 3, .001, 0.)
    stepped_delays = 100e-6 * clock_times + known_step
    biased_ppm, biased_offset = estimate_sro_ppm(clock_times, stepped_delays)
    corrected_ppm, corrected_offset = estimate_sro_ppm(clock_times, stepped_delays - known_step)
    results["E10-12"] = {
        "times_s": clock_times.tolist(), "delays_ms": (stepped_delays * 1000).tolist(),
        "whole_fit_ppm": biased_ppm, "whole_fit_offset_ms": biased_offset * 1000,
        "known_step_removed_ppm": corrected_ppm,
        "known_step_removed_offset_ms": corrected_offset * 1000,
    }
    results["E10-14"] = coupled_resource_budget()
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
    reference_words, errors = np.array([10, 90]), np.array([1, 45])
    results["E11-03"] = {
        "reference_words": reference_words.tolist(), "errors": errors.tolist(),
        "per_file_wer": (errors / reference_words).tolist(),
        "macro_average_wer": float(np.mean(errors / reference_words)),
        "pooled_wer": float(errors.sum() / reference_words.sum()),
    }
    latencies = np.array([*range(1, 20), 100.])
    probabilities = (.5, .95, .99)
    ranks = [math.ceil(p * len(latencies)) for p in probabilities]
    results["E11-04"] = {
        "latencies_ms": latencies.tolist(), "nearest_ranks": ranks,
        "nearest_rank_ms": [float(np.sort(latencies)[rank - 1]) for rank in ranks],
        "linear_interpolation_ms": np.quantile(latencies, probabilities, method="linear").tolist(),
    }
    ring_candidates = []
    for microphones in (4, 6):
        nearest_spacing = 2 * .04 * math.sin(math.pi / microphones)
        adjacent_boundary = 343 / (2 * nearest_spacing)
        ring_candidates.append({"microphones": microphones,
                                "nearest_neighbor_spacing_m": nearest_spacing,
                                "adjacent_pair_boundary_hz": adjacent_boundary,
                                "passes_adjacent_4khz_screen": adjacent_boundary >= 4000})
    results["E11-05"] = {"ring_radius_m": .04, "screen_frequency_hz": 4000,
                         "playback_aec_required": True, "candidates": ring_candidates,
                         "scope": "adjacent-pair spacing screen, not global circular-array alias guarantee"}
    meeting_cases = []
    for name, errors, latency, streams in (("single_beam", 25, 90, 1),
                                           ("beam_plus_wpe", 20, 115, 1),
                                           ("two_stream_separator", None, 180, 2)):
        meeting_cases.append({"name": name, "errors_per_100_reference_words": errors,
                              "wer": errors / 100 if errors is not None else None,
                              "score_protocol": ("same_reference_single_output_wer"
                                                 if streams == 1 else
                                                 "undefined_until_multistream_protocol_is_fixed"),
                              "latency_ms": latency,
                              "output_streams": streams,
                              "passes_hard_limits": latency <= 150 and streams == 1})
    results["E11-06"] = {"reference_words": 100, "latency_limit_ms": 150,
                         "required_output_streams": 1, "candidates": meeting_cases,
                         "scope": "constructed single-output word counts; no comparable multistream score or population ranking"}
    results["E11-07"] = {"relative_sro_ppm": 80., "allowed_drift_ms": .1,
                         "initial_offset_ms": 2., "candidates": [
                             {"realignment_interval_s": interval,
                              "drift_ms": 80e-6 * interval * 1000,
                              "passes_drift_limit": 80e-6 * interval * 1000 <= .1}
                             for interval in (10., 1.)],
                         "maximum_interval_s": .0001 / 80e-6,
                         "scope": "known constant SRO, no dropped samples or timestamp noise"}
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
    noise_covariance = np.diag([4., 1.])
    whitening = np.diag([.5, 1.])
    steering = np.ones(2)
    whitened_covariance = whitening @ noise_covariance @ whitening.conj().T
    cross_spectrum = (2 + 0j) * (1 + 0j).conjugate()
    results["E12-04"] = {
        "noise_covariance": noise_covariance.tolist(),
        "whitening_matrix": whitening.tolist(),
        "whitened_covariance": whitened_covariance.tolist(),
        "whitened_steering": (whitening @ steering).tolist(),
        "raw_eigenvalues": np.linalg.eigvalsh(noise_covariance).tolist(),
        "whitened_eigenvalues": np.linalg.eigvalsh(whitened_covariance).tolist(),
        "one_nonzero_cross_spectrum_phat": float((cross_spectrum / abs(cross_spectrum)).real),
        "theoretical_noise_cross_spectrum": 0.,
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
