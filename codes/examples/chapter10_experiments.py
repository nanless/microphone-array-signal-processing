"""E10-18..27: deterministic engineering calculations, not hardware benchmarks.

Run ``python -m codes.examples.chapter10_experiments``. Every answer is computed
from the stated inputs; no external model, network or recording is required.
"""
from __future__ import annotations
import json
import math
import numpy as np
from codes.array_tutorial.engineering import q15_dot, resample_sro_to_reference
from codes.array_tutorial.noise_suppression import power_spectral_subtraction
from codes.array_tutorial.audio_samples import agc_blocks_case
from codes.examples.sro_closed_loop_demo import StatefulLinearClockCorrector


def rtf_anchor():
    sample_rate, window, hop, service_ms = 16000, 512, 128, 4.
    audio_ms, services_ms = np.array([10., 100.]), np.array([1., 90.])
    return {'sample_rate_hz': sample_rate, 'window_samples': window, 'hop_samples': hop,
            'service_ms': service_ms, 'hop_rtf': service_ms / (1000 * hop / sample_rate),
            'wrong_window_rtf': service_ms / (1000 * window / sample_rate),
            'audio_ms': audio_ms.tolist(), 'services_ms': services_ms.tolist(),
            'aggregate_rtf': float(services_ms.sum() / audio_ms.sum()),
            'unweighted_mean_rtf': float(np.mean(services_ms / audio_ms))}


def subtraction_anchor():
    clean, noise = 1., np.array([1., -1.])
    observation = clean + noise
    raw_power = observation**2 - np.mean(noise**2)
    # The first bin-frame is a known noise-only observation; score the last two.
    enhanced, estimate = power_spectral_subtraction(np.array([[1., *observation]]),
                                                   np.array([0]), floor_ratio=0.)
    return {'clean': clean, 'noise': noise.tolist(), 'observation': observation.tolist(),
            'mean_cross_term': float(np.mean(2 * clean * noise)),
            'noise_power': float(estimate[0]), 'unclipped_power': raw_power.tolist(),
            'unclipped_mean': float(raw_power.mean()),
            'clipped_power': (np.abs(enhanced[0, 1:])**2).tolist(),
            'clipped_mean': float(np.mean(np.abs(enhanced[0, 1:])**2))}


def q15_anchor():
    x, h = np.array([1, 1], dtype=np.int16), np.array([16384, 16384], dtype=np.int16)
    products = x.astype(np.int64) * h
    worst = np.array([-32768, -32768], dtype=np.int16)
    return {'x': x.tolist(), 'h': h.tolist(), 'products': products.tolist(),
            'round_once': int(q15_dot(x, h)),
            'round_each': int(np.rint(products / 32768).sum()),
            'two_worst_products_sum': sum(int(v)**2 for v in worst),
            'int32_max': int(np.iinfo(np.int32).max),
            'saturated_result': int(q15_dot(worst, worst)),
            'safe_int64_length_bound': (2**63 - 1) // 2**30}


def critical_path_anchor():
    capture, a, b, c = 32., 4., 7., 3.
    finish_a, finish_b = capture + a, capture + b
    return {'capture_ms': capture, 'service_ms': {'A': a, 'B': b, 'C': c},
            'parallel_finish_a_ms': finish_a, 'parallel_finish_b_ms': finish_b,
            'parallel_finish_c_ms': max(finish_a, finish_b) + c,
            'single_worker_finish_c_ms': capture + a + b + c}


def phase_drift_anchor():
    epsilon, time, fs = 100e-6, 1., 16000
    exact_delay = time * epsilon / (1 + epsilon)
    approximate_delay = time * epsilon
    result = {}
    for frequency in (1000, 4000):
        exact = abs(math.cos(math.pi * frequency * exact_delay))
        approximate = abs(math.cos(math.pi * frequency * approximate_delay))
        result[str(frequency)] = {'exact_amplitude': exact, 'exact_db': 20 * math.log10(exact),
                                 'first_order_amplitude': approximate,
                                 'first_order_db': 20 * math.log10(approximate)}
    return {'sample_rate_hz': fs, 'epsilon': epsilon, 'reference_time_s': time,
            'exact_delay_s': exact_delay, 'first_order_delay_s': approximate_delay,
            'frequencies_hz': result}


def memory_anchor():
    persistent, scratch = 512, [128, 96]
    buffered = 3 * 4 * 160 * np.dtype(np.float32).itemsize / 1024
    return {'persistent_kib': persistent, 'scratch_kib': scratch,
            'serial_peak_kib': persistent + max(scratch),
            'parallel_peak_kib': persistent + sum(scratch), 'in_flight_kib': buffered,
            'serial_with_buffers_kib': persistent + max(scratch) + buffered,
            'parallel_with_buffers_kib': persistent + sum(scratch) + buffered}


def streaming_anchor():
    values = [n / 3 for n in range(6)]
    result = {}
    for name, chunks in [('complete', [[0, 1], [2, 3, 4, 5]]),
                         ('missing_index_2', [[0, 1], [3, 4, 5]])]:
        state = StatefulLinearClockCorrector(2., 3., 0., 4)
        output = []
        per_push = []
        for indices in chunks:
            emitted = state.push(indices, [values[i] for i in indices])
            output.extend(emitted)
            per_push.append(emitted)
        result[name] = {'chunks': chunks, 'outputs': output, 'per_push_outputs': per_push}
    return {'reference_rate_hz': 2, 'device_rate_hz': 3, 'device_start_s': 0,
            'device_values': values, 'device_capture_times_s': values, **result}


def alias_anchor():
    input_rate, output_rate, frequency, count = 8000, 4000, 3000, 17
    signal = np.cos(2 * np.pi * frequency * np.arange(count) / input_rate)
    output = resample_sro_to_reference(signal, (input_rate / output_rate - 1) * 1e6)
    alias = np.cos(2 * np.pi * 1000 * np.arange(len(output)) / output_rate)
    return {'input_rate_hz': input_rate, 'output_rate_hz': output_rate,
            'frequency_hz': frequency, 'input_samples': count,
            'output': output.tolist(), 'alias_frequency_hz': 1000,
            'max_alias_error': float(np.max(np.abs(output - alias))),
            'anti_alias_filter_run': False}


def telemetry_anchor():
    indices, timestamps = [0, 1, 3], [0, 10_000_000, 30_000_000]
    hop, sample_rate, channels = 160, 16000, 4
    missing_frames = sum(max(b - a - 1, 0) for a, b in zip(indices, indices[1:]))
    return {'frame_indices': indices, 'first_sample_timestamp_ns': timestamps,
            'expected_first_sample_timestamp_ns': [i * hop * 10**9 // sample_rate for i in indices],
            'hop_samples': hop, 'sample_rate_hz': sample_rate, 'channels': channels,
            'missing_time_samples': missing_frames * hop,
            'missing_scalar_values': missing_frames * hop * channels,
            'in_flight_blocks': 3, 'processing_blocks': 1, 'waiting_blocks': 3 - 1}


def agc_anchor():
    case = agc_blocks_case()
    return {key: case[key] for key in ('parameters', 'float_analysis', 'pcm_analysis', 'limits')} | {
        'common_export_gain': case['export_gain_override'],
        'files': [name + '.wav' for name in case['signals']],
        'availability': {name: {'first_block_end_s': record['blocks'][0]['available_time_s'],
                               'attack_alpha': record['attack_alpha'],
                               'release_alpha': record['release_alpha']}
                         for name, record in case['block_records'].items()}}


def run_experiments():
    functions = [rtf_anchor, subtraction_anchor, q15_anchor, critical_path_anchor,
                 phase_drift_anchor, memory_anchor, streaming_anchor, alias_anchor,
                 telemetry_anchor, agc_anchor]
    return {f'E10-{number:02d}': function() for number, function in enumerate(functions, 18)}


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
