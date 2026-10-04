"""Generate/check E02-19 digital sweep, complete/noisy/cut response controls.

The excitation has 32000 samples; the three responses have 32160 samples.
All are mathematical mono PCM16 signals at gain 1. --check only replays and
compares current-source assets; it never repairs or writes. This is not a
measured room, a Farina nonlinear measurement chain or a device benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import wave

import numpy as np

from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch00.io_contracts import (
    same_metadata, strict_json_loads, validate_asset_directory, validate_parent_chain,
)
from codes.chapters.ch02.core.deconvolution import (
    build_sweep_cases, regularized_inverse, score_impulse_response,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIRECTORY = ROOT / 'codes/chapters/ch02/sweep_audio'
SOURCE_PATHS = (
    'codes/chapters/ch02/core/deconvolution.py',
    'codes/chapters/ch02/examples/generate_sweep_audio.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
WAV_LENGTHS = {'sweep_source.wav': 32000, 'sweep_complete.wav': 32160,
               'sweep_noisy.wav': 32160, 'sweep_cut.wav': 32160}
MEMBERS = set(WAV_LENGTHS) | {'MANIFEST.json'}
N_FFT = 65536
RHO_VALUES = (0., 1e-8, 1e-6, 1e-4)


def _pcm_record(payload):
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        metadata = {'sample_rate_hz': stream.getframerate(),
                    'channels': stream.getnchannels(),
                    'samples_per_channel': stream.getnframes(),
                    'sample_width_bytes': stream.getsampwidth(),
                    'compression': stream.getcomptype()}
        integers = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.int64)
    # At these fixed lengths even the largest PCM16 square sum fits int64.
    energy = int(np.sum(integers * integers))
    denominator = len(integers) * 32768**2
    return metadata, {'squared_sum_E': energy, 'integer_denominator_D': denominator,
                      'samples': len(integers), 'mean_square': energy / denominator}


def _inverse_rows(source, signals, reference):
    """Keep every estimated IR sample; oracle support is a separate diagnostic."""
    reference_energy = float(reference @ reference)
    if not reference_energy > 0:
        raise ValueError('The controlled path must have positive reference energy')
    source_spectrum = np.fft.rfft(source, n=N_FFT)
    power = np.abs(source_spectrum)**2
    maximum_power = float(np.max(power))
    if not np.all(np.isfinite(power)) or not maximum_power > 0:
        raise ValueError('Controlled excitation spectrum is invalid')
    rows = []
    for key in ('sweep_complete', 'sweep_noisy', 'sweep_cut'):
        for rho in RHO_VALUES:
            epsilon = rho * maximum_power
            result = regularized_inverse(source, signals[key], n_fft=N_FFT,
                                         regularization=epsilon)
            estimate = np.asarray(result['impulse_response'])
            if estimate.shape != (N_FFT,) or not np.all(np.isfinite(estimate)):
                raise ValueError('Inverse must retain the finite complete FFT-length IR')
            score = score_impulse_response(estimate, reference, support_length=len(reference))
            full_error = score['error_energy']
            support_error = score['support_error_energy']
            outside = score['outside_support_energy']
            rows.append({'case': key.removeprefix('sweep_'), 'rho': rho,
                         'absolute_regularization': epsilon,
                         'reference_energy': reference_energy,
                         'full_ir_error_energy': full_error,
                         'full_ir_relative_error_energy': score['full_ir_relative_error_energy'],
                         'true_support_error_energy': support_error,
                         'true_support_relative_error_energy': score['true_support_relative_error_energy'],
                         'outside_support_energy': outside,
                         'outside_support_relative_energy': outside / reference_energy,
                         'max_ir_error': score['max_ir_error'],
                         'direct_estimate': float(estimate[0]),
                         'echo_estimate': float(estimate[160])})
    return {'source_minimum_spectral_power': float(np.min(power)),
            'source_maximum_spectral_power': maximum_power, 'rows': rows}


def expected_assets():
    case = build_sweep_cases()
    required_parameters = {
        'sample_rate_hz': 16000, 'source_samples': 32000,
        'response_samples': 32160, 'fft_length': N_FFT,
        'source_amplitude': .2, 'start_frequency_hz': 100.,
        'end_frequency_hz': 6000., 'phase_duration_s': 2.,
        'fade_samples': 160, 'direct_gain': .8, 'reflection_gain': .25,
        'reflection_delay_samples': 160, 'noise_seed': 2026100402,
        'noise_standard_deviation': .003, 'common_export_gain': 1.,
        'regularization_ratios': list(RHO_VALUES),
        'noise_position': 'after known digital FIR convolution',
    }
    actual_parameters = case.get('parameters', {})
    if not same_metadata({key: actual_parameters.get(key)
                          for key in required_parameters}, required_parameters):
        raise ValueError('Controlled case parameters differ from the declared asset contract')
    signals = {'sweep_source': np.asarray(case['source']),
               **{key: np.asarray(case['signals'][key]) for key in
                  ('sweep_complete', 'sweep_noisy', 'sweep_cut')}}
    reference = np.asarray(case['impulse_response'])
    expected_reference = np.zeros(161)
    expected_reference[[0, 160]] = [.8, .25]
    if not np.array_equal(reference, expected_reference):
        raise ValueError('Controlled FIR differs from the declared two-tap path')
    payloads = {}
    decoded = {}
    records = {}
    for filename, count in WAV_LENGTHS.items():
        signal = signals[Path(filename).stem]
        if signal.shape != (count,) or not np.all(np.isfinite(signal)):
            raise ValueError('Controlled waveform shape or finiteness differs: ' + filename)
        payload = pcm16_bytes(signal)
        metadata, integer_score = _pcm_record(payload)
        required = {'sample_rate_hz': 16000, 'channels': 1,
                    'samples_per_channel': count, 'sample_width_bytes': 2,
                    'compression': 'NONE'}
        if metadata != required:
            raise ValueError('Controlled WAV format differs: ' + filename)
        _, waveform = read_pcm16(payload)
        decoded[Path(filename).stem] = waveform[0]
        error = float(np.max(np.abs(waveform[0] - signal)))
        if error > .5 / 32768 + 1e-15:
            raise ValueError('PCM quantization exceeds half a step')
        payloads[filename] = payload
        records[filename] = {**metadata, 'sha256': hashlib.sha256(payload).hexdigest(),
                             'peak': float(np.max(np.abs(waveform))),
                             'quantization_max_abs_error': error,
                             'pcm_integer_measurements': integer_score}
    manifest = {
        'schema_version': 1, 'exercise_id': 'E02-19', 'sample_rate_hz': 16000,
        'common_export_gain': 1.,
        'source_sha256': {name: hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'byteorder': sys.byteorder, 'platform': platform.platform()},
        'parameters': {'source_samples': 32000, 'response_samples': 32160,
                       'source_duration_s': 2., 'response_duration_s': 32160 / 16000,
                       'design_frequency_range_hz': [100., 6000.], 'source_amplitude': .2,
                       'fade_samples': 160,
                       'fade': 'min(1,n/160,(31999-n)/160), n=0..31999',
                       'last_instantaneous_frequency_hz': float(100 * np.exp(31999 / 32000 * np.log(60))),
                       'known_path': {'length': 161, 'nonzero_taps': {'0': .8, '160': .25}},
                       'noise_location': 'additive after the path, before PCM quantization',
                       'noise_standard_deviation': .003, 'noise_seed': 2026100402,
                       'cut': 'zero response samples n>=32000; retain a 32160-sample record',
                       'fft_length': N_FFT, 'rho_values': list(RHO_VALUES),
                       'regularization': 'epsilon=rho*max(abs(rfft(excitation,65536))**2), separately per domain',
                       'initial_history': 'zero', 'complete_response_tail_samples': 160},
        'scoring': {'reference': 'known artificial FIR padded to 65536; no gain or delay fit',
                    'full_interval': [0, N_FFT], 'true_support_interval': [0, 161],
                    'outside_support_interval': [161, N_FFT],
                    'reference_energy': float(reference @ reference),
                    'true_support': 'oracle scoring diagnostic, never used to truncate the estimator',
                    'float_domain': 'float excitation and float response before export',
                    'pcm_domain': 'actual PCM16-decoded excitation and independently quantized response; estimated IR remains float'},
        'quantization': {'format': 'signed little-endian PCM16', 'scale': 32768,
                         'rounding': 'nearest, ties to even', 'clipping': 'rejected',
                         'max_rounding_error_bound': .5 / 32768,
                         'order': 'generate float path and output noise, then quantize each waveform independently'},
        'limits': 'One known digital LTI path and one fixed noise seed, no measured room, natural speech, '
                  'Farina weighted inverse-sweep/harmonic separation, calibration, blind parameter selection or industrial performance. '
                  'Design instantaneous frequencies do not imply a strictly band-limited finite DFT. '
                  'All 65536 estimated taps are scored; small oracle-support error cannot conceal outside-support error.',
        'listening': 'Start at low volume; compare excitation and responses at common gain 1, without autoplay. '
                     'Do not judge IR-estimation accuracy from response listening. No formal listening test.',
        'files': records,
        'float_measurements': _inverse_rows(signals['sweep_source'], signals, reference),
        'pcm_measurements': _inverse_rows(decoded['sweep_source'], decoded, reference),
    }
    return payloads, manifest


def check_assets(directory=DEFAULT_DIRECTORY):
    directory = validate_asset_directory(directory, MEMBERS, check=True)
    payloads, expected = expected_assets()
    actual = strict_json_loads((directory / 'MANIFEST.json').read_bytes())
    if not same_metadata(actual, expected):
        raise ValueError('Sweep manifest differs from current sources, configuration, environment or scores')
    for name, payload in payloads.items():
        actual_payload = (directory / name).read_bytes()
        if actual_payload != payload:
            raise ValueError('Stale or modified sweep WAV: ' + name)
        _pcm_record(actual_payload)
    return actual


def generate_assets(directory=DEFAULT_DIRECTORY):
    directory = validate_asset_directory(directory, MEMBERS, check=False)
    payloads, manifest = expected_assets()
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    (directory / 'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return check_assets(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    try:
        manifest = check_assets(args.output) if args.check else generate_assets(args.output)
    except (OSError, ValueError, TypeError, wave.Error) as error:
        parser.exit(1, f'Sweep verification failed: {error}\n')
    print(f"{'Checked' if args.check else 'Generated and checked'} {len(manifest['files'])} sweep WAVs")


if __name__ == '__main__':
    main()
