"""Generate or strictly read-check three E05-23 known-component phase WAVs.

Finite preflight checks do not eliminate races or guarantee crash durability.
--check reads actual PCM and reconstructs the fixture only in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import wave
import numpy as np

from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch00.io_contracts import (
    same_metadata, strict_json_loads, validate_asset_directory, validate_parent_chain,
)
from codes.chapters.ch05.core.phase_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, SCORING_INTERVAL,
    analytic_measurements, parameters, generate_signals, measure_signal,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch05/phase_audio'
SOURCE_PATHS = (
    'codes/chapters/ch05/core/phase_audio.py',
    'codes/chapters/ch05/examples/generate_phase_audio.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = set(FILE_NAMES.values()) | {'MANIFEST.json'}


# Fixed values and exact types, independent of the descriptive function at runtime.
REQUIRED_PARAMETERS = {'exercise_id': 'E05-23',
 'sample_rate_hz': 16000,
 'samples_per_channel': 32000,
 'frequencies_hz': [500.0, 1500.0],
 'source_amplitude_per_frequency': 0.1,
 'source_duration_s': 2.0,
 'fade_samples': 320,
 'fade': 'squared sine, endpoints included, common envelope',
 'common_export_gain': 1.0,
 'channel_order': ['mono_known_component_output'],
 'target_channel_model': 'two identical target channels for each known frequency component',
 'base_weights_real_imag': [[0.5, 0.0], [0.5, 0.0]],
 'high_weight_multiplier_real_imag': {'reference': [1.0, 0.0],
                                      'flip': [-1.0, 0.0],
                                      'quadrature': [0.0, 1.0]},
 'output_model': 'A[n]*0.1*(cos(500Hz)+cos(1500Hz)); high output -cos for flip, +sin for '
                 'quadrature',
 'application_convention': 'y=w^H*x; output multiplier is conjugate of weight multiplier',
 'generation_model': 'known real frequency components resynthesized before common envelope; no '
                     'STFT',
 'propagation_delay_samples': 0,
 'algorithm_delay_samples': 0,
 'scoring_interval_samples': [2400, 29600],
 'samples_in_scoring_window': 27200,
 'scoring_cycles_per_frequency': [850, 2550],
 'phasor_convention': 'cos coefficient minus j*sin coefficient; absolute sample clock',
 'alignment': 'original sample clock and gain; no fitted correction',
 'noise': 'none',
 'randomness': 'none'}

def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata(payload: bytes) -> dict:
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        return {'sample_rate_hz': stream.getframerate(), 'channels': stream.getnchannels(),
                'samples_per_channel': stream.getnframes(),
                'sample_width_bytes': stream.getsampwidth(), 'compression': stream.getcomptype()}


def _integers(payload: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), '<i2').reshape(-1, stream.getnchannels()).astype(np.int64)


def _integer_measurements(payload: bytes, reference: bytes) -> dict:
    start, stop = SCORING_INTERVAL
    pcm, ref = _integers(payload)[start:stop], _integers(reference)[start:stop]
    channels = pcm.shape[1]
    energies = [int(pcm[:, c]@pcm[:, c]) for c in range(channels)]
    errors = [int((pcm[:, c]-ref[:, 0])@(pcm[:, c]-ref[:, 0])) for c in range(channels)]
    ref_energy = int(ref[:, 0]@ref[:, 0])
    if ref_energy <= 0:
        raise ValueError('positive actual PCM reference energy required')
    denominator = (stop-start)*32768**2
    return {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
            'channels': channels, 'integer_squared_sum_E_per_channel': energies,
            'integer_denominator_D_per_channel': denominator,
            'mean_square_per_channel': [e/denominator for e in energies],
            'integer_squared_sum_E_all_channels': sum(energies),
            'integer_denominator_D_all_channels': channels*denominator,
            'mean_square_all_channels': sum(energies)/(channels*denominator),
            'integer_reference_squared_sum': ref_energy,
            'integer_error_squared_sum_per_channel': errors,
            'total_reference_mse_per_channel': [e/denominator for e in errors],
            'normalized_squared_reference_error_per_channel': [e/ref_energy for e in errors]}


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    true_parameters = parameters()
    if not same_metadata(true_parameters, REQUIRED_PARAMETERS):
        raise ValueError('phase true parameters differ from the fixed asset contract')
    signals = generate_signals()
    if set(signals) != set(FILE_NAMES):
        raise ValueError('expected reference, flip and quadrature')
    payloads, decoded, files, samples = {}, {}, {}, {}
    expected_format = {'sample_rate_hz': SAMPLE_RATE, 'channels': 1,
                       'samples_per_channel': SAMPLES, 'sample_width_bytes': 2, 'compression': 'NONE'}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        payload = pcm16_bytes(signal, SAMPLE_RATE)
        metadata = _metadata(payload)
        if metadata != expected_format:
            raise ValueError('unexpected phase PCM format: '+filename)
        _, actual = read_pcm16(payload)
        error = float(np.max(abs(actual-signal)))
        if error > .5/32768+1e-15:
            raise ValueError('phase quantization exceeded the common gain rounding bound')
        payloads[filename], decoded[name] = payload, actual
        files[filename] = {**metadata, 'sha256': _sha(payload),
                           'quantization_max_abs_error': error, 'peak': float(np.max(abs(actual)))}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        samples[name] = {'file': filename, 'analytic': analytic_measurements(name),
                         'float_measurements': measure_signal(signal, signals['reference']),
                         'pcm_measurements': measure_signal(decoded[name], decoded['reference']),
                         'pcm_integer_measurements': _integer_measurements(payloads[filename], payloads[FILE_NAMES['reference']])}
    manifest = {'schema_version': 1, 'exercise_id': 'E05-23',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'common_export_gain': 1.0,
                'origin': 'original deterministic known two-tone phase resynthesis',
                'source_sha256': {p: _sha(validate_parent_chain(ROOT/p).read_bytes()) for p in SOURCE_PATHS},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'parameters': true_parameters, 'files': files, 'samples': samples,
                'pcm': {'format': 'signed little-endian PCM16', 'scale': 32768,
                        'rounding': 'nearest, ties to even', 'dither': 'none', 'clipping': 'rejected',
                        'max_rounding_error_bound': .5/32768},
                'limits': LIMITS, 'listening': 'Start at low volume; never autoplay. Playback is not a formal listening study.'}
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return payloads, manifest


def check_assets(output: Path = DEFAULT_OUTPUT) -> dict:
    output = validate_asset_directory(output, MEMBERS, check=True)
    manifest = strict_json_loads((output/'MANIFEST.json').read_bytes())
    expected_payloads, expected = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('phase manifest differs from current sources, true parameter types or scores')
    actual = {name: (output/filename).read_bytes() for name, filename in FILE_NAMES.items()}
    _, reference = read_pcm16(actual['reference'])
    for name, filename in FILE_NAMES.items():
        payload = actual[name]
        if _metadata(payload) != {k: manifest['files'][filename][k] for k in _metadata(payload)} or _sha(payload) != manifest['files'][filename]['sha256']:
            raise ValueError('phase PCM format or SHA mismatch: '+filename)
        _, pcm = read_pcm16(payload)
        recorded = manifest['samples'][name]
        if (not same_metadata(measure_signal(pcm, reference), recorded['pcm_measurements'])
                or not same_metadata(_integer_measurements(payload, actual['reference']), recorded['pcm_integer_measurements'])):
            raise ValueError('phase actual PCM scores differ: '+filename)
        if payload != expected_payloads[filename]:
            raise ValueError('phase waveform differs from the complete source replay: '+filename)
    return manifest


def generate_assets(output: Path = DEFAULT_OUTPUT) -> dict:
    output = validate_asset_directory(output, MEMBERS, check=False)
    payloads, manifest = prepare_assets()
    output.mkdir(parents=True, exist_ok=True)
    validate_asset_directory(output, MEMBERS, check=False)
    for filename, payload in payloads.items():
        (output/filename).write_bytes(payload)
    (output/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return check_assets(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        manifest = check_assets(args.output) if args.check else generate_assets(args.output)
    except (OSError, ValueError, TypeError, wave.Error) as error:
        parser.exit(1, f'Phase audio verification failed: {error}\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
