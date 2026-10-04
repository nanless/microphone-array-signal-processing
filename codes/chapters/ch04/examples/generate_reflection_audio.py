"""Generate or strictly read-check four E04-24 coherent direct/reflection mathematical WAVs.

--check reconstructs all signals, current source hashes and numerical scores
in memory, and never repairs assets. Finite preflight checks do not eliminate
filesystem races or guarantee crash durability. Run from the repository root.
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
from codes.chapters.ch04.core.reflection_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, SCORING_INTERVAL,
    analytic_measurements, parameters, virtual_covariance_control,
    generate_signals, measure_signal,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / 'codes/chapters/ch04/reflection_audio'
SOURCE_PATHS = (
    'codes/chapters/ch04/core/reflection_audio.py',
    'codes/chapters/ch04/examples/generate_reflection_audio.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = set(FILE_NAMES.values()) | {'MANIFEST.json'}
# Fixed asset contract, checked independently of the descriptive function.
# A source edit must deliberately update both the signal and this contract;
# accepting a self-consistent but false declaration would corrupt provenance.
REQUIRED_PARAMETERS = {'exercise_id': 'E04-24',
 'sample_rate_hz': 16000,
 'source_samples': 32000,
 'samples_per_channel': 32012,
 'frequency_hz': 4000.0,
 'source_amplitude': 0.1,
 'sound_speed_m_s': 343.0,
 'microphone_spacing_m': 0.042875,
 'source_direction_convention': 'unit vector from array toward source; tau=-d*sin(theta)/c',
 'direct_angle_deg': 0.0,
 'reflection_angle_deg': 30.0,
 'direct_delay_samples': [8, 8],
 'reflection_delay_samples': [12, 11],
 'common_export_gain': 1.0,
 'fade_samples': 320,
 'fade': 'squared sine, endpoints included, before all delays',
 'waveform_model': 'reference=F; direct=[D8 F,D8 F]; reflection=[D12 F,D11 F]; '
                   'mixed=direct+reflection',
 'delay_implementation': 'integer causal delay, zero extension, complete 12-sample tail',
 'channel_order': ['microphone_0', 'microphone_1'],
 'scoring_interval_samples': [2400, 29600],
 'samples_in_scoring_window': 27200,
 'scoring_cycles': 6800,
 'phasor_convention': 'cos coefficient minus j*sin coefficient; absolute sample clock',
 'normalized_covariance': '(z/0.1)(z/0.1)^H; single known phasor outer product',
 'virtual_covariance_diagonal': 0.1,
 'virtual_covariance_scope': 'model control only; no noise added to any WAV',
 'randomness': 'none',
 'noise': 'none'}

def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata(payload: bytes) -> dict:
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        return {'sample_rate_hz': stream.getframerate(), 'channels': stream.getnchannels(),
                'samples_per_channel': stream.getnframes(),
                'sample_width_bytes': stream.getsampwidth(), 'compression': stream.getcomptype()}


def _integer_measurements(payload: bytes) -> dict:
    """Read actual integer PCM; include each channel and all-channel denominator."""
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        channels = stream.getnchannels()
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
    pcm = pcm.reshape(-1, channels).astype(np.int64)
    start, stop = SCORING_INTERVAL
    energies = [int(np.sum(pcm[start:stop, channel]**2)) for channel in range(channels)]
    denominator = (stop-start)*32768**2
    return {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
            'channels': channels, 'integer_squared_sum_E_per_channel': energies,
            'integer_denominator_D_per_channel': denominator,
            'mean_square_per_channel': [energy/denominator for energy in energies],
            'integer_squared_sum_E_all_channels': sum(energies),
            'integer_denominator_D_all_channels': channels*denominator,
            'mean_square_all_channels': sum(energies)/(channels*denominator)}


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    """Prepare the complete current source-bound fixture without writing."""
    true_parameters = parameters()
    if (not same_metadata(true_parameters, REQUIRED_PARAMETERS)):
        raise ValueError('reflection true parameters differ from the fixed asset contract')
    signals = generate_signals()
    if set(signals) != set(FILE_NAMES):
        raise ValueError('expected one reference and three stereo observations')
    payloads, files, samples = {}, {}, {}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        payload = pcm16_bytes(signal, SAMPLE_RATE)
        metadata = _metadata(payload)
        expected = {'sample_rate_hz': SAMPLE_RATE, 'channels': 1 if name == 'reference' else 2,
                    'samples_per_channel': SAMPLES, 'sample_width_bytes': 2, 'compression': 'NONE'}
        if metadata != expected:
            raise ValueError('unexpected reflection PCM format: '+filename)
        _, pcm = read_pcm16(payload)
        error = float(np.max(np.abs(pcm-signal)))
        if error > .5/32768 + 1e-15:
            raise ValueError('reflection PCM exceeded the common-gain rounding bound')
        analytic, floating, actual = analytic_measurements(name), measure_signal(signal), measure_signal(pcm)
        payloads[filename] = payload
        files[filename] = {**metadata, 'sha256': _sha(payload),
                           'peak': float(np.max(np.abs(pcm))),
                           'quantization_max_abs_error': error}
        samples[name] = {'file': filename, 'analytic': analytic,
                         'float_measurements': floating, 'pcm_measurements': actual,
                         'pcm_integer_measurements': _integer_measurements(payload)}
    manifest = {
        'schema_version': 1, 'exercise_id': 'E04-24', 'sample_rate_hz': SAMPLE_RATE,
        'samples_per_channel': SAMPLES, 'common_export_gain': 1.0,
        'origin': 'original deterministic coherent direct/reflection mathematical synthesis',
        'source_sha256': {path: _sha(validate_parent_chain(ROOT/path).read_bytes()) for path in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'system': platform.system(), 'machine': platform.machine()},
        'parameters': true_parameters,
        'pcm': {'format': 'signed little-endian PCM16', 'scale': 32768,
                'rounding': 'nearest, ties to even', 'dither': 'none', 'clipping': 'rejected',
                'max_rounding_error_bound': .5/32768},
        'files': files, 'samples': samples,
        'virtual_covariance_control': virtual_covariance_control(),
        'limits': LIMITS,
        'listening': 'Start at low volume; never autoplay. Stereo playback is not direct-sound or localization evidence.',
    }
    # Serialize now, before any filesystem mutation, to reject nonfinite metadata.
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return payloads, manifest


def check_assets(output: Path = DEFAULT_OUTPUT) -> dict:
    """Validate all five ordinary files and replay actual PCM without writing."""
    output = validate_asset_directory(output, MEMBERS, check=True)
    manifest = strict_json_loads((output/'MANIFEST.json').read_bytes())
    payloads, expected = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('reflection manifest differs from current sources, true parameter types or scores')
    for name, filename in FILE_NAMES.items():
        payload = (output/filename).read_bytes()
        metadata = _metadata(payload)
        required = {'sample_rate_hz': SAMPLE_RATE, 'channels': 1 if name == 'reference' else 2,
                    'samples_per_channel': SAMPLES, 'sample_width_bytes': 2, 'compression': 'NONE'}
        if metadata != required or _sha(payload) != manifest['files'][filename]['sha256']:
            raise ValueError('reflection WAV format or SHA differs: '+filename)
        _, actual = read_pcm16(payload)
        recorded = manifest['samples'][name]
        if (not same_metadata(measure_signal(actual), recorded['pcm_measurements'])
                or not same_metadata(_integer_measurements(payload), recorded['pcm_integer_measurements'])):
            raise ValueError('reflection actual PCM scores differ: '+filename)
        if payload != payloads[filename]:
            raise ValueError('reflection waveform differs from the complete source replay: '+filename)
    return manifest


def generate_assets(output: Path = DEFAULT_OUTPUT) -> dict:
    output = validate_asset_directory(output, MEMBERS, check=False)
    payloads, manifest = prepare_assets()
    output.mkdir(parents=True, exist_ok=True)
    # Recheck finite member/parent preconditions before replacing local assets.
    validate_asset_directory(output, MEMBERS, check=False)
    for filename, payload in payloads.items():
        (output/filename).write_bytes(payload)
    (output/'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return check_assets(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        manifest = check_assets(args.output) if args.check else generate_assets(args.output)
    except (OSError, ValueError, TypeError, wave.Error) as error:
        parser.exit(1, f'Reflection audio verification failed: {error}\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
