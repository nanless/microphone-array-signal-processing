"""Generate or strictly read-check six E03-18 known-direction mathematical WAVs.

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
from codes.chapters.ch03.core.baseline_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, SCORING_INTERVAL,
    analytic_measurements, baseline_parameters, calibration_results,
    generate_signals, measure_signal,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / 'codes/chapters/ch03/baseline_audio'
SOURCE_PATHS = (
    'codes/chapters/ch03/core/baseline_audio.py',
    'codes/chapters/ch03/examples/generate_baseline_audio.py',
    'codes/chapters/ch03/core/baseline_calibration.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = set(FILE_NAMES.values()) | {'MANIFEST.json'}
# Fixed asset contract, checked independently of the descriptive function.
# A source edit must deliberately update both the signal and this contract;
# accepting a self-consistent but false declaration would corrupt provenance.
REQUIRED_PARAMETERS = {
    'exercise_id': 'E03-18', 'sample_rate_hz': 16000,
    'samples_per_channel': 32000, 'duration_s': 2.0,
    'frequency_hz': 500.0, 'source_amplitude': .2,
    'source_active_interval_s': [.1, 1.9], 'linear_fade_duration_s': .02,
    'common_base_delay_s': .002, 'common_export_gain': 1.0,
    'sound_speed_m_s': 343.0, 'baseline_m': [.04, .03, .02],
    'fixed_channel_offset_s': 20e-6,
    'training_directions_xyz': [[1., 0., 0.], [-1., 0., 0.],
                                [0., 1., 0.], [0., 0., 1.]],
    'heldout_direction_xyz': [.6, .8, 0.],
    'channel_order': ['reference_microphone_0', 'microphone_1'],
    'direction_convention': 'unit vector from array toward source',
    'delay_model': 'tau=-u dot baseline/c+offset; positive tau means channel 1 arrives later',
    'waveform_model': 'x0=F(t-base), x1=F(t-base-tau); mono source is F(t-base)',
    'delay_implementation': 'continuous source evaluation, no sampled interpolation',
    'scoring_interval_samples': [2400, 29600],
    'scoring_samples_per_channel': 27200, 'scoring_cycles': 850,
    'phase_delay_prior_abs_upper_bound_s': .001,
    'phasor_convention': 'z=C-jD from cos/sin LS; tau=-arg(z1*conj(z0))/(2*pi*f)',
    'randomness': 'none', 'noise': 'none',
}


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
    parameters = baseline_parameters()
    if (type(parameters) is not dict or not same_metadata(
            {key: parameters.get(key) for key in REQUIRED_PARAMETERS}, REQUIRED_PARAMETERS)):
        raise ValueError('baseline true parameters differ from the fixed asset contract')
    signals = generate_signals()
    if set(signals) != set(FILE_NAMES):
        raise ValueError('expected exactly one source and five stereo observations')
    payloads, files, samples = {}, {}, {}
    domains = {'analytic': {}, 'float': {}, 'pcm': {}}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        payload = pcm16_bytes(signal, SAMPLE_RATE)
        metadata = _metadata(payload)
        expected = {'sample_rate_hz': SAMPLE_RATE, 'channels': 1 if name == 'source' else 2,
                    'samples_per_channel': SAMPLES, 'sample_width_bytes': 2, 'compression': 'NONE'}
        if metadata != expected:
            raise ValueError('unexpected baseline PCM format: '+filename)
        _, pcm = read_pcm16(payload)
        error = float(np.max(np.abs(pcm-signal)))
        if error > .5/32768 + 1e-15:
            raise ValueError('baseline PCM exceeded the common-gain rounding bound')
        analytic, floating, actual = analytic_measurements(name), measure_signal(signal), measure_signal(pcm)
        payloads[filename] = payload
        files[filename] = {**metadata, 'sha256': _sha(payload),
                           'peak': float(np.max(np.abs(pcm))),
                           'quantization_max_abs_error': error}
        samples[name] = {'file': filename, 'analytic': analytic,
                         'float_measurements': floating, 'pcm_measurements': actual,
                         'pcm_integer_measurements': _integer_measurements(payload)}
        for domain, result in zip(domains, (analytic, floating, actual)):
            domains[domain][name] = result
    manifest = {
        'schema_version': 1, 'exercise_id': 'E03-18', 'sample_rate_hz': SAMPLE_RATE,
        'samples_per_channel': SAMPLES, 'common_export_gain': 1.0,
        'origin': 'original deterministic mathematical known-direction synthesis',
        'source_sha256': {path: _sha(validate_parent_chain(ROOT/path).read_bytes()) for path in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'system': platform.system(), 'machine': platform.machine()},
        'parameters': parameters,
        'pcm': {'format': 'signed little-endian PCM16', 'scale': 32768,
                'rounding': 'nearest, ties to even', 'dither': 'none', 'clipping': 'rejected',
                'max_rounding_error_bound': .5/32768},
        'files': files, 'samples': samples,
        'calibration': {domain: calibration_results(measurements) for domain, measurements in domains.items()},
        'limits': LIMITS,
        'listening': 'Start at low volume; never autoplay. Stereo playback is not baseline calibration evidence.',
    }
    # Serialize now, before any filesystem mutation, to reject nonfinite metadata.
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return payloads, manifest


def check_assets(output: Path = DEFAULT_OUTPUT) -> dict:
    """Validate all seven ordinary files and replay actual PCM without writing."""
    output = validate_asset_directory(output, MEMBERS, check=True)
    manifest = strict_json_loads((output/'MANIFEST.json').read_bytes())
    payloads, expected = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('baseline manifest differs from current sources, true parameter types or scores')
    for name, filename in FILE_NAMES.items():
        payload = (output/filename).read_bytes()
        metadata = _metadata(payload)
        required = {'sample_rate_hz': SAMPLE_RATE, 'channels': 1 if name == 'source' else 2,
                    'samples_per_channel': SAMPLES, 'sample_width_bytes': 2, 'compression': 'NONE'}
        if metadata != required or _sha(payload) != manifest['files'][filename]['sha256']:
            raise ValueError('baseline WAV format or SHA differs: '+filename)
        _, actual = read_pcm16(payload)
        recorded = manifest['samples'][name]
        if (not same_metadata(measure_signal(actual), recorded['pcm_measurements'])
                or not same_metadata(_integer_measurements(payload), recorded['pcm_integer_measurements'])):
            raise ValueError('baseline actual PCM scores differ: '+filename)
        if payload != payloads[filename]:
            raise ValueError('baseline waveform differs from the complete source replay: '+filename)
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
        parser.exit(1, f'Baseline audio verification failed: {error}\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
