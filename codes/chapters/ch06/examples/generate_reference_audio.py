"""Generate or strictly read-check six E06-42 known-reference-location WAVs.

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
from codes.chapters.ch06.core.reference_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, WINDOWS,
    analytic_measurements, parameters, generate_signals, measure_signal, measure_pcm_tail_activity,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch06/reference_audio'
SOURCE_PATHS = (
    'codes/chapters/ch06/core/reference_audio.py',
    'codes/chapters/ch06/examples/generate_reference_audio.py',
    'codes/chapters/ch06/core/aec.py',
    'codes/chapters/ch06/core/aec_numeric.py',
    'codes/chapters/ch06/core/double_talk.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = set(FILE_NAMES.values()) | {'MANIFEST.json'}


# Fixed values and exact types, independent of the descriptive function at runtime.
REQUIRED_PARAMETERS = {'exercise_id': 'E06-42',
 'sample_rate_hz': 16000,
 'source_samples': 32000,
 'samples_per_channel': 33600,
 'convolution_tail_samples': 1600,
 'frequencies_hz': [500.0, 730.0],
 'source_amplitudes': [0.08, 0.06],
 'source_model': '0.08*cos(500Hz)+0.06*sin(730Hz); absolute clock, period 1600',
 'fade_samples': 320,
 'fade': 'squared sine, endpoints included, common envelope',
 'playback_gain_intervals': [[0, 16000, 1.0], [16000, 33600, 2.0]],
 'early_reference': 'source before time-varying playback gain, zero after 32000',
 'late_reference': 'gain times source before the fixed acoustic FIR, zero after 32000',
 'true_path_delays_samples': [0, 1600],
 'true_path_coefficients': [0.7, 0.35],
 'filter_length': 1601,
 'initial_weights': 'known true FIR, current sample first',
 'initial_history': '1600 zeros',
 'freeze': 'all 33600 samples; history still advances',
 'nlms_step_size': 0.5,
 'nlms_epsilon': 1e-08,
 'wrong_gain_prediction': 'current gain times fixed early-reference prediction, tail gain remains 2',
 'common_export_gain': 1.0,
 'channel_order': ['mono'],
 'scoring_intervals_samples': {'step': [16000, 17600], 'stable': [19200, 28800], 'tail': [32000, 33600]},
 'scoring_cycles_per_frequency': {'step': [50, 73], 'stable': [300, 438]},
 'tail_power': 'finite known enveloped samples, not a constant-envelope cycle average',
 'ncc_frame_samples': 160,
 'ncc_activity_rms': 0.001,
 'ncc_coherence_threshold': 0.85,
 'ncc_input': 'actual decoded late-reference and true-echo PCM',
 'ncc_availability': 'frame end; same-frame use requires a complete frame buffer',
 'alignment': 'same causal sample clock; no fitted delay or gain',
 'near_end': 'none',
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


def _integer_measurements(payload: bytes) -> dict:
    integers = _integers(payload)
    result = {}
    for label, (start, stop) in WINDOWS.items():
        x = integers[start:stop]
        energy = int(x[:, 0]@x[:, 0])
        denominator = (stop-start)*32768**2
        result[label] = {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start,
                         'channels': 1, 'integer_squared_sum_E_per_channel': [energy],
                         'integer_denominator_D_per_channel': denominator,
                         'mean_square_per_channel': [energy/denominator],
                         'integer_squared_sum_E_all_channels': energy,
                         'integer_denominator_D_all_channels': denominator,
                         'mean_square_all_channels': energy/denominator}
    return result


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    true_parameters = parameters()
    if not same_metadata(true_parameters, REQUIRED_PARAMETERS):
        raise ValueError('reference true parameters differ from the fixed asset contract')
    signals = generate_signals()
    if set(signals) != set(FILE_NAMES):
        raise ValueError('expected all six reference-location signals')
    payloads, decoded, files, samples = {}, {}, {}, {}
    expected_format = {'sample_rate_hz': SAMPLE_RATE, 'channels': 1,
                       'samples_per_channel': SAMPLES, 'sample_width_bytes': 2, 'compression': 'NONE'}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        payload = pcm16_bytes(signal, SAMPLE_RATE)
        metadata = _metadata(payload)
        if metadata != expected_format:
            raise ValueError('unexpected reference PCM format: '+filename)
        _, actual = read_pcm16(payload)
        error = float(np.max(abs(actual-signal)))
        if error > .5/32768+1e-15:
            raise ValueError('reference quantization exceeded the common gain rounding bound')
        payloads[filename], decoded[name] = payload, actual
        files[filename] = {**metadata, 'sha256': _sha(payload),
                           'quantization_max_abs_error': error, 'peak': float(np.max(abs(actual)))}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        samples[name] = {'file': filename, 'analytic': analytic_measurements(name),
                         'float_measurements': measure_signal(signal),
                         'pcm_measurements': measure_signal(decoded[name]),
                         'pcm_integer_measurements': _integer_measurements(payloads[filename])}
    manifest = {'schema_version': 1, 'exercise_id': 'E06-42',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'common_export_gain': 1.0,
                'origin': 'original deterministic known playback gain and causal FIR reference control',
                'source_sha256': {p: _sha(validate_parent_chain(ROOT/p).read_bytes()) for p in SOURCE_PATHS},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'parameters': true_parameters, 'files': files, 'samples': samples,
                'pcm_tail_activity': measure_pcm_tail_activity(decoded['late'], decoded['echo']),
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
        raise ValueError('reference manifest differs from current sources, true parameter types or scores')
    actual = {name: (output/filename).read_bytes() for name, filename in FILE_NAMES.items()}
    for name, filename in FILE_NAMES.items():
        payload = actual[name]
        if _metadata(payload) != {k: manifest['files'][filename][k] for k in _metadata(payload)} or _sha(payload) != manifest['files'][filename]['sha256']:
            raise ValueError('reference PCM format or SHA mismatch: '+filename)
        _, pcm = read_pcm16(payload)
        recorded = manifest['samples'][name]
        if (not same_metadata(measure_signal(pcm), recorded['pcm_measurements'])
                or not same_metadata(_integer_measurements(payload), recorded['pcm_integer_measurements'])):
            raise ValueError('reference actual PCM scores differ: '+filename)
        if payload != expected_payloads[filename]:
            raise ValueError('reference waveform differs from the complete source replay: '+filename)
    _, late = read_pcm16(actual['late'])
    _, echo = read_pcm16(actual['echo'])
    if not same_metadata(measure_pcm_tail_activity(late, echo), manifest['pcm_tail_activity']):
        raise ValueError('actual PCM tail activity differs')
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
        parser.exit(1, f'Reference audio verification failed: {error}\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
