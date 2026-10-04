"""Generate/check four E01-10 mathematical source/response WAVs, without HRTFs.

Two mono sources have 32000 samples; their complete two-channel FIR outputs
have 32001 frames. All use export gain 1. --check replays in memory and never
repairs a manifest or writes an asset. No direction, speech or listener data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import wave

import numpy as np

from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch00.io_contracts import (
    same_metadata, strict_json_loads, validate_asset_directory,
    validate_parent_chain,
)
from codes.chapters.ch01.core.spectral_cues import (
    CASES, FREQUENCIES, SAMPLE_RATE, WINDOW, build_cases, measure_response,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIRECTORY = ROOT / 'codes/chapters/ch01/spectral_audio'
SOURCE_PATHS = (
    'codes/chapters/ch01/core/spectral_cues.py',
    'codes/chapters/ch01/examples/generate_spectral_cues.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
WAV_CHANNELS = {'flat_source.wav': 1, 'flat_stereo.wav': 2,
                'tilted_source.wav': 1, 'tilted_stereo.wav': 2}
MEMBERS = set(WAV_CHANNELS) | {'MANIFEST.json'}


def _metadata(payload):
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        return {'sample_rate_hz': stream.getframerate(), 'channels': stream.getnchannels(),
                'samples_per_channel': stream.getnframes(),
                'sample_width_bytes': stream.getsampwidth(), 'compression': stream.getcomptype()}


def _integer_score(payload):
    with wave.open(io.BytesIO(payload), 'rb') as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
        pcm = pcm.reshape(-1, 2).astype(np.int64)
    start, stop = WINDOW
    energies = [int(np.sum(pcm[start:stop, channel] ** 2)) for channel in range(2)]
    if min(energies) <= 0:
        raise ValueError('Controlled stereo has no scoring energy')
    denominator = (stop - start) * 32768**2
    return {'left_squared_sum_E': energies[0], 'right_squared_sum_E': energies[1],
            'integer_denominator_D': denominator,
            'samples': stop - start,
            'left_mean_square': energies[0] / denominator,
            'right_mean_square': energies[1] / denominator,
            'ild_right_minus_left_db': 10 * math.log10(energies[1] / energies[0])}


def expected_assets():
    """Current source-bound reconstruction; separate analytic/float/PCM values."""
    signals = build_cases()
    if {name + '.wav' for name in signals} != set(WAV_CHANNELS):
        raise ValueError('Expected exactly two sources and two stereo responses')
    payloads = {name + '.wav': pcm16_bytes(signal) for name, signal in signals.items()}
    manifest = {
        'schema_version': 1, 'exercise_id': 'E01-10',
        'sample_rate_hz': SAMPLE_RATE, 'common_export_gain': 1.,
        'channel_order': ['left', 'right'],
        'source_sha256': {name: hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'byteorder': sys.byteorder, 'platform': platform.platform()},
        'parameters': {'frequencies_hz': list(FREQUENCIES),
                       'source_amplitudes': {name: list(values) for name, values in CASES.items()},
                       'filters_left_right': [[1., .5], [1., -.5]],
                       'source_samples': 32000, 'output_samples': 32001,
                       'source_duration_s': 2., 'output_duration_s': 32001 / SAMPLE_RATE,
                       'fade': 'min(1,t/0.02,(2-t)/0.02)', 'edge_fade_s': .02,
                       'scoring_window': list(WINDOW), 'scoring_samples': WINDOW[1] - WINDOW[0],
                       'initial_history': 'zero', 'tail': 'complete one-sample FIR tail, excluded from scoring',
                       'seed': None},
        'conventions': {'fourier': 'forward negative exponential',
                        'ild': 'right minus left; 10 log10 power ratio',
                        'ipd': 'arg(right*conjugate(left)); radians and degrees',
                        'spectral': 'known-frequency real sine/cosine projection on the stable integer-period window',
                        'power': 'uncentered digital mean square'},
        'quantization': {'format': 'signed little-endian PCM16', 'scale': 32768,
                         'rounding': 'nearest, ties to even', 'clipping': 'rejected',
                         'max_rounding_error_bound': .5 / 32768},
        'limits': 'Artificial known FIRs, two noiseless mathematical tone sources; not measured HRIR/HRTF, '
                  'human localization, speech, a blind response estimator or a direction manifold. '
                  'Sources have different total power. PCM ratios need not be exactly invariant.',
        'listening': 'Start at low volume, retain stereo channels; never autoplay. No formal listening test.',
        'files': {}, 'samples': {},
    }
    for filename, payload in payloads.items():
        metadata = _metadata(payload)
        channels = WAV_CHANNELS[filename]
        required = {'sample_rate_hz': 16000, 'channels': channels,
                    'samples_per_channel': 32000 if channels == 1 else 32001,
                    'sample_width_bytes': 2, 'compression': 'NONE'}
        if metadata != required:
            raise ValueError('Controlled PCM shape or format differs: ' + filename)
        _, decoded = read_pcm16(payload)
        error = float(np.max(np.abs(decoded - signals[Path(filename).stem])))
        if error > .5 / 32768 + 1e-15:
            raise ValueError('PCM exceeds rounding bound')
        manifest['files'][filename] = {**metadata,
                                      'sha256': hashlib.sha256(payload).hexdigest(),
                                      'peak': float(np.max(np.abs(decoded))),
                                      'quantization_max_abs_error': error}
    for scene, amplitudes in CASES.items():
        source_key, stereo_key = scene + '_source', scene + '_stereo'
        _, source_pcm = read_pcm16(payloads[source_key + '.wav'])
        _, stereo_pcm = read_pcm16(payloads[stereo_key + '.wav'])
        left_power = sum(a*a/2 * (1.25 + math.cos(2*math.pi*f/SAMPLE_RATE))
                         for a, f in zip(amplitudes, FREQUENCIES))
        right_power = sum(a*a/2 * (1.25 - math.cos(2*math.pi*f/SAMPLE_RATE))
                          for a, f in zip(amplitudes, FREQUENCIES))
        manifest['samples'][scene] = {
            'source_file': source_key + '.wav', 'stereo_file': stereo_key + '.wav',
            'analytic': {'source_mean_square': sum(a*a/2 for a in amplitudes),
                         'left_mean_square': left_power, 'right_mean_square': right_power,
                         'ild_right_minus_left_db': 10 * math.log10(right_power / left_power)},
            'float_measurements': measure_response(signals[source_key], signals[stereo_key]),
            'pcm_measurements': measure_response(source_pcm, stereo_pcm),
            'pcm_integer_measurements': _integer_score(payloads[stereo_key + '.wav']),
        }
    return payloads, manifest


def check_assets(directory=DEFAULT_DIRECTORY):
    directory = validate_asset_directory(directory, MEMBERS, check=True)
    payloads, expected = expected_assets()
    actual = strict_json_loads((directory / 'MANIFEST.json').read_bytes())
    if not same_metadata(actual, expected):
        raise ValueError('Spectral manifest differs from current sources, parameters, environment or scores')
    for name, payload in payloads.items():
        if (directory / name).read_bytes() != payload:
            raise ValueError('Stale or modified spectral WAV: ' + name)
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
        parser.exit(1, f'Spectral cue verification failed: {error}\n')
    print(f"{'Checked' if args.check else 'Generated and checked'} {len(manifest['files'])} spectral WAVs")


if __name__ == '__main__':
    main()
