"""Generate/check the independent five-WAV known-weight comparison.

--check validates ordinary paths, strict JSON, current sources and actual PCM,
then replays the complete model in memory. It never repairs a stale directory.
"""
from __future__ import annotations
if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes
from codes.chapters.appendix_a.core.weighted_audio import (
    SAMPLE_RATE, SAMPLES, STEMS, GAIN, build_fixture, analytic_results, analyze_fixture, analyze_pcm,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'codes/chapters/appendix_a/weighted_audio'
SOURCE_PATHS = (
    'codes/chapters/appendix_a/core/weighted_audio.py',
    'codes/chapters/appendix_a/examples/generate_weighted_audio.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = tuple(stem+'.wav' for stem in STEMS)+('MANIFEST.json',)


def validate_asset_directory(directory, *, check: bool):
    """Shared path preflight with this experiment's exact member policy."""
    return _shared_asset_directory(directory, MEMBERS, check=check)


def _strict_json(data: bytes):
    return strict_json_loads(data)


def _same_metadata(actual, expected, path='manifest'):
    if not same_metadata(actual, expected):
        raise ValueError(f'{path} has invalid types or differs from current source/PCM replay')


def expected_assets():
    """Pure full replay, with source hashes from the actual four dependencies."""
    fixture = build_fixture()
    buffers = {}
    for name, signal in fixture['signals'].items():
        if np.max(np.abs(signal*GAIN)) > 32767/32768:
            raise ValueError('fixture would clip PCM at common gain 1')
        buffers[name+'.wav'] = pcm16_bytes(signal*GAIN, SAMPLE_RATE)
    metadata = {
        'schema_version': 1, 'sample_rate_hz': SAMPLE_RATE, 'common_export_gain': GAIN,
        'origin': 'mathematical known-covariance fixed-weight example; not random noise, speech or a recording',
        'parameters': fixture['parameters'], 'analytic': analytic_results(), 'floating_point': analyze_fixture(fixture),
        'pcm_analysis': analyze_pcm(buffers),
        'quantization': 'signed little-endian PCM16; nearest-even; no dither; decoded amplitude=int16/32768',
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
        'files': {name: {'channels': 2 if name=='weighted_array.wav' else 1, 'samples_per_channel': SAMPLES,
                         'sha256': hashlib.sha256(data).hexdigest()} for name, data in buffers.items()},
    }
    buffers['MANIFEST.json'] = (json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    return buffers, metadata


def check_assets(directory=OUTPUT):
    """Strictly read actual bytes/score them, then compare full current replay."""
    directory = Path(directory)
    validate_asset_directory(directory, check=True)
    metadata = _strict_json((directory/'MANIFEST.json').read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError('manifest must be an object')
    actual = {name: (directory/name).read_bytes() for name in MEMBERS if name.endswith('.wav')}
    # This analysis runs on the directory bytes, not on regenerated stand-ins.
    measured = analyze_pcm(actual)
    expected, replay = expected_assets()
    _same_metadata(metadata, replay)
    _same_metadata(metadata['pcm_analysis'], measured, 'actual PCM analysis')
    for name, data in actual.items():
        if hashlib.sha256(data).hexdigest() != metadata['files'][name]['sha256']:
            raise ValueError(f'actual WAV digest differs: {name}')
        if data != expected[name]:
            raise ValueError(f'actual WAV differs from current complete replay: {name}')
    return metadata


def generate(directory=OUTPUT, *, check=False):
    validate_asset_directory(directory, check=check)
    if check:
        return check_assets(directory)
    buffers, metadata = expected_assets()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in buffers.items():
        (directory/name).write_bytes(data)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = generate(args.output, check=args.check)
    print(json.dumps({'files': len(result['files']), 'checked': args.check}))
