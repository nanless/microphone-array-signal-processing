"""Generate/check the independent five-WAV known-response comparison.

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
import math
import platform
from pathlib import Path
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes
from codes.chapters.appendix_b.core.response_audio import (
    SAMPLE_RATE, SAMPLES, STEMS, GAIN, build_fixture, analytic_results, analyze_fixture, analyze_pcm,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'codes/chapters/appendix_b/response_audio'
SOURCE_PATHS = (
    'codes/chapters/appendix_b/core/response_audio.py',
    'codes/chapters/appendix_b/examples/generate_response_audio.py',
    'codes/chapters/ch00/core/audio_samples.py',
)
MEMBERS = tuple(stem+'.wav' for stem in STEMS)+('MANIFEST.json',)


def validate_asset_directory(directory, *, check: bool):
    """Check every ancestor and exact ordinary members before reading/writing.

    Empty/new directories are allowed only for generation. The sole permitted
    path aliases are system /tmp, /var and /etc to their /private equivalents.
    """
    if type(check) is not bool:
        raise ValueError('check must be bool')
    if '..' in Path(directory).parts:
        raise ValueError('audio directory must not contain lexical parent traversal')
    directory = Path(directory).absolute()
    for component in (*reversed(directory.parents), directory):
        if component.is_symlink() and not (
                str(component) in ('/tmp', '/var', '/etc')
                and component.resolve() == Path('/private')/component.name):
            raise ValueError(f'audio ancestor must be an ordinary directory: {component}')
        if component.exists() and not component.is_dir():
            raise ValueError(f'audio ancestor must be a directory: {component}')
    if not directory.exists():
        if check:
            raise ValueError('audio directory is missing')
        return
    members = list(directory.iterdir())
    if any(member.is_symlink() or not member.is_file() for member in members):
        raise ValueError('audio members must be ordinary files')
    names = {member.name for member in members}
    if names != set(MEMBERS) and (check or names):
        raise ValueError('audio directory must contain exactly the six expected members')


def _strict_json(data: bytes):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def parse_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('JSON number exceeds finite float64 range')
        return number
    def parse_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=object_pairs,
                          parse_float=parse_float, parse_constant=parse_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('manifest must be strict UTF-8 JSON') from error


def _same_metadata(actual, expected, path='manifest'):
    if type(actual) is not type(expected):
        raise ValueError(f'{path} has an invalid type')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'{path} has missing or extra fields')
        for name in expected:
            _same_metadata(actual[name], expected[name], path+'.'+name)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'{path} has the wrong length')
        for i, (left, right) in enumerate(zip(actual, expected)):
            _same_metadata(left, right, path+f'[{i}]')
    elif actual != expected:
        raise ValueError(f'{path} differs from current source/PCM replay')


def expected_assets():
    """Pure full replay, with source hashes from the actual three dependencies."""
    fixture = build_fixture()
    buffers = {}
    for name, signal in fixture['signals'].items():
        if np.max(np.abs(signal*GAIN)) > 32767/32768:
            raise ValueError('fixture would clip PCM at common gain 1')
        buffers[name+'.wav'] = pcm16_bytes(signal*GAIN, SAMPLE_RATE)
    metadata = {
        'schema_version': 1, 'sample_rate_hz': SAMPLE_RATE, 'common_export_gain': GAIN,
        'origin': 'mathematical known RIR same-DRR contrast; not real room, speech or a recording',
        'parameters': fixture['parameters'], 'analytic': analytic_results(), 'floating_point': analyze_fixture(fixture),
        'pcm_analysis': analyze_pcm(buffers),
        'quantization': 'signed little-endian PCM16; nearest-even; no dither; decoded amplitude=int16/32768',
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
        'files': {name: {'channels': 1, 'samples_per_channel': SAMPLES,
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
