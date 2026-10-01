"""Generate/check two independent Chapter 9 audio files; no writes on import.

python -m codes.chapters.ch09.examples.chapter09_tracking_audio [--output DIRECTORY] [--check]
The directory contains source.wav, array_noisy.wav and MANIFEST.json only.
--check recomputes source/PCM analysis in memory and rejects stale assets; it
never regenerates files to make a check pass. Byte reproducibility is tied to
the recorded environment. No audio is played by this program.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from codes.chapters.ch09.core.tracking_audio import build_fixture

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'codes/chapters/ch09/tracking_audio'
SOURCE_PATHS = (
    'codes/chapters/ch09/examples/chapter09_tracking_audio.py', 'codes/chapters/ch09/core/tracking_audio.py',
    'codes/chapters/ch09/core/moving_source.py', 'codes/chapters/ch09/core/tracking.py',
    'codes/chapters/ch04/core/doa.py', 'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
)


def expected_assets():
    buffers, metadata = build_fixture()
    metadata['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    metadata['environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()}
    metadata['files'] = {name: {'channels': 1 if name == 'source.wav' else 2,
        'samples_per_channel': 32000, 'sha256': hashlib.sha256(data).hexdigest()} for name, data in buffers.items()}
    buffers['MANIFEST.json'] = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return buffers, metadata


def _validate_directory(directory, expected, *, check):
    """Reject links/special files before any output write or read-through check.

    Existing directories are either empty (new export) or contain exactly the
    fixed assets. Parent components are checked without resolving away links.
    """
    if not isinstance(check, bool):
        raise ValueError('check must be bool')
    directory = Path(directory).absolute()
    for parent in (*reversed(directory.parents), directory):
        if (parent.is_symlink() and not
                (str(parent) in ('/var', '/tmp', '/etc')
                 and parent.resolve() == Path('/private')/parent.name)):
            raise ValueError(f'audio directory ancestor is a symbolic link: {parent}')
        if parent.exists() and not parent.is_dir():
            raise ValueError(f'audio directory ancestor is not a directory: {parent}')
    if not directory.exists():
        if check:
            raise ValueError('audio directory is missing')
        return
    members = list(directory.iterdir())
    for member in members:
        if member.is_symlink() or not member.is_file():
            raise ValueError(f'audio member must be an ordinary file: {member}')
    names = {member.name for member in members}
    if names != set(expected) and (check or names):
        raise ValueError('audio directory must contain exactly the expected members')


def generate(out_dir=OUTPUT, *, check=False):
    out_dir = Path(out_dir)
    _validate_directory(out_dir, ("source.wav", "array_noisy.wav", "MANIFEST.json"), check=check)
    assets, metadata = expected_assets()
    if check:
        for name, expected in assets.items():
            path = out_dir/name
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f'missing or stale tracking asset: {path}')
        if {p.name for p in out_dir.iterdir()} != set(assets):
            raise ValueError('tracking audio directory contains unexpected files')
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in assets.items():
            (out_dir/name).write_bytes(content)
    return metadata


def check_assets(out_dir=OUTPUT):
    """Strict read-only replay of actual directory bytes, current source and PCM."""
    return generate(out_dir, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = generate(args.output, check=args.check)
    print(json.dumps({'mode': 'verified' if args.check else 'generated',
                      'directory': str(args.output), 'scores': report['pcm_analysis']['scores']}, indent=2))


if __name__ == '__main__':
    main()
