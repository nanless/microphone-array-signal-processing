"""Generate/check two independent Chapter 9 audio files; no writes on import.

python -m codes.examples.chapter09_tracking_audio [--output DIRECTORY] [--check]
The directory contains source.wav, array_noisy.wav and MANIFEST.json only.
--check recomputes source/PCM analysis in memory and rejects stale assets; it
never regenerates files to make a check pass. Byte reproducibility is tied to
the recorded environment. No audio is played by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from codes.array_tutorial.tracking_audio import build_fixture

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'codes/tracking_audio'
SOURCE_PATHS = (
    'codes/examples/chapter09_tracking_audio.py', 'codes/array_tutorial/tracking_audio.py',
    'codes/array_tutorial/moving_source.py', 'codes/array_tutorial/tracking.py',
    'codes/array_tutorial/doa.py', 'codes/array_tutorial/conventions.py',
    'codes/array_tutorial/audio_samples.py',
)


def expected_assets():
    buffers, metadata = build_fixture()
    metadata['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    metadata['environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()}
    metadata['files'] = {name: {'channels': 1 if name == 'source.wav' else 2,
        'samples_per_channel': 32000, 'sha256': hashlib.sha256(data).hexdigest()} for name, data in buffers.items()}
    buffers['MANIFEST.json'] = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return buffers, metadata


def generate(out_dir=OUTPUT, *, check=False):
    out_dir = Path(out_dir)
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
