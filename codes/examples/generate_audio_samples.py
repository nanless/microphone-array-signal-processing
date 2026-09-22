"""Generate small original synthetic WAVs, or verify them without writing.

Run from the repository root. --check compares generator inputs, file hashes,
PCM format and actual byte content; it never rewrites a failed artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codes.array_tutorial.audio_samples import prepare_exports, SEED  # noqa: E402

INPUTS = ['codes/examples/generate_audio_samples.py', 'codes/array_tutorial/audio_samples.py',
          'codes/array_tutorial/aec.py', 'codes/array_tutorial/dereverberation.py',
          'codes/array_tutorial/spectral.py', 'codes/array_tutorial/conventions.py',
          'codes/array_tutorial/geometry.py']


def generate(destination: Path, check: bool = False) -> dict:
    files, groups = prepare_exports()
    records = []
    for filename, (blob, info) in files.items():
        records.append({'file': filename, 'sha256': hashlib.sha256(blob).hexdigest(), **info})
    manifest = {'schema_version': 1, 'seed': SEED, 'origin': 'original deterministic mathematical synthesis',
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine(), 'float_dtype': 'float64'},
                'synthetic_signal': {'applies_to': '173/293 Hz harmonic targets in groups other than nonlinear and fractional_array; see each group for noise and transforms',
                                     'duration_s': 2, 'base_frequencies_hz': [173, 293],
                                     'harmonic_numbers': [1, 2, 3, 4, 5, 6], 'harmonic_amplitude': '1/(4*k)',
                                     'envelope': '0.3+0.7*sin(2*pi*2*t)^2', 'fade_duration_s': .02,
                                     'fade': 'squared sine, endpoint included'},
                'rights': 'No third-party recordings or model weights. Repository licensing remains unspecified.',
                'pcm': 'signed little-endian 16-bit, round-to-nearest-even; no dither; no per-file normalization',
                'listening': 'Start at low device volume. No automatic playback. No formal listening study performed.',
                'generator_inputs': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS},
                'groups': groups, 'files': records}
    target = destination/'MANIFEST.json'
    if check:
        if not target.is_file() or json.loads(target.read_text()) != manifest:
            raise ValueError('audio manifest missing or stale; regenerate explicitly')
        actual = {p.name for p in destination.glob('*.wav')}
        if actual != set(files):
            raise ValueError('audio file set differs from manifest')
        for filename, (blob, _) in files.items():
            if (destination/filename).read_bytes() != blob:
                raise ValueError(f'audio content differs: {filename}')
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for filename, (blob, _) in files.items():
            (destination/filename).write_bytes(blob)
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return {'files': len(files), 'groups': len(groups), 'bytes': sum(len(x[0]) for x in files.values()), 'checked': check}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'codes/audio')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.check)))
