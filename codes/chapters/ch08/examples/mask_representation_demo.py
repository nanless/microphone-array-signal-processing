"""Generate six E08-28 known-mask WAVs, or strictly read-check without repair."""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import wave
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch08.core.mask_representation import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, measure_signal, parameters, run_experiment,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch08/mask_audio'
SOURCE_PATHS = ('codes/chapters/ch08/core/mask_representation.py',
                'codes/chapters/ch08/examples/mask_representation_demo.py',
                'codes/chapters/ch02/core/conventions.py',
                'codes/chapters/ch00/core/audio_samples.py')


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {path: _sha((ROOT/path).read_bytes()) for path in SOURCE_PATHS}


def prepare_assets() -> tuple[dict, dict[str, bytes]]:
    report, arrays = run_experiment()
    contents, files, decoded, samples = {}, {}, {}, {}
    for key, value in arrays.items():
        blob = pcm16_bytes(value, SAMPLE_RATE)
        rate, pcm = read_pcm16(blob)
        filename = FILE_NAMES[key]
        contents[filename], decoded[key] = blob, pcm
        files[filename] = {'channels': 1, 'samples_per_channel': SAMPLES,
                           'sample_rate_hz': rate, 'sha256': _sha(blob)}
    for key in FILE_NAMES:
        samples[key] = {'file': FILE_NAMES[key],
                        'float_measurements': report['float_measurements'][key],
                        'pcm_measurements': measure_signal(decoded[key], decoded['target'], pcm=True),
                        'quantization_max_abs_error': float(np.max(abs(arrays[key]-decoded[key])))}
    manifest = {'schema_version': 1, 'origin': 'original known-bin offline mathematical representation',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'common_export_gain': 1.,
                'source_sha256': source_digests(), 'parameters': report['parameters'],
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'files': files, 'samples': samples,
                'unwindowed_selected_mixture_phasors_real_imag': report['unwindowed_selected_mixture_phasors_real_imag'],
                'discarded_bin_max_magnitude': report['discarded_bin_max_magnitude'],
                'pcm_encode_multiplier': 32768, 'pcm_decode_divisor': 32768,
                'pcm': 'signed little-endian PCM16, round-to-nearest-even, no dither or per-file gain',
                'pcm_quantization_step': 1/32768, 'pcm_half_step_error_bound': .5/32768,
                'limits': LIMITS, 'listening': 'Start at low volume; no automatic playback or listening study.'}
    contents['MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return manifest, contents


def _members(output, *, checking):
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError('mask_audio output must be an ordinary directory')
    if not output.exists():
        if checking:
            raise ValueError('mask_audio directory missing')
        return
    members = list(output.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in members):
        raise ValueError('mask_audio members must be ordinary files without symlinks')
    names, allowed = {p.name for p in members}, set(FILE_NAMES.values()) | {'MANIFEST.json'}
    if names != allowed and (checking or names):
        raise ValueError('mask_audio requires exactly six WAVs and MANIFEST.json')


def _strict_equal(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys() and
                all(_strict_equal(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected) and
                all(_strict_equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) in (int, float) and np.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def check_assets(directory: Path = DEFAULT_OUTPUT, *, replay=True) -> dict:
    """Validate real members, source SHA, PCM scores and optionally full replay."""
    if type(replay) is not bool:
        raise ValueError('replay must be boolean')
    directory = Path(directory)
    _members(directory, checking=True)
    try:
        manifest = json.loads((directory/'MANIFEST.json').read_text(),
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError('invalid mask_audio manifest') from exc
    if not isinstance(manifest, dict):
        raise ValueError('mask_audio manifest must be an object')
    if manifest.get('source_sha256') != source_digests():
        raise ValueError('mask_audio source set or SHA is stale')
    files, samples = manifest.get('files'), manifest.get('samples')
    if (not isinstance(files, dict) or not isinstance(samples, dict) or
            set(files) != set(FILE_NAMES.values()) or set(samples) != set(FILE_NAMES) or
            any(not isinstance(r, dict) for r in (*files.values(), *samples.values()))):
        raise ValueError('mask_audio manifest file set differs')
    if (not _strict_equal(manifest.get('parameters'), parameters()) or
            not _strict_equal(manifest.get('common_export_gain'), 1.) or
            not _strict_equal(manifest.get('sample_rate_hz'), SAMPLE_RATE) or
            not _strict_equal(manifest.get('samples_per_channel'), SAMPLES)):
        raise ValueError('mask_audio fixed parameters differ')
    decoded = {}
    for key, filename in FILE_NAMES.items():
        blob = (directory/filename).read_bytes()
        if _sha(blob) != files[filename].get('sha256'):
            raise ValueError('mask_audio WAV SHA mismatch: '+filename)
        try:
            with wave.open(io.BytesIO(blob)) as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                    reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, 1, SAMPLES, 2, 'NONE'):
                    raise ValueError('mask_audio PCM format mismatch: '+filename)
            _, decoded[key] = read_pcm16(blob)
        except (wave.Error, EOFError) as exc:
            raise ValueError('invalid mask_audio WAV: '+filename) from exc
        record = {'channels': 1, 'samples_per_channel': SAMPLES,
                  'sample_rate_hz': SAMPLE_RATE, 'sha256': _sha(blob)}
        if not _strict_equal(files[filename], record) or samples[key].get('file') != filename:
            raise ValueError('mask_audio PCM metadata differs: '+filename)
    for key in FILE_NAMES:
        actual = measure_signal(decoded[key], decoded['target'], pcm=True)
        if not _strict_equal(samples[key].get('pcm_measurements'), actual):
            raise ValueError('mask_audio actual PCM measurement mismatch: '+FILE_NAMES[key])
    if replay:
        expected, contents = prepare_assets()
        if not _strict_equal(manifest, expected):
            raise ValueError('mask_audio float/model measurements are stale')
        if any((directory/name).read_bytes() != blob for name, blob in contents.items()):
            raise ValueError('mask_audio differs from current source')
    return manifest


def generate_assets(directory: Path = DEFAULT_OUTPUT, *, check=False) -> dict:
    if type(check) is not bool:
        raise ValueError('check must be boolean')
    directory = Path(directory)
    if check:
        return check_assets(directory)
    _members(directory, checking=False)
    manifest, contents = prepare_assets()
    directory.mkdir(parents=True, exist_ok=True)
    for name, blob in contents.items():
        (directory/name).write_bytes(blob)
    return check_assets(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    result = generate_assets(args.output_dir, check=args.check)
    print(json.dumps({'status': 'checked' if args.check else 'generated', 'files': list(result['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
