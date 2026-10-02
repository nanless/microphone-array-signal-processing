"""Generate six independent E07-21 WAVs, or read-check them without repair."""
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
from codes.chapters.ch07.core.mint_teaching import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, measure_signal, parameters, run_experiment,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1]/'mint_audio'
SOURCE_PATHS = (
    'codes/chapters/ch07/core/mint_teaching.py',
    'codes/chapters/ch07/examples/mint_teaching_demo.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {p: _sha((ROOT/p).read_bytes()) for p in SOURCE_PATHS}


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    experiment = run_experiment()
    blobs, files, decoded, samples = {}, {}, {}, {}
    for key, value in experiment['signals'].items():
        blob = pcm16_bytes(value, SAMPLE_RATE)
        rate, pcm = read_pcm16(blob)
        filename = FILE_NAMES[key]
        blobs[filename], decoded[key] = blob, pcm
        files[filename] = {'channels': len(pcm), 'samples_per_channel': pcm.shape[1],
                           'sample_rate_hz': rate, 'sha256': _sha(blob)}
    for key, value in experiment['signals'].items():
        samples[key] = {'file': FILE_NAMES[key], 'float_measurements': experiment['float_measurements'][key],
                        'pcm_measurements': measure_signal(decoded[key], decoded['reference'], pcm=True),
                        'quantization_max_abs_error': float(np.max(abs(value-decoded[key])))}
    manifest = {'schema_version': 1, 'origin': 'original fixed known-path mathematical synthesis',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'common_export_gain': 1.,
                'source_sha256': source_digests(),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'parameters': experiment['parameters'], 'files': files, 'samples': samples,
                'float_decomposition': experiment['float_decomposition'],
                'finite_noise_statistics': experiment['finite_noise_statistics'],
                'pcm_encode_multiplier': 32768, 'pcm_decode_divisor': 32768,
                'pcm': 'signed little-endian PCM16, round-to-nearest-even, no dither or per-file gain',
                'pcm_quantization_step': 1/32768, 'pcm_half_step_error_bound': .5/32768,
                'limits': LIMITS, 'listening': 'Start at low volume; no automatic playback or listening study.'}
    return blobs, manifest


def _members(output, *, checking):
    return _shared_asset_directory(output, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=checking)


def _strict_equal(actual, expected):
    return same_metadata(actual, expected, allow_int_for_float=True)


def check_assets(output: Path = DEFAULT_OUTPUT, *, replay=True) -> dict:
    if type(replay) is not bool:
        raise ValueError('replay must be boolean')
    output = Path(output)
    _members(output, checking=True)
    try:
        manifest = strict_json_loads((output/'MANIFEST.json').read_text())
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('invalid mint_audio manifest') from error
    if not isinstance(manifest, dict):
        raise ValueError('mint_audio manifest must be an object')
    if manifest.get('source_sha256') != source_digests():
        raise ValueError('mint_audio source set or SHA is stale')
    files, samples = manifest.get('files'), manifest.get('samples')
    if (not isinstance(files, dict) or not isinstance(samples, dict)
            or set(files) != set(FILE_NAMES.values()) or set(samples) != set(FILE_NAMES)
            or any(not isinstance(record, dict) for record in (*files.values(), *samples.values()))):
        raise ValueError('mint_audio manifest file set differs')
    if (not _strict_equal(manifest.get('parameters'), parameters())
            or not _strict_equal(manifest.get('common_export_gain'), 1.)
            or type(manifest.get('sample_rate_hz')) is not int or manifest['sample_rate_hz'] != SAMPLE_RATE
            or type(manifest.get('samples_per_channel')) is not int or manifest['samples_per_channel'] != SAMPLES):
        raise ValueError('mint_audio fixed parameters differ')
    decoded = {}
    for key, filename in FILE_NAMES.items():
        blob = (output/filename).read_bytes()
        if _sha(blob) != manifest['files'][filename].get('sha256'):
            raise ValueError('mint_audio WAV SHA mismatch: '+filename)
        try:
            with wave.open(io.BytesIO(blob)) as reader:
                channels = 2 if key.endswith('_array') else 1
                if (reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth(),
                        reader.getcomptype()) != (SAMPLE_RATE, channels, SAMPLES, 2, 'NONE'):
                    raise ValueError('mint_audio PCM format mismatch: '+filename)
            _, decoded[key] = read_pcm16(blob)
        except (wave.Error, EOFError) as error:
            raise ValueError('invalid mint_audio WAV: '+filename) from error
        expected_record = {'channels': channels, 'samples_per_channel': SAMPLES,
                           'sample_rate_hz': SAMPLE_RATE, 'sha256': _sha(blob)}
        if not _strict_equal(manifest['files'][filename], expected_record):
            raise ValueError('mint_audio manifest PCM format differs: '+filename)
        if manifest['samples'][key].get('file') != filename:
            raise ValueError('mint_audio sample filename differs: '+filename)
    for key in FILE_NAMES:
        actual = measure_signal(decoded[key], decoded['reference'], pcm=True)
        recorded = manifest.get('samples', {}).get(key, {}).get('pcm_measurements')
        if not _strict_equal(recorded, actual):
            raise ValueError('mint_audio actual PCM measurement mismatch: '+FILE_NAMES[key])
    # Source replay supplements, never replaces, actual waveform re-scoring.
    if replay:
        expected_blobs, expected_manifest = prepare_assets()
        if not _strict_equal(manifest, expected_manifest):
            raise ValueError('mint_audio parameters or float/PCM measurements are stale')
        if any((output/name).read_bytes() != blob for name, blob in expected_blobs.items()):
            raise ValueError('mint_audio differs from current source')
    return manifest


def generate(output: Path = DEFAULT_OUTPUT, *, check=False) -> dict:
    if type(check) is not bool:
        raise ValueError('check must be boolean')
    output = Path(output)
    if check:
        return check_assets(output)
    _members(output, checking=False)  # preflight before any file is changed
    blobs, manifest = prepare_assets()
    output.mkdir(parents=True, exist_ok=True)
    for filename, blob in blobs.items():
        (output/filename).write_bytes(blob)
    (output/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return check_assets(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    manifest = generate(args.output_dir, check=args.check)
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output_dir), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
