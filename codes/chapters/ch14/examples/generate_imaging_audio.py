"""Generate five imaging WAVs; --check replays strictly without writing."""
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
from codes.chapters.ch00.io_contracts import validate_asset_directory, strict_json_loads, same_metadata
from codes.chapters.ch14.core.imaging_audio import (
    FILE_NAMES, SAMPLE_RATE, SAMPLES, LIMITS, parameters, run_experiment,
    measure_signal, measure_source_pairs,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch14/imaging_audio'
SOURCE_PATHS = (
    'codes/chapters/ch14/core/imaging.py', 'codes/chapters/ch14/core/imaging_audio.py',
    'codes/chapters/ch14/examples/generate_imaging_audio.py',
    'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {path: _sha((ROOT/path).read_bytes()) for path in SOURCE_PATHS}


def prepare_assets():
    report, arrays = run_experiment()
    contents, files, decoded, samples = {}, {}, {}, {}
    for key, array in arrays.items():
        blob = pcm16_bytes(array, SAMPLE_RATE)
        rate, pcm = read_pcm16(blob)
        filename = FILE_NAMES[key]
        contents[filename] = blob; decoded[key] = pcm
        files[filename] = {'channels': array.shape[0], 'samples_per_channel': SAMPLES,
                           'sample_rate_hz': rate, 'sha256': _sha(blob)}
        samples[key] = {'file': filename, 'float_measurements': report['float_measurements'][key],
                        'pcm_measurements': measure_signal(pcm, key, pcm=True),
                        'quantization_max_abs_error': float(np.max(abs(array-pcm)))}
    manifest = {'schema_version': 1, 'origin': 'original finite-snapshot known-tone mathematical imaging observations',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES,
                'common_export_gain': 1., 'source_sha256': source_digests(), 'parameters': parameters(),
                'files': files, 'samples': samples, 'float_source_pairs': report['float_source_pairs'],
                'pcm_source_pairs': measure_source_pairs(decoded),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'pcm_encode_multiplier': 32768, 'pcm_decode_divisor': 32768,
                'pcm_half_step_error_bound': .5/32768,
                'pcm': 'signed little-endian PCM16, nearest-even rounding; no dither, saturation or per-file gain',
                'limits': LIMITS, 'listening': 'Start at low device volume; no autoplay or listening study.'}
    contents['MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return manifest, contents


def check_assets(directory=DEFAULT_OUTPUT):
    """Strict six-member/source/type/actual-PCM/full-replay check; never repair."""
    directory = Path(directory)
    validate_asset_directory(directory, set(FILE_NAMES.values())|{'MANIFEST.json'}, check=True)
    manifest = strict_json_loads((directory/'MANIFEST.json').read_bytes())
    if not isinstance(manifest, dict) or manifest.get('source_sha256') != source_digests():
        raise ValueError('imaging audio source set or SHA differs')
    files, samples = manifest.get('files'), manifest.get('samples')
    if (not isinstance(files, dict) or set(files) != set(FILE_NAMES.values())
            or not isinstance(samples, dict) or set(samples) != set(FILE_NAMES)
            or any(not isinstance(item, dict) for item in (*files.values(), *samples.values()))):
        raise ValueError('imaging audio manifest member records differ')
    decoded = {}
    for key, filename in FILE_NAMES.items():
        blob = (directory/filename).read_bytes()
        channels = 2 if key.startswith('array_') else 1
        try:
            with wave.open(io.BytesIO(blob)) as reader:
                actual = (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                          reader.getsampwidth(), reader.getcomptype())
                if actual != (SAMPLE_RATE, channels, SAMPLES, 2, 'NONE'):
                    raise ValueError('imaging audio PCM format differs: '+filename)
            _, decoded[key] = read_pcm16(blob)
        except (wave.Error, EOFError) as exc:
            raise ValueError('invalid imaging audio WAV: '+filename) from exc
        record = {'channels': channels, 'samples_per_channel': SAMPLES,
                  'sample_rate_hz': SAMPLE_RATE, 'sha256': _sha(blob)}
        if not same_metadata(files[filename], record) or samples[key].get('file') != filename:
            raise ValueError('imaging audio WAV metadata/SHA differs: '+filename)
        if not same_metadata(samples[key].get('pcm_measurements'), measure_signal(decoded[key], key, pcm=True)):
            raise ValueError('imaging audio PCM scores differ: '+filename)
    if not same_metadata(manifest.get('pcm_source_pairs'), measure_source_pairs(decoded)):
        raise ValueError('imaging audio actual source CSM differs')
    expected, contents = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('imaging audio metadata/model replay differs')
    if any((directory/name).read_bytes() != blob for name, blob in contents.items()):
        raise ValueError('imaging audio byte replay differs from current source/environment')
    return manifest


def generate_assets(directory=DEFAULT_OUTPUT, *, check=False):
    if type(check) is not bool:
        raise ValueError('check must be bool')
    directory = Path(directory)
    if check:
        return check_assets(directory)
    validate_asset_directory(directory, set(FILE_NAMES.values())|{'MANIFEST.json'}, check=False)
    manifest, contents = prepare_assets()
    directory.mkdir(parents=True, exist_ok=True)
    # Recheck after in-memory preparation and before any ordinary member write.
    validate_asset_directory(directory, set(FILE_NAMES.values())|{'MANIFEST.json'}, check=False)
    for filename, blob in contents.items():
        (directory/filename).write_bytes(blob)
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
