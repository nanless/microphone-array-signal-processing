"""Generate 17 original distributed WAVs; --check is strict read-only replay."""
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
from codes.chapters.ch00.io_contracts import validate_asset_directory, validate_parent_chain, strict_json_loads, same_metadata
from codes.chapters.ch13.core.distributed_audio import (
    FILE_NAMES, OUTPUT_REFERENCES, SAMPLE_RATE, SAMPLES, LIMITS, parameters,
    run_experiment, measure_signal, validate_fixed_fixture, validate_fixed_pcm,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch13/distributed_audio'
SOURCE_PATHS = (
    'codes/chapters/ch13/core/distributed.py',
    'codes/chapters/ch13/core/distributed_audio.py',
    'codes/chapters/ch13/examples/generate_distributed_audio.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch10/sro_closed_loop_demo.py',
    'codes/chapters/ch10/core/engineering.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {name: _sha(validate_parent_chain(ROOT/name).read_bytes()) for name in SOURCE_PATHS}


def prepare_assets():
    """Validate the fixed model and all encoded bytes/scores before any write."""
    if type(SAMPLE_RATE) is not int or SAMPLE_RATE != 16000 or type(SAMPLES) is not int or SAMPLES != 32000:
        raise ValueError('distributed fixed fixture sample rate/length differs')
    report, arrays = run_experiment()
    declared_parameters = parameters()
    literal_signals = validate_fixed_fixture(report, arrays, declared_parameters, LIMITS, FILE_NAMES, OUTPUT_REFERENCES)
    contents, files, samples, decoded = {}, {}, {}, {}
    for key, array in arrays.items():
        filename = FILE_NAMES[key]; blob = pcm16_bytes(array, SAMPLE_RATE)
        rate, decoded[key] = read_pcm16(blob)
        try:
            with wave.open(io.BytesIO(blob)) as reader:
                actual = (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                          reader.getsampwidth(), reader.getcomptype())
                if actual != (16000, array.shape[0], 32000, 2, 'NONE'):
                    raise ValueError('distributed fixed fixture encoded WAV format differs: '+key)
                integers = np.frombuffer(reader.readframes(32000), dtype='<i2')
            actual_pcm = integers.reshape(32000, array.shape[0]).T.astype(float)/32768
        except (wave.Error, EOFError) as error:
            raise ValueError('distributed fixed fixture encoded WAV is invalid: '+key) from error
        if type(rate) is not int or rate != 16000 or not np.array_equal(decoded[key], actual_pcm):
            raise ValueError('distributed fixed fixture PCM decoder differs from actual WAV: '+key)
        contents[filename] = blob
        files[filename] = {'channels': array.shape[0], 'samples_per_channel': SAMPLES,
                           'sample_rate_hz': rate, 'sha256': _sha(blob)}
    for key, array in arrays.items():
        ref = decoded.get(OUTPUT_REFERENCES.get(key))
        samples[key] = {'file': FILE_NAMES[key], 'float_measurements': report['float_measurements'][key],
                        'finite_window_analytic_expected': report['analytic_expected'].get(key),
                        'float_components': report['float_components'].get(key),
                        'pcm_measurements': measure_signal(decoded[key], key, reference=ref, pcm=True),
                        'quantization_max_abs_error': float(np.max(abs(array-decoded[key])))}
    validate_fixed_pcm(decoded, {key: sample['pcm_measurements'] for key, sample in samples.items()},
                       literal_signals, OUTPUT_REFERENCES)
    manifest = {'schema_version': 1, 'origin': 'original known-covariance instantaneous mathematical node mixtures',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'common_export_gain': 1.,
                'source_sha256': source_digests(), 'parameters': declared_parameters, 'files': files, 'samples': samples,
                'float_clock_control': report['clock'], 'float_transport_control': report['transport'],
                'finite_steady_basis_covariance': report['finite_steady_basis_covariance'],
                'finite_steady_source_noise_cross': report['finite_steady_source_noise_cross'],
                'finite_steady_target_power': report['finite_steady_target_power'],
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'pcm_encode_multiplier': 32768, 'pcm_decode_divisor': 32768,
                'pcm_half_step_error_bound': .5/32768,
                'pcm': 'signed little-endian PCM16, nearest-even rounding; no dither, saturation or per-file gain',
                'limits': LIMITS, 'listening': 'Start at low device volume; no autoplay or listening study.',
                'io_scope': 'ordinary-member preflight for local non-concurrent use; not race-proof or crash-durable publication'}
    contents['MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return manifest, contents


def check_assets(directory=DEFAULT_OUTPUT):
    """Require exactly 18 ordinary members and replay current source completely."""
    directory = Path(directory)
    validate_asset_directory(directory, set(FILE_NAMES.values())|{'MANIFEST.json'}, check=True)
    manifest = strict_json_loads((directory/'MANIFEST.json').read_bytes())
    if not isinstance(manifest, dict) or manifest.get('source_sha256') != source_digests():
        raise ValueError('distributed audio source set or SHA differs')
    files, samples = manifest.get('files'), manifest.get('samples')
    if (not isinstance(files, dict) or set(files) != set(FILE_NAMES.values())
            or not isinstance(samples, dict) or set(samples) != set(FILE_NAMES)
            or any(not isinstance(v, dict) for v in (*files.values(), *samples.values()))):
        raise ValueError('distributed audio manifest record sets differ')
    decoded = {}
    for key, filename in FILE_NAMES.items():
        blob = (directory/filename).read_bytes(); channels = 4 if key.startswith('array_') else 1
        try:
            with wave.open(io.BytesIO(blob)) as reader:
                actual = (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                          reader.getsampwidth(), reader.getcomptype())
                if actual != (SAMPLE_RATE, channels, SAMPLES, 2, 'NONE'):
                    raise ValueError('distributed audio PCM format differs: '+filename)
            _, decoded[key] = read_pcm16(blob)
        except (wave.Error, EOFError) as error:
            raise ValueError('invalid distributed WAV: '+filename) from error
        record = {'channels': channels, 'samples_per_channel': SAMPLES,
                  'sample_rate_hz': SAMPLE_RATE, 'sha256': _sha(blob)}
        if not same_metadata(files[filename], record) or samples[key].get('file') != filename:
            raise ValueError('distributed audio WAV metadata/SHA differs: '+filename)
    for key in FILE_NAMES:
        reference = decoded.get(OUTPUT_REFERENCES.get(key))
        score = measure_signal(decoded[key], key, reference=reference, pcm=True)
        if not same_metadata(samples[key].get('pcm_measurements'), score):
            raise ValueError('distributed audio actual PCM scores differ: '+key)
    expected, contents = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('distributed audio metadata/model replay differs')
    if any((directory/name).read_bytes() != blob for name, blob in contents.items()):
        raise ValueError('distributed audio byte replay differs from current source/environment')
    return manifest


def generate_assets(directory=DEFAULT_OUTPUT, *, check=False):
    if type(check) is not bool:
        raise ValueError('check must be bool')
    if check:
        return check_assets(directory)
    directory = Path(directory)
    members = set(FILE_NAMES.values())|{'MANIFEST.json'}
    validate_asset_directory(directory, members, check=False)
    manifest, contents = prepare_assets()
    directory.mkdir(parents=True, exist_ok=True)
    validate_asset_directory(directory, members, check=False)
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
