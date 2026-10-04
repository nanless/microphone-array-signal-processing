"""Generate six E06-39 WAVs, or read/check them without changing any file.

Run python -m codes.chapters.ch06.examples.generate_apa_audio [--check].
Only the generation mode writes assets. --check reads all actual PCM, checks
source/file SHA and scores it; deterministic source replay is supplemental.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import wave
import numpy as np

from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch06.core.apa_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, measure_pcm, run_experiment, parameters,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1]/'apa_audio'
SOURCE_PATHS = (
    'codes/chapters/ch06/core/apa_audio.py',
    'codes/chapters/ch06/examples/generate_apa_audio.py',
    'codes/chapters/ch06/core/aec.py',
    'codes/chapters/ch06/core/aec_numeric.py',
    'codes/chapters/ch06/core/aec_affine_projection.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)


# Literal fixed values and exact types; checked before simulation and encoding.
REQUIRED_PARAMETERS = {'exercise_id': 'E06-39',
 'sample_rate_hz': 16000,
 'source_samples': 32000,
 'samples_per_channel': 32013,
 'convolution_tail_samples': 13,
 'seed': 20261001,
 'random_bit_generator': 'PCG64',
 'ar_coefficient': 0.98,
 'ar_initial_previous_sample': 0.0,
 'excitation_standard_deviation': 0.015,
 'true_path_current_first': [0.6,
                             0.0,
                             0.0,
                             -0.2,
                             0.0,
                             0.0,
                             0.0,
                             0.1,
                             0.0,
                             0.0,
                             0.0,
                             0.0,
                             0.0,
                             0.05,
                             0.0,
                             0.0],
 'filter_length': 16,
 'step_size': 0.2,
 'regularization': 0.001,
 'regularization_units': 'squared digital signal amplitude',
 'projection_orders': [1, 2, 4],
 'observation_noise_standard_deviation': 0.003,
 'random_draw_order': '32000 AR excitation draws, then 32013 independent observation-noise draws',
 'training_interval_samples': [0, 24000],
 'holdout_interval_samples': [24000, 32000],
 'holdout_sample_denominator': 8000,
 'trace_interval_samples': 160,
 'initial_weights': 'all zero',
 'initial_reference_and_projection_history': 'empty/zero past reference',
 'output': 'current prior prediction/error before any microphone-dependent update',
 'freeze': 'all samples at and after 24000; reference and projection histories still advance',
 'common_export_gain': 1.0,
 'alignment': 'same causal sample clock, no fitted compensation',
 'export_condition': 'observation noise standard deviation .003',
 'noiseless_control': 'in-memory generated component, no extra WAV'}

def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_assets():
    if not same_metadata(parameters(), REQUIRED_PARAMETERS):
        raise ValueError("APA true parameters differ from the fixed asset contract")
    signals, experiment = run_experiment()
    if not same_metadata(experiment.get("parameters"), REQUIRED_PARAMETERS):
        raise ValueError("APA returned parameters differ from the fixed asset contract")
    blobs, decoded, files = {}, {}, {}
    for key, x in signals.items():
        name = FILE_NAMES[key]
        blob = pcm16_bytes(x, SAMPLE_RATE)
        rate, actual = read_pcm16(blob)
        blobs[name], decoded[key] = blob, actual
        files[name] = {'sha256': sha(blob), 'sample_rate_hz': rate,
                       'channels': actual.shape[0], 'samples_per_channel': actual.shape[1]}
    manifest = {'schema_version': 1, 'sample_rate_hz': SAMPLE_RATE,
                'samples_per_channel': SAMPLES, 'common_export_gain': 1.,
                'source_sha256': {p: sha((ROOT/p).read_bytes()) for p in SOURCE_PATHS},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__},
                'files': files, 'experiment': experiment, 'pcm_measurements': measure_pcm(decoded),
                'limits': LIMITS, 'pcm_decode_divisor': 32768,
                'listening': 'Synthetic noise only. Start at low volume; controls never autoplay.'}
    return blobs, manifest


def check_assets(output=DEFAULT_OUTPUT, *, replay=True):
    if type(replay) is not bool:
        raise ValueError('replay must be bool')
    output = Path(output)
    _shared_asset_directory(output, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=True)
    expected = set(FILE_NAMES.values()) | {'MANIFEST.json'}
    if (output.is_symlink() or not output.is_dir()
            or {p.name for p in output.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in output.iterdir())):
        raise ValueError('apa_audio must contain exactly six ordinary WAVs and MANIFEST.json')
    manifest = strict_json_loads((output/'MANIFEST.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be an object')
    if manifest.get('source_sha256') != {p: sha((ROOT/p).read_bytes()) for p in SOURCE_PATHS}:
        raise ValueError('APA generator source set or SHA is stale')
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or not isinstance(manifest.get('files'), dict)
            or any(not isinstance(row, dict) for row in manifest['files'].values())
            or type(manifest.get('sample_rate_hz')) is not int or type(manifest.get('samples_per_channel')) is not int
            or type(manifest.get('common_export_gain')) not in (int, float)
            or set(manifest['files']) != set(FILE_NAMES.values()) or manifest['sample_rate_hz'] != SAMPLE_RATE
            or manifest['samples_per_channel'] != SAMPLES or manifest['common_export_gain'] != 1):
        raise ValueError('APA manifest file set, clock, length or shared gain differs')
    decoded = {}
    for key, name in FILE_NAMES.items():
        blob = (output/name).read_bytes()
        record = manifest['files'][name]
        if not same_metadata(record, {'sha256': sha(blob), 'sample_rate_hz': SAMPLE_RATE,
                                      'channels': 1, 'samples_per_channel': SAMPLES}):
            raise ValueError('APA file SHA or record differs: '+name)
        with wave.open(io.BytesIO(blob), 'rb') as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, 1, SAMPLES, 2, 'NONE'):
                raise ValueError('APA PCM format differs: '+name)
            if len(reader.readframes(SAMPLES)) != SAMPLES*2:
                raise ValueError('APA PCM payload truncated: '+name)
        _, decoded[key] = read_pcm16(blob)
    if not same_metadata(manifest.get('pcm_measurements'), measure_pcm(decoded)):
        raise ValueError('APA actual PCM scores differ')
    if replay:
        blobs, current = prepare_assets()
        if not same_metadata(manifest, current) or any((output/name).read_bytes() != blob for name, blob in blobs.items()):
            raise ValueError('APA manifest or audio differs from current deterministic source')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.check:
        manifest = check_assets(args.output_dir)
    else:
        _shared_asset_directory(args.output_dir, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=False)
        if args.output_dir.is_symlink():
            raise ValueError('output must be an ordinary directory')
        names = set(FILE_NAMES.values()) | {'MANIFEST.json'}
        if args.output_dir.exists() and (not args.output_dir.is_dir()
                or {p.name for p in args.output_dir.iterdir()} - names
                or any(p.is_symlink() or not p.is_file() for p in args.output_dir.iterdir())):
            raise ValueError('existing output members must be expected ordinary files')
        blobs, manifest = prepare_assets()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in blobs.items():
            (args.output_dir/name).write_bytes(blob)
        (args.output_dir/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        check_assets(args.output_dir, replay=False)
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output_dir), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
