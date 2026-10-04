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


# Reviewed fixed fixture values and types, independent of runtime metadata.
REQUIRED_PARAMETERS = {'exercise_id': 'E07-21',
 'sample_rate_hz': 16000,
 'samples_per_channel': 32512,
 'source_samples': 32000,
 'source_frequencies_hz': [200, 500, 900, 1200],
 'amplitude_per_frequency': 0.05,
 'source_phase': 'cosines on absolute sample clock, initial phase zero',
 'source_active_interval_samples': [0, 32000],
 'fade_samples_each_end': 320,
 'fade': 'sin(linspace(0,pi/2,320)) squared, endpoint included; reversed at end',
 'reflection_delay_samples': 512,
 'reflection_delay_seconds': 0.032,
 'paths': {'first': [1.0, 0.5], 'well_second': [1.0, -0.5], 'near_second': [1.0, 0.49]},
 'path_tap_positions': [0, 512],
 'negative_time_samples': 'zero',
 'complete_tail_samples': 512,
 'noise_model': 'two independent Rademacher draws times sqrt(5e-7), added after each path at ALL 32512 '
                'samples',
 'noise_variance_per_channel': 5e-07,
 'noise_shared_across_conditions': True,
 'seed': 20261001,
 'random_bit_generator': 'PCG64',
 'random_draw': 'integers(0,2,size=(2,32512),dtype=int64), map 0/1 to -1/+1',
 'steady_source_population_mean_square': 0.005,
 'regularization': 0.0001,
 'regularization_interpretation': 'noise variance / declared steady source power = 5e-7/.005',
 'designs': {'well_exact': {'weights': [0.5, 0.5],
                            'weights_exact': ['1/2', '1/2'],
                            'direct_response': 1.0,
                            'reflection_response': 0.0,
                            'weight_norm_squared': 0.5,
                            'reflection_cost': 0.0,
                            'regularization_cost': 0.0,
                            'total_cost': 0.0,
                            'regularization': 0.0,
                            'scope': 'direct coefficient sum is one; only reflection and weight norm are '
                                     'penalized',
                            'population_noise_mean_square': 2.5e-07,
                            'population_clean_error_mean_square': 0.0,
                            'population_total_reference_mse': 2.5e-07},
             'near_exact': {'weights': [-49.0, 50.0],
                            'weights_exact': ['-49', '50'],
                            'direct_response': 1.0,
                            'reflection_response': 0.0,
                            'weight_norm_squared': 4901.0,
                            'reflection_cost': 0.0,
                            'regularization_cost': 0.0,
                            'total_cost': 0.0,
                            'regularization': 0.0,
                            'scope': 'direct coefficient sum is one; only reflection and weight norm are '
                                     'penalized',
                            'population_noise_mean_square': 0.0024505,
                            'population_clean_error_mean_square': 0.0,
                            'population_total_reference_mse': 0.0024505},
             'near_regularized': {'weights': [-16.0, 17.0],
                                  'weights_exact': ['-16', '17'],
                                  'direct_response': 1.0,
                                  'reflection_response': 0.33,
                                  'weight_norm_squared': 545.0,
                                  'reflection_cost': 0.1089,
                                  'regularization_cost': 0.0545,
                                  'total_cost': 0.1634,
                                  'regularization': 0.0001,
                                  'scope': 'direct coefficient sum is one; only reflection and weight norm '
                                           'are penalized',
                                  'population_noise_mean_square': 0.0002725,
                                  'population_clean_error_mean_square': 0.0005445000000000001,
                                  'population_total_reference_mse': 0.000817}},
 'scoring_interval_samples': [2400, 29600],
 'scoring_samples_per_channel': 27200,
 'tail_interval_samples': [32000, 32512],
 'alignment': 'same source sample clock; fixed causal one-tap outputs; no fitted gain or delay',
 'common_export_gain': 1.0,
 'channel_order': ['first_path', 'second_path'],
 'statistics': 'one finite fixed-seed fixture; population expectations and actual cross terms are distinct'}
REQUIRED_LIMITS = 'Original known-path mathematical synthesis; no speech, measured room or listening study. Noise is added after the two paths, with common draws across conditions. Rademacher population means/cross moments are not substituted for finite-record moments. Fixed causal one-tap filters preserve the direct coefficient sum, not every frequency response. The constrained regularizer penalizes the delayed reflection and weight norm; it is not a general MINT optimum. Float component scores and actual PCM total errors are separate. No posterior gain/time fitting, clipping, dither or per-file normalization. A 512-sample reflection has 512 polynomial zeros; coefficient difference is not zero distance.'


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {p: _sha((ROOT/p).read_bytes()) for p in SOURCE_PATHS}


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    # Reject changed declarations before encoding or writing any asset.
    if (not same_metadata(parameters(), REQUIRED_PARAMETERS)
            or not same_metadata(LIMITS, REQUIRED_LIMITS)):
        raise ValueError('mint true parameters differ from the fixed asset contract')
    experiment = run_experiment()
    if (not isinstance(experiment, dict)
            or not same_metadata(experiment.get('parameters'), REQUIRED_PARAMETERS)
            or not same_metadata(experiment.get('limits'), REQUIRED_LIMITS)):
        raise ValueError('mint experiment parameters differ from the fixed asset contract')
    if (not isinstance(experiment.get('signals'), dict)
            or set(experiment['signals']) != set(FILE_NAMES)):
        raise ValueError('mint experiment must contain all six fixed signals')
    for key, value in experiment['signals'].items():
        channels = 2 if key.endswith('_array') else 1
        if (not isinstance(value, np.ndarray) or value.dtype.kind != 'f'
                or value.shape != (channels, SAMPLES) or not np.isfinite(value).all()
                or np.any(value < -1) or np.any(value > 32767/32768)):
            raise ValueError('mint fixed finite unclipped waveform shape differs: '+key)
    actual_float = {key: measure_signal(value, experiment['signals']['reference'])
                    for key, value in experiment['signals'].items()}
    if not same_metadata(experiment.get('float_measurements'), actual_float):
        raise ValueError('mint float measurements differ from supplied waveforms')
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
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return blobs, manifest


def _members(output, *, checking):
    return _shared_asset_directory(output, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=checking)


def _strict_equal(actual, expected):
    return same_metadata(actual, expected)


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
