"""Generate or strictly read-check six known propagation-time control WAVs.

Preflight protects a local non-concurrent write, not concurrent races or crash
persistence. Checking re-scores actual PCM and replays all sources in memory.
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
from codes.chapters.ch00.io_contracts import same_metadata, strict_json_loads, validate_asset_directory, validate_parent_chain
from codes.chapters.ch07.core.delay_audio import (FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, SCORING_INTERVAL,
    analytic_measurements, parameters, generate_experiment, measure_signal, measure_regression)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch07/delay_audio'
SOURCE_PATHS = (
    'codes/chapters/ch07/core/delay_audio.py',
    'codes/chapters/ch07/examples/generate_delay_audio.py',
    'codes/chapters/ch07/core/prediction_delays.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = set(FILE_NAMES.values()) | {'MANIFEST.json'}

# Reviewed exact values/types, independent of the runtime declaration function.
REQUIRED_PARAMETERS = {'exercise_id': 'E07-24',
 'sample_rate_hz': 16000,
 'hop_samples': 128,
 'source_samples': 32256,
 'samples_per_channel': 33024,
 'source_period_samples': 1536,
 'source_period_frames': 12,
 'source_periods': 21,
 'pulse_start_frames_in_period': [0, 4],
 'pulse_samples': 128,
 'frequency_hz': 500.0,
 'amplitude': 0.1,
 'pulse_phase': 'cosine at pulse-local sample clock, phase zero; four cycles per pulse',
 'pulse_envelope': 'sin(linspace(0,pi,128)) squared, both endpoints included',
 'pulse_repetition': 'repeat the same finite pulse values; source zero outside the two pulse supports',
 'observation_delays_samples': {'reference': 384, 'auxiliary': 0},
 'common_auxiliary_history_delay_samples': 384,
 'aligned_auxiliary_history_delay_samples': 768,
 'reference_protection_delay_samples': 384,
 'source_time_gaps_samples': {'common': 0, 'aligned': 384},
 'physical_propagation_tail_samples': 384,
 'maximum_history_padding_samples': 768,
 'negative_time_samples': 'zero',
 'complete_retained_support_samples': [0, 33024],
 'channel_order': ['late_reference', 'early_auxiliary'],
 'regression': 'one auxiliary real history, lambda=1; ordinary transpose; no ridge or reference-history '
               'column',
 'scoring_interval_samples': [1536, 30720],
 'scoring_samples_per_channel': 29184,
 'scoring_source_periods': 19,
 'common_export_gain': 1.0,
 'alignment': 'same causal receiver sample clock; no fitted propagation delay or export gain',
 'noise': 'none',
 'randomness': 'none'}
REQUIRED_LIMITS = 'Original deterministic 500 Hz pulse control, not speech, STFT-WPE, a room measurement or device audio. Known integer propagation and history delays; one real regressor, unit prescribed powers, no ridge. The corrected history has disjoint source-time support only in this declared pulse fixture. Alignment does not generally prevent predictable-target cancellation. No delay/gain fitting to a clean output, random independence, noise, human listening test or industrial performance claim. The common residual is deliberately zero; the aligned residual deliberately duplicates the reference PCM. Physical propagation tail 384 and maximum history padding 768 are different quantities.'


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _metadata(blob):
    with wave.open(io.BytesIO(blob), 'rb') as reader:
        return {'sample_rate_hz': reader.getframerate(), 'channels': reader.getnchannels(),
                'samples_per_channel': reader.getnframes(), 'sample_width_bytes': reader.getsampwidth(),
                'compression': reader.getcomptype()}


def _integers(blob):
    with wave.open(io.BytesIO(blob), 'rb') as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != 'NONE':
            raise ValueError('uncompressed PCM16 required')
        return np.frombuffer(reader.readframes(reader.getnframes()), '<i2').reshape(-1, reader.getnchannels()).T.astype(np.int64)


def _integer_measurements(blob, reference_blob):
    integers, reference = _integers(blob), _integers(reference_blob)
    start, stop = SCORING_INTERVAL
    x, truth = integers[:, start:stop], reference[0, start:stop]
    samples, channels = stop-start, len(x)
    energies = [int(v@v) for v in x]
    errors = [int(v@v) for v in x-truth]
    cross = [int(v@truth) for v in x]
    denominator, reference_energy = samples*32768**2, int(truth@truth)
    if reference_energy <= 0:
        raise ValueError('positive actual PCM reference energy required')
    return {'scoring_interval_samples': [start, stop], 'samples_per_channel': samples, 'channels': channels,
            'integer_squared_sum_E_per_channel': energies, 'integer_denominator_D_per_channel': denominator,
            'mean_square_per_channel': [v/denominator for v in energies],
            'integer_squared_sum_E_all_channels': sum(energies),
            'integer_denominator_D_all_channels': channels*denominator,
            'mean_square_all_channels': sum(energies)/(channels*denominator),
            'integer_reference_squared_sum_per_channel': reference_energy,
            'integer_reference_error_squared_sum_per_channel': errors,
            'integer_output_reference_cross_sum_per_channel': cross,
            'reference_error_mean_square_per_channel': [v/denominator for v in errors],
            'reference_NMSE_per_channel': [v/reference_energy for v in errors],
            'projection_gain_per_channel': [v/reference_energy for v in cross],
            'pcm_decode_divisor': 32768}


def prepare_assets():
    true_parameters = parameters()
    if not same_metadata(true_parameters, REQUIRED_PARAMETERS) or not same_metadata(LIMITS, REQUIRED_LIMITS):
        raise ValueError('delay true parameters differ from the fixed asset contract')
    experiment = generate_experiment()
    signals = experiment['signals']
    if set(signals) != set(FILE_NAMES):
        raise ValueError('all six known propagation controls required')
    blobs, decoded, files, samples = {}, {}, {}, {}
    for name, signal in signals.items():
        channels = 2 if name == 'array' else 1
        if (not isinstance(signal, np.ndarray) or signal.dtype.kind != 'f'
                or signal.shape != (channels, 33024) or not np.isfinite(signal).all()
                or np.max(abs(signal)) >= 1):
            raise ValueError('fixed finite unclipped propagation waveform required: '+name)
        blob = pcm16_bytes(signal, 16000)
        metadata = _metadata(blob)
        if metadata != {'sample_rate_hz': 16000, 'channels': channels, 'samples_per_channel': 33024,
                        'sample_width_bytes': 2, 'compression': 'NONE'}:
            raise ValueError('unexpected delay PCM format: '+name)
        _, pcm = read_pcm16(blob)
        error = float(np.max(abs(signal-pcm)))
        if error > .5/32768+1e-15:
            raise ValueError('quantization exceeded the shared gain-1 rounding bound')
        filename = FILE_NAMES[name]
        blobs[filename], decoded[name] = blob, pcm
        files[filename] = {**metadata, 'sha256': _sha(blob), 'quantization_max_abs_error': error,
                           'peak': float(np.max(abs(pcm)))}
    for name, signal in signals.items():
        filename = FILE_NAMES[name]
        samples[name] = {'file': filename, 'analytic': analytic_measurements(name),
                         'float_measurements': measure_signal(signal, signals['reference']),
                         'pcm_measurements': measure_signal(decoded[name], decoded['reference']),
                         'pcm_integer_measurements': _integer_measurements(blobs[filename], blobs[FILE_NAMES['reference']])}
    manifest = {'schema_version': 1, 'exercise_id': 'E07-24', 'sample_rate_hz': 16000,
                'samples_per_channel': 33024, 'common_export_gain': 1.0,
                'origin': 'original deterministic known propagation and history-delay control',
                'source_sha256': {path: _sha(validate_parent_chain(ROOT/path).read_bytes()) for path in SOURCE_PATHS},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'parameters': true_parameters, 'files': files, 'samples': samples,
                'float_regression': measure_regression(signals['reference'], signals['common_history'], signals['aligned_history']),
                'pcm_regression': measure_regression(decoded['reference'], decoded['common_history'], decoded['aligned_history']),
                'pcm': {'format': 'signed little-endian PCM16', 'scale': 32768,
                        'rounding': 'nearest, ties to even', 'dither': 'none', 'clipping': 'rejected',
                        'max_rounding_error_bound': .5/32768},
                'limits': LIMITS, 'listening': 'Start at low volume; never autoplay. No formal listening study.'}
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return blobs, manifest


def check_assets(output=DEFAULT_OUTPUT):
    output = validate_asset_directory(output, MEMBERS, check=True)
    manifest = strict_json_loads((output/'MANIFEST.json').read_bytes())
    expected_blobs, expected = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('delay manifest differs from current source identities, true parameters or scores')
    actual = {name: (output/filename).read_bytes() for name, filename in FILE_NAMES.items()}
    decoded = {}
    for name, filename in FILE_NAMES.items():
        blob = actual[name]
        metadata = _metadata(blob)
        record = manifest['files'][filename]
        if metadata != {key: record[key] for key in metadata} or _sha(blob) != record['sha256']:
            raise ValueError('delay actual format or SHA differs: '+filename)
        _, decoded[name] = read_pcm16(blob)
    for name, filename in FILE_NAMES.items():
        recorded = manifest['samples'][name]
        if (not same_metadata(measure_signal(decoded[name], decoded['reference']), recorded['pcm_measurements'])
                or not same_metadata(_integer_measurements(actual[name], actual['reference']), recorded['pcm_integer_measurements'])):
            raise ValueError('delay actual PCM scores differ: '+filename)
        if actual[name] != expected_blobs[filename]:
            raise ValueError('delay waveform differs from full in-memory source replay: '+filename)
    if not same_metadata(measure_regression(decoded['reference'], decoded['common_history'], decoded['aligned_history']),
                         manifest['pcm_regression']):
        raise ValueError('delay actual PCM regression differs')
    return manifest


def generate_assets(output=DEFAULT_OUTPUT):
    output = validate_asset_directory(output, MEMBERS, check=False)
    blobs, manifest = prepare_assets()
    output.mkdir(parents=True, exist_ok=True)
    validate_asset_directory(output, MEMBERS, check=False)
    for filename, blob in blobs.items():
        (output/filename).write_bytes(blob)
    (output/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return check_assets(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        manifest = check_assets(args.output) if args.check else generate_assets(args.output)
    except (OSError, ValueError, TypeError, wave.Error) as error:
        parser.exit(1, f'Delay audio verification failed: {error}\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated', 'directory': str(args.output),
                      'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
