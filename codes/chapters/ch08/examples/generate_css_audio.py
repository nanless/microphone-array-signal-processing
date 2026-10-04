"""Generate or strictly read-check four known-slot polarity/gain control WAVs.

Local preflight is not a concurrent-write or crash-persistence guarantee.
Checking reads actual PCM, recomputes scores, and replays all sources in memory.
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
from codes.chapters.ch08.core.css import overlap_application_gain
from codes.chapters.ch08.core.css_audio import (FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, SCORING_INTERVALS,
    analytic_measurements, parameters, generate_experiment, measure_signal)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch08/css_audio'
SOURCE_PATHS = ('codes/chapters/ch08/core/css_audio.py',
                'codes/chapters/ch08/examples/generate_css_audio.py',
                'codes/chapters/ch08/core/css.py',
                'codes/chapters/ch02/core/conventions.py',
                'codes/chapters/ch00/core/audio_samples.py',
                'codes/chapters/ch00/io_contracts.py')
MEMBERS = set(FILE_NAMES.values()) | {'MANIFEST.json'}
# Exact fixed contract independent of the runtime parameters declaration.
REQUIRED_PARAMETERS = {'exercise_id': 'E08-32', 'sample_rate_hz': 16000, 'samples_per_channel': 32000,
 'frequencies_hz': [500.0, 1500.0], 'amplitudes': [0.1, 0.1],
 'channel_order': ['fixed_source_slot_1', 'fixed_source_slot_2'],
 'phase': 'sine at the common absolute sample clock, phase zero',
 'envelope': 'sin(linspace(0,pi/2,320)) squared at each end, reversed at end; endpoints zero',
 'fade_samples_each_end': 320, 'first_block_samples': [0, 19200], 'second_block_samples': [12800, 32000],
 'overlap_samples': [12800, 19200], 'overlap_samples_per_channel': 6400,
 'overlap_cycles_per_channel': [200, 600], 'second_block_multiplier': -2.0,
 'crossfade': 'complementary linear weights: current r=linspace(0,1,6400), previous 1-r, endpoints included',
 'slot_matching': 'absolute centered correlation on observed overlap; apply current indices for previous',
 'application_gain_fit': 'ordinary real raw-sample LS current dot previous / current dot current; no centering',
 'polarity_control': 'sign of the overlap application gain only; magnitude remains 2',
 'scoring_intervals_samples': {'primary': [2400, 29600], 'post_overlap': [19200, 29600]},
 'scoring_samples_per_channel': {'primary': 27200, 'post_overlap': 10400},
 'post_overlap_cycles_per_channel': [325, 975], 'common_export_gain': 1.0,
 'noise': 'none', 'randomness': 'none',
 'alignment': 'same known slot order and receiving sample clock; no delay or clean-reference gain fit'}
REQUIRED_LIMITS = ('Original known two-slot sinusoid stitching control, not speech, a blind CSS separator, '
          'STFT separation, speaker recognition or industrial performance. The source-slot order and '
          'sample-clock correspondence are fixed. Absolute correlation assigns slots but does not '
          'repair polarity or gain. A real scalar is fitted only between observed overlapping blocks, '
          'not to clean truth; it cannot repair arbitrary distortion, delay or frequency-dependent phase. '
          'Reference and fully corrected PCM deliberately match. No per-file normalization, fitted '
          'reference delay, noise, random independence or formal listening study. Fixed-window '
          'uncompensated MSE/NMSE measures the gain error; SI-SDR is not used as a gain check.')


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _metadata(blob):
    with wave.open(io.BytesIO(blob), 'rb') as reader:
        return {'sample_rate_hz': reader.getframerate(), 'channels': reader.getnchannels(),
                'samples_per_channel': reader.getnframes(), 'sample_width_bytes': reader.getsampwidth(),
                'compression': reader.getcomptype()}


def _integers(blob):
    with wave.open(io.BytesIO(blob), 'rb') as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth(), reader.getcomptype()) != (16000, 2, 32000, 2, 'NONE'):
            raise ValueError('fixed two-channel PCM16 required')
        return np.frombuffer(reader.readframes(32000), '<i2').reshape(-1, 2).T.astype(np.int64)


def _integer_measurements(blob, reference_blob):
    integers, reference = _integers(blob), _integers(reference_blob)
    result = {}
    for key, (start, stop) in SCORING_INTERVALS.items():
        x, truth = integers[:, start:stop], reference[:, start:stop]
        energies = [int(v@v) for v in x]
        errors = [int(v@v) for v in x-truth]
        cross = [int(v@r) for v, r in zip(x, truth)]
        ref = [int(v@v) for v in truth]
        denominator = (stop-start)*32768**2
        if min(ref) <= 0:
            raise ValueError('positive actual PCM reference energy required')
        result[key] = {'scoring_interval_samples': [start, stop], 'samples_per_channel': stop-start, 'channels': 2,
                      'integer_squared_sum_E_per_channel': energies, 'integer_denominator_D_per_channel': denominator,
                      'mean_square_per_channel': [v/denominator for v in energies],
                      'integer_squared_sum_E_all_channels': sum(energies),
                      'integer_denominator_D_all_channels': 2*denominator,
                      'mean_square_all_channels': sum(energies)/(2*denominator),
                      'integer_reference_squared_sum_per_channel': ref,
                      'integer_reference_error_squared_sum_per_channel': errors,
                      'integer_output_reference_cross_sum_per_channel': cross,
                      'reference_error_mean_square_per_channel': [v/denominator for v in errors],
                      'reference_NMSE_per_channel': [v/r for v, r in zip(errors, ref)],
                      'projection_gain_per_channel': [v/r for v, r in zip(cross, ref)], 'pcm_decode_divisor': 32768}
    return result


def prepare_assets():
    true_parameters = parameters()
    if (not same_metadata(true_parameters, REQUIRED_PARAMETERS) or not same_metadata(LIMITS, REQUIRED_LIMITS)
            or type(SAMPLE_RATE) is not int or SAMPLE_RATE != 16000
            or type(SAMPLES) is not int or SAMPLES != 32000
            or not same_metadata(FILE_NAMES, {k: 'css_'+k+'.wav' for k in ('reference', 'naive', 'polarity', 'corrected')})
            or not same_metadata(SCORING_INTERVALS, {'primary': (2400, 29600), 'post_overlap': (19200, 29600)})):
        raise ValueError('CSS true parameters differ from the fixed asset contract')
    experiment = generate_experiment()
    signals = experiment['signals']
    if set(signals) != set(FILE_NAMES):
        raise ValueError('all four CSS controls required')
    if experiment['overlap_matching']['current_indices_for_previous'] != [0, 1]:
        raise ValueError('fixed overlap matching differs')
    gains = np.asarray(experiment['overlap_application_gain'])
    if gains.shape != (2,) or not np.allclose(gains, [-.5, -.5], atol=2e-15, rtol=0):
        raise ValueError('observed overlap gain differs from the prescribed control')
    blobs, decoded, files, samples = {}, {}, {}, {}
    for name, signal in signals.items():
        if (not isinstance(signal, np.ndarray) or signal.dtype.kind != 'f' or signal.shape != (2, 32000)
                or not np.isfinite(signal).all() or np.max(abs(signal)) >= 1
                or np.any(signal[:, 0] != 0) or np.any(signal[:, -1] != 0)):
            raise ValueError('fixed finite unclipped CSS waveform required: '+name)
        blob = pcm16_bytes(signal, 16000)
        metadata = _metadata(blob)
        if metadata != {'sample_rate_hz': 16000, 'channels': 2, 'samples_per_channel': 32000,
                        'sample_width_bytes': 2, 'compression': 'NONE'}:
            raise ValueError('unexpected CSS PCM format')
        _, pcm = read_pcm16(blob)
        error = float(np.max(abs(signal-pcm)))
        if error > .5/32768+1e-15:
            raise ValueError('quantization exceeded the gain-1 rounding bound')
        filename = FILE_NAMES[name]
        blobs[filename], decoded[name] = blob, pcm
        files[filename] = {**metadata, 'sha256': _sha(blob), 'quantization_max_abs_error': error,
                           'peak': float(np.max(abs(pcm)))}
    for name, signal in signals.items():
        samples[name] = {'file': FILE_NAMES[name], 'analytic': analytic_measurements(name),
                         'float_measurements': measure_signal(signal, signals['reference']),
                         'pcm_measurements': measure_signal(decoded[name], decoded['reference']),
                         'pcm_integer_measurements': _integer_measurements(blobs[FILE_NAMES[name]], blobs[FILE_NAMES['reference']])}
    overlap = experiment['observed_overlap']
    overlap_pcm = {}
    for key in ('previous', 'current'):
        _, overlap_pcm[key] = read_pcm16(pcm16_bytes(overlap[key], 16000))
    manifest = {'schema_version': 1, 'exercise_id': 'E08-32', 'sample_rate_hz': 16000,
                'samples_per_channel': 32000, 'common_export_gain': 1.0,
                'origin': 'original fixed-slot same-clock polarity/gain stitching control',
                'source_sha256': {path: _sha(validate_parent_chain(ROOT/path).read_bytes()) for path in SOURCE_PATHS},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'parameters': true_parameters, 'files': files, 'samples': samples,
                'overlap_fit': {'matching': experiment['overlap_matching'],
                                'float_application_gain': gains.tolist(),
                                'decoded_pcm_overlap_application_gain': overlap_application_gain(overlap_pcm['previous'], overlap_pcm['current']).tolist(),
                                'pcm_overlap_scope': 'separate in-memory PCM encoding of the two observed blocks; not a new export or re-fit of corrected WAV',
                                'application': 'floating overlap gain applied before common PCM export; decoded gain is diagnostic only'},
                'pcm': {'format': 'signed little-endian PCM16', 'scale': 32768, 'rounding': 'nearest, ties to even',
                        'dither': 'none', 'clipping': 'rejected', 'max_rounding_error_bound': .5/32768},
                'limits': LIMITS, 'listening': 'Start at low volume; never autoplay. No formal listening study.'}
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return blobs, manifest


def check_assets(output=DEFAULT_OUTPUT):
    output = validate_asset_directory(output, MEMBERS, check=True)
    manifest = strict_json_loads((output/'MANIFEST.json').read_bytes())
    expected_blobs, expected = prepare_assets()
    if not same_metadata(manifest, expected):
        raise ValueError('CSS manifest differs from current source identities, true parameters or scores')
    actual = {name: (output/filename).read_bytes() for name, filename in FILE_NAMES.items()}
    decoded = {}
    for name, filename in FILE_NAMES.items():
        blob, record = actual[name], manifest['files'][filename]
        metadata = _metadata(blob)
        if metadata != {key: record[key] for key in metadata} or _sha(blob) != record['sha256']:
            raise ValueError('CSS actual format or SHA differs: '+filename)
        _, decoded[name] = read_pcm16(blob)
    for name, filename in FILE_NAMES.items():
        record = manifest['samples'][name]
        if (not same_metadata(measure_signal(decoded[name], decoded['reference']), record['pcm_measurements'])
                or not same_metadata(_integer_measurements(actual[name], actual['reference']), record['pcm_integer_measurements'])):
            raise ValueError('CSS actual PCM scores differ: '+filename)
        if actual[name] != expected_blobs[filename]:
            raise ValueError('CSS waveform differs from complete in-memory replay: '+filename)
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
        parser.exit(1, f'CSS audio verification failed: {error}\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated', 'directory': str(args.output),
                      'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
