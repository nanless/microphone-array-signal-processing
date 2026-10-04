"""Generate/check six fixed known-channel-removal controls; never autoplay.

--check reads actual PCM and strictly replays the fixed model in memory without
repairing files. Parent/member preflight is not a race or crash guarantee.
"""
from __future__ import annotations
if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import wave
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes
from codes.chapters.ch00.io_contracts import same_metadata, strict_json_loads, validate_asset_directory, validate_parent_chain
from codes.chapters.ch10.core.channel_audio import (FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES,
    SCORING_INTERVAL, parameters, generate_experiment, analytic_measurements, measure_signal)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch10/channel_audio'
SOURCE_PATHS = ('codes/chapters/ch10/core/channel_audio.py',
    'codes/chapters/ch10/examples/generate_channel_audio.py', 'codes/chapters/ch10/core/channel_selection.py',
    'codes/chapters/ch05/core/beamforming.py', 'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py')
MEMBERS = tuple(FILE_NAMES.values())+('MANIFEST.json',)
REQUIRED_PARAMETERS = {'exercise_id': 'E10-34', 'sample_rate_hz': 16000, 'samples_per_channel': 32000,
    'target_frequency_hz': 500.0, 'noise_basis_frequencies_hz': [1500.0, 2500.0, 3500.0],
    'amplitude_each': 0.1, 'phase': 'sine, phase zero at the common absolute sample origin',
    'fade_samples_each_end': 320,
    'envelope': 'sin(linspace(0,pi/2,320)) squared; reversed at end; both endpoints zero',
    'common_export_gain': 1.0, 'tail_samples': 0,
    'scoring_interval_samples': [2400, 29600], 'scoring_samples_per_channel': 27200,
    'target_cycles_in_score': 850, 'noise_basis_cycles_in_score': [2550, 4250, 5950],
    'mixing_matrix_B': [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
    'steering': [1.0, 1.0, 1.0], 'known_failed_channel': 0, 'selected_channels': [1, 2],
    'fault': 'replace channel zero by zeros before applying the former full weights',
    'noise_covariance': 'known score-window time covariance 0.005*B*B.T; fixed real weights across all tones',
    'solver': 'unique Chapter05 MVDR; selection by Chapter10 channel_selection; no loading',
    'scoring': 'same samples and common reference; no output gain/time matching',
    'randomness': 'none', 'noise_scope': 'three deterministic orthogonal basis tones, not random noise'}
REQUIRED_LIMITS = ('Known failed-channel, fixed-real-weight time-domain covariance control. '
    'The target and three noise bases are mathematical sinusoids; finite-window orthogonality '
    'is not random independence. The broadband noise covariance is known, not a per-frequency '
    'SCM estimate. Channel zero is prescribed failed and replaced by zero, not diagnosed. '
    'Selection removes both covariance axes and the matching steering and observation entries; '
    'it does not preserve the former noise optimum. No physical propagation, speech, device '
    'recording, blind fault detection, per-file normalization, reference gain/delay fitting '
    'or formal listening study. Equal analytic total NMSE does not imply equal target retention; '
    'the two actual PCM NMSE values need not be exactly equal.')
EXPECTED_NAMES = {name: 'channel_'+name+'.wav' for name in (
    'reference', 'healthy_array', 'faulty_array', 'healthy_output', 'stale_output', 'recomputed_output')}
REQUIRED_ANALYTIC = {
    'healthy_output': {'target_gain': 1.0, 'target_mean_square': .005, 'noise_mean_square': .0025,
        'output_mean_square': .0075, 'reference_error_mean_square': .0025, 'reference_NMSE': .5},
    'stale_output': {'target_gain': 0.0, 'target_mean_square': 0.0, 'noise_mean_square': .0025,
        'output_mean_square': .0025, 'reference_error_mean_square': .0075, 'reference_NMSE': 1.5},
    'recomputed_output': {'target_gain': 1.0, 'target_mean_square': .005, 'noise_mean_square': .0075,
        'output_mean_square': .0125, 'reference_error_mean_square': .0075, 'reference_NMSE': 1.5}}


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _codes(blob, channels):
    if type(blob) is not bytes:
        raise ValueError('actual PCM buffer must be bytes')
    try:
        with wave.open(io.BytesIO(blob), 'rb') as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                    reader.getsampwidth(), reader.getcomptype()) != (16000, channels, 32000, 2, 'NONE'):
                raise ValueError('fixed channel PCM16 format required')
            raw = reader.readframes(32000)
        if len(raw) != 64000*channels:
            raise ValueError('truncated channel PCM')
        return np.array(struct.unpack('<'+'h'*(32000*channels), raw), dtype=np.int64).reshape(-1, channels).T
    except (wave.Error, EOFError, struct.error) as error:
        raise ValueError('invalid PCM buffer') from error


def integer_measurements(blob, reference_blob, channels):
    values, reference = _codes(blob, channels), _codes(reference_blob, 1)
    lo, hi = (2400, 29600)
    x, truth = values[:, lo:hi], reference[0, lo:hi]
    power = [sum(int(v)**2 for v in row) for row in x]
    error = [sum((int(v)-int(t))**2 for v, t in zip(row, truth)) for row in x]
    cross = [sum(int(v)*int(t) for v, t in zip(row, truth)) for row in x]
    ref = sum(int(v)**2 for v in truth)
    if ref <= 0:
        raise ValueError('positive actual PCM reference energy required')
    denominator = 27200*32768**2
    return {'scoring_interval_samples': [lo, hi], 'samples_per_channel': 27200, 'channels': channels,
        'integer_squared_sum_E_per_channel': power, 'integer_denominator_D_per_channel': denominator,
        'mean_square_per_channel': [v/denominator for v in power],
        'integer_squared_sum_E_all_channels': sum(power), 'integer_denominator_D_all_channels': channels*denominator,
        'mean_square_all_channels': sum(power)/(channels*denominator),
        'integer_reference_squared_sum_per_channel': [ref]*channels,
        'integer_reference_error_squared_sum_per_channel': error,
        'integer_output_reference_cross_sum_per_channel': cross,
        'reference_error_mean_square_per_channel': [v/denominator for v in error],
        'reference_NMSE_per_channel': [v/ref for v in error],
        'projection_gain_per_channel': [v/ref for v in cross], 'pcm_decode_divisor': 32768,
        'projection_scope': 'reference projection diagnostic only; no gain compensation applied'}


def _validate_experiment(experiment):
    keys = {'signals', 'noise_bases', 'noise_covariance', 'full_weights', 'selected_covariance',
            'selected_steering', 'selected_weights', 'selection_matrix', 'stale_remaining_weights'}
    if type(experiment) is not dict or set(experiment) != keys:
        raise ValueError('fixed experiment fields required')
    signals = experiment['signals']
    if type(signals) is not dict or set(signals) != set(EXPECTED_NAMES):
        raise ValueError('exact six signal roles required')
    for name, signal in signals.items():
        channels = 3 if name.endswith('_array') else 1
        if (type(signal) is not np.ndarray or signal.dtype != np.dtype('float64')
                or signal.shape != (channels, 32000) or not np.isfinite(signal).all()
                or np.max(abs(signal)) > 32767/32768 or np.any(signal[:, 0] != 0)
                or np.any(signal[:, -1] != 0)):
            raise ValueError('finite fixed-shape unclipped float64 waveform required: '+name)
    # Literal model equations guard the trusted builder before any publication.
    time = np.arange(32000)/16000
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    envelope = np.ones(32000); envelope[:320], envelope[-320:] = fade, fade[::-1]
    reference = .1*np.sin(2*np.pi*500*time)*envelope
    bases = np.array([.1*np.sin(2*np.pi*f*time)*envelope for f in (1500, 2500, 3500)])
    matrix = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.]])
    healthy = reference[None]+matrix@bases; faulty = healthy.copy(); faulty[0] = 0
    exact = {'noise_bases': bases, 'noise_covariance': .005*(matrix@matrix.T),
             'selected_covariance': .005*np.array([[2., 1.], [1., 2.]]),
             'selected_steering': np.ones(2), 'selection_matrix': np.array([[0., 1., 0.], [0., 0., 1.]])}
    for key, expected in exact.items():
        actual = experiment[key]
        if (type(actual) is not np.ndarray or actual.dtype.kind != 'f'
                or actual.shape != expected.shape or not np.array_equal(actual, expected)):
            raise ValueError('declared model field differs: '+key)
    for key, expected in [('full_weights', [1., -.5, .5]), ('selected_weights', [.5, .5]),
                          ('stale_remaining_weights', [-.5, .5])]:
        actual = experiment[key]
        if (type(actual) is not np.ndarray or actual.dtype.kind != 'f' or actual.shape != (len(expected),)
                or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=2e-15, rtol=0)):
            raise ValueError('fixed MVDR control differs: '+key)
    from codes.chapters.ch05.core.beamforming import apply_beamformer
    expected_signals = {'reference': reference[None], 'healthy_array': healthy, 'faulty_array': faulty,
        'healthy_output': apply_beamformer(healthy[:, None, :], experiment['full_weights']).real,
        'stale_output': apply_beamformer(faulty[:, None, :], experiment['full_weights']).real,
        'recomputed_output': apply_beamformer(faulty[[1, 2], None, :], experiment['selected_weights']).real}
    for key, expected in expected_signals.items():
        if not np.array_equal(signals[key], expected):
            raise ValueError('waveform differs from declared model: '+key)


def prepare_assets():
    p = parameters()
    if (not same_metadata(p, REQUIRED_PARAMETERS) or not same_metadata(LIMITS, REQUIRED_LIMITS)
            or type(SAMPLE_RATE) is not int or SAMPLE_RATE != 16000
            or type(SAMPLES) is not int or SAMPLES != 32000
            or not same_metadata(FILE_NAMES, EXPECTED_NAMES)
            or not same_metadata(MEMBERS, tuple(EXPECTED_NAMES.values())+('MANIFEST.json',))
            or not same_metadata(SCORING_INTERVAL, (2400, 29600))):
        raise ValueError('fixed channel parameters, types or scope differ')
    sources = {}
    for name in SOURCE_PATHS:
        path = validate_parent_chain(ROOT/name)
        if not path.is_file():
            raise ValueError('missing ordinary source: '+name)
        sources[name] = _sha(path.read_bytes())
    experiment = generate_experiment(); _validate_experiment(experiment)
    analytic = analytic_measurements()
    if not same_metadata(analytic, REQUIRED_ANALYTIC):
        raise ValueError('fixed analytic control values/types differ')
    blobs, files, samples = {}, {}, {}
    for name, signal in experiment['signals'].items():
        blob = pcm16_bytes(signal, 16000)
        channels = signal.shape[0]; decoded = _codes(blob, channels).astype(float)/32768
        quantization_error = float(np.max(abs(decoded-signal)))
        if quantization_error > .5/32768+1e-15:
            raise ValueError('PCM rounding bound exceeded')
        filename = FILE_NAMES[name]; blobs[filename] = blob
        files[filename] = {'channels': channels, 'samples_per_channel': 32000, 'sample_rate_hz': 16000,
            'sample_width_bytes': 2, 'sha256': _sha(blob), 'peak': float(np.max(abs(decoded))),
            'quantization_max_abs_error': quantization_error}
    reference_blob = blobs[FILE_NAMES['reference']]
    for name, signal in experiment['signals'].items():
        blob = blobs[FILE_NAMES[name]]; channels = signal.shape[0]
        decoded = _codes(blob, channels).astype(float)/32768
        ref = _codes(reference_blob, 1).astype(float)/32768
        samples[name] = {'file': FILE_NAMES[name], 'float_measurements': measure_signal(signal, experiment['signals']['reference']),
            'pcm_measurements': measure_signal(decoded, ref),
            'pcm_integer_measurements': integer_measurements(blob, reference_blob, channels)}
    manifest = {'schema_version': 1, 'exercise_id': 'E10-34', 'sample_rate_hz': 16000,
        'samples_per_channel': 32000, 'common_export_gain': 1.0,
        'origin': 'mathematical known-channel-removal fixed-real-weight control', 'parameters': p,
        'analytic': analytic, 'files': files, 'samples': samples,
        'model': {key: value.tolist() for key, value in experiment.items() if key not in ('signals', 'noise_bases')},
        'noise_basis_gram_in_score': (experiment['noise_bases'][:, 2400:29600]@experiment['noise_bases'][:, 2400:29600].T/27200).tolist(),
        'source_sha256': sources, 'limits': LIMITS,
        'pcm': {'encoding': 'signed little-endian PCM16', 'divisor': 32768, 'rounding': 'nearest-even',
                'dither': 'none', 'clipping': 'rejected'},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'system': platform.system(), 'machine': platform.machine()}}
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return blobs, manifest


def check_assets(output=DEFAULT_OUTPUT):
    output = validate_asset_directory(output, MEMBERS, check=True)
    recorded = strict_json_loads((output/'MANIFEST.json').read_bytes())
    expected_blobs, expected = prepare_assets()
    if not same_metadata(recorded, expected):
        raise ValueError('channel manifest differs from fixed current source/model replay')
    for name, filename in FILE_NAMES.items():
        blob = (output/filename).read_bytes(); channels = 3 if name.endswith('_array') else 1
        measured = integer_measurements(blob, (output/FILE_NAMES['reference']).read_bytes(), channels)
        if not same_metadata(recorded['samples'][name]['pcm_integer_measurements'], measured):
            raise ValueError('actual PCM measurement differs: '+filename)
        if _sha(blob) != recorded['files'][filename]['sha256'] or blob != expected_blobs[filename]:
            raise ValueError('actual PCM SHA or full model bytes differ: '+filename)
    return recorded


def generate_assets(output=DEFAULT_OUTPUT):
    output = validate_asset_directory(output, MEMBERS, check=False)
    blobs, manifest = prepare_assets()
    manifest_blob = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    output.mkdir(parents=True, exist_ok=True)
    for name, blob in blobs.items():
        (output/name).write_bytes(blob)
    (output/'MANIFEST.json').write_bytes(manifest_blob)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = check_assets(args.output) if args.check else generate_assets(args.output)
    print(json.dumps({'mode': 'verified' if args.check else 'generated', 'directory': str(args.output),
                      'pcm_integer_measurements': {name: row['pcm_integer_measurements'] for name, row in manifest['samples'].items()}}, indent=2))


if __name__ == '__main__':
    main()
