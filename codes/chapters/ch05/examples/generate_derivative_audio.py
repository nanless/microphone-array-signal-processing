"""Generate or strictly read-check four independent E05-22 WAVs.

Run python -m codes.chapters.ch05.examples.generate_derivative_audio [--check].
The check reads actual PCM, re-scores it and never writes or repairs assets.
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
from codes.chapters.ch05.core.derivative_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, generate_signals, measure_signal,
    measure_float_decomposition, parameters,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1]/'derivative_audio'
SOURCE_PATHS = (
    'codes/chapters/ch05/core/derivative_audio.py',
    'codes/chapters/ch05/examples/generate_derivative_audio.py',
    'codes/chapters/ch05/core/beamforming.py',
    'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch03/core/geometry.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)


# Fixed values and exact types, independent of the descriptive function at runtime.
REQUIRED_PARAMETERS = {'exercise_id': 'E05-22',
 'sample_rate_hz': 16000,
 'samples_per_channel': 32002,
 'source_active_interval_s': [0.0, 2.0],
 'sound_speed_m_s': 343.0,
 'positions_m': [[-0.04, 0.0], [0.0, 0.0], [0.04, 0.0]],
 'channel_order': [0, 1, 2],
 'azimuth_deg': 7.843941679416739,
 'phase_centre': 'middle microphone/array centre',
 'frequencies_hz': [1000.0, 3000.0],
 'phase_per_spacing_rad': [0.1, 0.3],
 'source_amplitude_per_frequency': 0.08,
 'fade_duration_s': 0.02,
 'fade': 'continuous linear envelope min(clip(t/.02),clip((2-t)/.02)); zero outside [0,2]',
 'common_propagation_delay_samples': 1.0,
 'propagation_model': 'reference=F(t-1/fs); x_m_clean=F(t-1/fs+r_m dot u/c)',
 'noise_model': 'independent zero-mean Gaussian observation noise at every recording sample, '
                'including the two tail samples',
 'noise_variance_per_channel': [0.0004, 0.0008, 0.0016],
 'seed': 20260522,
 'random_bit_generator': 'PCG64',
 'single_weights': [0.5714285714285714, 0.2857142857142857, 0.14285714285714285],
 'constrained_weights': [0.30769230769230765, 0.3846153846153846, 0.30769230769230765],
 'weight_model': 'nominal broadside exact diagonal covariance; real causal one-tap filter-and-sum',
 'scoring_interval_samples': [2400, 29600],
 'scoring_samples_per_channel': 27200,
 'phasor_convention': 'cos coefficient minus j*sin coefficient; absolute sample clock; '
                      'output/reference ratio',
 'expected_population_steady_window': {'single': {'response_real_imag': [[0.9964315466271612,
                                                                          -0.04278574999149778],
                                                                         [0.9680974922325757,
                                                                          -0.1266515171405741]],
                                                  'steady_clean_distortion_mean_square': 6.04855393845401e-05,
                                                  'white_noise_mean_square': 0.00022857142857142857,
                                                  'total_reference_mean_square': 0.0002890569679559687,
                                                  'weight_norm_squared': 0.42857142857142855},
                                       'constrained': {'response_real_imag': [[0.9969256401710926,
                                                                               7.002913214210914e-19],
                                                                              [0.9725147625388343,
                                                                               -4.951944239072687e-18]],
                                                       'steady_clean_distortion_mean_square': 2.4476478932936524e-06,
                                                       'white_noise_mean_square': 0.00030769230769230765,
                                                       'total_reference_mean_square': 0.0003101399555856013,
                                                       'weight_norm_squared': 0.33727810650887563}},
 'expected_clean_reference_mean_square': 0.0064,
 'common_export_gain': 1.0,
 'alignment': 'fixed shared propagation clock and gain, no fitted correction'}

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    true_parameters = parameters()
    if not same_metadata(true_parameters, REQUIRED_PARAMETERS):
        raise ValueError("derivative true parameters differ from the fixed asset contract")
    signals = generate_signals()
    blobs, decoded, files, samples = {}, {}, {}, {}
    for key, x in signals.items():
        filename = FILE_NAMES[key]
        blob = pcm16_bytes(x, SAMPLE_RATE)
        rate, actual = read_pcm16(blob)
        decoded[key] = actual
        blobs[filename] = blob
        files[filename] = {'sample_rate_hz': rate, 'channels': len(actual),
                           'samples_per_channel': actual.shape[1], 'sha256': _sha(blob)}
    for key, x in signals.items():
        samples[key] = {'file': FILE_NAMES[key],
                        'float_measurements': measure_signal(x, signals['reference']),
                        'pcm_measurements': measure_signal(decoded[key], decoded['reference'], pcm=True),
                        'quantization_max_abs_error': float(np.max(abs(decoded[key]-x)))}
    manifest = {'schema_version': 1, 'sample_rate_hz': SAMPLE_RATE,
                'samples_per_channel': SAMPLES, 'common_export_gain': 1.,
                'source_sha256': {p: _sha((ROOT/p).read_bytes()) for p in SOURCE_PATHS},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'origin': 'original fixed derivative-constraint free-field two-tone synthesis',
                'parameters': true_parameters, 'files': files, 'samples': samples,
                'float_decomposition': measure_float_decomposition(),
                'pcm': 'signed little-endian PCM16, round-to-nearest-even, no dither or per-file gain',
                'pcm_encode_multiplier': 32768, 'pcm_decode_divisor': 32768,
                'pcm_quantization_step': 1/32768, 'pcm_half_step_error_bound': .5/32768,
                'limits': LIMITS, 'listening': 'Start at low volume; no automatic playback. Three-channel playback/downmixing is device dependent.'}
    strict_json_loads(json.dumps(manifest, allow_nan=False))
    return blobs, manifest


def check_assets(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output)
    _shared_asset_directory(output, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=True)
    if not output.is_dir() or {p.name for p in output.iterdir()} != set(FILE_NAMES.values()) | {'MANIFEST.json'}:
        raise ValueError('derivative_audio must contain exactly four WAVs and MANIFEST.json')
    manifest = strict_json_loads((output/'MANIFEST.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be an object')
    sources = {p: _sha((ROOT/p).read_bytes()) for p in SOURCE_PATHS}
    if manifest.get('source_sha256') != sources:
        raise ValueError('derivative_audio source set or SHA is stale')
    expected_blobs, expected_manifest = prepare_assets()
    if not same_metadata(manifest, expected_manifest):
        raise ValueError('derivative_audio manifest parameters or numerical measurements are stale')
    decoded = {}
    for key, filename in FILE_NAMES.items():
        blob = (output/filename).read_bytes()
        if _sha(blob) != manifest['files'][filename]['sha256']:
            raise ValueError('derivative_audio WAV SHA mismatch: '+filename)
        with wave.open(io.BytesIO(blob), 'rb') as reader:
            channels = 3 if key == 'array' else 1
            if (reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, channels, SAMPLES, 2, 'NONE'):
                raise ValueError('derivative_audio PCM format mismatch: '+filename)
        _, decoded[key] = read_pcm16(blob)
    for key, filename in FILE_NAMES.items():
        if measure_signal(decoded[key], decoded['reference'], pcm=True) != manifest['samples'][key]['pcm_measurements']:
            raise ValueError('derivative_audio actual PCM measurements mismatch: '+filename)
        # Source replay supplements, never replaces, the actual PCM readback.
        if (output/filename).read_bytes() != expected_blobs[filename]:
            raise ValueError('derivative_audio PCM differs from current continuous source: '+filename)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.check:
        manifest = check_assets(args.output_dir)
    else:
        _shared_asset_directory(args.output_dir, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=False)
        blobs, manifest = prepare_assets()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, blob in blobs.items():
            (args.output_dir/filename).write_bytes(blob)
        (args.output_dir/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        check_assets(args.output_dir)
    print(json.dumps({'status': 'checked' if args.check else 'generated',
                      'directory': str(args.output_dir), 'files': list(manifest['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
