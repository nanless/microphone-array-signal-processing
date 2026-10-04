"""Generate/check two independent Chapter 9 audio files; no writes on import.

python -m codes.chapters.ch09.examples.chapter09_tracking_audio [--output DIRECTORY] [--check]
The directory contains source.wav, array_noisy.wav and MANIFEST.json only.
--check recomputes source/PCM analysis in memory and rejects stale assets; it
never regenerates files to make a check pass. Byte reproducibility is tied to
the recorded environment. No audio is played by this program.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from codes.chapters.ch00.io_contracts import same_metadata, strict_json_loads, validate_asset_directory
from codes.chapters.ch09.core.tracking_audio import build_fixture
from codes.chapters.ch09.core import tracking_audio as _model

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'codes/chapters/ch09/tracking_audio'
SOURCE_PATHS = (
    'codes/chapters/ch09/examples/chapter09_tracking_audio.py', 'codes/chapters/ch09/core/tracking_audio.py',
    'codes/chapters/ch09/core/moving_source.py', 'codes/chapters/ch09/core/tracking.py',
    'codes/chapters/ch04/core/doa.py', 'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py',
)


def expected_assets():
    buffers, metadata = build_fixture()
    _validate_model(buffers, metadata)
    metadata['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    metadata['environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()}
    metadata['files'] = {name: {'channels': 1 if name == 'source.wav' else 2,
        'samples_per_channel': 32000, 'sha256': hashlib.sha256(data).hexdigest()} for name, data in buffers.items()}
    buffers['MANIFEST.json'] = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return buffers, metadata


# Reviewed literal values/types, independent of build_fixture return metadata.
REQUIRED_MODEL = {'model': 'continuous multisine with exact retarded time and assigned 1/r gain; independent '
          'Gaussian sensor noise; no room, speech or device recording',
 'sample_rate_hz': 16000,
 'duration_s': 2.0,
 'seed': 9001,
 'rng': 'PCG64; 32 phases then channel-major noise samples',
 'frequencies_hz': [300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                    1400,
                    1500,
                    1600,
                    1700,
                    1800,
                    1900,
                    2000,
                    2100,
                    2200,
                    2300,
                    2400,
                    2500,
                    2600,
                    2700,
                    2800,
                    2900,
                    3000,
                    3100,
                    3200,
                    3300,
                    3400],
 'source_amplitude_per_tone': 0.17677669529663687,
 'noise_std_before_export': 0.01,
 'emission_envelope': 'linear 20-ms onset/end; fade out .8-.82 s, zero .82-1.08 s, fade in '
                      '1.08-1.10 s',
 'microphones_xy_m': [[-0.05, 0.0], [0.05, 0.0]],
 'source_start_xy_m': [-0.8, 1.5],
 'source_velocity_xy_m_s': [0.8, 0.0],
 'sound_speed_m_s': 343.0,
 'export_peak_ceiling': 0.7,
 'pcm': '16-bit little endian; codes.chapters.ch00.core.audio_samples.pcm16_bytes; decoded /32768; '
        'export gain undone only for fixed RMS gate',
 'analysis_config': {'window_samples': 512,
                     'hop_samples': 160,
                     'window': 'numpy.hanning(512), symmetric',
                     'state_timestamp': '(start_sample+255.5)/16000, receiver window center',
                     'measurement_timestamp': 'receiver window center only when GCC observation is '
                                              'valid; otherwise null',
                     'last_valid_measurement_timestamp': 'most recent valid GCC window center; '
                                                         'null before initialization and unchanged '
                                                         'across missing frames',
                     'availability_timestamp': '(start_sample+512)/16000, half-open block complete',
                     'lookahead_from_state_ms': 16.03125,
                     'rms_threshold_before_export': 0.04,
                     'gcc': 'gcc_phat(ch1, ch0), tau10=t1-t0; max_tau=.1/343; parabolic '
                            'interpolation; no truth input',
                     'angle_conversion': 'asin(-343*tau10/.1), far-field approximation; outside '
                                         '[-1,1] -> missing',
                     'initial_state': '[first observed angle, 0 deg/s]',
                     'initial_covariance': [[4.0, 0.0], [0.0, 400.0]],
                     'measurement_variance_deg2': 4.0,
                     'white_acceleration_density_deg2_s3': 100.0,
                     'update_order': 'initialize once; thereafter predict each 10 ms then update '
                                     'only with valid observation',
                     'truth': 'array-center retarded emission direction; tau10 compares the same '
                              'emission event at both mics',
                     'scope': 'one fixed synthetic sequence; overlapping observations are '
                              'correlated; assigned R/Q are not calibrated coverage; no identity '
                              'or beamformer output'}}

def _validate_model(buffers, metadata):
    """Validate fixed model and actual PCM before creating or replacing files.

    Replay uses the unique continuous-source, propagation, PCM and analysis
    kernels; this trusted-code contract is not a hostile-code sandbox.
    """
    dynamic = {'phases_rad', 'common_export_gain', 'retarded_equation_max_residual_s',
               'float_analysis', 'pcm_analysis'}
    if type(metadata) is not dict or set(metadata) != set(REQUIRED_MODEL) | dynamic:
        raise ValueError('tracking model metadata fields differ')
    if not same_metadata({key: metadata[key] for key in REQUIRED_MODEL}, REQUIRED_MODEL):
        raise ValueError('tracking fixed model values or true types differ')
    if type(buffers) is not dict or set(buffers) != {'source.wav', 'array_noisy.wav'}:
        raise ValueError('two declared tracking PCM members required')
    if any(type(blob) is not bytes for blob in buffers.values()):
        raise ValueError('tracking PCM buffers must be actual bytes')
    rng = np.random.default_rng(9001)
    phases = rng.uniform(-np.pi, np.pi, 32)
    times = np.arange(32000) / 16000
    microphones = np.array([[-.05, 0.], [.05, 0.]])
    start, velocity = np.array([-.8, 1.5]), np.array([.8, 0.])
    emission = _model.retarded_emission_times(times, microphones,
        source_start_xy=start, source_velocity_xy=velocity)
    distance = np.linalg.norm(start + emission[:, :, None]*velocity-microphones[:, None, :], axis=2)
    source = _model.continuous_source(times, phases)[None]
    array = _model.continuous_source(emission, phases)/distance + .01*rng.standard_normal((2, 32000))
    gain = .7 / max(float(np.max(abs(source))), float(np.max(abs(array))))
    expected_buffers = {'source.wav': _model.pcm16_bytes(source*gain, 16000),
                        'array_noisy.wav': _model.pcm16_bytes(array*gain, 16000)}
    if buffers != expected_buffers:
        raise ValueError('tracking PCM differs from the declared shared-kernel model')
    expected_dynamic = {'phases_rad': phases.tolist(), 'common_export_gain': gain,
        'retarded_equation_max_residual_s': float(np.max(abs(emission+distance/343.-times))),
        'float_analysis': _model.analyze_array(array),
        'pcm_analysis': _model.analyze_array(_model.read_pcm16(buffers['array_noisy.wav']), export_gain=gain)}
    for key, expected in expected_dynamic.items():
        if not same_metadata(metadata[key], expected):
            raise ValueError('tracking model phase, gain or float/actual-PCM analysis differs: ' + key)


def _validate_directory(directory, expected, *, check):
    """Local preflight only; no concurrency or crash-persistence guarantee."""
    return validate_asset_directory(directory, expected, check=check)


def generate(out_dir=OUTPUT, *, check=False):
    out_dir = Path(out_dir)
    _validate_directory(out_dir, ("source.wav", "array_noisy.wav", "MANIFEST.json"), check=check)
    assets, metadata = expected_assets()
    if check:
        recorded = strict_json_loads((out_dir/'MANIFEST.json').read_bytes())
        if not same_metadata(recorded, metadata):
            raise ValueError('manifest differs from current fixed model, source identities or scores')
        for name, expected in assets.items():
            path = out_dir/name
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f'missing or stale tracking asset: {path}')
        if {p.name for p in out_dir.iterdir()} != set(assets):
            raise ValueError('tracking audio directory contains unexpected files')
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in assets.items():
            (out_dir/name).write_bytes(content)
    return metadata


def check_assets(out_dir=OUTPUT):
    """Strict read-only replay of actual directory bytes, current source and PCM."""
    return generate(out_dir, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = generate(args.output, check=args.check)
    print(json.dumps({'mode': 'verified' if args.check else 'generated',
                      'directory': str(args.output), 'scores': report['pcm_analysis']['scores']}, indent=2))


if __name__ == '__main__':
    main()
