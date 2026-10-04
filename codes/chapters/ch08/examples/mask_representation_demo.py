"""Generate six E08-28 known-mask WAVs, or strictly read-check without repair."""
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
from codes.chapters.ch08.core.mask_representation import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, measure_signal, parameters, run_experiment,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata, validate_parent_chain,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch08/mask_audio'
SOURCE_PATHS = (
    'codes/chapters/ch08/core/mask_representation.py',
    'codes/chapters/ch08/examples/mask_representation_demo.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)


# Reviewed fixed model values/types; not obtained from the runtime report.
REQUIRED_PARAMETERS = {'exercise_id': 'E08-28',
 'sample_rate_hz': 16000,
 'samples_per_channel': 32000,
 'frequencies_hz': [500, 1250],
 'positive_rfft_bins': [1000, 2500],
 'source': 'target=.1*cos(500Hz)-.1*sin(1250Hz); other=-.08*cos(500Hz)+.1*cos(1250Hz)',
 'fft': 'full 32000-sample unwindowed periodic record; numpy rfft/irfft; only bins 1000/2500 retained',
 'target_phasors': [[0.1, 0.0], [0.0, 0.1]],
 'mixture_phasors': [[0.02, 0.0], [0.1, 0.1]],
 'masks': {'bounded_real': [[1.0, 0.0], [0.5, 0.0]],
           'unbounded_real': [[5.0, 0.0], [0.5, 0.0]],
           'complex_oracle': [[5.0, 0.0], [0.5, 0.5]]},
 'envelope': 'after FFT operator: sin(linspace(0,pi/2,320)) squared at each end, reversed at end; '
             'endpoints zero',
 'fade_samples_each_end': 320,
 'scoring_interval_samples': [2400, 29600],
 'scoring_samples_per_channel': 27200,
 'alignment': 'same absolute source clock; no propagation or fitted gain/delay',
 'common_export_gain': 1.0,
 'algorithm_lookahead': 'entire 2-second record; noncausal offline known mask',
 'analytic_steady_reference_power': 0.01,
 'analytic_steady_mse': {'target': 0.0,
                         'other': 0.0262,
                         'mixture': 0.0082,
                         'bounded_real': 0.0057,
                         'unbounded_real': 0.0025,
                         'complex_oracle': 0.0}}
REQUIRED_LIMITS = 'Known-bin full-record offline representation, not an estimated mask, blind separator, speech recording or listening evaluation. The same sin-squared listening envelope is applied after the rFFT operator. Nonselected numerical-leakage bins are set to zero. No delay/gain fitting, per-file normalization, dither or clipping. Phase LS measures the fixed absolute sample clock; it does not compensate the output.'

def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {path: _sha(validate_parent_chain(ROOT/path).read_bytes()) for path in SOURCE_PATHS}


def prepare_assets() -> tuple[dict, dict[str, bytes]]:
    true_parameters = parameters()
    report, arrays = run_experiment()
    if (not same_metadata(true_parameters, REQUIRED_PARAMETERS)
            or not same_metadata(report.get('parameters'), REQUIRED_PARAMETERS)
            or not same_metadata(LIMITS, REQUIRED_LIMITS)
            or not same_metadata(report.get('limits'), REQUIRED_LIMITS)
            or type(SAMPLE_RATE) is not int or SAMPLE_RATE != 16000
            or type(SAMPLES) is not int or SAMPLES != 32000
            or not same_metadata(FILE_NAMES, {key: 'mask_'+key+'.wav' for key in
                ('target', 'other', 'mixture', 'bounded_real', 'unbounded_real', 'complex_oracle')})):
        raise ValueError('mask true parameters differ from fixed asset contract')
    if (not isinstance(arrays, dict) or set(arrays) != set(FILE_NAMES)
            or any(not isinstance(value, np.ndarray) or value.dtype.kind != 'f'
                   or value.shape != (1, 32000) or not np.isfinite(value).all()
                   or np.max(abs(value)) >= 1 for value in arrays.values())):
        raise ValueError('fixed finite unclipped mask waveforms required')
    actual_float = {key: measure_signal(value, arrays['target']) for key, value in arrays.items()}
    if not same_metadata(report.get('float_measurements'), actual_float):
        raise ValueError('mask float report differs from generated waveforms')
    contents, files, decoded, samples = {}, {}, {}, {}
    for key, value in arrays.items():
        blob = pcm16_bytes(value, SAMPLE_RATE)
        rate, pcm = read_pcm16(blob)
        filename = FILE_NAMES[key]
        contents[filename], decoded[key] = blob, pcm
        files[filename] = {'channels': 1, 'samples_per_channel': SAMPLES,
                           'sample_rate_hz': rate, 'sha256': _sha(blob)}
    for key in FILE_NAMES:
        samples[key] = {'file': FILE_NAMES[key],
                        'float_measurements': report['float_measurements'][key],
                        'pcm_measurements': measure_signal(decoded[key], decoded['target'], pcm=True),
                        'quantization_max_abs_error': float(np.max(abs(arrays[key]-decoded[key])))}
    manifest = {'schema_version': 1, 'origin': 'original known-bin offline mathematical representation',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'common_export_gain': 1.,
                'source_sha256': source_digests(), 'parameters': report['parameters'],
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'system': platform.system(), 'machine': platform.machine()},
                'files': files, 'samples': samples,
                'unwindowed_selected_mixture_phasors_real_imag': report['unwindowed_selected_mixture_phasors_real_imag'],
                'discarded_bin_max_magnitude': report['discarded_bin_max_magnitude'],
                'pcm_encode_multiplier': 32768, 'pcm_decode_divisor': 32768,
                'pcm': 'signed little-endian PCM16, round-to-nearest-even, no dither or per-file gain',
                'pcm_quantization_step': 1/32768, 'pcm_half_step_error_bound': .5/32768,
                'limits': LIMITS, 'listening': 'Start at low volume; no automatic playback or listening study.'}
    contents['MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    return manifest, contents


def _members(output, *, checking):
    return _shared_asset_directory(output, set(FILE_NAMES.values()) | {'MANIFEST.json'}, check=checking)


def _strict_equal(actual, expected):
    return same_metadata(actual, expected, allow_int_for_float=True)


def check_assets(directory: Path = DEFAULT_OUTPUT, *, replay=True) -> dict:
    """Validate real members, source SHA, PCM scores and optionally full replay."""
    if type(replay) is not bool:
        raise ValueError('replay must be boolean')
    directory = Path(directory)
    _members(directory, checking=True)
    try:
        manifest = strict_json_loads((directory/'MANIFEST.json').read_text())
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError('invalid mask_audio manifest') from exc
    if not isinstance(manifest, dict):
        raise ValueError('mask_audio manifest must be an object')
    if manifest.get('source_sha256') != source_digests():
        raise ValueError('mask_audio source set or SHA is stale')
    files, samples = manifest.get('files'), manifest.get('samples')
    if (not isinstance(files, dict) or not isinstance(samples, dict) or
            set(files) != set(FILE_NAMES.values()) or set(samples) != set(FILE_NAMES) or
            any(not isinstance(r, dict) for r in (*files.values(), *samples.values()))):
        raise ValueError('mask_audio manifest file set differs')
    if (not _strict_equal(manifest.get('parameters'), parameters()) or
            not _strict_equal(manifest.get('common_export_gain'), 1.) or
            not _strict_equal(manifest.get('sample_rate_hz'), SAMPLE_RATE) or
            not _strict_equal(manifest.get('samples_per_channel'), SAMPLES)):
        raise ValueError('mask_audio fixed parameters differ')
    decoded = {}
    for key, filename in FILE_NAMES.items():
        blob = (directory/filename).read_bytes()
        if _sha(blob) != files[filename].get('sha256'):
            raise ValueError('mask_audio WAV SHA mismatch: '+filename)
        try:
            with wave.open(io.BytesIO(blob)) as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                    reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, 1, SAMPLES, 2, 'NONE'):
                    raise ValueError('mask_audio PCM format mismatch: '+filename)
            _, decoded[key] = read_pcm16(blob)
        except (wave.Error, EOFError) as exc:
            raise ValueError('invalid mask_audio WAV: '+filename) from exc
        record = {'channels': 1, 'samples_per_channel': SAMPLES,
                  'sample_rate_hz': SAMPLE_RATE, 'sha256': _sha(blob)}
        if not _strict_equal(files[filename], record) or samples[key].get('file') != filename:
            raise ValueError('mask_audio PCM metadata differs: '+filename)
    for key in FILE_NAMES:
        actual = measure_signal(decoded[key], decoded['target'], pcm=True)
        if not _strict_equal(samples[key].get('pcm_measurements'), actual):
            raise ValueError('mask_audio actual PCM measurement mismatch: '+FILE_NAMES[key])
    if replay:
        expected, contents = prepare_assets()
        if not _strict_equal(manifest, expected):
            raise ValueError('mask_audio float/model measurements are stale')
        if any((directory/name).read_bytes() != blob for name, blob in contents.items()):
            raise ValueError('mask_audio differs from current source')
    return manifest


def generate_assets(directory: Path = DEFAULT_OUTPUT, *, check=False) -> dict:
    if type(check) is not bool:
        raise ValueError('check must be boolean')
    directory = Path(directory)
    if check:
        return check_assets(directory)
    _members(directory, checking=False)
    manifest, contents = prepare_assets()
    directory.mkdir(parents=True, exist_ok=True)
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
