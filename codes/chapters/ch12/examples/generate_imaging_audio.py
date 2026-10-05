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
from codes.chapters.ch12.core.imaging_audio import (
    FILE_NAMES, SAMPLE_RATE, SAMPLES, LIMITS, parameters, run_experiment,
    measure_signal, measure_source_pairs,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT/'codes/chapters/ch12/imaging_audio'
SOURCE_PATHS = (
    'codes/chapters/ch12/core/imaging.py', 'codes/chapters/ch12/core/imaging_audio.py',
    'codes/chapters/ch12/examples/generate_imaging_audio.py',
    'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)

_FIXED_FILES = {name: name+'.wav' for name in (
    'source_1', 'source_2_phase_code', 'source_2_coherent',
    'array_phase_code', 'array_coherent',
)}
_FIXED_LIMITS = ('Known 2kHz tones, deterministic balanced phase code and exactly known four-point propagation; '
                 'not random-process independence, speech, a measured array, blind imaging, an inverse waveform '
                 'or a listening evaluation. Digital amplitude squared is not Pa^2 or acoustic watts. '
                 'No per-file normalization, clipping, fitted gain/delay, or circular shift.')


def _fixed_parameters():
    """Publication fixture, independent of the model's metadata callbacks."""
    return {'exercise_ids': ['E12-04', 'E12-05', 'E12-10'], 'sample_rate_hz': 24000,
            'frequency_hz': 2000, 'samples_per_channel': 48004,
            'body_samples': 48000, 'complete_tail_samples': 4,
            'block_samples': 2400, 'blocks': 20, 'fade_samples_each_end_of_block': 240,
            'envelope': 'min(r/240,(2399-r)/240,1), r=n mod 2400, common to both sources',
            'phase_code': [1, -1]*10, 'source_peak_amplitudes': [.2, .1],
            'source_steady_mean_square': [.02, .005],
            'second_source_second_microphone_delay_samples': 4,
            'positive_frequency_convention': 'x[n]=Re(z exp(+j*2*pi*f*n/fs)); positive DFT uses exp(-j*...)',
            'scoring_intervals_samples': [[2400*j+252, 2400*j+2160] for j in range(20)],
            'samples_per_snapshot': 1908, 'integer_cycles_per_snapshot': 159,
            'scoring_samples_per_channel': 38160,
            'complex_peak_amplitude': 'z=(2/1908)*sum(x[n]*exp(-j*2*pi*f*n/fs)) on each declared interval',
            'csm': 'R=(1/(2*20))*sum(z_l z_l^H), digital mean-square; interior-frequency tone only',
            'normalization_power_scale': .2**2/2, 'common_export_gain': 1.,
            'channel_order': 'microphone 1, microphone 2; only source 2 is delayed at microphone 2',
            'statistical_scope': 'zero cross term in specified balanced finite snapshot ensemble, not random independence'}


def _validate_fixed_contract():
    if (type(SAMPLE_RATE) is not int or SAMPLE_RATE != 24000
            or type(SAMPLES) is not int or SAMPLES != 48004
            or not same_metadata(FILE_NAMES, _FIXED_FILES)
            or not same_metadata(LIMITS, _FIXED_LIMITS)
            or not same_metadata(parameters(), _fixed_parameters())):
        raise ValueError('imaging fixed parameters/types/limits differ')


def _fixed_waveforms():
    """Independently reconstruct the declared five signals, including the tail."""
    n = np.arange(48000)
    r = n % 2400
    envelope = np.minimum(np.minimum(r/240, (2399-r)/240), 1.)
    first = .2*envelope*np.cos(2*np.pi*2000*n/24000)
    second = first/2
    coded = second*np.where((n//2400) % 2 == 0, 1., -1.)
    first, second, coded = [np.pad(x, (0, 4)) for x in (first, second, coded)]
    return {'source_1': first[None, :], 'source_2_phase_code': coded[None, :],
            'source_2_coherent': second[None, :],
            'array_phase_code': np.vstack([first+coded, first+np.pad(coded[:-4], (4, 0))]),
            'array_coherent': np.vstack([first+second, first+np.pad(second[:-4], (4, 0))])}


def _pairs(value):
    return np.stack([np.asarray(value).real, np.asarray(value).imag], axis=-1).tolist()


def _independent_measurements(x, key, *, pcm=False):
    """Direct DFT/outer-product and scalar two-cell oracles; no model scorer."""
    intervals = _fixed_parameters()['scoring_intervals_samples']
    z = np.column_stack([2/1908*np.sum(x[:, a:b]*np.exp(-2j*np.pi*2000*np.arange(a, b)/24000), axis=1)
                         for a, b in intervals])
    csm = z@z.conj().T/40
    analytic = {'source_1': np.array([[.02]]),
                'source_2_phase_code': np.array([[.005]]), 'source_2_coherent': np.array([[.005]]),
                'array_phase_code': .2**2/2*np.array([[1.25, .875+1j*np.sqrt(3)/8], [.875-1j*np.sqrt(3)/8, 1.25]]),
                'array_coherent': .2**2/2*np.array([[2.25, 1.125+3j*np.sqrt(3)/8], [1.125-3j*np.sqrt(3)/8, .75]])}[key]
    result = {'snapshot_peak_amplitudes_real_imag': _pairs(z), 'csm_real_imag': _pairs(csm),
              'csm_units': 'digital amplitude squared, single-tone mean-square',
              'analytic_csm_real_imag': _pairs(analytic),
              'analytic_csm_max_abs_error': float(np.max(abs(csm-analytic))),
              'normalized_csm_max_abs_error': float(np.max(abs(csm-analytic))/(.2**2/2)),
              'scoring_samples_per_channel': 38160, 'snapshot_denominator': 20,
              'samples_per_snapshot': 1908, 'peak_all_samples': float(np.max(abs(x)))}
    if x.shape[0] == 2:
        a = np.array([[1, 1], [1, np.exp(-2j*np.pi/3)]])
        w = a/2
        scan = np.array([float(np.vdot(w[:, k], csm@w[:, k]).real) for k in range(2)])
        q = np.array([16/15*(scan[0]-scan[1]/4), 16/15*(scan[1]-scan[0]/4)])
        model = (a*q)@a.conj().T
        delta = model-csm
        squared = float(np.sum(abs(delta)**2))
        reference_squared = float(np.sum(abs(csm)**2))
        result.update({'scan_mean_square': scan.tolist(),
                       'normalized_scan_mean_square': (scan/(.2**2/2)).tolist(),
                       'inverse_source_mean_square': q.tolist(),
                       'normalized_inverse_source_mean_square': (q/(.2**2/2)).tolist(),
                       'scan_residual': (np.array([[1, .25], [.25, 1]])@q-scan).tolist(),
                       'full_csm_residual': {'frobenius_squared': squared,
                           'reference_frobenius_squared': reference_squared,
                           'relative_frobenius': float(np.sqrt(squared/reference_squared)),
                           'upper_unweighted_squared': float(np.sum(abs(np.triu(delta))**2)),
                           'squared_underflow': {'frobenius_squared': False,
                               'reference_frobenius_squared': False, 'upper_unweighted_squared': False}}})
    if pcm:
        integers = np.rint(x*32768).astype(np.int64)
        result.update({'pcm_decode_divisor': 32768,
                       'integer_scored_squared_sum_by_channel': [
                           sum(int(v)**2 for start, stop in intervals for v in channel[start:stop])
                           for channel in integers],
                       'integer_power_sample_denominator': 38160, 'integer_power_scale_denominator': 32768**2})
    return result


def _independent_source_pairs(signals):
    intervals = _fixed_parameters()['scoring_intervals_samples']
    def amplitudes(x):
        return np.array([2/1908*np.sum(x[0, a:b]*np.exp(-2j*np.pi*2000*np.arange(a, b)/24000))
                         for a, b in intervals])
    first = amplitudes(signals['source_1'])
    result = {}
    for name, key in [('phase_code', 'source_2_phase_code'), ('coherent', 'source_2_coherent')]:
        z = np.vstack([first, amplitudes(signals[key])])
        r = z@z.conj().T/40
        result[name] = {'source_csm_real_imag': _pairs(r), 'source_cross_term_real_imag': _pairs(r[0, 1]),
                        'magnitude_squared_coherence': float(abs(r[0, 1])**2/(r[0, 0].real*r[1, 1].real))}
    return result


def _numeric_metadata(actual, expected):
    """Exact structures/types; allow rounding only in independently computed floats."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_numeric_metadata(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_numeric_metadata(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return np.isfinite(actual) and bool(np.isclose(actual, expected, rtol=2e-12, atol=1e-13))
    return actual == expected


def _validate_model(report, arrays):
    fixed = _fixed_waveforms()
    if (type(report) is not dict or not same_metadata(report.get('parameters'), _fixed_parameters())
            or not same_metadata(report.get('limits'), _FIXED_LIMITS)):
        raise ValueError('imaging model fixed metadata/types/limits differ')
    if type(arrays) is not dict or arrays.keys() != fixed.keys():
        raise ValueError('imaging fixture members differ')
    for key, expected in fixed.items():
        array = arrays[key]
        if (type(array) is not np.ndarray or array.dtype != np.dtype('float64')
                or array.shape != expected.shape or not np.all(np.isfinite(array))
                or not np.array_equal(array, expected)):
            raise ValueError('imaging fixed waveform/type/shape differs: '+key)
    expected_report = {'parameters': _fixed_parameters(),
                       'float_measurements': {k: _independent_measurements(x, k) for k, x in fixed.items()},
                       'float_source_pairs': _independent_source_pairs(fixed), 'limits': _FIXED_LIMITS}
    if not _numeric_metadata(report, expected_report):
        raise ValueError('imaging independent float metadata/score differs')


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_digests():
    return {path: _sha((ROOT/path).read_bytes()) for path in SOURCE_PATHS}


def prepare_assets():
    _validate_fixed_contract()
    report, arrays = run_experiment()
    _validate_model(report, arrays)
    contents, files, decoded, samples = {}, {}, {}, {}
    for key, array in arrays.items():
        blob = pcm16_bytes(array, SAMPLE_RATE)
        rate, pcm = read_pcm16(blob)
        with wave.open(io.BytesIO(blob)) as reader:
            channels = array.shape[0]
            if (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                    reader.getsampwidth(), reader.getcomptype()) != (24000, channels, 48004, 2, 'NONE'):
                raise ValueError('imaging in-memory WAV format differs')
            integers = np.frombuffer(reader.readframes(48004), dtype='<i2').reshape(-1, channels).T
        if (type(rate) is not int or rate != 24000 or not np.array_equal(integers, np.rint(array*32768))
                or type(pcm) is not np.ndarray or pcm.dtype != np.dtype('float64')
                or not np.array_equal(pcm, integers.astype(np.float64)/32768)):
            raise ValueError('imaging independent PCM encoding/decoding differs')
        filename = FILE_NAMES[key]
        contents[filename] = blob; decoded[key] = pcm
        files[filename] = {'channels': array.shape[0], 'samples_per_channel': SAMPLES,
                           'sample_rate_hz': rate, 'sha256': _sha(blob)}
        pcm_scores = measure_signal(pcm, key, pcm=True)
        if not _numeric_metadata(pcm_scores, _independent_measurements(pcm, key, pcm=True)):
            raise ValueError('imaging independent PCM metadata/score differs: '+key)
        samples[key] = {'file': filename, 'float_measurements': report['float_measurements'][key],
                        'pcm_measurements': pcm_scores,
                        'quantization_max_abs_error': float(np.max(abs(array-pcm)))}
    pcm_pairs = measure_source_pairs(decoded)
    if not _numeric_metadata(pcm_pairs, _independent_source_pairs(decoded)):
        raise ValueError('imaging independent PCM source pairs differ')
    manifest = {'schema_version': 1, 'origin': 'original finite-snapshot known-tone mathematical imaging observations',
                'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES,
                'common_export_gain': 1., 'source_sha256': source_digests(), 'parameters': parameters(),
                'files': files, 'samples': samples, 'float_source_pairs': report['float_source_pairs'],
                'pcm_source_pairs': pcm_pairs,
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
    _validate_fixed_contract()
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
    _validate_fixed_contract()
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
