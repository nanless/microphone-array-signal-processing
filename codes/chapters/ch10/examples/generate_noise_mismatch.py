"""Generate/check the independent six-WAV fixed-noise-estimate experiment.

--check validates ordinary paths, strict JSON, current sources and actual PCM,
then replays the complete model in memory. It never repairs a stale directory.
"""
from __future__ import annotations
if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes
from codes.chapters.ch10.core.noise_mismatch import (
    SAMPLE_RATE, SAMPLES, STEMS, build_fixture, analyze_fixture, analyze_pcm,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata, validate_parent_chain,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'codes/chapters/ch10/noise_audio'
SOURCE_PATHS = (
    'codes/chapters/ch10/core/noise_mismatch.py',
    'codes/chapters/ch10/examples/generate_noise_mismatch.py',
    'codes/chapters/ch10/core/noise_suppression.py',
    'codes/chapters/ch02/core/spectral.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = tuple(stem+'.wav' for stem in STEMS)+('MANIFEST.json',)


# Literal publication contract: a changed trusted builder must fail before the
# first write, rather than publish unchanged PCM with a different model claim.
REQUIRED_PARAMETERS = {'sample_rate_hz': 16000,
 'samples_per_channel': 32000,
 'seed': 20261001,
 'target_frequencies_hz': [500, 1500],
 'target_amplitudes': [0.12, 0.08],
 'noise_standard_deviations': [0.03, 0.12],
 'noise_step_sample': 19200,
 'noise_type': 'independent Gaussian mathematical samples with a deterministic variance step',
 'noise_only_sample_interval': [0, 6400],
 'polluted_estimation_sample_interval': [8000, 12800],
 'target_onset_fade_sample_interval': [6400, 6720],
 'target_end_fade_samples': 320,
 'fade': 'sin squared, including exact endpoint zero',
 'n_fft': 512,
 'hop_samples': 128,
 'window': 'DFT-periodic Hann',
 'center': True,
 'pure_noise_frame_indices': [2,
                              3,
                              4,
                              5,
                              6,
                              7,
                              8,
                              9,
                              10,
                              11,
                              12,
                              13,
                              14,
                              15,
                              16,
                              17,
                              18,
                              19,
                              20,
                              21,
                              22,
                              23,
                              24,
                              25,
                              26,
                              27,
                              28,
                              29,
                              30,
                              31,
                              32,
                              33,
                              34,
                              35,
                              36,
                              37,
                              38,
                              39,
                              40,
                              41,
                              42,
                              43,
                              44,
                              45,
                              46,
                              47,
                              48],
 'polluted_frame_indices': [65,
                            66,
                            67,
                            68,
                            69,
                            70,
                            71,
                            72,
                            73,
                            74,
                            75,
                            76,
                            77,
                            78,
                            79,
                            80,
                            81,
                            82,
                            83,
                            84,
                            85,
                            86,
                            87,
                            88,
                            89,
                            90,
                            91,
                            92,
                            93,
                            94,
                            95,
                            96,
                            97,
                            98],
 'oversubtraction': 1.0,
 'floor_ratio': 0.04,
 'minimum_positive_bin_amplitude_gain': 0.2,
 'known_variance_power': 'D_t=sum_n sigma[n]^2*w[n]^2; include all points of each window across the '
                         'step; padded variance is zero',
 'known_variance_scope': 'offline expected noise power from model, not observed noise realization, '
                         'online tracking or clean recovery',
 'score_windows': {'before_step': [9600, 16000], 'after_step': [22400, 28800]},
 'alignment': 'common sample origin; no gain/time matching; centered WOLA uses future samples',
 'common_export_gain': 1.0}
REQUIRED_STEMS = ('noise_reference', 'noise_component', 'noise_mixture', 'noise_fixed',
                  'noise_polluted', 'noise_known_variance')


def _validate_fixture(fixture):
    if (type(SAMPLE_RATE) is not int or SAMPLE_RATE != 16000
            or type(SAMPLES) is not int or SAMPLES != 32000
            or not same_metadata(STEMS, REQUIRED_STEMS)
            or not same_metadata(MEMBERS, tuple(x+'.wav' for x in REQUIRED_STEMS)+('MANIFEST.json',))
            or type(fixture) is not dict
            or set(fixture) != {'signals', 'components', 'parameters', 'spectral'}
            or not same_metadata(fixture['parameters'], REQUIRED_PARAMETERS)):
        raise ValueError('fixed noise fixture parameters/types differ')
    signals, components, spectral = (fixture[k] for k in ('signals', 'components', 'spectral'))
    if type(signals) is not dict or set(signals) != set(REQUIRED_STEMS):
        raise ValueError('fixed six noise signals required')
    def array(value, shape, kind='f'):
        if (type(value) is not np.ndarray or value.shape != shape
                or value.dtype.kind != kind or not np.isfinite(value).all()):
            raise ValueError('finite declared fixture array required')
    for signal in signals.values():
        array(signal, (1, 32000))
        if signal.dtype != np.dtype('float64') or np.max(abs(signal)) > 32767/32768:
            raise ValueError('unclipped fixed float64 noise signal required')
    # Independent literal source equations and RNG identity. The signal kernels
    # below remain the unique STFT/subtraction implementations, not copied DSP.
    time = np.arange(32000)/16000
    target = .12*np.cos(2*np.pi*500*time)+.08*np.cos(2*np.pi*1500*time)
    target[:6400] = 0
    target[6400:6720] *= np.sin(np.linspace(0, np.pi/2, 320))**2
    target[-320:] *= np.sin(np.linspace(np.pi/2, 0, 320))**2
    sigma = np.where(time < 1.2, .03, .12)
    noise = sigma*np.random.default_rng(20261001).standard_normal(32000)
    for name, expected in [('noise_reference', target), ('noise_component', noise),
                           ('noise_mixture', target+noise)]:
        if not np.array_equal(signals[name], expected[None]):
            raise ValueError('noise source differs from fixed model: '+name)
    from codes.chapters.ch02.core.spectral import stft, istft, periodic_hann
    from codes.chapters.ch10.core.noise_suppression import power_spectral_subtraction
    Y, S, V = [stft(x, n_fft=512, hop_length=128)[0] for x in (target+noise, target, noise)]
    centers = np.arange(Y.shape[1])*128
    padded = np.pad(sigma**2, (256, 256)); window = periodic_hann(512)
    D = np.array([float(np.sum(padded[c:c+512]*window**2)) for c in centers])
    if (type(spectral) is not dict or set(spectral) != {'frame_centers_samples', 'fixed_noise_power',
            'polluted_noise_power', 'known_variance_power', 'gains', 'mixture'}):
        raise ValueError('fixed noise spectral fields required')
    array(spectral['frame_centers_samples'], centers.shape, 'i')
    if not np.array_equal(spectral['frame_centers_samples'], centers):
        raise ValueError('noise frame clock differs')
    array(spectral['mixture'], Y.shape, 'c')
    if not np.array_equal(spectral['mixture'], Y):
        raise ValueError('noise spectrum differs')
    pure = np.arange(2, 49); polluted = np.arange(65, 99)
    fixed, D_fixed = power_spectral_subtraction(Y, pure)
    contaminated, D_polluted = power_spectral_subtraction(Y, polluted)
    for key, expected in [('known_variance_power', D), ('fixed_noise_power', D_fixed),
                           ('polluted_noise_power', D_polluted)]:
        array(spectral[key], expected.shape)
        if not np.array_equal(spectral[key], expected):
            raise ValueError('noise spectral power differs: '+key)
    names = {'noise_fixed', 'noise_polluted', 'noise_known_variance'}
    if (type(components) is not dict or set(components) != names
            or type(spectral['gains']) is not dict or set(spectral['gains']) != names):
        raise ValueError('fixed three noise decompositions required')
    # Validate known-power subtraction through its unique kernel too.
    known = np.empty_like(Y)
    for index, power in enumerate(D):
        proxy = np.full((Y.shape[0], 1), np.sqrt(power), dtype=complex)
        known[:, index] = power_spectral_subtraction(np.concatenate((proxy, Y[:, index:index+1]), axis=1),
                                                    np.array([0]))[0][:, 1]
    for name, Z in [('noise_fixed', fixed), ('noise_polluted', contaminated), ('noise_known_variance', known)]:
        gain = np.zeros(Y.shape); gain[Y != 0] = (Z[Y != 0]/Y[Y != 0]).real
        array(spectral['gains'][name], gain.shape)
        if not np.array_equal(spectral['gains'][name], gain):
            raise ValueError('noise gain differs: '+name)
        if type(components[name]) is not dict or set(components[name]) != {'target', 'noise'}:
            raise ValueError('fixed target/noise components required')
        for role, spectrum in [('target', S), ('noise', V)]:
            array(components[name][role], (1, 32000))
            expected = istft((gain*spectrum)[None], n_fft=512, hop_length=128, length=32000)
            if not np.array_equal(components[name][role], expected):
                raise ValueError('noise component differs: '+name+'/'+role)
        expected = istft(Z[None], n_fft=512, hop_length=128, length=32000)
        if not np.array_equal(signals[name], expected):
            raise ValueError('noise output differs: '+name)


def validate_asset_directory(directory, *, check: bool):
    """Shared path preflight with this experiment's exact member policy."""
    return _shared_asset_directory(directory, MEMBERS, check=check)


def _strict_json(data: bytes):
    return strict_json_loads(data)


def _same_metadata(actual, expected, path='manifest'):
    if not same_metadata(actual, expected):
        raise ValueError(f'{path} has invalid types or differs from current source/PCM replay')


def expected_assets():
    """Pure full replay, with source hashes from the actual seven dependencies."""
    sources = {}
    for name in SOURCE_PATHS:
        path = validate_parent_chain(ROOT/name)
        if not path.is_file():
            raise ValueError('missing ordinary source: '+name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fixture = build_fixture()
    _validate_fixture(fixture)
    buffers = {}
    for name, signal in fixture['signals'].items():
        if np.max(np.abs(signal)) > 32767/32768:
            raise ValueError('fixture would clip PCM at common gain 1')
        buffers[name+'.wav'] = pcm16_bytes(signal, SAMPLE_RATE)
    metadata = {
        'schema_version': 1, 'sample_rate_hz': SAMPLE_RATE, 'common_export_gain': 1.,
        'origin': 'mathematical synthetic two-tone/noise example; not speech or a recording',
        'parameters': fixture['parameters'], 'floating_point': analyze_fixture(fixture),
        'pcm_analysis': analyze_pcm(buffers),
        'quantization': 'signed little-endian PCM16; nearest-even; no dither; decoded amplitude=int16/32768',
        'source_sha256': sources,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
        'files': {name: {'channels': 1, 'samples_per_channel': SAMPLES,
                         'sha256': hashlib.sha256(data).hexdigest()} for name, data in buffers.items()},
    }
    buffers['MANIFEST.json'] = (json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    return buffers, metadata


def check_assets(directory=OUTPUT):
    """Strictly read actual bytes/score them, then compare full current replay."""
    directory = Path(directory)
    validate_asset_directory(directory, check=True)
    metadata = _strict_json((directory/'MANIFEST.json').read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError('manifest must be an object')
    actual = {name: (directory/name).read_bytes() for name in MEMBERS if name.endswith('.wav')}
    # This analysis runs on the directory bytes, not on regenerated stand-ins.
    measured = analyze_pcm(actual)
    expected, replay = expected_assets()
    _same_metadata(metadata, replay)
    _same_metadata(metadata['pcm_analysis'], measured, 'actual PCM analysis')
    for name, data in actual.items():
        if hashlib.sha256(data).hexdigest() != metadata['files'][name]['sha256']:
            raise ValueError(f'actual WAV digest differs: {name}')
        if data != expected[name]:
            raise ValueError(f'actual WAV differs from current complete replay: {name}')
    return metadata


def generate(directory=OUTPUT, *, check=False):
    validate_asset_directory(directory, check=check)
    if check:
        return check_assets(directory)
    buffers, metadata = expected_assets()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in buffers.items():
        (directory/name).write_bytes(data)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    metadata = generate(args.output, check=args.check)
    print(json.dumps({'mode': 'verified' if args.check else 'generated',
                      'directory': str(args.output), 'pcm_analysis': metadata['pcm_analysis']}, indent=2))


if __name__ == '__main__':
    main()
