"""Known fixed-weight least-squares comparison; deterministic orthogonal tones.

The two channels have the same target, no propagation or processing delay.
Noise tones are orthogonal in the specified finite scoring window, not random
independent noise. We do not estimate a covariance or fit a compensating gain.
"""
from __future__ import annotations

import io
import struct
import wave
import numpy as np
from codes.chapters.ch00.io_contracts import same_metadata

SAMPLE_RATE = 16000
SAMPLES = 32000
GAIN = 1.0
SCORE = (1600, 30400)
STEMS = ('weighted_target', 'weighted_array', 'weighted_ols', 'weighted_gls', 'weighted_reversed')
WEIGHTS = {'weighted_ols': (.5, .5), 'weighted_gls': (.8, .2), 'weighted_reversed': (.2, .8)}


def build_fixture():
    n = np.arange(SAMPLES)
    envelope = np.minimum(1., np.minimum(n/640, (SAMPLES-1-n)/640))
    target = .2*np.cos(2*np.pi*700*n/SAMPLE_RATE)*envelope
    noises = np.array([.03*np.cos(2*np.pi*3500*n/SAMPLE_RATE),
                       .06*np.cos(2*np.pi*4000*n/SAMPLE_RATE)])*envelope
    array = target[None, :]+noises
    signals = {'weighted_target': target, 'weighted_array': array}
    components = {}
    for name, weights in WEIGHTS.items():
        clean = sum(weights)*target
        noise = np.asarray(weights)@noises
        signals[name] = np.asarray(weights)@array
        components[name] = {'clean': clean, 'noise': noise}
    return {'signals': signals, 'components': components, 'parameters': {
        'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES,
        'source_score': list(SCORE), 'scored_samples': SCORE[1]-SCORE[0],
        'target_frequency_hz': 700, 'target_amplitude': .2,
        'noise_frequencies_hz': [3500, 4000], 'noise_amplitudes': [.03, .06],
        'envelope': 'min(1,n/640,(31999-n)/640), n=0..31999; endpoints exactly zero',
        'weights': {key: list(value) for key, value in WEIGHTS.items()},
        'noise_covariance_on_score': [[.00045, 0.], [0., .0018]],
        'common_export_gain': GAIN, 'delay_samples': 0,
        'limits': 'deterministic finite-window orthogonality, not random independence, estimated covariance, speech or listening-study evidence',
    }}


def analytic_results():
    return {'target_power': .02, 'noise_covariance': [[.00045, 0.], [0., .0018]],
            'candidates': {name+'.wav': {'target_distortion_power': 0.,
                'residual_noise_power': .00045*w[0]**2+.0018*w[1]**2,
                'cross_term': 0., 'total_error_power': .00045*w[0]**2+.0018*w[1]**2,
                'nmse': (.00045*w[0]**2+.0018*w[1]**2)/.02}
                for name, w in WEIGHTS.items()}}


def analyze_fixture(fixture=None):
    """Float components keep finite cross terms; PCM receives no decomposition."""
    fixture = build_fixture() if fixture is None else fixture
    lo, hi = SCORE
    reference = fixture['signals']['weighted_target'][lo:hi]
    power = float(np.mean(reference**2))
    report = {'scored_samples': hi-lo, 'reference_power': power, 'candidates': {}}
    for name, component in fixture['components'].items():
        distortion = component['clean'][lo:hi]-reference
        noise = component['noise'][lo:hi]
        output = fixture['signals'][name][lo:hi]
        report['candidates'][name+'.wav'] = {
            'target_distortion_power': float(np.mean(distortion**2)),
            'residual_noise_power': float(np.mean(noise**2)),
            'cross_term': float(2*np.mean(distortion*noise)),
            'total_error_power': float(np.mean((output-reference)**2)),
            'nmse': float(np.mean((output-reference)**2)/power),
            'component_identity_max_error': float(np.max(np.abs(output-component['clean'][lo:hi]-noise))),
        }
    return report


def decode_pcm(data, *, channels=1):
    """Actual uncompressed PCM16 integer samples, channels x samples."""
    try:
        with wave.open(io.BytesIO(data), 'rb') as reader:
            if (reader.getframerate(), reader.getnframes(), reader.getnchannels(),
                reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, SAMPLES, channels, 2, 'NONE'):
                raise ValueError('unexpected weighted PCM format')
            raw = reader.readframes(SAMPLES)
    except (wave.Error, EOFError) as error:
        raise ValueError('invalid weighted PCM WAV') from error
    if len(raw) != SAMPLES*channels*2:
        raise ValueError('truncated weighted PCM')
    values = struct.unpack('<'+'h'*(SAMPLES*channels), raw)
    return np.array(values, dtype=np.int64).reshape(SAMPLES, channels).T


def analyze_pcm(buffers):
    """Total integer error against actual reference, with no gain/time fitting."""
    expected = {stem+'.wav' for stem in STEMS}
    if set(buffers) != expected:
        raise ValueError('weighted PCM set must contain exactly five WAVs')
    integers = {name: decode_pcm(data, channels=2 if name=='weighted_array.wav' else 1)
                for name, data in buffers.items()}
    lo, hi = SCORE
    reference = integers['weighted_target.wav'][0, lo:hi]
    denominator = sum(int(x)**2 for x in reference)
    if denominator <= 0:
        raise ValueError('PCM reference power must be positive')
    candidates = {}
    for name in WEIGHTS:
        filename = name+'.wav'
        numerator = sum((int(x)-int(y))**2 for x, y in zip(integers[filename][0, lo:hi], reference))
        candidates[filename] = {'integer_error_squared_sum': numerator,
            'integer_reference_squared_sum': denominator, 'scored_samples': hi-lo,
            'mse': numerator/((hi-lo)*32768**2), 'nmse': numerator/denominator,
            'score': [lo, hi], 'gain_fitting': False, 'delay_fitting': False}
    return {'scored_samples': hi-lo, 'integer_reference_squared_sum': denominator,
            'decoded_amplitude_denominator': 32768, 'candidates': candidates}


def _literal_contract():
    """Independent fixed fixture, rather than a second call to build_fixture.

    Literal constants deliberately do not follow mutable generation constants.
    This checks trusted internal model drift; it is not an untrusted-code sandbox.
    """
    weights = {'weighted_ols': [.5, .5], 'weighted_gls': [.8, .2],
               'weighted_reversed': [.2, .8]}
    parameters = {
        'sample_rate_hz': 16000, 'samples_per_channel': 32000,
        'source_score': [1600, 30400], 'scored_samples': 28800,
        'target_frequency_hz': 700, 'target_amplitude': .2,
        'noise_frequencies_hz': [3500, 4000], 'noise_amplitudes': [.03, .06],
        'envelope': 'min(1,n/640,(31999-n)/640), n=0..31999; endpoints exactly zero',
        'weights': weights, 'noise_covariance_on_score': [[.00045, 0.], [0., .0018]],
        'common_export_gain': 1.0, 'delay_samples': 0,
        'limits': 'deterministic finite-window orthogonality, not random independence, estimated covariance, speech or listening-study evidence',
    }
    n = np.arange(32000)
    envelope = np.minimum(1., np.minimum(n/640, (31999-n)/640))
    target = .2*np.cos(2*np.pi*700*n/16000)*envelope
    first = .03*np.cos(2*np.pi*3500*n/16000)*envelope
    second = .06*np.cos(2*np.pi*4000*n/16000)*envelope
    signals = {'weighted_target': target,
               'weighted_array': np.stack((target+first, target+second))}
    components = {}
    for name, (w1, w2) in weights.items():
        noise = w1*first+w2*second
        signals[name] = target+noise
        components[name] = {'clean': target.copy(), 'noise': noise}
    return {'parameters': parameters, 'signals': signals, 'components': components}


def _fixed_array(actual, expected, label):
    if (type(actual) is not np.ndarray or actual.dtype.kind != 'f'
            or actual.shape != expected.shape or not np.all(np.isfinite(actual))):
        raise ValueError(f'{label} must be a finite real floating array of the fixed shape')
    if not np.allclose(actual, expected, rtol=0., atol=3e-13):
        raise ValueError(f'{label} differs from the fixed literal waveform')
    if np.any(actual[..., (0, -1)] != 0.):
        raise ValueError(f'{label} fixed envelope endpoints must be exactly zero')


def validate_fixed_fixture(fixture):
    """Check exact typed declarations and all unquantized samples before IO."""
    literal = _literal_contract()
    if type(fixture) is not dict or fixture.keys() != literal.keys():
        raise ValueError('fixed weighted fixture must have exactly parameters/signals/components')
    if not same_metadata(fixture['parameters'], literal['parameters']):
        raise ValueError('fixed weighted parameters differ in value, structure or type')
    for section in ('signals', 'components'):
        if type(fixture[section]) is not dict or fixture[section].keys() != literal[section].keys():
            raise ValueError(f'fixed weighted {section} member set differs')
    for name, expected in literal['signals'].items():
        _fixed_array(fixture['signals'][name], expected, name)
    for name, expected in literal['components'].items():
        actual = fixture['components'][name]
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError('fixed weighted component fields differ')
        for key, values in expected.items():
            _fixed_array(actual[key], values, name+'.'+key)
    return literal


def _fixed_report(actual, expected, label):
    """Fixed report structure/types; powers nonnegative, signed cross allowed."""
    if type(actual) is not type(expected):
        raise ValueError(f'{label} has an incorrect metadata type')
    if type(expected) is dict:
        if actual.keys() != expected.keys():
            raise ValueError(f'{label} fields differ')
        for key, value in expected.items():
            _fixed_report(actual[key], value, label+'.'+key)
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(f'{label} length differs')
        for index, value in enumerate(expected):
            _fixed_report(actual[index], value, label+f'[{index}]')
    elif type(expected) is float:
        if not np.isfinite(actual):
            raise ValueError(f'{label} must be finite')
        if label.rsplit('.', 1)[-1] != 'cross_term' and expected >= 0 and actual < 0:
            raise ValueError(f'{label} power/error must not be negative')
        if not np.isclose(actual, expected, rtol=1e-12, atol=2e-15):
            raise ValueError(f'{label} differs from the literal model')
    elif actual != expected:
        raise ValueError(f'{label} differs from the literal model')


def validate_fixed_export(fixture, analytic, floating, buffers, pcm):
    """Pure preflight of declarations, float statistics and actual encoded PCM.

    Output PCM is independently quantized from the literal target/noise sum.
    No gain or delay is fitted. Small float summation residuals remain allowed;
    negative cross terms are legal, unlike negative squared powers.
    """
    literal = validate_fixed_fixture(fixture)
    powers = {'weighted_ols': 9/16000, 'weighted_gls': 9/25000,
              'weighted_reversed': 117/100000}
    expected_analytic = {'target_power': .02, 'noise_covariance': [[.00045, 0.], [0., .0018]],
                         'candidates': {}}
    expected_float = {'scored_samples': 28800, 'reference_power': .02, 'candidates': {}}
    for name, power in powers.items():
        row = {'target_distortion_power': 0., 'residual_noise_power': power,
               'cross_term': 0., 'total_error_power': power, 'nmse': power/.02}
        expected_analytic['candidates'][name+'.wav'] = row
        expected_float['candidates'][name+'.wav'] = {**row, 'component_identity_max_error': 0.}
    _fixed_report(analytic, expected_analytic, 'analytic')
    _fixed_report(floating, expected_float, 'floating')
    expected_names = {name+'.wav' for name in literal['signals']}
    if type(buffers) is not dict or buffers.keys() != expected_names:
        raise ValueError('fixed weighted export must contain exactly five WAV buffers')
    quantized = {}
    for name, values in literal['signals'].items():
        expected = np.rint(np.atleast_2d(values)*32768).astype(np.int64)
        blob = buffers[name+'.wav']
        if type(blob) is not bytes:
            raise ValueError('fixed weighted WAV buffers must be bytes')
        actual = decode_pcm(blob, channels=2 if name == 'weighted_array' else 1)
        if not np.array_equal(actual, expected):
            raise ValueError(f'{name} PCM differs from independent literal quantization')
        quantized[name] = expected
    reference = quantized['weighted_target'][0, 1600:30400]
    denominator = sum(int(value)**2 for value in reference)
    expected_pcm = {'scored_samples': 28800, 'integer_reference_squared_sum': denominator,
                    'decoded_amplitude_denominator': 32768, 'candidates': {}}
    for name in powers:
        numerator = sum((int(x)-int(y))**2 for x, y in zip(
            quantized[name][0, 1600:30400], reference))
        expected_pcm['candidates'][name+'.wav'] = {
            'integer_error_squared_sum': numerator, 'integer_reference_squared_sum': denominator,
            'scored_samples': 28800, 'mse': numerator/(28800*32768**2), 'nmse': numerator/denominator,
            'score': [1600, 30400], 'gain_fitting': False, 'delay_fitting': False}
    if not same_metadata(pcm, expected_pcm):
        raise ValueError('fixed weighted PCM scores differ in value, structure or type')
