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
