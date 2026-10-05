"""Same RIR DRR, different known in-band complex responses.

Five deterministic mathematical signals, not measured rooms or a blind
estimator. Full linear convolution precedes the complete two-sample tail.
Scoring keeps the common clock/gain: no delay or compensating-gain fit.
"""
from __future__ import annotations
import io
import math
import struct
import wave
import numpy as np

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
SAMPLES = 32002
GAIN = 1.0
SCORE = (1600, 30400)
FREQUENCIES = (2000, 4000)
STEMS = ('response_source', 'response_reflection_a', 'response_reflection_b',
         'response_full_a', 'response_full_b')
REFLECTIONS = {'a': (0., .5, .5), 'b': (0., .5, -.5)}


def _pairs(values):
    return [[float(v.real), float(v.imag)] for v in values]


def build_fixture():
    n = np.arange(SOURCE_SAMPLES)
    envelope = np.minimum(1., np.minimum(n/320, (SOURCE_SAMPLES-1-n)/320))
    source = envelope*sum(.1*np.cos(2*np.pi*f*n/SAMPLE_RATE) for f in FREQUENCIES)
    signals = {'response_source': np.pad(source, (0, 2))}
    for label, rir in REFLECTIONS.items():
        reflection = np.convolve(source, rir)
        signals['response_reflection_'+label] = reflection
        signals['response_full_'+label] = signals['response_source']+reflection
    return {'signals': signals, 'parameters': {
        'sample_rate_hz': SAMPLE_RATE, 'source_samples': SOURCE_SAMPLES,
        'samples_per_channel': SAMPLES, 'full_tail_samples': 2,
        'source_score': list(SCORE), 'scored_samples': SCORE[1]-SCORE[0],
        'frequencies_hz': list(FREQUENCIES), 'amplitudes': [.1, .1],
        'envelope': 'min(1,n/320,(31999-n)/320), n=0..31999; endpoints exactly zero',
        'direct_rir': [1., 0., 0.],
        'reflection_rirs': {key: list(value) for key, value in REFLECTIONS.items()},
        'common_export_gain': GAIN, 'delay_fitting': False, 'gain_fitting': False,
        'scope': 'known complete mathematical convolution; not real room, speech, blind RIR/DRR estimation or listening study',
    }}


def analytic_results():
    z = np.exp(-2j*np.pi*np.array(FREQUENCIES)/SAMPLE_RATE)
    source_power = .01
    report = {'source_power': source_power, 'source_tone_powers': [.005, .005],
              'rir_drr_db': 10*math.log10(2), 'candidates': {}, 'reflections': {}}
    for label, rir in REFLECTIONS.items():
        hr = rir[1]*z+rir[2]*z*z
        power = float(.005*np.sum(np.abs(hr)**2))
        cross = float(.01*np.sum(hr.real))
        report['reflections']['response_reflection_'+label+'.wav'] = {
            'complex_response_real_imag': _pairs(hr), 'output_power': power,
            'source_to_reflection_output_energy_ratio_db': 10*math.log10(source_power/power)}
        report['candidates']['response_full_'+label+'.wav'] = {
            'complex_response_real_imag': _pairs(1+hr), 'total_error_power': power,
            'nmse': power/source_power, 'direct_output_power': source_power,
            'reflection_output_power': power, 'full_energy_cross_term': cross,
            'full_output_power': source_power+power+cross}
    return report


def _phasors(values):
    """Joint real cos/sin LS is a measurement, not an output compensation."""
    lo, hi = SCORE
    n = np.arange(lo, hi)
    columns = [v for f in FREQUENCIES for v in (np.cos(2*np.pi*f*n/SAMPLE_RATE),
                                               np.sin(2*np.pi*f*n/SAMPLE_RATE))]
    weights = np.linalg.lstsq(np.array(columns).T, values[lo:hi], rcond=None)[0]
    return weights[::2]-1j*weights[1::2]


def analyze_fixture(fixture=None):
    fixture = build_fixture() if fixture is None else fixture
    signals = fixture['signals']
    if set(signals) != set(STEMS):
        raise ValueError('response fixture requires exactly five named signals')
    for signal in signals.values():
        raw = np.asarray(signal)
        if raw.shape != (SAMPLES,) or raw.dtype.kind not in 'iuf' or not np.all(np.isfinite(raw)):
            raise ValueError('response signals must be finite real 32002-point arrays')
    lo, hi = SCORE
    reference = signals['response_source'][lo:hi]
    power = float(np.mean(reference**2))
    if not math.isfinite(power) or power <= 0:
        raise ValueError('float reference power must be positive and finite')
    result = {'scored_samples': hi-lo, 'reference_power': power, 'candidates': {},
              'signals': {key+'.wav': {'complex_tone_amplitudes_real_imag': _pairs(_phasors(value)),
                  'output_power': float(np.mean(value[lo:hi]**2))} for key, value in signals.items()}}
    for label in REFLECTIONS:
        reflection = signals['response_reflection_'+label][lo:hi]
        full = signals['response_full_'+label][lo:hi]
        mse = float(np.mean((full-reference)**2))
        result['candidates']['response_full_'+label+'.wav'] = {
            'direct_output_power': power, 'reflection_output_power': float(np.mean(reflection**2)),
            'full_energy_cross_term': float(2*np.mean(reference*reflection)),
            'full_output_power': float(np.mean(full**2)), 'total_error_power': mse, 'nmse': mse/power,
            'component_identity_max_error': float(np.max(np.abs(full-reference-reflection)))}
    return result


def decode_pcm(data):
    try:
        with wave.open(io.BytesIO(data), 'rb') as reader:
            if (reader.getframerate(), reader.getnframes(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, SAMPLES, 1, 2, 'NONE'):
                raise ValueError('unexpected response PCM format')
            raw = reader.readframes(SAMPLES)
    except (wave.Error, EOFError) as error:
        raise ValueError('invalid response WAV') from error
    if len(raw) != SAMPLES*2:
        raise ValueError('truncated response PCM')
    return np.array(struct.unpack('<'+str(SAMPLES)+'h', raw), dtype=np.int64)


def analyze_pcm(buffers):
    if set(buffers) != {stem+'.wav' for stem in STEMS}:
        raise ValueError('response PCM set requires exactly five WAVs')
    values = {name: decode_pcm(data) for name, data in buffers.items()}
    lo, hi = SCORE
    reference = values['response_source.wav'][lo:hi]
    denominator = sum(int(v)**2 for v in reference)
    if denominator <= 0:
        raise ValueError('actual PCM reference power must be positive')
    report = {'scored_samples': hi-lo, 'integer_reference_squared_sum': denominator,
              'decoded_amplitude_denominator': 32768, 'candidates': {}, 'signals': {}}
    for name, samples in values.items():
        report['signals'][name] = {'complex_tone_amplitudes_real_imag': _pairs(_phasors(samples/32768)),
            'integer_output_squared_sum': sum(int(v)**2 for v in samples[lo:hi]),
            'scored_samples': hi-lo}
    for label in REFLECTIONS:
        name = 'response_full_'+label+'.wav'
        numerator = sum((int(x)-int(y))**2 for x, y in zip(values[name][lo:hi], reference))
        report['candidates'][name] = {'integer_error_squared_sum': numerator,
            'integer_reference_squared_sum': denominator, 'scored_samples': hi-lo,
            'mse': numerator/((hi-lo)*32768**2), 'nmse': numerator/denominator,
            'score': [lo, hi], 'gain_fitting': False, 'delay_fitting': False}
    return report


def _fixed_literal():
    """Independent published fixture: sample shifts, not producer convolution.

    This guards trusted implementation drift, not arbitrary room data. It is
    an in-memory preflight; filesystem checks do not eliminate concurrent races.
    """
    n = np.arange(32000)
    envelope = np.minimum(1., np.minimum(n/320, (31999-n)/320))
    source = envelope*(.1*np.cos(2*np.pi*2000*n/16000)
                       + .1*np.cos(2*np.pi*4000*n/16000))
    padded = np.pad(source, (0, 2))
    signals = {'response_source': padded}
    for label, sign in (('a', 1), ('b', -1)):
        reflection = .5*np.pad(source, (1, 1))+sign*.5*np.pad(source, (2, 0))
        signals['response_reflection_'+label] = reflection
        signals['response_full_'+label] = padded+reflection
    parameters = {
        'sample_rate_hz': 16000, 'source_samples': 32000,
        'samples_per_channel': 32002, 'full_tail_samples': 2,
        'source_score': [1600, 30400], 'scored_samples': 28800,
        'frequencies_hz': [2000, 4000], 'amplitudes': [.1, .1],
        'envelope': 'min(1,n/320,(31999-n)/320), n=0..31999; endpoints exactly zero',
        'direct_rir': [1., 0., 0.],
        'reflection_rirs': {'a': [0., .5, .5], 'b': [0., .5, -.5]},
        'common_export_gain': 1.0, 'delay_fitting': False, 'gain_fitting': False,
        'scope': 'known complete mathematical convolution; not real room, speech, blind RIR/DRR estimation or listening study',
    }
    return signals, parameters


def _fixed_compare(actual, expected, path, *, measured=False):
    if type(actual) is not type(expected):
        raise ValueError(f'{path}: invalid fixed-contract type')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'{path}: missing or extra fixed-contract fields')
        for key in expected:
            _fixed_compare(actual[key], expected[key], path+'.'+key, measured=measured)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'{path}: invalid fixed-contract length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            _fixed_compare(a, b, path+f'[{index}]', measured=measured)
    elif type(expected) is float:
        # Signed energy cross terms and complex coordinates are legitimate.
        if not math.isfinite(actual) or (actual != expected and
                not (measured and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14))):
            raise ValueError(f'{path}: differs from independent fixed calculation')
        if expected >= 0 and any(path.endswith('.'+key) for key in
                ('output_power', 'source_power', 'reference_power', 'total_error_power',
                 'direct_output_power', 'reflection_output_power', 'full_output_power',
                 'nmse', 'mse', 'component_identity_max_error')) and actual < 0:
            raise ValueError(f'{path}: negative power or error')
    elif actual != expected:
        raise ValueError(f'{path}: differs from fixed contract')


def validate_fixed_fixture(fixture):
    """Require the complete typed literal before encoding or directory creation."""
    signals, parameters = _fixed_literal()
    if type(fixture) is not dict or fixture.keys() != {'signals', 'parameters'}:
        raise ValueError('response fixture must have exactly signals and parameters')
    _fixed_compare(fixture['parameters'], parameters, 'parameters')
    actual = fixture['signals']
    if type(actual) is not dict or actual.keys() != signals.keys():
        raise ValueError('fixed response signal names differ')
    for name, expected in signals.items():
        value = actual[name]
        if (type(value) is not np.ndarray or value.dtype != np.dtype('float64')
                or value.shape != (32002,) or not np.all(np.isfinite(value))
                or not np.array_equal(value, expected)):
            raise ValueError(f'{name}: differs from finite float64 literal waveform')


def validate_fixed_reports(buffers, analytic, floating, pcm):
    """Independently check closed responses, full floats and actual PCM readback.

    Fourier projections here use the declared integer-cycle scoring interval;
    they are not gain/delay compensation. Producer LS reports retain their
    numerical roundoff. PCM integers and every complete quantized sample are
    exact; report floats allow only a small calculation-roundoff tolerance.
    """
    signals, _ = _fixed_literal()
    filenames = {name+'.wav' for name in signals}
    if type(buffers) is not dict or buffers.keys() != filenames:
        raise ValueError('fixed response requires five PCM buffers')
    integers = {}
    for name, signal in signals.items():
        data = buffers[name+'.wav']
        if type(data) is not bytes:
            raise ValueError('fixed response WAV must be bytes')
        try:
            with wave.open(io.BytesIO(data), 'rb') as reader:
                if (reader.getframerate(), reader.getnframes(), reader.getnchannels(),
                        reader.getsampwidth(), reader.getcomptype()) != (16000, 32002, 1, 2, 'NONE'):
                    raise ValueError('fixed response WAV format differs')
                raw = reader.readframes(32002)
        except (wave.Error, EOFError) as error:
            raise ValueError('invalid fixed response WAV') from error
        expected = np.rint(signal*32768).astype('<i2').tobytes()
        if raw != expected:
            raise ValueError('actual PCM differs from independent literal quantization')
        integers[name+'.wav'] = np.array(struct.unpack('<32002h', raw), dtype=np.int64)
    root2 = math.sqrt(2)
    responses = {'a': [complex(root2/4, -(root2+2)/4), complex(-.5, -.5)],
                 'b': [complex(root2/4, (2-root2)/4), complex(.5, -.5)]}
    a = {'source_power': .01, 'source_tone_powers': [.005, .005],
         'rir_drr_db': 10*math.log10(2), 'candidates': {}, 'reflections': {}}
    for label, response in responses.items():
        power = .005*sum(abs(v)**2 for v in response)
        cross = .01*sum(v.real for v in response)
        a['reflections']['response_reflection_'+label+'.wav'] = {
            'complex_response_real_imag': _pairs(response), 'output_power': power,
            'source_to_reflection_output_energy_ratio_db': 10*math.log10(.01/power)}
        a['candidates']['response_full_'+label+'.wav'] = {
            'complex_response_real_imag': _pairs([1+v for v in response]),
            'total_error_power': power, 'nmse': power/.01, 'direct_output_power': .01,
            'reflection_output_power': power, 'full_energy_cross_term': cross,
            'full_output_power': .01+power+cross}
    n = np.arange(1600, 30400)
    def measured_pairs(value):
        scored = value[1600:30400]
        return [[float(2*np.dot(scored, np.cos(2*np.pi*f*n/16000))/28800),
                 float(-2*np.dot(scored, np.sin(2*np.pi*f*n/16000))/28800)]
                for f in (2000, 4000)]
    reference = signals['response_source'][1600:30400]
    power = float(np.mean(reference**2))
    f = {'scored_samples': 28800, 'reference_power': power, 'candidates': {},
         'signals': {name+'.wav': {'complex_tone_amplitudes_real_imag': measured_pairs(value),
                     'output_power': float(np.mean(value[1600:30400]**2))}
                     for name, value in signals.items()}}
    reference_pcm = integers['response_source.wav'][1600:30400]
    d = sum(int(v)**2 for v in reference_pcm)
    p = {'scored_samples': 28800, 'integer_reference_squared_sum': d,
         'decoded_amplitude_denominator': 32768, 'candidates': {},
         'signals': {name: {'complex_tone_amplitudes_real_imag': measured_pairs(value/32768),
                     'integer_output_squared_sum': sum(int(v)**2 for v in value[1600:30400]),
                     'scored_samples': 28800} for name, value in integers.items()}}
    for label in ('a', 'b'):
        name = 'response_full_'+label+'.wav'
        reflection = signals['response_reflection_'+label][1600:30400]
        full = signals['response_full_'+label][1600:30400]
        mse = float(np.mean((full-reference)**2))
        f['candidates'][name] = {
            'direct_output_power': power, 'reflection_output_power': float(np.mean(reflection**2)),
            'full_energy_cross_term': float(2*np.mean(reference*reflection)),
            'full_output_power': float(np.mean(full**2)), 'total_error_power': mse,
            'nmse': mse/power,
            'component_identity_max_error': float(np.max(np.abs(full-reference-reflection)))}
        e = sum((int(x)-int(y))**2 for x, y in zip(integers[name][1600:30400], reference_pcm))
        p['candidates'][name] = {'integer_error_squared_sum': e,
            'integer_reference_squared_sum': d, 'scored_samples': 28800,
            'mse': e/(28800*32768**2), 'nmse': e/d, 'score': [1600, 30400],
            'gain_fitting': False, 'delay_fitting': False}
    for name, actual, expected in (('analytic', analytic, a), ('floating', floating, f), ('pcm', pcm, p)):
        _fixed_compare(actual, expected, name, measured=True)
