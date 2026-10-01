"""Two task compositions, the same causal FIRs, and explicit selection weights.

Known deterministic tones, not speech/ASR/device or listening performance.
Both candidates receive only their scene's mixture. All eight recordings retain
32000 source samples and the complete eight-sample maximum convolution tail.
"""
from __future__ import annotations
import io
import math
import struct
import wave
import numpy as np

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
SAMPLES = SOURCE_SAMPLES + 8
GAIN = .8
SCORE = (1600, 30400)
SCENES = ('single', 'dual')
STEMS = tuple(f'selection_{scene}_{kind}' for scene in SCENES
              for kind in ('target', 'mixture', 'fir3', 'fir9'))
FREQUENCIES = (500, 1500, 3500)


def _filters():
    return {f'fir{length}': np.ones(length) / abs(sum(
        np.exp(-2j*np.pi*500*k/SAMPLE_RATE) for k in range(length)))
            for length in (3, 9)}


def _aggregate(candidates, metric):
    values = {key: {scene: candidates[f'selection_{scene}_{key}.wav'][metric]
                    for scene in SCENES} for key in ('fir3', 'fir9')}
    by_weight = {str(q): {key: q*v['single']+(1-q)*v['dual'] for key, v in values.items()}
                 for q in (.25, .75)}
    difference_single = values['fir3']['single']-values['fir9']['single']
    difference_dual = values['fir3']['dual']-values['fir9']['dual']
    return {'q_definition': 'single-tone scene weight; dual weight is 1-q',
            'at_q': by_weight, 'equal_cost_single_weight': -difference_dual/(difference_single-difference_dual),
            'q_interval': [.25, .75],
            'worst_over_q_interval': {key: max(row[key] for row in by_weight.values()) for key in values},
            'worst_scene': {key: max(v.values()) for key, v in values.items()}}


def build_fixture():
    t = np.arange(SOURCE_SAMPLES)/SAMPLE_RATE
    fade = np.minimum(np.clip(t/.02, 0, 1), np.clip((2-t)/.02, 0, 1))
    tones = {f: .2*np.cos(2*np.pi*f*t)*fade for f in FREQUENCIES}
    filters = _filters()
    signals, components = {}, {}
    for scene in SCENES:
        target = tones[500] + (tones[1500] if scene == 'dual' else 0)
        noise = tones[3500]
        signals[f'selection_{scene}_target'] = np.pad(target, (0, 8))
        signals[f'selection_{scene}_mixture'] = np.pad(target+noise, (0, 8))
        for key, taps in filters.items():
            name = f'selection_{scene}_{key}'
            signals[name] = np.pad(np.convolve(target+noise, taps), (0, 9-len(taps)))
            components[name] = {'clean': np.pad(np.convolve(target, taps), (0, 9-len(taps))),
                                'noise': np.pad(np.convolve(noise, taps), (0, 9-len(taps)))}
    parameters = {'sample_rate_hz': SAMPLE_RATE, 'source_samples': SOURCE_SAMPLES,
                  'samples_per_channel': SAMPLES, 'source_duration_s': 2.,
                  'export_duration_s': SAMPLES/SAMPLE_RATE, 'component_amplitude': .2,
                  'source_score': list(SCORE), 'common_export_gain': GAIN,
                  'fade': 'linear min(1,t/.02,(2-t)/.02), n=0..31999; then eight zeros',
                  'scene_target_frequencies_hz': {'single': [500], 'dual': [500, 1500]},
                  'noise_frequency_hz': 3500, 'target_steady_power': {'single': .02, 'dual': .04},
                  'input_steady_snr_db': {'single': 0., 'dual': 10*math.log10(2)},
                  'filters': {key: {'taps': h.tolist(), 'group_delay_samples': (len(h)-1)//2,
                      'output_score': [SCORE[0]+(len(h)-1)//2, SCORE[1]+(len(h)-1)//2]}
                      for key, h in filters.items()},
                  'convolution': 'causal full linear convolution with zero initial history; common 32008 length',
                  'score': 'known group delay only; no gain/time fitting; no scoring of fade/startup/tail',
                  'pcm_fit': 'DC and cos/sin orthogonal projections at three whole-cycle frequencies; no compensation'}
    return {'signals': signals, 'components': components, 'parameters': parameters}


def analytic_results(fixture):
    candidates = {}
    for scene in SCENES:
        ps = .02 if scene == 'single' else .04
        for key, config in fixture['parameters']['filters'].items():
            L = len(config['taps'])
            base = abs(math.sin(L*math.pi*500/SAMPLE_RATE)/(L*math.sin(math.pi*500/SAMPLE_RATE)))
            response = {str(f): math.sin(L*math.pi*f/SAMPLE_RATE)/(L*math.sin(math.pi*f/SAMPLE_RATE))/base
                        for f in FREQUENCIES}
            distortion = 0. if scene == 'single' else .02*(response['1500']-1)**2
            residual = .02*response['3500']**2
            candidates[f'selection_{scene}_{key}.wav'] = {
                'signed_aligned_response': response, 'reference_power': ps,
                'target_distortion_power': distortion, 'residual_noise_power': residual,
                'cross_term_power': 0., 'total_error_power': distortion+residual,
                'aligned_total_nmse': (distortion+residual)/ps}
    return {'candidates': candidates, 'selection': _aggregate(candidates, 'aligned_total_nmse')}


def analyze_fixture(fixture):
    """Float component powers before the common .8 export gain."""
    candidates = {}
    lo, hi = SCORE
    for scene in SCENES:
        reference = fixture['signals'][f'selection_{scene}_target'][lo:hi]
        ps = float(np.mean(reference**2))
        for key, config in fixture['parameters']['filters'].items():
            name = f'selection_{scene}_{key}'
            start, stop = config['output_score']
            clean = fixture['components'][name]['clean'][start:stop]
            noise = fixture['components'][name]['noise'][start:stop]
            error_target = clean-reference
            error = fixture['signals'][name][start:stop]-reference
            candidates[name+'.wav'] = {'reference_power': ps,
                'target_distortion_power': float(np.mean(error_target**2)),
                'residual_noise_power': float(np.mean(noise**2)),
                'cross_term_power': float(2*np.mean(error_target*noise)),
                'total_error_power': float(np.mean(error**2)),
                'aligned_total_nmse': float(np.sum(error**2)/np.sum(reference**2))}
    return {'candidates': candidates, 'selection': _aggregate(candidates, 'aligned_total_nmse')}


def decode_pcm(data, *, samples=SAMPLES):
    """Actual mono PCM16 bytes -> integer samples, with exact format/length."""
    try:
        with wave.open(io.BytesIO(data), 'rb') as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes(), w.getcomptype()) != (
                    SAMPLE_RATE, 1, 2, samples, 'NONE'):
                raise ValueError('invalid selection PCM format')
            raw = w.readframes(samples)
        if len(raw) != 2*samples:
            raise ValueError('truncated selection PCM')
        return struct.unpack('<'+'h'*samples, raw)
    except (wave.Error, EOFError, TypeError) as error:
        raise ValueError('invalid selection WAV') from error


def measure_pcm(reference, mixture, output, delay, *, target_1500_present=True):
    lo, hi = SCORE
    start, stop = lo+delay, hi+delay
    r, y = reference[lo:hi], output[start:stop]
    denominator = sum(int(v)**2 for v in r)
    if denominator <= 0:
        raise ValueError('actual PCM reference energy must be positive')
    numerator = sum((int(a)-int(b))**2 for a, b in zip(y, r))
    def phasors(signal, a, b):
        indices = np.arange(a,b)
        segment = np.asarray(signal[a:b], dtype=float)/32768
        return {str(f): complex(2*np.dot(segment, np.exp(-2j*np.pi*f*indices/SAMPLE_RATE))/(b-a))
                for f in FREQUENCIES}
    clean, mixed, fitted = phasors(reference,lo,hi), phasors(mixture,lo,hi), phasors(output,start,stop)
    noise_ratio = abs(fitted['3500'])/abs(mixed['3500']) if abs(mixed['3500']) else None
    return {'integer_error_squared_sum': numerator, 'integer_reference_squared_sum': denominator,
            'scored_samples': hi-lo, 'reference_score': [lo,hi], 'output_score': [start,stop],
            'aligned_total_nmse': numerator/denominator,
            'aligned_total_nmse_db': 10*math.log10(numerator/denominator) if numerator else None,
            'fitted_amplitudes': {f:abs(v) for f,v in fitted.items()},
            'fitted_phasors_real_imag': {f:[v.real,v.imag] for f,v in fitted.items()},
            'target_500_retention': abs(fitted['500'])/abs(clean['500']),
            'target_1500_retention': abs(fitted['1500'])/abs(clean['1500']) if target_1500_present else None,
            'noise_3500_retention': noise_ratio,
            'noise_attenuation_db': -20*math.log10(noise_ratio) if noise_ratio else None}


def analyze_pcm(buffers):
    expected = {name+'.wav' for name in STEMS}
    if set(buffers) != expected:
        raise ValueError('exactly eight selection PCM files required')
    pcm = {name: decode_pcm(data) for name,data in buffers.items()}
    candidates = {}
    for scene in SCENES:
        ref, mixed = [pcm[f'selection_{scene}_{key}.wav'] for key in ('target','mixture')]
        for L in (3,9):
            name = f'selection_{scene}_fir{L}.wav'
            candidates[name] = measure_pcm(ref,mixed,pcm[name],(L-1)//2, target_1500_present=scene == 'dual')
    return {'candidates': candidates, 'selection': _aggregate(candidates,'aligned_total_nmse')}
