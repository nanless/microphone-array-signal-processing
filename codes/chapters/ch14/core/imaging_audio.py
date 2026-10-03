"""Five known-tone imaging observations, not speech or measured arrays.

Balanced block phase codes cancel source cross terms in the declared finite
snapshot ensemble. They are not a proof of statistical independence. No WAV
is a power map, inverse waveform, or blind source estimate.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array, validate_waveforms
from codes.chapters.ch14.core.imaging import (
    csm_from_amplitudes, scan_power, source_power_csm, csm_residual,
    two_cell_experiment,
)

SAMPLE_RATE = 24000
FREQUENCY = 2000
BLOCK_SAMPLES = 2400
BLOCKS = 20
FADE_SAMPLES = 240
DELAY_SAMPLES = 4
SAMPLES = BLOCK_SAMPLES*BLOCKS+DELAY_SAMPLES
FRAME_OFFSET = 252
FRAME_STOP = 2160
FRAME_SAMPLES = FRAME_STOP-FRAME_OFFSET
SCORING_SAMPLES = BLOCKS*FRAME_SAMPLES
POWER_SCALE = .2**2/2
FILE_NAMES = {name: name+'.wav' for name in (
    'source_1', 'source_2_phase_code', 'source_2_coherent',
    'array_phase_code', 'array_coherent',
)}
LIMITS = ('Known 2kHz tones, deterministic balanced phase code and exactly known four-point propagation; '
          'not random-process independence, speech, a measured array, blind imaging, an inverse waveform '
          'or a listening evaluation. Digital amplitude squared is not Pa^2 or acoustic watts. '
          'No per-file normalization, clipping, fitted gain/delay, or circular shift.')


def complex_pairs(value):
    array = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(array)):
        raise ValueError('complex metadata must be finite')
    return np.stack([array.real, array.imag], axis=-1).tolist()


def scoring_intervals():
    return [[j*BLOCK_SAMPLES+FRAME_OFFSET, j*BLOCK_SAMPLES+FRAME_STOP] for j in range(BLOCKS)]


def parameters():
    return {'exercise_ids': ['E14-04', 'E14-05', 'E14-10'], 'sample_rate_hz': SAMPLE_RATE,
            'frequency_hz': FREQUENCY, 'samples_per_channel': SAMPLES,
            'body_samples': BLOCKS*BLOCK_SAMPLES, 'complete_tail_samples': DELAY_SAMPLES,
            'block_samples': BLOCK_SAMPLES, 'blocks': BLOCKS, 'fade_samples_each_end_of_block': FADE_SAMPLES,
            'envelope': 'min(r/240,(2399-r)/240,1), r=n mod 2400, common to both sources',
            'phase_code': [1 if j % 2 == 0 else -1 for j in range(BLOCKS)],
            'source_peak_amplitudes': [.2, .1], 'source_steady_mean_square': [.02, .005],
            'second_source_second_microphone_delay_samples': DELAY_SAMPLES,
            'positive_frequency_convention': 'x[n]=Re(z exp(+j*2*pi*f*n/fs)); positive DFT uses exp(-j*...)',
            'scoring_intervals_samples': scoring_intervals(), 'samples_per_snapshot': FRAME_SAMPLES,
            'integer_cycles_per_snapshot': 159, 'scoring_samples_per_channel': SCORING_SAMPLES,
            'complex_peak_amplitude': 'z=(2/1908)*sum(x[n]*exp(-j*2*pi*f*n/fs)) on each declared interval',
            'csm': 'R=(1/(2*20))*sum(z_l z_l^H), digital mean-square; interior-frequency tone only',
            'normalization_power_scale': POWER_SCALE, 'common_export_gain': 1.,
            'channel_order': 'microphone 1, microphone 2; only source 2 is delayed at microphone 2',
            'statistical_scope': 'zero cross term in specified balanced finite snapshot ensemble, not random independence'}


def make_signals():
    n = np.arange(BLOCKS*BLOCK_SAMPLES)
    r = n % BLOCK_SAMPLES
    envelope = np.minimum(np.minimum(r/FADE_SAMPLES, (BLOCK_SAMPLES-1-r)/FADE_SAMPLES), 1.)
    first = .2*envelope*np.cos(2*np.pi*FREQUENCY*n/SAMPLE_RATE)
    second = first/2
    coded = second*np.where((n//BLOCK_SAMPLES) % 2 == 0, 1., -1.)
    pad = lambda x: np.pad(x, (0, DELAY_SAMPLES))
    delay = lambda x: np.pad(x, (DELAY_SAMPLES, 0))
    return {'source_1': pad(first)[None, :], 'source_2_phase_code': pad(coded)[None, :],
            'source_2_coherent': pad(second)[None, :],
            'array_phase_code': np.vstack([pad(first)+pad(coded), pad(first)+delay(coded)]),
            'array_coherent': np.vstack([pad(first)+pad(second), pad(first)+delay(second)])}


def extract_snapshot_amplitudes(samples):
    x = validate_waveforms(samples)
    if x.shape[1] != SAMPLES or x.shape[0] not in (1, 2):
        raise ValueError('require one/two channels with exactly 48004 samples')
    indices = np.array([np.arange(start, stop) for start, stop in scoring_intervals()])
    phasors = np.exp(-2j*np.pi*FREQUENCY*indices/SAMPLE_RATE)
    with np.errstate(over='raise', invalid='raise'):
        try:
            result = (2/FRAME_SAMPLES)*np.einsum('cjl,jl->cj', x[:, indices], phasors)
        except FloatingPointError as exc:
            raise ValueError('snapshot calculation exceeds float64 support') from exc
    if not np.all(np.isfinite(result)):
        raise ValueError('snapshot amplitudes must be finite')
    return result


def analytic_csm(key):
    if key not in FILE_NAMES:
        raise ValueError('unknown imaging fixture')
    if key == 'source_1':
        return np.array([[.02]], complex)
    if key.startswith('source_2'):
        return np.array([[.005]], complex)
    # Scalar analytic entries independently derived from the two source tones.
    if key == 'array_phase_code':
        return POWER_SCALE*np.array([[1.25, .875+1j*np.sqrt(3)/8], [.875-1j*np.sqrt(3)/8, 1.25]])
    return POWER_SCALE*np.array([[2.25, 1.125+3j*np.sqrt(3)/8], [1.125-3j*np.sqrt(3)/8, .75]])


def measure_signal(samples, key, *, pcm=False):
    if type(pcm) is not bool:
        raise ValueError('pcm must be bool')
    x = validate_waveforms(samples)
    channels = 2 if key.startswith('array_') else 1
    if key not in FILE_NAMES or x.shape != (channels, SAMPLES):
        raise ValueError('imaging fixture key/shape differs')
    amplitudes = extract_snapshot_amplitudes(x)
    r = csm_from_amplitudes(amplitudes)
    expected = analytic_csm(key)
    result = {'snapshot_peak_amplitudes_real_imag': complex_pairs(amplitudes),
              'csm_real_imag': complex_pairs(r), 'csm_units': 'digital amplitude squared, single-tone mean-square',
              'analytic_csm_real_imag': complex_pairs(expected),
              'analytic_csm_max_abs_error': float(np.max(abs(r-expected))),
              'normalized_csm_max_abs_error': float(np.max(abs(r-expected))/POWER_SCALE),
              'scoring_samples_per_channel': SCORING_SAMPLES, 'snapshot_denominator': BLOCKS,
              'samples_per_snapshot': FRAME_SAMPLES, 'peak_all_samples': float(np.max(abs(x)))}
    if channels == 2:
        model = two_cell_experiment(); b = scan_power(r, model['W'])
        estimate = np.linalg.solve(model['P'], b)
        result.update({'scan_mean_square': b.tolist(), 'normalized_scan_mean_square': (b/POWER_SCALE).tolist(),
                       'inverse_source_mean_square': estimate.tolist(),
                       'normalized_inverse_source_mean_square': (estimate/POWER_SCALE).tolist(),
                       'scan_residual': (model['P']@estimate-b).tolist(),
                       'full_csm_residual': csm_residual(r, source_power_csm(model['A'], estimate))})
    if pcm:
        scaled = x*32768
        if np.any(x < -1) or np.any(x > 32767/32768) or not np.array_equal(scaled, np.rint(scaled)):
            raise ValueError('PCM measurements require genuine int16 values decoded with /32768')
        integers = scaled.astype(np.int64)
        result.update({'pcm_decode_divisor': 32768,
                       'integer_scored_squared_sum_by_channel': [
                           sum(int(v)**2 for start, stop in scoring_intervals() for v in channel[start:stop])
                           for channel in integers],
                       'integer_power_sample_denominator': SCORING_SAMPLES,
                       'integer_power_scale_denominator': 32768**2})
    return result


def measure_source_pairs(signals):
    first = extract_snapshot_amplitudes(signals['source_1'])
    result = {}
    for name, key in [('phase_code', 'source_2_phase_code'), ('coherent', 'source_2_coherent')]:
        r = csm_from_amplitudes(np.vstack([first, extract_snapshot_amplitudes(signals[key])]))
        denominator = float(r[0, 0].real*r[1, 1].real)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError('source coherency requires representable positive source powers')
        result[name] = {'source_csm_real_imag': complex_pairs(r),
                        'source_cross_term_real_imag': complex_pairs(r[0, 1]),
                        'magnitude_squared_coherence': float(abs(r[0, 1])**2/denominator)}
    return result


def run_experiment():
    signals = make_signals()
    return ({'parameters': parameters(), 'float_measurements': {
        key: measure_signal(value, key) for key, value in signals.items()},
        'float_source_pairs': measure_source_pairs(signals), 'limits': LIMITS}, signals)
