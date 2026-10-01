"""Offline known-bin mask representation, not blind separation or speech.

The full-record rFFT acts on two unfaded, periodic tones. Only their two bins
are retained; floating-point leakage elsewhere is explicitly discarded. The
same playback envelope is applied AFTER that operation to all six signals.
These are not STFT masks estimated from the windowed listening files.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array

SAMPLE_RATE = 16000
SAMPLES = 32000
FADE_SAMPLES = 320
FREQUENCIES = (500, 1250)
SCORING_INTERVAL = (2400, 29600)
FILE_NAMES = {key: 'mask_'+key+'.wav' for key in
              ('target', 'other', 'mixture', 'bounded_real', 'unbounded_real', 'complex_oracle')}
LIMITS = ('Known-bin full-record offline representation, not an estimated mask, blind separator, '
          'speech recording or listening evaluation. The same sin-squared listening envelope is '
          'applied after the rFFT operator. Nonselected numerical-leakage bins are set to zero. '
          'No delay/gain fitting, per-file normalization, dither or clipping. Phase LS measures '
          'the fixed absolute sample clock; it does not compensate the output.')


def parameters() -> dict:
    return {'exercise_id': 'E08-28', 'sample_rate_hz': SAMPLE_RATE,
            'samples_per_channel': SAMPLES, 'frequencies_hz': list(FREQUENCIES),
            'positive_rfft_bins': [1000, 2500],
            'source': 'target=.1*cos(500Hz)-.1*sin(1250Hz); other=-.08*cos(500Hz)+.1*cos(1250Hz)',
            'fft': 'full 32000-sample unwindowed periodic record; numpy rfft/irfft; only bins 1000/2500 retained',
            'target_phasors': [[.1, 0.], [0., .1]],
            'mixture_phasors': [[.02, 0.], [.1, .1]],
            'masks': {'bounded_real': [[1., 0.], [.5, 0.]],
                      'unbounded_real': [[5., 0.], [.5, 0.]],
                      'complex_oracle': [[5., 0.], [.5, .5]]},
            'envelope': 'after FFT operator: sin(linspace(0,pi/2,320)) squared at each end, reversed at end; endpoints zero',
            'fade_samples_each_end': FADE_SAMPLES,
            'scoring_interval_samples': list(SCORING_INTERVAL),
            'scoring_samples_per_channel': SCORING_INTERVAL[1]-SCORING_INTERVAL[0],
            'alignment': 'same absolute source clock; no propagation or fitted gain/delay',
            'common_export_gain': 1., 'algorithm_lookahead': 'entire 2-second record; noncausal offline known mask',
            'analytic_steady_reference_power': .01,
            'analytic_steady_mse': {'target': 0., 'other': .0262, 'mixture': .0082,
                                    'bounded_real': .0057, 'unbounded_real': .0025, 'complex_oracle': 0.}}


def _validate(samples, reference):
    x = finite_real_array(samples, 'samples')
    r = finite_real_array(reference, 'reference')
    if x.shape != (1, SAMPLES) or r.shape != x.shape:
        raise ValueError('require mono samples/reference of 32000 samples')
    return x, r


def measure_signal(samples, reference, *, pcm=False) -> dict:
    """Uncompensated MSE/NMSE and four-column absolute-clock phase LS."""
    if not isinstance(pcm, (bool, np.bool_)):
        raise ValueError('pcm must be boolean')
    x, r = _validate(samples, reference)
    start, stop = SCORING_INTERVAL
    truth, output = r[0, start:stop], x[0, start:stop]
    with np.errstate(over='raise', invalid='raise'):
        try:
            denominator = float(truth@truth)
            error = output-truth
            numerator = float(error@error)
        except FloatingPointError as exc:
            raise ValueError('reference/error power exceeds float64 support') from exc
    if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        raise ValueError('positive finite scoring reference power required')
    if numerator == 0 and np.any(error != 0):
        raise ValueError('nonzero scoring error power underflows float64 support')
    time = np.arange(start, stop)/SAMPLE_RATE
    design = np.column_stack([fn(2*np.pi*f*time) for f in FREQUENCIES for fn in (np.cos, np.sin)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, output, rcond=None)
    phasors = coefficients[::2]-1j*coefficients[1::2]
    result = {'scoring_interval_samples': [start, stop], 'sample_denominator': stop-start,
              'reference_squared_sum': denominator, 'error_squared_sum': numerator,
              'total_reference_mse': numerator/(stop-start), 'relative_squared_reference_error': numerator/denominator,
              'phase_ls': {'columns': 'cos500,sin500,cos1250,sin1250 on absolute sample clock',
                           'rank': int(rank), 'sample_denominator': stop-start,
                           'coefficients': coefficients.tolist(),
                           'phasors_real_imag': [[float(v.real), float(v.imag)] for v in phasors],
                           'amplitudes': abs(phasors).tolist(), 'phases_radians': np.angle(phasors).tolist(),
                           'residual_squared_sum': float(np.sum((output-design@coefficients)**2)),
                           'gain_or_time_compensation': False},
              'peak': float(np.max(abs(x)))}
    if pcm:
        integers = []
        for value in (x, r):
            scaled = value*32768
            if (np.any(value < -1) or np.any(value > 32767/32768) or
                    not np.array_equal(scaled, np.rint(scaled))):
                raise ValueError('PCM must be actual signed int16 samples decoded with /32768')
            integers.append(scaled.astype(np.int64)[0, start:stop])
        xi, ri = integers
        # Python integer sums explicitly preserve the scoring numerator.
        denominator = sum(int(v)**2 for v in ri)
        if denominator <= 0:
            raise ValueError('positive PCM scoring reference power required')
        numerator = sum((int(a)-int(b))**2 for a, b in zip(xi, ri))
        result.update({'pcm_decode_divisor': 32768, 'integer_reference_squared_sum': denominator,
                       'integer_error_squared_sum': numerator,
                       'integer_mse_denominator': (stop-start)*32768**2,
                       'reference_squared_sum': denominator/32768**2,
                       'error_squared_sum': numerator/32768**2,
                       'total_reference_mse': numerator/((stop-start)*32768**2),
                       'relative_squared_reference_error': numerator/denominator})
    return result


def run_experiment() -> tuple[dict, dict[str, np.ndarray]]:
    """Pure full-record FFT model, floating signals and their measurements."""
    time = np.arange(SAMPLES)/SAMPLE_RATE
    target = .1*np.cos(2*np.pi*500*time)-.1*np.sin(2*np.pi*1250*time)
    other = -.08*np.cos(2*np.pi*500*time)+.1*np.cos(2*np.pi*1250*time)
    mixture = target+other
    spectrum = np.fft.rfft(mixture)
    bins = np.array([1000, 2500])
    arrays = {'target': target, 'other': other, 'mixture': mixture}
    masks = {'bounded_real': [1., .5], 'unbounded_real': [5., .5],
             'complex_oracle': [5., .5+.5j]}
    for key, mask in masks.items():
        selected = np.zeros_like(spectrum)
        selected[bins] = spectrum[bins]*mask
        arrays[key] = np.fft.irfft(selected, n=SAMPLES)
    envelope = np.ones(SAMPLES)
    fade = np.sin(np.linspace(0, np.pi/2, FADE_SAMPLES))**2
    envelope[:FADE_SAMPLES], envelope[-FADE_SAMPLES:] = fade, fade[::-1]
    envelope[0] = envelope[-1] = 0.
    arrays = {key: (value*envelope)[None] for key, value in arrays.items()}
    report = {'parameters': parameters(), 'limits': LIMITS,
              'float_measurements': {key: measure_signal(value, arrays['target']) for key, value in arrays.items()},
              'unwindowed_selected_mixture_phasors_real_imag':
                  [[float(v.real), float(v.imag)] for v in 2*spectrum[bins]/SAMPLES],
              'discarded_bin_max_magnitude': float(np.max(abs(np.delete(spectrum, bins))))}
    return report, arrays
