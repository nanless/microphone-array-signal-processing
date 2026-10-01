"""Known two-path inversion and a constrained noise/distortion tradeoff.

This is a fixed mathematical fixture, not blind MINT, a room measurement or
speech enhancement. Arrays are channels x samples; delays are causal integer
delays. The same observation-side noise is used in both path conditions.
"""
from __future__ import annotations

from fractions import Fraction
import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
REFLECTION_DELAY = 512
SAMPLES = SOURCE_SAMPLES + REFLECTION_DELAY
FADE_SAMPLES = 320
FREQUENCIES = (200, 500, 900, 1200)
AMPLITUDE = .05
SEED = 20261001
NOISE_VARIANCE = 5e-7
REGULARIZATION = Fraction(1, 10000)
SCORING_INTERVAL = (2400, 29600)
FILE_NAMES = {
    'reference': 'mint_reference.wav',
    'well_array': 'mint_well_array.wav',
    'near_array': 'mint_near_array.wav',
    'well_exact': 'mint_well_exact.wav',
    'near_exact': 'mint_near_exact.wav',
    'near_regularized': 'mint_near_regularized.wav',
}
LIMITS = ('Original known-path mathematical synthesis; no speech, measured room or listening study. '
          'Noise is added after the two paths, with common draws across conditions. '
          'Rademacher population means/cross moments are not substituted for finite-record moments. '
          'Fixed causal one-tap filters preserve the direct coefficient sum, not every frequency response. '
          'The constrained regularizer penalizes the delayed reflection and weight norm; '
          'it is not a general MINT optimum. Float component scores and actual PCM total errors are separate. '
          'No posterior gain/time fitting, clipping, dither or per-file normalization. '
          'A 512-sample reflection has 512 polynomial zeros; coefficient difference is not zero distance.')


def _fraction(value, name):
    if isinstance(value, Fraction):
        return value
    return Fraction.from_float(finite_real_scalar(value, name))


def _finite_float(value, name):
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(name + ' is outside float64 support') from error
    if not np.isfinite(result) or (value != 0 and result == 0):
        raise ValueError(name + ' is outside float64 support')
    return result


def regularized_weights(a, b, regularization) -> np.ndarray:
    """Minimize (a*u1+b*u2)^2 + lambda*||u||^2 with u1+u2=1.

    Fraction inputs preserve exact rational parameters; other inputs preserve
    their binary floating-point values. A common coefficient with lambda=0
    has nonunique minimizers and is rejected, rather than called an inverse.
    """
    a, b, lam = (_fraction(v, n) for v, n in
                 ((a, 'a'), (b, 'b'), (regularization, 'regularization')))
    if lam < 0:
        raise ValueError('regularization must be nonnegative')
    difference = a-b
    denominator = difference*difference+2*lam
    if denominator == 0:
        raise ValueError('common path coefficients with zero regularization are nonunique')
    first = (lam-b*difference)/denominator
    return np.array([_finite_float(first, 'first weight'),
                     _finite_float(1-first, 'second weight')])


def constrained_design(a, b, regularization) -> dict:
    a, b, lam = (_fraction(v, n) for v, n in
                 ((a, 'a'), (b, 'b'), (regularization, 'regularization')))
    regularized_weights(a, b, lam)  # validates and checks supported conversion
    first = (lam-b*(a-b))/((a-b)**2+2*lam)
    weights = (first, 1-first)
    reflection = a*weights[0]+b*weights[1]
    norm = sum(w*w for w in weights)
    total = reflection**2+lam*norm
    return {'weights': [_finite_float(w, 'weight') for w in weights],
            'weights_exact': [str(w) for w in weights], 'direct_response': 1.,
            'reflection_response': _finite_float(reflection, 'reflection'),
            'weight_norm_squared': _finite_float(norm, 'norm'),
            'reflection_cost': _finite_float(reflection**2, 'reflection cost'),
            'regularization_cost': _finite_float(lam*norm, 'regularization cost'),
            'total_cost': _finite_float(total, 'total cost'),
            'regularization': _finite_float(lam, 'regularization'),
            'scope': 'direct coefficient sum is one; only reflection and weight norm are penalized'}


def parameters() -> dict:
    a, near, well = Fraction(1, 2), Fraction(49, 100), Fraction(-1, 2)
    designs = {'well_exact': constrained_design(a, well, 0),
               'near_exact': constrained_design(a, near, 0),
               'near_regularized': constrained_design(a, near, REGULARIZATION)}
    for design in designs.values():
        design['population_noise_mean_square'] = NOISE_VARIANCE*design['weight_norm_squared']
        design['population_clean_error_mean_square'] = .005*design['reflection_response']**2
        design['population_total_reference_mse'] = (design['population_noise_mean_square']
                                                   + design['population_clean_error_mean_square'])
    return {'exercise_id': 'E07-21', 'sample_rate_hz': SAMPLE_RATE,
            'samples_per_channel': SAMPLES, 'source_samples': SOURCE_SAMPLES,
            'source_frequencies_hz': list(FREQUENCIES), 'amplitude_per_frequency': AMPLITUDE,
            'source_phase': 'cosines on absolute sample clock, initial phase zero',
            'source_active_interval_samples': [0, SOURCE_SAMPLES], 'fade_samples_each_end': FADE_SAMPLES,
            'fade': 'sin(linspace(0,pi/2,320)) squared, endpoint included; reversed at end',
            'reflection_delay_samples': REFLECTION_DELAY, 'reflection_delay_seconds': .032,
            'paths': {'first': [1., .5], 'well_second': [1., -.5], 'near_second': [1., .49]},
            'path_tap_positions': [0, REFLECTION_DELAY], 'negative_time_samples': 'zero',
            'complete_tail_samples': REFLECTION_DELAY,
            'noise_model': 'two independent Rademacher draws times sqrt(5e-7), added after each path at ALL 32512 samples',
            'noise_variance_per_channel': NOISE_VARIANCE, 'noise_shared_across_conditions': True,
            'seed': SEED, 'random_bit_generator': 'PCG64',
            'random_draw': 'integers(0,2,size=(2,32512),dtype=int64), map 0/1 to -1/+1',
            'steady_source_population_mean_square': .005,
            'regularization': float(REGULARIZATION),
            'regularization_interpretation': 'noise variance / declared steady source power = 5e-7/.005',
            'designs': designs, 'scoring_interval_samples': list(SCORING_INTERVAL),
            'scoring_samples_per_channel': SCORING_INTERVAL[1]-SCORING_INTERVAL[0],
            'tail_interval_samples': [SOURCE_SAMPLES, SAMPLES],
            'alignment': 'same source sample clock; fixed causal one-tap outputs; no fitted gain or delay',
            'common_export_gain': 1., 'channel_order': ['first_path', 'second_path'],
            'statistics': 'one finite fixed-seed fixture; population expectations and actual cross terms are distinct'}


def generate_components() -> dict:
    time = np.arange(SOURCE_SAMPLES)/SAMPLE_RATE
    source = AMPLITUDE*sum(np.cos(2*np.pi*f*time) for f in FREQUENCIES)
    fade = np.sin(np.linspace(0, np.pi/2, FADE_SAMPLES))**2
    source[:FADE_SAMPLES] *= fade
    source[-FADE_SAMPLES:] *= fade[::-1]
    reference = np.pad(source, (0, REFLECTION_DELAY))
    delayed = np.r_[np.zeros(REFLECTION_DELAY), source]
    rng = np.random.Generator(np.random.PCG64(SEED))
    noise = (2*rng.integers(0, 2, size=(2, SAMPLES), dtype=np.int64)-1)*np.sqrt(NOISE_VARIANCE)
    return {'reference': reference[None], 'delayed_source': delayed[None], 'noise_array': noise,
            'well_clean_array': np.vstack((reference+.5*delayed, reference-.5*delayed)),
            'near_clean_array': np.vstack((reference+.5*delayed, reference+.49*delayed))}


def _signal_arrays(samples, reference):
    x, r = finite_real_array(samples, 'samples'), finite_real_array(reference, 'reference')
    if x.shape not in ((1, SAMPLES), (2, SAMPLES)) or r.shape != (1, SAMPLES):
        raise ValueError('require mono or two-channel samples and mono reference, all 32512 samples')
    return x, r


def measure_signal(samples, reference, *, pcm=False) -> dict:
    """Uncompensated reference error; PCM mode requires actual int16/32768."""
    if not isinstance(pcm, (bool, np.bool_)):
        raise ValueError('pcm must be boolean')
    x, r = _signal_arrays(samples, reference)
    start, stop = SCORING_INTERVAL
    try:
        with np.errstate(over='raise', invalid='raise'):
            truth = r[0, start:stop]
            denominator = float(truth@truth)
            if denominator <= 0:
                raise ValueError('positive scoring reference power required')
            delta = x[:, start:stop]-truth
            errors = np.sum(delta*delta, axis=1)
            cross = x[:, start:stop]@truth
            result = {'scoring_interval_samples': [start, stop], 'sample_denominator_per_channel': stop-start,
                      'reference_squared_sum': denominator, 'error_squared_sum_per_channel': errors.tolist(),
                      'total_reference_mse_per_channel': (errors/(stop-start)).tolist(),
                      'relative_squared_reference_error_per_channel': (errors/denominator).tolist(),
                      'projection_gain_per_channel': (cross/denominator).tolist(),
                      'tail_interval_samples': [SOURCE_SAMPLES, SAMPLES], 'tail_samples_per_channel': REFLECTION_DELAY,
                      'tail_mean_square_per_channel': np.mean(x[:, SOURCE_SAMPLES:]**2, axis=1).tolist(),
                      'peak_per_channel': np.max(abs(x), axis=1).tolist()}
    except FloatingPointError as error:
        raise ValueError('measurement exceeds float64 support') from error
    if pcm:
        integers = []
        for value in (x, r):
            if np.any(value < -1) or np.any(value > 32767/32768):
                raise ValueError('PCM samples must lie in the signed int16 range')
            scaled = value*32768
            if not np.array_equal(scaled, np.rint(scaled)):
                raise ValueError('PCM samples must be exactly decoded int16/32768')
            integers.append(scaled.astype(np.int64))
        xi, ri = integers
        truth = ri[0, start:stop]
        denominator = int(truth@truth)
        if denominator <= 0:
            raise ValueError('positive PCM scoring reference power required')
        numerator = [int(d@d) for d in xi[:, start:stop]-truth]
        cross = [int(v@truth) for v in xi[:, start:stop]]
        tail = [int(v@v) for v in xi[:, SOURCE_SAMPLES:]]
        result.update({'pcm_decode_divisor': 32768, 'integer_reference_squared_sum': denominator,
                       'integer_error_squared_sum_per_channel': numerator,
                       'integer_output_reference_cross_sum_per_channel': cross,
                       'integer_tail_squared_sum_per_channel': tail,
                       'total_reference_mse_per_channel': [v/(32768**2*(stop-start)) for v in numerator],
                       'relative_squared_reference_error_per_channel': [v/denominator for v in numerator],
                       'projection_gain_per_channel': [v/denominator for v in cross],
                       'tail_mean_square_per_channel': [v/(32768**2*REFLECTION_DELAY) for v in tail]})
    return result


def run_experiment() -> dict:
    """Pure model, arrays and actual finite float component scores for plotting."""
    p, components = parameters(), generate_components()
    noise = components['noise_array']
    signals = {'reference': components['reference'],
               'well_array': components['well_clean_array']+noise,
               'near_array': components['near_clean_array']+noise}
    decomposition = {}
    start, stop = SCORING_INTERVAL
    for key in ('well_exact', 'near_exact', 'near_regularized'):
        condition = 'well' if key == 'well_exact' else 'near'
        w = np.array(p['designs'][key]['weights'])
        clean = (w@components[condition+'_clean_array'])[None]
        output_noise = (w@noise)[None]
        signals[key] = (w@signals[condition+'_array'])[None]
        error = (clean-components['reference'])[0, start:stop]
        n = output_noise[0, start:stop]
        components[key+'_clean'] = clean
        components[key+'_noise'] = output_noise
        ce, ne, cross = float(error@error), float(n@n), float(2*error@n)
        decomposition[key] = {'sample_denominator': stop-start, 'clean_error_squared_sum': ce,
                              'noise_squared_sum': ne, 'twice_clean_error_noise_cross_sum': cross,
                              'clean_error_mean_square': ce/(stop-start), 'noise_mean_square': ne/(stop-start),
                              'twice_clean_error_noise_cross_mean': cross/(stop-start),
                              'decomposed_total_mean_square': (ce+ne+cross)/(stop-start)}
    return {'parameters': p, 'signals': signals, 'components': components,
            'float_measurements': {k: measure_signal(v, signals['reference']) for k, v in signals.items()},
            'float_decomposition': decomposition,
            'finite_noise_statistics': {'scoring_interval_samples': [start, stop],
                'sample_denominator': stop-start, 'mean_per_channel': np.mean(noise[:, start:stop], axis=1).tolist(),
                'uncentered_second_moment': (noise[:, start:stop]@noise[:, start:stop].T/(stop-start)).tolist()},
            'limits': LIMITS}
