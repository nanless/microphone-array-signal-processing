"""E05-22: fixed three-microphone derivative constraints under real propagation.

An original free-field two-tone fixture with observation-side white noise.
No speech, room, online covariance estimator, fractional-delay FIR or device
is modeled. Fixed real weights are causal one-tap filter-and-sum weights.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array
from codes.chapters.ch03.core.geometry import plane_wave_delays
from codes.chapters.ch05.core.beamforming import apply_beamformer, lcmv_weights, mvdr_weights

SAMPLE_RATE = 16000
SAMPLES = 32002
SOURCE_SECONDS = 2.
FREQUENCIES = (1000., 3000.)
AMPLITUDE = .08
NOISE_STD = .02
SEED = 20260522
SCORING_INTERVAL = (2400, 29600)
FILE_NAMES = {'reference': 'derivative_reference.wav', 'array': 'derivative_array.wav',
              'single': 'derivative_single.wav', 'constrained': 'derivative_constrained.wav'}
LIMITS = ('Original mathematical free-field fixture, not speech, room or device recording. '
          'Propagation evaluates a continuous two-tone source with its envelope, not a sampled fractional-delay filter. '
          'Noise is added independently at the three observations; its exact diagonal covariance is supplied '
          'to fixed one-tap weights, not estimated online. Derivative constraints impose local complex-response '
          'flatness on the middle-microphone phase centre and increase output noise in this model. '
          'Float component truth and actual PCM total-reference measurements are separate. '
          'No posterior delay/gain fitting or per-file normalization is applied; playback is not a DOA or quality test.')


def parameters() -> dict:
    positions = np.column_stack(([-.04, 0., .04], np.zeros(3)))
    theta = np.arcsin(.1*343/(2*np.pi*1000*.04))
    covariance = np.diag([1., 2., 4.])
    # Solve the exact nominal E18 model once for fixed one-tap weights. No
    # covariance is estimated from this recording and no online update runs.
    target, derivative = np.ones(3), np.array([-1j, 0., 1j])
    single = mvdr_weights(covariance, target)
    constrained = lcmv_weights(covariance, np.column_stack((target, derivative)), [1., 0.])
    if max(np.max(abs(single.imag)), np.max(abs(constrained.imag))) > 1e-14:
        raise ValueError('this real one-tap fixture requires real weights')
    expected = {}
    for key, w in [('single', single), ('constrained', constrained)]:
        response = [np.vdot(w, np.exp(1j*phi*np.array([-1., 0., 1.]))) for phi in (.1, .3)]
        distortion = AMPLITUDE**2/2*sum(abs(b-1)**2 for b in response)
        noise = NOISE_STD**2*float(np.vdot(w, covariance@w).real)
        expected[key] = {'response_real_imag': complex_pairs(response),
                         'steady_clean_distortion_mean_square': float(distortion),
                         'white_noise_mean_square': noise,
                         'total_reference_mean_square': float(distortion+noise),
                         'weight_norm_squared': float(np.vdot(w, w).real)}
    return {'exercise_id': 'E05-22', 'sample_rate_hz': SAMPLE_RATE,
            'samples_per_channel': SAMPLES, 'source_active_interval_s': [0., SOURCE_SECONDS],
            'sound_speed_m_s': 343., 'positions_m': positions.tolist(), 'channel_order': [0, 1, 2],
            'azimuth_deg': float(np.rad2deg(theta)), 'phase_centre': 'middle microphone/array centre',
            'frequencies_hz': list(FREQUENCIES), 'phase_per_spacing_rad': [.1, .3],
            'source_amplitude_per_frequency': AMPLITUDE, 'fade_duration_s': .02,
            'fade': 'continuous linear envelope min(clip(t/.02),clip((2-t)/.02)); zero outside [0,2]',
            'common_propagation_delay_samples': 1.,
            'propagation_model': 'reference=F(t-1/fs); x_m_clean=F(t-1/fs+r_m dot u/c)',
            'noise_model': 'independent zero-mean Gaussian observation noise at every recording sample, including the two tail samples',
            'noise_variance_per_channel': (NOISE_STD**2*np.array([1., 2., 4.])).tolist(),
            'seed': SEED, 'random_bit_generator': 'PCG64',
            'single_weights': single.real.tolist(), 'constrained_weights': constrained.real.tolist(),
            'weight_model': 'nominal broadside exact diagonal covariance; real causal one-tap filter-and-sum',
            'scoring_interval_samples': list(SCORING_INTERVAL),
            'scoring_samples_per_channel': SCORING_INTERVAL[1]-SCORING_INTERVAL[0],
            'phasor_convention': 'cos coefficient minus j*sin coefficient; absolute sample clock; output/reference ratio',
            'expected_population_steady_window': expected,
            'expected_clean_reference_mean_square': AMPLITUDE**2,
            'common_export_gain': 1., 'alignment': 'fixed shared propagation clock and gain, no fitted correction'}


def complex_pairs(values) -> list:
    z = np.asarray(values)
    return np.stack((z.real, z.imag), axis=-1).tolist()


def _source(time: np.ndarray) -> np.ndarray:
    envelope = np.minimum(np.clip(time/.02, 0, 1), np.clip((SOURCE_SECONDS-time)/.02, 0, 1))
    return AMPLITUDE*envelope*sum(np.cos(2*np.pi*f*time) for f in FREQUENCIES)


def generate_components() -> dict[str, np.ndarray]:
    """Return finite float truth, preserving all fractional propagation tails."""
    p = parameters()
    time = np.arange(SAMPLES)/SAMPLE_RATE
    delays = plane_wave_delays(np.asarray(p['positions_m']), np.deg2rad(p['azimuth_deg']), reference=1)
    reference = _source(time-1/SAMPLE_RATE)[None, :]
    clean = np.array([_source(time-1/SAMPLE_RATE-delay) for delay in delays])
    rng = np.random.Generator(np.random.PCG64(SEED))
    noise = rng.standard_normal((3, SAMPLES))*np.sqrt(np.asarray(p['noise_variance_per_channel']))[:, None]
    return {'reference': reference, 'clean_array': clean, 'noise_array': noise, 'array': clean+noise}


def generate_signals() -> dict[str, np.ndarray]:
    p, truth = parameters(), generate_components()
    result = {'reference': truth['reference'], 'array': truth['array']}
    for key in ('single', 'constrained'):
        output = apply_beamformer(truth['array'][:, None, :], np.asarray(p[key+'_weights']))
        if np.any(output.imag != 0):
            raise ValueError('real fixed-weight fixture produced a complex output')
        result[key] = output.real
    return result


def measure_signal(samples: np.ndarray, reference: np.ndarray, *, pcm: bool = False) -> dict:
    """Four-column real LS response estimates and uncompensated total errors.

    Noise and quantization affect the measured phasors. Ratios describe the
    observation, not an exact noise-free transfer. No fit is used to rescale
    samples before the independently computed total-reference score.
    """
    x, r = finite_real_array(samples, 'samples'), finite_real_array(reference, 'reference')
    if x.shape not in ((1, SAMPLES), (3, SAMPLES)) or r.shape != (1, SAMPLES):
        raise ValueError('require one or three channels and a mono reference, all 32002 samples')
    start, stop = SCORING_INTERVAL
    n = np.arange(start, stop)
    design = np.column_stack([g(2*np.pi*f*n/SAMPLE_RATE) for f in FREQUENCIES for g in (np.cos, np.sin)])
    coefficients = np.linalg.lstsq(design, x[:, start:stop].T, rcond=None)[0]
    ref_coefficients = np.linalg.lstsq(design, r[:, start:stop].T, rcond=None)[0]
    z = np.array([coefficients[2*k]-1j*coefficients[2*k+1] for k in range(2)])
    ref_z = np.array([ref_coefficients[2*k, 0]-1j*ref_coefficients[2*k+1, 0] for k in range(2)])
    if np.any(abs(ref_z) == 0):
        raise ValueError('two positive reference phasors required')
    ratios = z/ref_z[:, None]
    error = x[:, start:stop]-r[:, start:stop]
    reference_energy = float(np.sum(r[:, start:stop]**2))
    error_energy = np.sum(error**2, axis=1)
    result = {'scoring_interval_samples': [start, stop], 'sample_denominator_per_channel': stop-start,
              'channels': len(x), 'real_ls_design_columns': 4,
              'reference_squared_sum': reference_energy,
              'error_squared_sum_per_channel': error_energy.tolist(),
              'total_reference_mse_per_channel': (error_energy/(stop-start)).tolist(),
              'normalized_reference_error_per_channel': np.sqrt(error_energy/reference_energy).tolist(),
              'reference_phasor_real_imag': complex_pairs(ref_z),
              'phasor_real_imag': complex_pairs(z), 'response_estimate_real_imag': complex_pairs(ratios),
              'response_estimate_magnitude': abs(ratios).tolist(),
              'response_estimate_phase_rad': np.angle(ratios).tolist(),
              'response_estimate_error_squared': (abs(ratios-1)**2).tolist(),
              'response_scope': 'input/reference LS estimate, affected by any noise/quantization in the supplied input; no component separation',
              'peak': float(np.max(abs(x)))}
    if pcm:
        integers, ref_integers = np.rint(x*32768), np.rint(r*32768)
        if not np.array_equal(integers, x*32768) or not np.array_equal(ref_integers, r*32768):
            raise ValueError('PCM measurements require exactly decoded int16/32768 samples')
        integers, ref_integers = integers.astype(np.int64), ref_integers.astype(np.int64)
        ri = ref_integers[0, start:stop]
        denominator = int(ri@ri)
        delta = integers[:, start:stop]-ri
        numerators = [int(y@y) for y in delta]
        result.update({'pcm_decode_divisor': 32768, 'integer_reference_squared_sum': denominator,
                       'integer_error_squared_sum_per_channel': numerators,
                       'integer_cross_sum_per_channel': [int(y@ri) for y in integers[:, start:stop]],
                       'total_reference_mse_per_channel': [e/(32768**2*(stop-start)) for e in numerators],
                       'normalized_reference_error_per_channel': [float(np.sqrt(e/denominator)) for e in numerators]})
    return result


def measure_float_decomposition() -> dict:
    p, truth = parameters(), generate_components()
    start, stop = SCORING_INTERVAL
    result = {}
    for name in ('single', 'constrained'):
        w = np.asarray(p[name+'_weights'])
        clean_error = (w@truth['clean_array']-truth['reference'][0])[start:stop]
        noise = (w@truth['noise_array'])[start:stop]
        clean_power, noise_power = float(np.mean(clean_error**2)), float(np.mean(noise**2))
        cross = float(2*np.mean(clean_error*noise))
        total = float(np.mean((clean_error+noise)**2))
        result[name] = {'sample_denominator': stop-start, 'scoring_interval_samples': [start, stop],
                        'clean_distortion_mean_square': clean_power, 'noise_mean_square': noise_power,
                        'twice_clean_error_noise_cross_mean': cross, 'total_reference_mean_square': total,
                        'decomposition_sum': clean_power+noise_power+cross,
                        'clean_measurements': measure_signal((w@truth['clean_array'])[None, :], truth['reference']),
                        'scope': 'actual unquantized generated components; no claim of PCM component separation'}
    return result
