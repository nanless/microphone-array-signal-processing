"""Seventeen shared-input mathematical WAVs for extension II.

These are instantaneous real linear mixtures, not speech, a room, STFT/WOLA,
blind covariance estimation, or a listening study. Known-time linear clock
correction calls the unique Chapter 10 stateful implementation. Its extra
frequency-dependent attenuation changes the input model and may reduce total
MSE below the unfiltered synchronous control; this is not an oracle victory.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import validate_waveforms, finite_real_array
from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch10.sro_closed_loop_demo import StatefulLinearClockCorrector
from codes.chapters.ch15.core.distributed import known_models, mwf_weights, compressed_mwf, mse_components

SAMPLE_RATE = 16000
SAMPLES = 32000
SEED = 1502026
SCORING_START, SCORING_STOP = 1600, 30400
SCORING_SAMPLES = SCORING_STOP-SCORING_START
PACKET_START, PACKET_STOP = 16000, 16800
POWER = .1**2/2+.06**2/2
CLOCK_PPM = 100.
CLOCK_CHUNK = 257
FILE_NAMES = {key: key+'.wav' for key in (
    'reference_node1', 'reference_node2', 'array_white', 'array_correlated',
    'local_node1', 'central_white', 'compressed_white', 'central_correlated',
    'compressed_correlated', 'stale_correlated', 'central_node2_correlated',
    'remote_scalar_white', 'transport_pcm16_white', 'clock_misaligned_white',
    'clock_linear_corrected_white', 'packet_zerofill_white', 'packet_local_fallback_white',
)}
OUTPUT_REFERENCES = {key: 'reference_node1' for key in FILE_NAMES
                     if key not in ('reference_node1', 'reference_node2', 'array_white',
                                    'array_correlated', 'remote_scalar_white')}
OUTPUT_REFERENCES['central_node2_correlated'] = 'reference_node2'
LIMITS = ('Known instantaneous four-channel real mixtures and deterministic orthogonal tones; '
          'not speech, measured rooms, temporal white noise, statistical independence, blind DANSE, '
          'STFT/WOLA, an acoustic array response, hardware clocks or industrial performance. '
          'No fitted gain/delay, per-file normalization, clipping or listening evaluation. '
          'Known-rate linear interpolation changes spectral attenuation; its total MSE is not '
          'the original synchronous-covariance optimum. Float processing precedes final PCM '
          'except the explicitly encoded/read-back remote PCM16 transport case.')


def scoring_windows():
    return {'steady': [SCORING_START, SCORING_STOP], 'packet': [PACKET_START, PACKET_STOP]}


def parameters():
    return {'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'body_samples': SAMPLES,
            'complete_convolution_tail_samples': 0, 'tail_scope': 'instantaneous mixture; no FIR tail exists',
            'seed': SEED, 'target_frequencies_hz': [700, 1300], 'target_peak_amplitudes': [.1, .06],
            'noise_basis_frequencies_hz': [1900, 2300, 2900, 3500], 'noise_basis_peak_amplitude': float(np.sqrt(2*POWER)),
            'phase_rule': 'default_rng(1502026).uniform(-pi,pi,6), two target phases then four noise phases',
            'envelope': 'clip(min(n/1600,(31999-n)/1600),0,1), with n=fs*t on device clock',
            'target_steady_mean_square': POWER, 'a': [1., .5, 2., -.5],
            'nodes_zero_based': [[0, 1], [2, 3]], 'reference_microphones_zero_based': [0, 2],
            'reference_target_factors': [1., 2.], 'channel_order': 'x1,x2,x3,x4',
            'covariance_units': 'digital amplitude squared, not Pa squared',
            'scoring_windows_samples': scoring_windows(), 'steady_scoring_samples': SCORING_SAMPLES,
            'packet_scoring_samples': PACKET_STOP-PACKET_START, 'common_export_gain': 1.,
            'clock': {'true_ppm': CLOCK_PPM, 'true_device_rate_hz': SAMPLE_RATE*(1+CLOCK_PPM*1e-6),
                      'initial_offset_s': 0., 'chunk_samples': CLOCK_CHUNK,
                      'device_samples_for_right_endpoint_support': int(np.ceil((SAMPLES-1)*(1+CLOCK_PPM*1e-6)))+1,
                      'rate_is_known': True, 'estimator_run': False,
                      'correction': 'existing StatefulLinearClockCorrector, state retained across calls; no antialias filter'},
            'packet': {'missing_interval_samples': [PACKET_START, PACKET_STOP], 'known_flag': True,
                       'packet_samples': 160, 'consecutive_missing_packets': 5,
                       'reference_timeline_preserved': True, 'policies': ['remote zero with frozen global weights', 'explicit local MWF fallback']},
            'transport': {'remote_vector': [2., -.5], 'receiver_coefficient': 2/13,
                          'quantized_signal': 'z=2*x3-.5*x4', 'format': 'PCM16 encode/read back before receiver fusion',
                          'output_error_half_step_bound': (2/13)*(.5/32768)},
            'orthogonality_scope': 'finite declared integer-cycle windows only, not random-process independence'}


def continuous_components(times_s):
    """Evaluate only this fixture's finite [0,2.001] s support budget."""
    times = finite_real_array(times_s, 'times')
    if (times.ndim != 1 or not len(times) or np.any(times < 0)
            or np.any(times > (SAMPLES+16)/SAMPLE_RATE)):
        raise ValueError('times must be nonempty 1D values within the fixture support [0,2.001] seconds')
    phase = np.random.default_rng(SEED).uniform(-np.pi, np.pi, 6)
    n = times*SAMPLE_RATE
    envelope = np.clip(np.minimum(n/1600, (SAMPLES-1-n)/1600), 0, 1)
    target = envelope*(.1*np.cos(2*np.pi*700*times+phase[0])
                       +.06*np.cos(2*np.pi*1300*times+phase[1]))
    noise = envelope*np.sqrt(2*POWER)*np.cos(
        2*np.pi*np.array([1900, 2300, 2900, 3500])[:, None]*times+phase[2:, None])
    return target, noise


def _clock_correct(indices, values, device_rate, chunk):
    corrector = StatefulLinearClockCorrector(SAMPLE_RATE, device_rate, 0., SAMPLES)
    emitted = []
    for start in range(0, len(indices), chunk):
        emitted.extend(corrector.push(indices[start:start+chunk].tolist(), values[start:start+chunk].tolist()))
    if [j for j, _ in emitted] != list(range(SAMPLES)) or any(v is None for _, v in emitted):
        raise ValueError('known-clock fixture must produce every reference sample using real right endpoint support')
    return np.array([v for _, v in emitted]), emitted


def generate_components():
    """Return signals and physical target/noise/error components without IO."""
    model = known_models(); a = model['a']; rs = model['Rs']
    target, basis = continuous_components(np.arange(SAMPLES)/SAMPLE_RATE)
    noises = {'white': basis, 'correlated': np.linalg.cholesky(model['correlated'])@basis}
    arrays = {key: a[:, None]*target+noise for key, noise in noises.items()}
    signals = {'reference_node1': target[None, :], 'reference_node2': (2*target)[None, :],
               'array_white': arrays['white'], 'array_correlated': arrays['correlated']}
    components = {}; analytic = {}

    def output(key, weights, scene='white', reference=0):
        weights = np.asarray(weights).real
        y = weights@arrays[scene]
        yt = (weights@a)*target; yn = weights@noises[scene]
        signals[key] = y[None, :]
        components[key] = {'target': yt, 'noise': yn, 'other_error': y-yt-yn}
        value = mse_components(POWER*rs, POWER*model[scene], weights, reference)
        analytic[key] = {name: dict(value) for name in scoring_windows()}
        return y, yt, yn

    local = compressed_mwf(rs, model['white'], np.eye(4)[:2])['weights'].real
    output('local_node1', local)
    for scene in ('white', 'correlated'):
        w = mwf_weights(rs, model[scene]).real
        output('central_'+scene, w, scene)
        t = np.eye(4)[:2].tolist()+[[0., 0., *w[2:]]]
        compressed = compressed_mwf(rs, model[scene], t)['weights'].real
        output('compressed_'+scene, compressed, scene)
    t = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 2., -.5]]
    output('stale_correlated', compressed_mwf(rs, model['correlated'], t)['weights'].real, 'correlated')
    output('central_node2_correlated', mwf_weights(rs, model['correlated'], 2).real, 'correlated', 2)
    remote = 2*arrays['white'][2]-.5*arrays['white'][3]
    signals['remote_scalar_white'] = remote[None, :]
    _, remote_pcm = read_pcm16(pcm16_bytes(remote, SAMPLE_RATE))
    transport_y = (2*arrays['white'][0]+arrays['white'][1]+2*remote_pcm[0])/13
    ref_component = components['central_white']
    signals['transport_pcm16_white'] = transport_y[None, :]
    components['transport_pcm16_white'] = {'target': ref_component['target'], 'noise': ref_component['noise'],
                                         'other_error': transport_y-ref_component['target']-ref_component['noise']}
    analytic['transport_pcm16_white'] = None  # Nonlinear rounding is actually measured separately.
    transport = {'remote_quantization_max_abs_error': float(np.max(abs(remote_pcm[0]-remote))),
                 'output_max_abs_difference_from_unquantized': float(np.max(abs(transport_y-signals['central_white'][0]))),
                 'independent_half_step_output_bound': (2/13)*(.5/32768)}
    # A fast remote clock samples the SAME continuous target and noise basis.
    fd = SAMPLE_RATE*(1+CLOCK_PPM*1e-6)
    indices = np.arange(int(np.ceil((SAMPLES-1)/SAMPLE_RATE*fd))+1)
    device_target, device_noise = continuous_components(indices/fd)
    peer_target = 4.25*device_target
    peer_noise = 2*device_noise[2]-.5*device_noise[3]
    peer = peer_target+peer_noise
    own_target = (5/26)*target
    own_noise = (2*basis[0]+basis[1])/13
    raw_target = own_target+(2/13)*peer_target[:SAMPLES]
    raw_noise = own_noise+(2/13)*peer_noise[:SAMPLES]
    raw = (2*arrays['white'][0]+arrays['white'][1])/13+(2/13)*peer[:SAMPLES]
    signals['clock_misaligned_white'] = raw[None, :]
    components['clock_misaligned_white'] = {'target': raw_target, 'noise': raw_noise, 'other_error': raw-raw_target-raw_noise}
    analytic['clock_misaligned_white'] = None
    corrected_peer, emitted = _clock_correct(indices, peer, fd, CLOCK_CHUNK)
    corrected_target, _ = _clock_correct(indices, peer_target, fd, CLOCK_CHUNK)
    corrected_noise, _ = _clock_correct(indices, peer_noise, fd, CLOCK_CHUNK)
    corrected = (2*arrays['white'][0]+arrays['white'][1])/13+(2/13)*corrected_peer
    yt = own_target+(2/13)*corrected_target; yn = own_noise+(2/13)*corrected_noise
    signals['clock_linear_corrected_white'] = corrected[None, :]
    components['clock_linear_corrected_white'] = {'target': yt, 'noise': yn, 'other_error': corrected-yt-yn}
    analytic['clock_linear_corrected_white'] = None
    whole, _ = _clock_correct(indices, peer, fd, len(indices))
    steady = slice(SCORING_START, SCORING_STOP)
    reference_peer_energy = float(remote[steady]@remote[steady])
    clock = {'device_input_samples': len(indices), 'reference_output_samples': len(emitted),
             'invalid_reference_indices': [], 'whole_vs_chunked_exact_equal': bool(np.array_equal(whole, corrected_peer)),
             'uncorrected_peer_reconstruction_nmse': float(np.sum((peer[:SAMPLES][steady]-remote[steady])**2)/reference_peer_energy),
             'corrected_peer_reconstruction_nmse': float(np.sum((corrected_peer[steady]-remote[steady])**2)/reference_peer_energy),
             'peer_reconstruction_reference_energy': reference_peer_energy,
             'same_index_drift_samples_last': float((SAMPLES-1)*CLOCK_PPM*1e-6/(1+CLOCK_PPM*1e-6)),
             'same_index_time_definition': 'reference n/fs minus device n/fd, positive for faster device',
             'limitations': 'known true rate; interpolation changes frequency response; not blind SRO estimation or original synchronous model'}
    packet = slice(PACKET_START, PACKET_STOP)
    for key, fallback in [('packet_zerofill_white', False), ('packet_local_fallback_white', True)]:
        y = signals['central_white'][0].copy(); yt = ref_component['target'].copy(); yn = ref_component['noise'].copy()
        if fallback:
            y[packet] = signals['local_node1'][0, packet]
            yt[packet] = components['local_node1']['target'][packet]
            yn[packet] = components['local_node1']['noise'][packet]
            during_weights = local
        else:
            y[packet] = (2*arrays['white'][0, packet]+arrays['white'][1, packet])/13
            yt[packet] = own_target[packet]; yn[packet] = own_noise[packet]
            during_weights = np.array([2., 1., 0., 0.])/13
        signals[key] = y[None, :]; components[key] = {'target': yt, 'noise': yn, 'other_error': y-yt-yn}
        during = mse_components(POWER*rs, POWER*model['white'], during_weights)
        normal = analytic['central_white']['steady']; fraction = (PACKET_STOP-PACKET_START)/SCORING_SAMPLES
        mixed = {name: (1-fraction)*normal[name]+fraction*during[name]
                 for name in ('target_distortion', 'noise_power', 'total_mse')}
        mixed.update({'reference_target_power': POWER, 'normalized_mse': mixed['total_mse']/POWER,
                      'target_noise_cross_term': 0., 'cross_term_scope': 'both declared subwindows are integer-cycle windows'})
        analytic[key] = {'steady': mixed, 'packet': during}
    if set(signals) != set(FILE_NAMES) or any(not np.all(np.isfinite(v)) for v in signals.values()):
        raise ValueError('distributed signal set or finite values differ')
    return {'signals': signals, 'components': components, 'analytic': analytic,
            'target': target, 'noise_basis': basis, 'clock': clock, 'transport': transport}


def measure_float_components(samples, reference, physical_components):
    y = validate_waveforms(samples); reference = validate_waveforms(reference)
    if y.shape != (1, SAMPLES) or reference.shape != (1, SAMPLES):
        raise ValueError('float decomposition requires one channel with the exact fixture length')
    parts = {key: finite_real_array(physical_components[key], key) for key in ('target', 'noise', 'other_error')}
    if any(v.shape != (SAMPLES,) for v in parts.values()):
        raise ValueError('physical components must each contain exactly 32000 samples')
    y = y[0]; reference = reference[0]
    target_error = parts['target']-reference
    noise = parts['noise']; other = parts['other_error']
    result = {}
    for name, (start, stop) in scoring_windows().items():
        d, n, o = target_error[start:stop], noise[start:stop], other[start:stop]
        error = y[start:stop]-reference[start:stop]; power = float(np.mean(reference[start:stop]**2))
        record = {'samples': stop-start, 'reference_power': power,
                  'target_distortion_power': float(np.mean(d*d)), 'noise_power': float(np.mean(n*n)),
                  'other_error_power': float(np.mean(o*o)), 'target_noise_cross_power': float(2*np.mean(d*n)),
                  'target_other_cross_power': float(2*np.mean(d*o)), 'noise_other_cross_power': float(2*np.mean(n*o)),
                  'total_mse': float(np.mean(error*error)), 'normalized_mse': float(np.mean(error*error)/power)}
        record['component_sum'] = sum(record[k] for k in ('target_distortion_power', 'noise_power', 'other_error_power',
                                                         'target_noise_cross_power', 'target_other_cross_power', 'noise_other_cross_power'))
        record['decomposition_residual'] = record['total_mse']-record['component_sum']
        result[name] = record
    return result


def measure_signal(samples, key, *, reference=None, pcm=False):
    if type(pcm) is not bool or key not in FILE_NAMES:
        raise ValueError('invalid distributed measurement key/pcm flag')
    x = validate_waveforms(samples)
    channels = 4 if key.startswith('array_') else 1
    if x.shape != (channels, SAMPLES):
        raise ValueError('distributed fixture channel count or exact sample length differs')
    is_output = key in OUTPUT_REFERENCES
    if is_output and reference is None:
        raise ValueError('output scores require the actual same-stage reference')
    ref = validate_waveforms(reference) if reference is not None else None
    if ref is not None and ref.shape != (1, SAMPLES):
        raise ValueError('reference must be one channel with exact fixture length')
    result = {'channels': channels, 'samples_per_channel': SAMPLES, 'peak_all_samples': float(np.max(abs(x))),
              'reference_key': OUTPUT_REFERENCES.get(key), 'windows': {}}
    integer_x = integer_ref = None
    if pcm:
        for value in (x, ref):
            if value is not None and (np.any(value < -1) or np.any(value > 32767/32768)
                                      or not np.array_equal(value*32768, np.rint(value*32768))):
                raise ValueError('PCM scores require genuine int16 values decoded with /32768')
        integer_x = np.rint(x*32768).astype(np.int64)
        integer_ref = np.rint(ref*32768).astype(np.int64) if ref is not None else None
        result.update({'pcm_decode_divisor': 32768, 'integer_power_scale_denominator': 32768**2})
    for window, (start, stop) in scoring_windows().items():
        record = {'samples': stop-start, 'mean_square_by_channel': np.mean(x[:, start:stop]**2, axis=1).tolist()}
        if is_output:
            error = x[0, start:stop]-ref[0, start:stop]
            denominator = float(np.sum(ref[0, start:stop]**2))
            record.update({'reference_energy': denominator, 'total_mse': float(np.mean(error**2)),
                           'normalized_mse': float(np.sum(error**2)/denominator) if denominator else None})
        if pcm:
            record['integer_squared_sum_by_channel'] = [sum(int(v)**2 for v in row[start:stop]) for row in integer_x]
            if is_output:
                e = sum(int(v)**2 for v in integer_x[0, start:stop]-integer_ref[0, start:stop])
                d = sum(int(v)**2 for v in integer_ref[0, start:stop])
                record.update({'integer_error_squared_sum_E': e, 'integer_reference_squared_sum_D': d,
                               'integer_sample_denominator': stop-start, 'integer_nmse_E_over_D': e/d if d else None})
        result['windows'][window] = record
    return result


def run_experiment():
    data = generate_components(); signals = data['signals']
    floats = {}; components = {}
    for key, value in signals.items():
        reference = signals.get(OUTPUT_REFERENCES.get(key))
        floats[key] = measure_signal(value, key, reference=reference)
        if key in OUTPUT_REFERENCES:
            components[key] = measure_float_components(value, reference, data['components'][key])
    basis = data['noise_basis'][:, SCORING_START:SCORING_STOP]
    source = data['target'][SCORING_START:SCORING_STOP]
    report = {'parameters': parameters(), 'analytic_expected': data['analytic'], 'float_measurements': floats,
              'float_components': components, 'clock': data['clock'], 'transport': data['transport'],
              'finite_steady_basis_covariance': (basis@basis.T/SCORING_SAMPLES).tolist(),
              'finite_steady_source_noise_cross': (basis@source/SCORING_SAMPLES).tolist(),
              'finite_steady_target_power': float(source@source/SCORING_SAMPLES), 'limits': LIMITS}
    return report, signals
