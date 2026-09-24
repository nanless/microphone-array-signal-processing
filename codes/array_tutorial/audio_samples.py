"""Deterministic synthetic listening fixtures; not speech or room measurements.

Arrays use channels x samples. All cases have known inputs, no downloaded
recordings, and no pretrained model. Export applies ONE gain per comparison
group, including its references. Listen at a low device volume.
"""
from __future__ import annotations

import io
import wave

import numpy as np

from .aec import nlms
from .aec_ipnlms import ipnlms
from .aec_rls import rls
from .aec_kalman_matrix import KalmanAECState
from .aec_subband import haar_synthesize, two_tap_subband_outputs
from .dereverberation import offline_wpe
from .geometry import plane_wave_delays
from .noise_suppression import power_spectral_subtraction
from .spectral import stft, istft

SAMPLE_RATE = 16000
SEED = 20260922


def delay_samples(signal: np.ndarray, samples: int) -> np.ndarray:
    """Zero-padded causal delay, preserving length; never circularly wrap."""
    x = np.asarray(signal)
    if np.iscomplexobj(x):
        raise ValueError("signal must be real")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be finite and one-dimensional")
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 0:
        raise ValueError("delay must be a non-negative integer")
    out = np.zeros_like(x)
    if samples < x.size:
        out[samples:] = x[:x.size-samples]
    return out


def pcm16_bytes(waveforms: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode CxN audio without implicit normalization, saturation or dither."""
    x = np.asarray(waveforms)
    if np.iscomplexobj(x):
        raise ValueError("PCM input must be real")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("audio must be finite nonempty CxN")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample rate must be a positive integer")
    if np.max(np.abs(x)) > 32767 / 32768:
        raise ValueError("PCM headroom exceeded; choose an explicit common gain")
    quantized = np.rint(x.T * 32768).astype('<i2')
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as out:
        out.setnchannels(x.shape[0])
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(quantized.tobytes())
    return stream.getvalue()


def read_pcm16(data: bytes) -> tuple[int, np.ndarray]:
    """Read the uncompressed 16-bit PCM format produced by this module."""
    with wave.open(io.BytesIO(data), 'rb') as source:
        if source.getsampwidth() != 2 or source.getcomptype() != 'NONE':
            raise ValueError("expected uncompressed PCM16")
        rate, channels, frames = source.getframerate(), source.getnchannels(), source.getnframes()
        raw = source.readframes(frames)
    if len(raw) != frames * channels * 2:
        raise ValueError("truncated PCM payload")
    return rate, np.frombuffer(raw, dtype='<i2').reshape(-1, channels).T.astype(float) / 32768


def _tone(t: np.ndarray, frequency: float) -> np.ndarray:
    # An AM harmonic complex, deliberately labelled synthetic rather than speech.
    envelope = np.sin(np.pi * np.minimum(t / .02, 1) / 2)**2
    envelope *= np.sin(np.pi * np.minimum((t[-1] - t) / .02, 1) / 2)**2
    envelope *= .3 + .7 * np.sin(2 * np.pi * 2 * t)**2
    return envelope * sum(np.sin(2*np.pi*frequency*k*t) / k for k in range(1, 7)) / 4


def build_cases() -> dict:
    """Return thirteen experiments with model parameters and references.

    Each entry has ``signals`` (filename stem -> CxN array), ``parameters`` and
    ``limits``. Signals are pre-export floats; no group uses peak matching.
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    target = _tone(t, 173)
    other = _tone(t, 293)
    noise = .07 * rng.standard_normal((2, t.size))
    microphones = np.vstack((delay_samples(target, 3), target)) + noise
    # Align to the later microphone by delaying mic 2; both estimates share delay 3.
    aligned = .5 * (microphones[0] + delay_samples(microphones[1], 3))
    unaligned = .5 * (microphones[0] + microphones[1])

    far = .12 * rng.standard_normal(t.size) + .2 * other
    path = np.zeros(25)
    path[[0, 12, 24]] = [.7, -.3, .15]
    echo = np.convolve(far, path)[:t.size]
    active = (t >= 1.2) & (t < 1.8)
    near = .5 * target * active
    microphone = echo + near
    frozen, _, _ = nlms(far, microphone, 32, step_size=.4, freeze=active)
    free, _, _ = nlms(far, microphone, 32, step_size=.4)

    # Retain 0.25 s of tail. This is a sparse FIR echo, not a measured room/T60.
    dry = np.pad(target, (0, SAMPLE_RATE // 4))
    reverberant = dry + .6 * delay_samples(dry, 640) + .3 * delay_samples(dry, 1280)
    spectrum = stft(reverberant, n_fft=512, hop_length=128)
    processed = offline_wpe(spectrum.transpose(1, 0, 2), taps=4, delay=3, iterations=3)
    wpe = istft(processed.transpose(1, 0, 2), n_fft=512, hop_length=128, length=dry.size)[0]

    matrix = np.array([[1., .5], [.2, 1.]])
    sources = np.vstack((target, other))
    mixture = matrix @ sources
    recovered = np.linalg.solve(matrix, mixture)
    damaged = target.copy()
    damaged[(t >= .9) & (t < 1.)] = 0
    clipped = np.clip(3 * target, -.3, .3) / 3
    pan = np.linspace(0, np.pi / 2, t.size)
    # Separate streams keep the first six experiments unchanged when adding cases.
    contrast_rng = np.random.default_rng(SEED + 1)
    independent_noise = .07 * contrast_rng.standard_normal((2, t.size))
    independent_mean = target + independent_noise.mean(axis=0)
    common_mean = target + independent_noise[0]
    polarity_array = np.vstack((target, -.9 * target))
    measurement_noise = .003 * contrast_rng.standard_normal(sources.shape)
    well = np.array([[1., .5], [.5, 1.]])
    ill = np.array([[1., .99], [.99, 1.]])
    well_input, ill_input = well @ sources + measurement_noise, ill @ sources + measurement_noise
    # Exactly 1000 periods. No taper: preserve sinusoidal orthogonality over
    # the scoring interval. This is an analytic fixture, not a listening test.
    nonlinear_reference = .4 * np.sin(2 * np.pi * 500 * t)
    nonlinear_echo = nonlinear_reference + 2 * nonlinear_reference**3
    best_linear_gain = float(nonlinear_reference @ nonlinear_echo
                             / (nonlinear_reference @ nonlinear_reference))
    nonlinear_estimate = best_linear_gain * nonlinear_reference
    # A short, deliberately synthetic, shared-input AEC contrast. Use a new
    # stream so earlier fixtures and their PCM bytes do not change. The path
    # changes at exactly sample 6000; the microphone has known echo plus
    # independent background noise, never near-end speech.
    method_rng = np.random.default_rng(SEED + 3)
    method_samples = 12000
    method_reference = .12 * method_rng.standard_normal(method_samples)
    method_reference += .09 * np.sin(2 * np.pi * 310 * np.arange(method_samples) / SAMPLE_RATE)
    method_path_before = np.array([.65, 0., -.2, 0.])
    method_path_after = np.array([.1, -.45, 0., .5])
    method_change = 6000
    method_before = np.convolve(method_reference, method_path_before)[:method_samples]
    method_after = np.convolve(method_reference, method_path_after)[:method_samples]
    method_echo = method_before.copy()
    method_echo[method_change:] = method_after[method_change:]
    method_background = .005 * method_rng.standard_normal(method_samples)
    method_microphone = method_echo + method_background
    method_nlms, _, _ = nlms(method_reference, method_microphone, 4, step_size=.4)
    method_ipnlms, _, _ = ipnlms(method_reference, method_microphone, 4,
                                step_size=.4, kappa=0.,
                                denominator_floor=1e-8, gain_floor=1e-8)
    method_rls, _, _ = rls(method_reference, method_microphone, 4,
                           forgetting_factor=.995, initial_regularization=1.)
    method_kalman = KalmanAECState(
        4, transition=1., process_covariance=np.eye(4) * 1e-5,
        observation_variance=.0025, initial_covariance=np.eye(4),
    ).process_block(method_reference, method_microphone)['prior_error']
    # A separate exact model-identification contrast, not an adaptive run.
    subband_rng = np.random.default_rng(SEED + 4)
    subband_reference = .12 * subband_rng.standard_normal(8000)
    subband_reference += .08 * np.sin(2 * np.pi * 2100 * np.arange(8000) / SAMPLE_RATE)
    _, subband_diagonal_bands, subband_echo = two_tap_subband_outputs(
        subband_reference, [0., 1.])
    subband_diagonal = haar_synthesize(subband_diagonal_bands)
    # Four-microphone far-field example. A separate RNG leaves all older
    # experiments unchanged. Linear interpolation defines the fractional-
    # sample signal model; the corresponding alignment uses the same model.
    array_positions = np.column_stack((.04 * np.arange(4), np.zeros(4)))
    relative_delays = plane_wave_delays(array_positions, np.deg2rad(30.0))
    arrivals = .002 + relative_delays - np.min(relative_delays)
    fractional_rng = np.random.default_rng(SEED + 2)
    fractional_source = _tone(t, 383) + .04 * fractional_rng.standard_normal(t.size)
    fractional_clean = np.vstack([
        np.interp(t - arrival, t, fractional_source, left=0., right=0.)
        for arrival in arrivals
    ])
    fractional_array = fractional_clean + .02 * fractional_rng.standard_normal(fractional_clean.shape)
    fractional_reference = fractional_clean[0]
    fractional_unaligned = np.mean(fractional_array, axis=0)
    fractional_aligned = np.mean(np.vstack([
        np.interp(t - (arrivals[0] - arrival), t, channel, left=0., right=0.)
        for arrival, channel in zip(arrivals, fractional_array)
    ]), axis=0)
    subtraction_rng = np.random.default_rng(SEED + 5)
    subtraction_clean = target.copy()
    noise_only_samples = 6400
    subtraction_clean[:noise_only_samples] = 0.
    subtraction_clean[noise_only_samples:noise_only_samples + 320] *= (
        np.sin(np.linspace(0., np.pi / 2, 320, endpoint=False)) ** 2
    )
    subtraction_noise = .07 * subtraction_rng.standard_normal(t.size)
    subtraction_noisy = subtraction_clean + subtraction_noise
    subtraction_spectrum = stft(subtraction_noisy, n_fft=512, hop_length=128)[0]
    centers = np.arange(subtraction_spectrum.shape[1]) * 128
    noise_frames = np.flatnonzero((centers >= 256) & (centers + 256 <= noise_only_samples))
    subtraction_soft, subtraction_noise_power = power_spectral_subtraction(
        subtraction_spectrum, noise_frames, floor_ratio=.04
    )
    subtraction_zero, _ = power_spectral_subtraction(
        subtraction_spectrum, noise_frames, floor_ratio=0.
    )
    subtraction_soft_audio = istft(subtraction_soft[None], n_fft=512, hop_length=128,
                                   length=t.size)[0]
    subtraction_zero_audio = istft(subtraction_zero[None], n_fft=512, hop_length=128,
                                   length=t.size)[0]
    return {
        'spatial': {
            'signals': {'spatial_reference': delay_samples(target, 3),
                        'spatial_array': microphones, 'spatial_mic1': microphones[0],
                        'spatial_unaligned': unaligned, 'spatial_aligned': aligned},
            'parameters': {'delay_samples': 3, 'delay_seconds': 3/SAMPLE_RATE,
                           'noise_std_each_channel': .07, 'noise_correlation': 0,
                           'reference': 'target delayed by 3 samples; ignore first 3 samples when scoring',
                           'channel_order': ['mic1_later', 'mic2_earlier'],
                           'alignment': 'delay mic2 by 3, no noncausal advance'},
            'limits': 'Integer-delay broadband model; not binaural HRTF, measured array or speech.'},
        'fractional_array': {
            'signals': {'fractional_reference': fractional_reference,
                        'fractional_array': fractional_array,
                        'fractional_unaligned': fractional_unaligned,
                        'fractional_aligned': fractional_aligned},
            'parameters': {'seed': SEED + 2, 'sample_rate_hz': SAMPLE_RATE,
                           'positions_m': array_positions.tolist(), 'azimuth_deg': 30.,
                           'sound_speed_m_s': 343., 'relative_arrival_seconds': relative_delays.tolist(),
                           'arrival_seconds': arrivals.tolist(),
                           'arrival_difference_adjacent_samples': float((arrivals[0] - arrivals[1]) * SAMPLE_RATE),
                           'source': '383 Hz six-harmonic tone plus seeded broadband noise, std 0.04',
                           'sensor_noise_std_each_channel': .02,
                           'fractional_delay_model': 'linear interpolation with zero outside the 2 s source',
                           'reference': 'clean microphone 1 at its physical arrival time; no gain or delay fitting',
                           'alignment': 'causally delay microphones 2--4 to microphone 1 using linear interpolation',
                           'channel_order': ['mic1_x0', 'mic2_x0.04', 'mic3_x0.08', 'mic4_x0.12']},
            'limits': 'Far-field plane wave with simulated fractional delays and independent sensor noise; '
                      'no measured room, microphone directivity or physical moving-source recording.'},
        'aec': {
            'signals': {'aec_far': far, 'aec_near': near, 'aec_microphone': microphone,
                        'aec_frozen': frozen, 'aec_unfrozen': free},
            'parameters': {'path_nonzero_samples': [0, 12, 24], 'path_values': [.7, -.3, .15],
                           'filter_length': 32, 'step_size': .4, 'epsilon': 1e-8,
                           'oracle_freeze_interval_s': [1.2, 1.8],
                           'far_end_only_scoring_interval_s': [.6, 1.1], 'algorithm_delay_samples': 0},
            'limits': 'Exactly matched linear path, known double-talk mask (not an implemented DTD); no real-room ERLE claim.'},
        'aec_methods': {
            'signals': {
                'aec_methods_reference': method_reference,
                'aec_methods_true_echo': method_echo,
                'aec_methods_microphone': method_microphone,
                'aec_methods_nlms_residual': method_nlms,
                'aec_methods_ipnlms_residual': method_ipnlms,
                'aec_methods_rls_residual': method_rls,
                'aec_methods_kalman_residual': method_kalman,
            },
            'parameters': {
                'seed': SEED + 3, 'sample_rate_hz': SAMPLE_RATE,
                'samples': method_samples, 'path_change_sample': method_change,
                'path_before': method_path_before.tolist(),
                'path_after': method_path_after.tolist(),
                'reference': 'seeded white Gaussian std 0.12 plus 310 Hz sine amplitude 0.09',
                'microphone_model': 'causal 4-tap FIR, path selected per output sample, plus independent Gaussian background noise std 0.005; no near end',
                'background_noise_seed': SEED + 3,
                'nlms': {'step_size': .4, 'epsilon': 1e-8},
                'ipnlms': {'step_size': .4, 'kappa': 0., 'denominator_floor': 1e-8,
                           'gain_floor': 1e-8},
                'rls': {'forgetting_factor': .995, 'initial_regularization': 1.},
                'kalman': {'transition': 1., 'process_covariance_diagonal': 1e-5,
                           'observation_variance': .0025, 'initial_covariance_diagonal': 1.},
                'score_windows_samples': {'before_change': [3000, 6000],
                                          'immediate_after_change': [6000, 6200],
                                          'after_change': [9000, 12000]},
                'algorithm_delay_samples': 0,
                'output': 'causal prior linear residual before each sample update',
            },
            'limits': 'Seeded mathematical signal and exactly known 4-tap path; single run, no near end, '
                      'DTD, real loudspeaker, room or subjective speech test. Do not rank industrial AEC from these WAVs.',
        },
        'aec_subband': {
            'signals': {
                'aec_subband_reference': subband_reference,
                'aec_subband_true_echo': subband_echo,
                'aec_subband_diagonal_model': subband_diagonal,
                'aec_subband_missing_cross_terms': subband_echo - subband_diagonal,
            },
            'parameters': {
                'seed': SEED + 4, 'sample_rate_hz': SAMPLE_RATE, 'samples': 8000,
                'reference': 'seeded Gaussian std 0.12 plus 2100 Hz sine amplitude 0.08',
                'path': [0., 1.], 'analysis': 'nonoverlapping orthonormal two-sample Haar',
                'model': 'known current/previous block 2x2 crossband matrices; diagonal_model removes exact off-diagonal terms',
                'near_end': 'none', 'noise': 'none', 'algorithm_delay_samples': 0,
                'score_interval_samples': [0, 8000],
            },
            'limits': 'Known-path system representation only; diagonal model has NOT been adapted or fitted. '
                      'The difference is missing cross terms, not a measured AEC residual or subband ERLE.',
        },
        'wpe': {
            'signals': {'wpe_dry': dry, 'wpe_reverberant': reverberant, 'wpe_output': wpe},
            'parameters': {'echo_delays_samples': [0, 640, 1280], 'echo_gains': [1, .6, .3],
                           'tail_padding_samples': 4000, 'n_fft': 512, 'hop': 128,
                           'taps': 4, 'delay_frames': 3, 'iterations': 3, 'first_valid_frame': 6,
                           'relative_diagonal_loading': 1e-6, 'relative_power_floor': 1e-5,
                           'window': 'periodic Hann', 'center': True, 'synthesis_denominator_floor': 1e-12,
                           'causal': False, 'propagation_delay_samples': 0},
            'limits': 'Offline WPE may remove predictable harmonic content; this sparse FIR is not a room or speech-quality benchmark.'},
        'separation': {
            'signals': {'separation_source1': target, 'separation_source2': other,
                        'separation_mixture': mixture, 'separation_recovered1': recovered[0],
                        'separation_recovered2': recovered[1]},
            'parameters': {'mixing_matrix': matrix.tolist(), 'condition_number': float(np.linalg.cond(matrix)),
                           'algorithm_delay_samples': 0, 'permutation': [0, 1]},
            'limits': 'Known instantaneous mixing matrix; solving it is not blind separation, ICA or neural inference.'},
        'engineering': {
            'signals': {'engineering_reference': target, 'engineering_clipped': clipped,
                        'engineering_low_level': .02*target, 'engineering_dropout': damaged},
            'parameters': {'clip_operation': 'clip(3*x,-0.3,0.3)/3', 'low_level_gain': .02,
                           'dropout_interval_s': [.9, 1.], 'algorithm_delay_samples': 0},
            'limits': 'Clipping and dropout are deliberate faults, not enhancement algorithms.'},
        'tracking': {
            'signals': {'tracking_pan': np.vstack((np.cos(pan)*target, np.sin(pan)*target))},
            'parameters': {'left_gain': 'cos(phi)', 'right_gain': 'sin(phi)', 'phi_range_rad': [0, np.pi/2]},
            'limits': 'Equal-power stereo panning; not a physical moving-source array recording and not valid DOA ground truth.'},
        'correlation': {
            'signals': {'correlation_reference': target,
                        'correlation_single': target + independent_noise[0],
                        'correlation_independent': independent_mean,
                        'correlation_common': common_mean},
            'parameters': {'seed': SEED + 1, 'noise_std_each_channel': .07,
                           'model': 'Both target channels are already aligned; average weights [0.5, 0.5].',
                           'noise_correlations': {'independent': 0, 'common': 1},
                           'expected_noise_power_ratios': {'independent': .5, 'common': 1},
                           'reference': 'correlation_reference, no delay; score all samples',
                           'algorithm_delay_samples': 0},
            'limits': 'Population variances predict 3.0103 dB and 0 dB; one finite realization is not a statistical performance estimate.'},
        'polarity': {
            'signals': {'polarity_reference': target, 'polarity_array': polarity_array,
                        'polarity_uncorrected': polarity_array.mean(axis=0),
                        'polarity_corrected': (polarity_array[0] - polarity_array[1]) / 2},
            'parameters': {'channel_gains': [1, -.9], 'channel_order': ['normal', 'inverted_0.9'],
                           'uncorrected_target_gain': .05, 'corrected_target_gain': .95,
                           'relative_amplitude_db': float(20*np.log10(.05/.95)),
                           'reference': 'polarity_reference, no delay; no gain fitting when scoring',
                           'algorithm_delay_samples': 0},
            'limits': 'Known polarity correction, not blind calibration. Array WAV is raw stereo; compare mono averages to study cancellation.'},
        'conditioning': {
            'signals': {'conditioning_reference': sources,
                        'conditioning_well_input': well_input, 'conditioning_ill_input': ill_input,
                        'conditioning_well_output': np.linalg.solve(well, well_input),
                        'conditioning_ill_output': np.linalg.solve(ill, ill_input)},
            'parameters': {'seed': SEED + 1, 'rng_order': 'after correlation noise draw (2, 32000)',
                           'well_matrix': well.tolist(), 'ill_matrix': ill.tolist(),
                           'noise_std_each_channel': .003, 'same_noise_in_both_inputs': True,
                           'condition_numbers_2norm': [3., 199.],
                           'reference': 'conditioning_reference; compare each source channel without delay or gain fitting',
                           'input_channel_order': ['microphone1', 'microphone2'],
                           'output_and_reference_channel_order': ['source1', 'source2'],
                           'algorithm_delay_samples': 0},
            'limits': 'Both mixing matrices are known; this is inverse-problem noise amplification, not blind separation or a benchmark of AuxIVA.'},
        'nonlinear': {
            'signals': {'nonlinear_reference': nonlinear_reference,
                        'nonlinear_echo': nonlinear_echo,
                        'nonlinear_estimate': nonlinear_estimate,
                        'nonlinear_residual': nonlinear_echo - nonlinear_estimate},
            'parameters': {'sample_rate_hz': SAMPLE_RATE, 'samples': t.size,
                           'frequency_hz': 500, 'amplitude': .4, 'cubic_coefficient': 2.,
                           'model': 'd=x+2*x**3; best whole-segment scalar least-squares fit g*x',
                           'fit_gain': best_linear_gain, 'analysis_interval_samples': [0, t.size],
                           'reference': 'nonlinear_echo for echo-power ratio; nonlinear_reference is playback',
                           'taper': 'none; 1000 complete periods', 'algorithm_delay_samples': 0,
                           'noise': 'none', 'near_end': 'none', 'randomness': 'none'},
            'limits': 'Steady sinusoid, memoryless cubic distortion and batch scalar projection; '
                      'not NLMS convergence, a measured loudspeaker or speech-quality evaluation.'},
        'spectral_subtraction': {
            'signals': {'spectral_clean': subtraction_clean,
                        'spectral_noise': subtraction_noise,
                        'spectral_noisy': subtraction_noisy,
                        'spectral_floor04': subtraction_soft_audio,
                        'spectral_floor00': subtraction_zero_audio},
            'parameters': {'seed': SEED + 5, 'sample_rate_hz': SAMPLE_RATE,
                           'samples': t.size, 'noise_only_samples': noise_only_samples,
                           'noise_only_stft_frame_indices': noise_frames.tolist(),
                           'noise_model': 'independent white Gaussian, standard deviation 0.07',
                           'clean_model': '173 Hz six-harmonic synthetic tone, silent for first 0.4 s, 20 ms squared-sine onset',
                           'stft': {'n_fft': 512, 'hop_length': 128, 'window': 'periodic Hann',
                                    'center': True, 'weighted_overlap_add': True},
                           'noise_estimation': 'per-bin arithmetic mean of |Y|^2 over complete noise-only frames',
                           'mean_noise_power_sum': float(np.sum(subtraction_noise_power)),
                           'oversubtraction': 1., 'floor_ratios': [0., .04],
                           'spectral_floor_reference': 'fraction of observed noisy-bin power',
                           'noisy_phase_retained': True, 'algorithm_delay_samples': 'offline STFT demonstration; no online latency claim',
                           'reference': 'spectral_clean; no delay or gain fitting; first 0.4 s is noise-only'},
            'limits': 'Single seeded mathematical tone plus stationary Gaussian noise; fixed oracle-marked noise-only preamble. '
                      'The zero-floor output may make sparse tonal residuals audible; no human listening study, '
                      'natural speech, VAD, adaptive noise tracker, MMSE-STSA or MMSE-LSA.'},
    }


def prepare_exports(cases: dict | None = None) -> tuple[dict, dict]:
    """Apply common headroom per group; return encoded WAVs and metadata."""
    cases = build_cases() if cases is None else cases
    files, groups = {}, {}
    for name, case in cases.items():
        peak = max(float(np.max(np.abs(x))) for x in case['signals'].values())
        gain = min(1., .8 / peak) if peak > 0 else 1.
        groups[name] = {k: v for k, v in case.items() if k != 'signals'}
        groups[name]['common_export_gain'] = gain
        for stem, array in case['signals'].items():
            x = np.atleast_2d(array) * gain
            blob = pcm16_bytes(x)
            _, decoded = read_pcm16(blob)
            files[stem + '.wav'] = (blob, {
                'group': name, 'sample_rate_hz': SAMPLE_RATE, 'channels': x.shape[0],
                'samples': x.shape[1], 'duration_s': x.shape[1]/SAMPLE_RATE,
                'peak': float(np.max(np.abs(decoded))), 'rms': float(np.sqrt(np.mean(decoded**2))),
                'common_export_gain': gain, 'quantization_max_abs_error': float(np.max(np.abs(decoded-x))),
            })
    return files, groups
