"""E06-39: one colored-reference fixture, not a speech/room/device benchmark.

Training uses floating-point observations. Exported int16 is read back only
for output measurements; PCM is not fed into the adaptive filters. Every
prediction precedes the update, and the final 13 convolution-tail samples are
retained but excluded from training and holdout scores.
"""
from __future__ import annotations

import numpy as np

from codes.chapters.ch06.core.aec import NLMSState
from codes.chapters.ch06.core.aec_affine_projection import APAState

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
SAMPLES = 32013
TRAIN_STOP = 24000
SEED = 20261001
FILE_NAMES = {name: 'apa_'+name+'.wav' for name in (
    'reference', 'true_echo', 'microphone', 'nlms_residual',
    'apa2_residual', 'apa4_residual')}
LIMITS = ('Original one-seed mathematical colored-noise fixture, not speech, '
          'room or device recording. The exact stationary linear 16-tap path '
          'and observation-side white noise are supplied by the generator. '
          'No delay search, double-talk control, residual suppression, timing '
          'benchmark or listening-quality claim. Higher projection order is '
          'not assumed to improve noisy holdout error. Float component truth '
          'and actual PCM total power are different measurements. No per-file '
          'normalization, fitted delay or fitted gain is used.')


def parameters() -> dict:
    path = np.zeros(16)
    path[[0, 3, 7, 13]] = [.6, -.2, .1, .05]
    return {'exercise_id': 'E06-39', 'sample_rate_hz': SAMPLE_RATE,
            'source_samples': SOURCE_SAMPLES, 'samples_per_channel': SAMPLES,
            'convolution_tail_samples': 13, 'seed': SEED,
            'random_bit_generator': 'PCG64', 'ar_coefficient': .98,
            'ar_initial_previous_sample': 0., 'excitation_standard_deviation': .015,
            'true_path_current_first': path.tolist(), 'filter_length': 16,
            'step_size': .2, 'regularization': .001,
            'regularization_units': 'squared digital signal amplitude',
            'projection_orders': [1, 2, 4],
            'observation_noise_standard_deviation': .003,
            'random_draw_order': '32000 AR excitation draws, then 32013 independent observation-noise draws',
            'training_interval_samples': [0, TRAIN_STOP],
            'holdout_interval_samples': [TRAIN_STOP, SOURCE_SAMPLES],
            'holdout_sample_denominator': SOURCE_SAMPLES-TRAIN_STOP,
            'trace_interval_samples': 160, 'initial_weights': 'all zero',
            'initial_reference_and_projection_history': 'empty/zero past reference',
            'output': 'current prior prediction/error before any microphone-dependent update',
            'freeze': 'all samples at and after 24000; reference and projection histories still advance',
            'common_export_gain': 1., 'alignment': 'same causal sample clock, no fitted compensation',
            'export_condition': 'observation noise standard deviation .003',
            'noiseless_control': 'in-memory generated component, no extra WAV'}


def generate_components() -> dict:
    p = parameters()
    rng = np.random.Generator(np.random.PCG64(SEED))
    excitation = rng.standard_normal(SOURCE_SAMPLES)*.015
    reference = np.zeros(SAMPLES)
    previous = 0.
    for index, sample in enumerate(excitation):
        previous = .98*previous+sample
        reference[index] = previous
    echo = np.convolve(reference[:SOURCE_SAMPLES], p['true_path_current_first'])[:SAMPLES]
    noise = rng.standard_normal(SAMPLES)*.003
    return {'reference': reference, 'true_echo': echo, 'noise': noise,
            'microphone': echo+noise}


def _run(reference, observation, true_echo, path, order) -> dict:
    state = (NLMSState(16, step_size=.2, epsilon=.001) if order == 1
             else APAState(16, order, step_size=.2, regularization=.001))
    prior_echo = np.zeros(SAMPLES)
    prior_error = np.zeros(SAMPLES)
    trace, training_weights = [], None
    # TRAIN_STOP is divisible by 160, so no block straddles the freeze switch.
    for start in range(0, SAMPLES, 160):
        stop = min(start+160, SAMPLES)
        frozen = np.full(stop-start, start >= TRAIN_STOP, dtype=bool)
        if order == 1:
            error, prediction = state.process(reference[start:stop], observation[start:stop], freeze=frozen)
        else:
            result = state.process_block(reference[start:stop], observation[start:stop], freeze_mask=frozen)
            error, prediction = result['prior_error'], result['prior_echo']
        prior_echo[start:stop], prior_error[start:stop] = prediction, error
        weights = state.weights
        relative = float(np.linalg.norm(weights-path)/np.linalg.norm(path))
        trace.append({'state_after_samples': stop, 'relative_path_error': relative,
                      'weights': weights.tolist(), 'frozen': bool(start >= TRAIN_STOP)})
        if stop == TRAIN_STOP:
            training_weights = weights.copy()
    if training_weights is None or not np.array_equal(training_weights, state.weights):
        raise ValueError('the entire holdout and tail must preserve frozen weights')
    selection = slice(TRAIN_STOP, SOURCE_SAMPLES)
    clean_error = true_echo[selection]-prior_echo[selection]
    total_error = prior_error[selection]
    disturbance = observation[selection]-true_echo[selection]
    clean_mse = float(np.mean(clean_error**2))
    noise_mse = float(np.mean(disturbance**2))
    cross = float(2*np.mean(clean_error*disturbance))
    return {'prior_echo': prior_echo, 'prior_error': prior_error,
            'trace': trace, 'training_final_weights': training_weights.tolist(),
            'training_final_relative_path_error': float(np.linalg.norm(training_weights-path)/np.linalg.norm(path)),
            'holdout_float': {'sample_denominator': 8000,
                'clean_echo_prediction_error_mse': clean_mse,
                'observation_noise_mse': noise_mse,
                'twice_clean_error_noise_cross_mean': cross,
                'total_prior_residual_mse': float(np.mean(total_error**2)),
                'decomposition_sum': clean_mse+noise_mse+cross}}


def run_experiment() -> tuple[dict, dict]:
    p, truth = parameters(), generate_components()
    path = np.asarray(p['true_path_current_first'])
    conditions = {}
    for label, observation in [('noiseless', truth['true_echo']), ('noisy', truth['microphone'])]:
        conditions[label] = {str(k): _run(truth['reference'], observation, truth['true_echo'], path, k)
                             for k in (1, 2, 4)}
    signals = {name: truth[name][None, :] for name in ('reference', 'true_echo', 'microphone')}
    for name, k in [('nlms_residual', 1), ('apa2_residual', 2), ('apa4_residual', 4)]:
        signals[name] = conditions['noisy'][str(k)]['prior_error'][None, :]
    report = {'parameters': p, 'scope': LIMITS, 'conditions': {label: {
        k: {key: value for key, value in run.items() if key not in ('prior_echo', 'prior_error')}
        for k, run in methods.items()} for label, methods in conditions.items()}}
    if any(not np.isfinite(x).all() or np.max(abs(x)) >= 1 for x in signals.values()):
        raise ValueError('finite unclipped signals at shared gain 1 required')
    return signals, report


def measure_pcm(decoded: dict[str, np.ndarray]) -> dict:
    """Actual int16 decoded output power; no inferred PCM component truth."""
    if set(decoded) != set(FILE_NAMES):
        raise ValueError('six actual PCM files required')
    powers = {}
    for name, samples in decoded.items():
        x = np.asarray(samples)
        if x.dtype.kind not in 'iuf' or x.shape != (1, SAMPLES) or not np.isfinite(x).all():
            raise ValueError('one finite mono channel of 32013 samples required')
        integers = x*32768
        if (not np.array_equal(integers, np.rint(integers))
                or np.any(integers < -32768) or np.any(integers > 32767)):
            raise ValueError('measurements require actual decoded PCM16 values')
        window = integers[0, TRAIN_STOP:SOURCE_SAMPLES].astype(np.int64)
        squared_sum = sum(int(item)*int(item) for item in window)
        powers[name] = {'integer_squared_sum': squared_sum, 'sample_denominator': 8000,
                        'mean_square': squared_sum/(32768**2*8000)}
    input_sum = powers['microphone']['integer_squared_sum']
    if input_sum <= 0:
        raise ValueError('positive microphone energy required')
    ratios = {}
    for name in ('nlms_residual', 'apa2_residual', 'apa4_residual'):
        denominator = powers[name]['integer_squared_sum']
        ratios[name] = {'integer_numerator': input_sum, 'integer_denominator': denominator,
                       'microphone_to_residual_total_power_ratio_db':
                           float(10*np.log10(input_sum/denominator)) if denominator else None,
                       'zero_output': denominator == 0}
    return {'holdout_interval_samples': [TRAIN_STOP, SOURCE_SAMPLES],
            'pcm_decode_divisor': 32768, 'powers': powers, 'ratios': ratios,
            'interpretation': 'ratio includes generated observation noise; it is not clean-component ERLE, speech quality or training on PCM'}
