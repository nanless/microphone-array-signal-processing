"""Fixed noise estimates under a known variance step and target contamination.

The fixture is a mathematical two-tone/white-noise example, not speech or a
recorded environment. The known-variance control uses offline model knowledge;
it is neither noise tracking nor a perfect clean-signal oracle. All outputs
retain the same origin and common gain, with no fitted gain or time shift.
"""
from __future__ import annotations

import io
import struct
import wave
import numpy as np

from codes.chapters.ch02.core.spectral import stft, istft, periodic_hann
from codes.chapters.ch02.core.conventions import validate_waveforms
from codes.chapters.ch10.core.noise_suppression import power_spectral_subtraction

SAMPLE_RATE = 16000
SAMPLES = 32000
N_FFT, HOP = 512, 128
SEED = 20261001
STEMS = ('noise_reference', 'noise_component', 'noise_mixture', 'noise_fixed',
         'noise_polluted', 'noise_known_variance')
SCORE_WINDOWS = {'before_step': (9600, 16000), 'after_step': (22400, 28800)}


def build_fixture() -> dict:
    """Return pre-export waveforms, gain-conditioned components and diagnostics.

    ``components`` describes floating-point components conditioned on the gain
    actually selected from Y. It does not split quantization error into noise.
    The 512-point centered window needs future samples: this is offline WOLA.
    """
    time = np.arange(SAMPLES) / SAMPLE_RATE
    target = .12*np.cos(2*np.pi*500*time) + .08*np.cos(2*np.pi*1500*time)
    target[:6400] = 0
    target[6400:6720] *= np.sin(np.linspace(0, np.pi/2, 320))**2
    target[-320:] *= np.sin(np.linspace(np.pi/2, 0, 320))**2
    sigma = np.where(time < 1.2, .03, .12)
    noise = sigma*np.random.default_rng(SEED).standard_normal(SAMPLES)
    mixture = target+noise
    Y, S, V = [stft(x, n_fft=N_FFT, hop_length=HOP)[0] for x in (mixture, target, noise)]
    centers = np.arange(Y.shape[1])*HOP
    pure = np.flatnonzero((centers >= 256) & (centers+256 <= 6400))
    polluted = np.flatnonzero((centers-256 >= 8000) & (centers+256 <= 12800))
    fixed, D_fixed = power_spectral_subtraction(Y, pure)
    contaminated, D_polluted = power_spectral_subtraction(Y, polluted)
    window = periodic_hann(N_FFT)
    # For independent zero-mean noise, each unnormalized complex DFT bin has
    # E|V_ft|²=sum_n sigma[n]²*w[n]². Padded points have zero variance.
    padded_variance = np.pad(sigma**2, (256, 256))
    D_known = np.array([float(np.sum(padded_variance[c:c+N_FFT]*window**2)) for c in centers])
    known = np.empty_like(Y)
    for t, expected_power in enumerate(D_known):
        # One artificial noise coefficient encodes the known expected power.
        # Delegate the subtraction formula to its unique teaching kernel.
        proxy = np.full((Y.shape[0], 1), np.sqrt(expected_power), dtype=complex)
        coefficients = np.concatenate((proxy, Y[:, t:t+1]), axis=1)
        result, _ = power_spectral_subtraction(coefficients, np.array([0]))
        known[:, t] = result[:, 1]
    spectra = {'noise_fixed': fixed, 'noise_polluted': contaminated, 'noise_known_variance': known}
    signals = {'noise_reference': target[None], 'noise_component': noise[None], 'noise_mixture': mixture[None]}
    components, gains = {}, {}
    for name, Z in spectra.items():
        gain = np.zeros(Y.shape)
        gain[Y != 0] = (Z[Y != 0]/Y[Y != 0]).real
        gains[name] = gain
        signals[name] = istft(Z[None], n_fft=N_FFT, hop_length=HOP, length=SAMPLES)
        components[name] = {role: istft((gain*spectrum)[None], n_fft=N_FFT, hop_length=HOP,
                                      length=SAMPLES)
                            for role, spectrum in [('target', S), ('noise', V)]}
    parameters = {
        'sample_rate_hz': SAMPLE_RATE, 'samples_per_channel': SAMPLES, 'seed': SEED,
        'target_frequencies_hz': [500, 1500], 'target_amplitudes': [.12, .08],
        'noise_standard_deviations': [.03, .12], 'noise_step_sample': 19200,
        'noise_type': 'independent Gaussian mathematical samples with a deterministic variance step',
        'noise_only_sample_interval': [0, 6400], 'polluted_estimation_sample_interval': [8000, 12800],
        'target_onset_fade_sample_interval': [6400, 6720], 'target_end_fade_samples': 320,
        'fade': 'sin squared, including exact endpoint zero',
        'n_fft': N_FFT, 'hop_samples': HOP, 'window': 'DFT-periodic Hann', 'center': True,
        'pure_noise_frame_indices': pure.tolist(), 'polluted_frame_indices': polluted.tolist(),
        'oversubtraction': 1., 'floor_ratio': .04, 'minimum_positive_bin_amplitude_gain': .2,
        'known_variance_power': 'D_t=sum_n sigma[n]^2*w[n]^2; include all points of each window across the step; padded variance is zero',
        'known_variance_scope': 'offline expected noise power from model, not observed noise realization, online tracking or clean recovery',
        'score_windows': {name: list(value) for name, value in SCORE_WINDOWS.items()},
        'alignment': 'common sample origin; no gain/time matching; centered WOLA uses future samples',
        'common_export_gain': 1.,
    }
    return {'signals': signals, 'components': components, 'parameters': parameters,
            'spectral': {'frame_centers_samples': centers, 'fixed_noise_power': D_fixed,
                         'polluted_noise_power': D_polluted, 'known_variance_power': D_known,
                         'gains': gains, 'mixture': Y}}


def analyze_fixture(fixture: dict) -> dict:
    """Separate conditioned float distortion/noise/cross term from total error."""
    signals = fixture['signals']
    if set(signals) != set(STEMS):
        raise ValueError('fixture must contain exactly the six experiment signals')
    checked = {name: validate_waveforms(value) for name, value in signals.items()}
    if any(value.shape != (1, SAMPLES) for value in checked.values()):
        raise ValueError('all fixture signals must be mono with 32000 samples')
    if set(fixture['components']) != {'noise_fixed', 'noise_polluted', 'noise_known_variance'}:
        raise ValueError('fixture must contain the three gain-conditioned decompositions')
    scores = {}
    for window_name, (start, stop) in SCORE_WINDOWS.items():
        reference = checked['noise_reference'][0, start:stop]
        denominator = float(np.sum(reference**2))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('nonzero reference energy is required')
        rows = {}
        for name in ('noise_mixture', 'noise_fixed', 'noise_polluted', 'noise_known_variance'):
            error = checked[name][0, start:stop]-reference
            sse = float(np.sum(error**2))
            if not np.isfinite(sse):
                raise ValueError('float error energy is outside supported range')
            row = {'error_squared_sum': sse, 'mse': sse/(stop-start), 'nmse': sse/denominator}
            if name in fixture['components']:
                parts = fixture['components'][name]
                clean_part = validate_waveforms(parts['target'])
                noise_part = validate_waveforms(parts['noise'])
                if clean_part.shape != (1, SAMPLES) or noise_part.shape != (1, SAMPLES):
                    raise ValueError('components must be mono with 32000 samples')
                distortion = clean_part[0, start:stop]-reference
                residual = noise_part[0, start:stop]
                row.update(target_distortion_mse=float(np.mean(distortion**2)),
                           residual_noise_mse=float(np.mean(residual**2)),
                           twice_cross_term=float(2*np.mean(distortion*residual)))
                row['component_sum_mse'] = (row['target_distortion_mse'] + row['residual_noise_mse']
                                            + row['twice_cross_term'])
                if not all(np.isfinite(value) for value in row.values()):
                    raise ValueError('component energy is outside supported range')
                if not np.allclose(clean_part+noise_part, checked[name], rtol=1e-12, atol=1e-14):
                    raise ValueError('components do not reconstruct the reported output')
            rows[name+'.wav'] = row
        scores[window_name] = {'sample_interval': [start, stop], 'samples': stop-start,
                               'reference_squared_sum': denominator, 'reference_power': denominator/(stop-start),
                               'scores': rows}
    return {'score_windows': scores,
            'scope': 'gain-conditioned float decomposition; finite cross terms are not assumed zero; no PCM decomposition'}


def analyze_pcm(buffers: dict[str, bytes]) -> dict:
    """Read actual PCM bytes and retain exact integer SSE/reference sums."""
    if set(buffers) != {stem+'.wav' for stem in STEMS}:
        raise ValueError('PCM analysis requires the exact six WAV files')
    codes = {}
    for name, data in buffers.items():
        try:
            with wave.open(io.BytesIO(data), 'rb') as reader:
                if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                        reader.getnframes(), reader.getcomptype()) != (1, 2, SAMPLE_RATE, SAMPLES, 'NONE'):
                    raise ValueError('WAV must be 16 kHz mono PCM16 with 32000 frames')
                frames = reader.readframes(SAMPLES)
            if len(frames) != 2*SAMPLES:
                raise ValueError('truncated PCM frames')
            codes[name] = struct.unpack('<'+'h'*SAMPLES, frames)
        except (wave.Error, EOFError, struct.error) as error:
            raise ValueError(f'invalid PCM WAV: {name}') from error
    scores = {}
    for window_name, (start, stop) in SCORE_WINDOWS.items():
        reference = codes['noise_reference.wav'][start:stop]
        denominator = sum(value*value for value in reference)
        if denominator <= 0:
            raise ValueError('actual PCM reference energy must be positive')
        rows = {}
        for name in ('noise_mixture.wav', 'noise_fixed.wav', 'noise_polluted.wav', 'noise_known_variance.wav'):
            numerator = sum((value-ref)**2 for value, ref in zip(codes[name][start:stop], reference))
            rows[name] = {'error_squared_sum_pcm_integer': numerator,
                          'mse': numerator/((stop-start)*32768**2), 'nmse': numerator/denominator}
        scores[window_name] = {'sample_interval': [start, stop], 'samples': stop-start,
                               'reference_squared_sum_pcm_integer': denominator,
                               'pcm_amplitude_denominator': 32768, 'scores': rows}
    return {'score_windows': scores,
            'scope': 'actual PCM total error against actual PCM reference; no fitted gain, shift or noise decomposition'}
