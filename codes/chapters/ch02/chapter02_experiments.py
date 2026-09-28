"""E02-09--15: geometry, integration, decay, information and sample algebra.

Run ``.venv/bin/python -m codes.chapters.ch02.chapter02_experiments``. Results are
mathematical examples, not room/device or listening measurements. The module
uses only NumPy and the repository baseline. Imports do not run experiments.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json
import numpy as np

from codes.chapters.ch00.core.audio_samples import room_decay_case
from codes.chapters.ch03.core.covariance import spatial_covariance
from codes.chapters.ch02.core.spectral import periodic_hann


def near_far_errors() -> dict:
    """E02-09: center-referenced free-field amplitude and phase separately."""
    positions = np.array([-.1, 0., .1])
    frequency, speed = 1000., 343.
    rows = []
    for distance in (.2, 2.):
        for angle in (0., 90.):
            theta = np.deg2rad(angle)
            source = distance * np.array([np.sin(theta), np.cos(theta)])
            ranges = np.hypot(source[0] - positions, source[1])
            exact = ranges - distance
            planar = -positions * np.sin(theta)
            residual = exact - planar
            rows.append({'distance_m': distance, 'azimuth_deg': angle,
                         'source_xy_m': source.tolist(), 'ranges_m': ranges.tolist(),
                         'spherical_relative_path_m': exact.tolist(),
                         'planar_relative_path_m': planar.tolist(),
                         'path_error_m': residual.tolist(),
                         'phase_error_deg': (-360 * frequency * residual / speed).tolist(),
                         'spherical_amplitudes': (distance / ranges).tolist(),
                         'planar_amplitudes': [1., 1., 1.]})
    return {'microphone_x_m': positions.tolist(), 'reference_channel_zero_based': 1,
            'frequency_hz': frequency, 'sound_speed_m_s': speed, 'cases': rows,
            'phase_error_definition': 'arg(a_spherical/a_planar), unwrapped: -360*f*(exact_relative_path-planar_relative_path)/c',
            'limits': 'Free-field monopole without scattering; known source positions, not estimated ranges. Zero phase error alone does not imply equal amplitudes.'}


def diffuse_integration() -> dict:
    """E02-10: uniform direction cosine in 3D differs from a horizontal ring.

    The line array lies on x and points toward +y. u is the x direction cosine.
    Integrate a continuous angular model by trapezoidal quadrature; these grid
    points are not statistical trials or independent snapshots.
    """
    count, grid_count = 4, 20001
    u = np.linspace(-1., 1., grid_count)
    phi = np.linspace(0., 2 * np.pi, grid_count)

    def power(direction_cosine):
        steering = np.exp(1j * np.pi * np.arange(count)[:, None] * direction_cosine)
        return np.abs(np.mean(steering, axis=0))**2

    sphere = float(np.trapezoid(power(u), u) / 2)
    ring = float(np.trapezoid(power(np.sin(phi)), phi) / (2*np.pi))
    return {'microphones': count, 'spacing_wavelengths': .5, 'look_azimuth_deg': 0.,
            'quadrature_points_per_integral': grid_count,
            'sphere_mean_power': sphere, 'sphere_di_db': float(-10*np.log10(sphere)),
            'ring_mean_power': ring, 'ring_directivity_db': float(-10*np.log10(ring)),
            'wng_linear': float(count), 'wng_db': float(10*np.log10(count)),
            'limits': 'The horizontal-ring result is valid for that different noise model; it is not the 3D DI defined in this chapter. Both use unit target response.'}


def normalized_edc(impulse_response: np.ndarray) -> np.ndarray:
    """Normalized remaining ENERGY, including the sample at each index.

    Real finite nonzero one-dimensional filters only. Scale before squaring
    to keep invariance for extremely small or large but finite amplitudes.
    Exact zero remaining energy remains zero, not an arbitrary logarithm floor.
    """
    raw = np.asarray(impulse_response)
    if np.iscomplexobj(raw):
        raise ValueError('impulse_response must be real')
    h = np.asarray(raw, dtype=float)
    if h.ndim != 1 or h.size == 0 or not np.all(np.isfinite(h)):
        raise ValueError('impulse_response must be a finite nonempty vector')
    peak = np.max(np.abs(h))
    if peak == 0.:
        raise ValueError('zero energy has no normalized decay curve')
    remaining = np.cumsum((h[::-1] / peak)**2)[::-1]
    return remaining / remaining[0]


def rir_decay() -> dict:
    """E02-11: a hand-computable filter followed by separate listening assets."""
    h = np.array([1., 0., .5, 0., .25])
    delayed = np.pad(h, (2, 0))
    return {'hand_example': {
                'rir': h.tolist(), 'squared_samples': (h*h).tolist(),
                'remaining_energy': np.cumsum(h[::-1]**2)[::-1].tolist(),
                'normalized_edc': normalized_edc(h).tolist(),
                'direct_energy': 1., 'reflection_energy': 5/16,
                'rir_drr_db': float(10*np.log10(16/5)),
                'delayed_rir': delayed.tolist(), 'delay_samples': 2,
                'delayed_normalized_edc': normalized_edc(delayed).tolist(),
                'limits': 'Three sparse taps do not define an exponential reverberation time. A silent prefix shifts recording indices, not elapsed decay from direct arrival.'},
            'audio': room_decay_case()['parameters']}


def crb_parameter_changes() -> dict:
    """E02-12: formula (2-14), deterministic unknown complex amplitudes."""
    microphones, spacing, wavelength = 6, .014, .343
    rows = []
    for snapshots, snr_db, angle in ((100, 10, 0), (100, 20, 0),
                                      (1000, 10, 0), (100, 10, 60)):
        linear_snr = 10 ** (snr_db / 10)
        factor = (2*np.pi*spacing/wavelength)**2
        variance = 6 / (snapshots * linear_snr * microphones * (microphones**2-1)
                        * factor * np.cos(np.deg2rad(angle))**2)
        rows.append({'independent_snapshots': snapshots, 'snr_db': snr_db,
                     'linear_snr': linear_snr, 'azimuth_deg': angle,
                     'variance_bound_rad2': float(variance),
                     'standard_deviation_bound_deg': float(np.rad2deg(np.sqrt(variance)))})
    return {'microphones': microphones, 'spacing_m': spacing,
            'physical_aperture_m': (microphones-1)*spacing,
            'frequency_hz': 1000, 'sound_speed_m_s': 343,
            'wavelength_m': wavelength, 'cases': rows,
            'limits': 'Single narrowband source, unbiased DOA estimator, independent snapshots, proper complex spatially white Gaussian noise; not an achieved localization error or a promise at low SNR.'}


def four_sample_stft() -> dict:
    """E02-13: separate a single analysis frame from local WOLA support."""
    signal = np.array([1., 2., 3., 4.])
    window = periodic_hann(4)
    windowed = signal * window
    spectrum = np.fft.rfft(windowed)
    # Two windows, starting at n=0 and n=2; synthesis weights equal analysis.
    denominator = np.zeros(6)
    denominator[:4] += window**2
    denominator[2:] += window**2
    extended = np.arange(1., 7.)
    numerator = np.zeros(6)
    for start in (0, 2):
        analysis = np.fft.rfft(extended[start:start+4] * window)
        numerator[start:start+4] += np.fft.irfft(analysis, n=4) * window
    return {'signal': signal.tolist(), 'periodic_hann': window.tolist(),
            'windowed_frame': windowed.tolist(),
            'rfft_real': spectrum.real.tolist(), 'rfft_imag': spectrum.imag.tolist(),
            'inverse_rfft': np.fft.irfft(spectrum, n=4).tolist(),
            'window_start_samples': [0, 2], 'window_squared_sum': denominator.tolist(),
            'two_frame_signal': extended.tolist(), 'synthesis_numerator': numerator.tolist(),
            'local_supported_indices': [2, 3],
            'local_restored_samples': (numerator[[2, 3]] / denominator[[2, 3]]).tolist(),
            'unsupported_index': 0,
            'limits': 'The inverse FFT returns the windowed frame. Local nonzero support at n=2,3 does not prove reconstruction of every boundary sample; n=0 has zero support.'}


def sample_centering() -> dict:
    """E02-14: population-mean zero is not a zero finite sample mean."""
    x = np.array([[1., 1.], [0., 2.]])  # channels x frames
    mean = x.mean(axis=1)
    second = spatial_covariance(x[:, None, :])[0]
    centered = spatial_covariance(x[:, None, :], demean=True)[0]
    count = x.shape[1]
    return {'snapshots_as_columns': x.tolist(), 'sample_mean': mean.tolist(),
            'uncentered_second_moment': second.real.tolist(),
            'centered_divided_by_L': centered.real.tolist(),
            'centered_divided_by_L_minus_1': (centered * count/(count-1)).real.tolist(),
            'mean_outer_product': np.outer(mean, mean).tolist(),
            'identity_residual_max_abs': float(np.max(np.abs(second-centered-np.outer(mean, mean)))),
            'limits': 'L-1 is the unbiased covariance divisor under IID sampling with estimated mean, not a universal correction for correlated or weighted STFT snapshots.'}


def short_fir_convolution() -> dict:
    """E02-15: explicitly retain full tails and the direct/reflection cross term."""
    source = np.array([1., 2., 0., -1.])
    h = np.array([1., 0., .5])
    output = np.convolve(source, h, mode='full')
    direct = np.pad(source, (0, 2))
    reflected = .5 * np.pad(source, (2, 0))
    return {'source': source.tolist(), 'rir': h.tolist(),
            'full_output': output.tolist(), 'direct_output': direct.tolist(),
            'reflection_output': reflected.tolist(),
            'direct_energy': float(direct @ direct),
            'reflection_energy': float(reflected @ reflected),
            'twice_cross_energy': float(2 * (direct @ reflected)),
            'total_energy': float(output @ output),
            'cropped_first_four_energy': float(output[:4] @ output[:4]),
            'rir_direct_energy': 1., 'rir_reflection_energy': .25,
            'limits': 'A delayed half-amplitude reflection has quarter energy before source interactions. The two convolved components overlap and need their cross term; this sparse filter has no T60 model.'}


def run_exercises() -> dict:
    return {'E02-09': near_far_errors(), 'E02-10': diffuse_integration(),
            'E02-11': rir_decay(), 'E02-12': crb_parameter_changes(),
            'E02-13': four_sample_stft(), 'E02-14': sample_centering(),
            'E02-15': short_fir_convolution()}


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
