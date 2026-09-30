"""E02-09--18: geometry, statistics, windows and convolution boundaries.

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

import hashlib
import io
import itertools
import json
from pathlib import Path
import wave
import numpy as np

from codes.chapters.ch00.core.audio_samples import read_pcm16, room_decay_case
from codes.chapters.ch03.core.covariance import spatial_covariance
from codes.chapters.ch02.core.spectral import istft, periodic_hann, stft
from codes.chapters.ch02.core.stft_convolution import finite_window_example

ROOT = Path(__file__).resolve().parents[3]


def published_room_decay_measurements(root: Path = ROOT) -> dict:
    """Read the committed room-decay reference and three WAVs; never repair.

    Validate generating-source and published-file hashes, actual PCM headers,
    one common gain and half-LSB quantization. RIR decay/DRR remain float filter
    properties; PCM output energies are not called RIR DRR or measured T60.
    """
    root = Path(root)
    manifest_path = root / "codes/chapters/ch00/audio/MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = [row for row in manifest["files"] if row["group"] == "room_decay"]
    stems = {"room_decay_dry", "room_decay_short_drr0", "room_decay_long_drr0", "room_decay_long_drr6"}
    if {Path(row["file"]).stem for row in records} != stems or len(records) != 4:
        raise ValueError("Room decay manifest must contain exactly the four expected WAVs")
    gain = manifest["groups"]["room_decay"]["common_export_gain"]
    if gain != 1.:
        raise ValueError("Room decay fixtures require the common export gain 1")
    inputs = manifest["generator_inputs"]
    required_sources = {"codes/chapters/ch00/core/audio_samples.py",
                        "codes/chapters/ch00/examples/generate_audio_samples.py",
                        "codes/chapters/ch02/core/spectral.py"}
    if not required_sources.issubset(inputs):
        raise ValueError("Room decay source provenance is incomplete")
    for path, digest in inputs.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Room decay generating source is stale: {path}")
    case = room_decay_case()
    decoded, assets = {}, {}
    required_format = {"sample_rate_hz": 16000, "channels": 1, "samples_per_channel": 40000,
                       "sample_width_bytes": 2, "compression": "NONE"}
    for row in records:
        if row["chapter"] != "ch07":
            raise ValueError("Room decay files must follow the fixed ch07 audio group mapping")
        path = f"codes/chapters/ch07/audio/{row['file']}"
        payload = (root/path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with wave.open(io.BytesIO(payload), "rb") as wav:
            metadata = {"sample_rate_hz": wav.getframerate(), "channels": wav.getnchannels(),
                        "samples_per_channel": wav.getnframes(), "sample_width_bytes": wav.getsampwidth(),
                        "compression": wav.getcomptype()}
        if digest != row["sha256"] or metadata != required_format:
            raise ValueError(f"Published room decay SHA/PCM format mismatch: {path}")
        if (row["sample_rate_hz"], row["channels"], row["samples"], row["common_export_gain"]) != (16000, 1, 40000, gain):
            raise ValueError(f"Room decay manifest format/gain mismatch: {path}")
        stem = Path(row["file"]).stem
        decoded[stem] = read_pcm16(payload)[1][0]
        error = float(np.max(np.abs(decoded[stem]-gain*case["signals"][stem])))
        if error > .5/32768+1e-15:
            raise ValueError(f"Published room decay quantization mismatch: {path}")
        assets[stem] = {"path": path, "sha256": digest, **metadata,
                        "quantization_max_abs_error": error}
    scores = {}
    for domain, signals in (("float", case["signals"]), ("pcm", decoded)):
        reference = signals["room_decay_dry"]
        direct_energy = float(np.sum(reference**2))
        rows = []
        for name in ("short_drr0", "long_drr0", "long_drr6"):
            output = signals["room_decay_"+name]
            residual = output-reference
            rows.append({"name": name, "direct_reference_energy": direct_energy,
                         "output_minus_reference_energy": float(np.sum(residual**2)),
                         "twice_reference_residual_cross_energy": float(2*np.sum(reference*residual)),
                         "total_output_energy": float(np.sum(output**2)),
                         "total_output_mean_square": float(np.mean(output**2))})
        scores[domain+"_measurements"] = {"cases": rows}
    return {"manifest_path": str(manifest_path.relative_to(root)),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": inputs, "assets": assets, "common_export_gain": gain,
            "energy_interval_samples": [0, 40000], "mean_square_denominator_samples": 40000,
            "energy_units": "sum of squared dimensionless digital samples",
            "reference": "actual dry WAV has the same known 192-sample direct arrival; no gain/delay fit",
            "pcm_residual": "difference of independently quantized output and reference, not a separately quantized reflection component",
            **scores}


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
            'audio': room_decay_case()['parameters'],
            'published_audio': published_room_decay_measurements()}


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
            'unsupported_index': 0, 'endpoint_projection': endpoint_projection(),
            'limits': 'The inverse FFT returns the windowed frame. Local nonzero support at n=2,3 does not prove reconstruction of every boundary sample; n=0 has zero support.'}


def endpoint_projection() -> dict:
    """E02-13: modified one-sided endpoint semantics, including odd n_fft."""
    source = np.arange(1., 7.)
    rows = []
    for n_fft in (4, 5):
        original = stft(source, n_fft=n_fft, hop_length=2)
        baseline = istft(original, n_fft=n_fft, hop_length=2, length=6)[0]
        bins = ((0, 3.), (n_fft//2, 5.))
        for bin_index, imaginary_increment in bins:
            modified = original.copy()
            modified[:, bin_index, :] += imaginary_increment*1j
            before = modified.copy()
            synthesized = istft(modified, n_fft=n_fft, hop_length=2, length=6)[0]
            rows.append({"n_fft": n_fft, "bin_index": bin_index,
                         "imaginary_increment": imaginary_increment,
                         "is_dc_or_even_nyquist": bin_index == 0 or n_fft % 2 == 0,
                         "output_difference": (synthesized-baseline).tolist(),
                         "max_abs_output_difference": float(np.max(np.abs(synthesized-baseline))),
                         "input_spectrum_unchanged": bool(np.array_equal(modified, before))})
    return {"source": source.tolist(), "hop_length": 2, "center": True, "cases": rows,
            "limits": "Real-input DFT endpoints are real. For arbitrary modified spectra, irfft ignores DC and even-Nyquist imaginary parts; an odd final bin remains a conjugate pair. WOLA synthesis does not guarantee STFT consistency of modified coefficients."}


def correlated_source_noise() -> dict:
    """E02-17: spatially white noise need not be uncorrelated with the source."""
    outcomes = np.array(list(itertools.product((-1., 1.), repeat=3)))
    s, u, v = outcomes.T
    noise = np.vstack((-.5*s+.5*u+v/np.sqrt(2), -.5*s+.5*u-v/np.sqrt(2)))
    target = np.vstack((s, s))
    mixture = target+noise
    count = outcomes.shape[0]
    rnn, rxx = noise@noise.T/count, mixture@mixture.T/count
    signal = target@target.T/count
    cross = target@noise.T/count+noise@target.T/count
    return {"outcomes_s_u_v": outcomes.tolist(), "equal_probability": 1/count,
            "source_as_row": s.tolist(), "noise_as_rows": noise.tolist(), "mixture_as_rows": mixture.tolist(),
            "source_mean": float(np.mean(s)), "noise_mean": np.mean(noise, axis=1).tolist(),
            "source_variance": float(np.mean(s*s)), "noise_second_moment": rnn.tolist(),
            "source_noise_cross_row": np.mean(s[None, :]*noise, axis=1).tolist(),
            "signal_second_moment": signal.tolist(), "mixture_second_moment": rxx.tolist(),
            "cross_contribution": cross.tolist(), "incorrect_uncorrelated_sum": (signal+rnn).tolist(),
            "mixture_eigenvalues": np.linalg.eigvalsh(rxx).tolist(),
            "limits": "Eight equally weighted population outcomes, not eight random estimates. Noise is spatially white in this second-moment model, but source-noise cross moments are nonzero. No temporal whiteness or Gaussian law is assumed; one physical source need not create an eigenvalue above noise variance when the usual uncorrelated-source/noise assumption fails."}


def finite_window_convolution() -> dict:
    """E02-16; the numerical implementation is uniquely archived in core."""
    return finite_window_example()


def window_calibration() -> dict:
    """E02-18: coherent amplitude versus noise-energy window normalization."""
    n_fft, bin_index = 8, 2
    window = periodic_hann(n_fft)
    coherent_gain = float(np.mean(window))
    power_gain = float(np.mean(window**2))
    tone = np.cos(2*np.pi*bin_index*np.arange(n_fft)/n_fft)
    spectrum = np.fft.rfft(window*tone)
    tone_power = float(np.mean((window*tone)**2))
    noise = np.array(list(itertools.product((-1., 1.), repeat=n_fft)))
    windowed_power = float(np.mean((noise*window)**2))
    return {"n_fft": n_fft, "periodic_hann": window.tolist(),
            "window_sum": float(np.sum(window)), "window_squared_sum": float(np.sum(window**2)),
            "coherent_gain": coherent_gain, "power_gain": power_gain,
            "enbw_bins": power_gain/coherent_gain**2,
            "tone": {"bin": bin_index, "samples": tone.tolist(), "amplitude": 1.,
                     "original_mean_square": float(np.mean(tone**2)),
                     "raw_bin_magnitude": float(abs(spectrum[bin_index])),
                     "naive_amplitude": float(2*abs(spectrum[bin_index])/n_fft),
                     "corrected_amplitude": float(2*abs(spectrum[bin_index])/np.sum(window)),
                     "windowed_mean_square": tone_power, "divided_by_U": tone_power/power_gain,
                     "divided_by_G_squared": tone_power/coherent_gain**2},
            "white_noise_ensemble": {"outcomes": len(noise), "equal_probability": 1/len(noise),
                                     "mean": np.mean(noise, axis=0).tolist(),
                                     "second_moment": (noise.T@noise/len(noise)).tolist(),
                                     "original_mean_square": float(np.mean(noise**2)),
                                     "windowed_mean_square": windowed_power,
                                     "divided_by_U": windowed_power/power_gain,
                                     "divided_by_G_squared": windowed_power/coherent_gain**2},
            "limits": "The tone is an exact interior FFT-bin example, not an arbitrary-tone calibration. All 256 sign vectors form an exact equal-weight white-noise ensemble, not a random trial or a speech recording. G corrects coherent tone amplitude; U normalizes noise PSD energy. Finite weighted-record power need not equal raw-record power after any universal correction."}


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
            'E02-15': short_fir_convolution(), 'E02-16': finite_window_convolution(),
            'E02-17': correlated_source_noise(), 'E02-18': window_calibration()}


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
