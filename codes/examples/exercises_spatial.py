"""Executable exercises E01--E05; NumPy only, no files or audio are written.

Run from the repository root with ``.venv/bin/python
codes/examples/exercises_spatial.py``.  Outputs are mathematical/synthetic
examples, not speech-quality or device-performance measurements.  All powers
are dimensionless mean-square quantities relative to the stated input.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.array_tutorial.beamforming import lcmv_weights, mvdr_weights
from codes.array_tutorial.covariance import recursive_covariance, spatial_covariance
from codes.array_tutorial.doa import gcc_phat, mdl_source_count, music_spectrum
from codes.array_tutorial.geometry import direction_vector, near_field_steering, plane_wave_delays, plane_wave_steering
from codes.array_tutorial.spectral import istft, stft

SEED = 20260922


def correlated_noise() -> dict:
    """E01-01: exact covariance calculation for four aligned channels."""
    weight = np.ones(4) / 4
    rows = []
    for rho in (0.0, 0.5, 1.0):
        covariance = (1 - rho) * np.eye(4) + rho * np.ones((4, 4))
        power = float(weight @ covariance @ weight)
        rows.append({"rho": rho, "noise_power": power, "gain_db": float(-10 * np.log10(power))})
    return {"channels": 4, "input_noise_power_per_channel": 1.0, "cases": rows}


def amplitude_and_power() -> dict:
    """E01-02: same finite signal, common scale, no loudness normalization."""
    sample = np.array([0.0, 1.0, 0.0, -1.0])
    quieter = 0.5 * sample
    ratio = float(np.mean(quieter**2) / np.mean(sample**2))
    return {"amplitude_ratio": 0.5, "power_ratio": ratio, "level_change_db": float(10 * np.log10(ratio))}


def stft_framing() -> dict:
    """E02-01: distinguish boundary extension from within-frame FFT padding."""
    signal = np.ones(8000)
    uncentered = stft(signal, n_fft=256, hop_length=64, center=False)
    centered = stft(signal, n_fft=256, hop_length=64, center=True)
    return {
        "sample_rate_hz": 8000, "samples": 8000, "window_samples": 256, "hop_samples": 64,
        "uncentered_frames": uncentered.shape[2], "centered_frames": centered.shape[2],
        "rfft_bins_256": np.fft.rfft(signal[:256], n=256).size,
        "rfft_bins_512": np.fft.rfft(signal[:256], n=512).size,
        "grid_spacing_hz": [8000 / 256, 8000 / 512],
    }


def complex_covariance() -> dict:
    """E02-02: channel 1 leads channel 2 by 30 degrees; two equal-power snapshots."""
    a = np.sqrt(2.0) * np.array([np.exp(1j * np.pi / 6), 1.0])
    spectra = (a[:, None] * np.array([1.0, -1.0])[None, :])[:, None, :]
    matrix = spatial_covariance(spectra)[0]
    exchanged = spatial_covariance(spectra[::-1])[0]
    return {"matrix_real": matrix.real.tolist(), "matrix_imag": matrix.imag.tolist(),
            "eigenvalues": np.linalg.eigvalsh(matrix).tolist(),
            "swapped_offdiagonal_phase_deg": float(np.rad2deg(np.angle(exchanged[0, 1])))}


def stft_roundtrip() -> dict:
    """E02-03: 1 s random input, periodic Hann, explicit synthesis support."""
    rng = np.random.default_rng(SEED)
    signal = rng.standard_normal(8000)
    spectrum = stft(signal, n_fft=256, hop_length=64, center=True)
    restored = istft(spectrum, n_fft=256, hop_length=64, center=True, length=signal.size)[0]
    try:
        istft(stft(signal, n_fft=256, hop_length=64, center=False),
              n_fft=256, hop_length=64, center=False, length=signal.size)
    except ValueError:
        rejected = True
    else:
        rejected = False
    return {"seed": SEED, "max_abs_error": float(np.max(np.abs(restored - signal))),
            "uncentered_hann_rejected": rejected}


def difference_coarray() -> dict:
    """E03-01: integer positions in half-wavelength units, ordered pairs include self."""
    result = {}
    for name, positions in {"ula": [0, 1, 2, 3, 4, 5], "nested": [0, 1, 2, 3, 7, 11]}.items():
        positions = np.asarray(positions)
        lags, counts = np.unique(positions[:, None] - positions[None, :], return_counts=True)
        result[name] = {"lags": lags.tolist(), "multiplicities": counts.tolist(), "ordered_pair_count": int(counts.sum())}
    return result


def endfire_error() -> dict:
    """E03-02: exact inversion; out-of-domain values are not silently clipped."""
    rows = []
    for angle in (0.0, 85.0):
        true_tau = 0.04 * np.sin(np.deg2rad(angle)) / 343
        for error_us in (-10.0, 10.0):
            sine = (true_tau + error_us * 1e-6) * 343 / 0.04
            valid = bool(abs(sine) <= 1.0)
            estimate = float(np.rad2deg(np.arcsin(sine))) if valid else None
            rows.append({"true_angle_deg": angle, "delay_error_us": error_us,
                         "valid": valid, "estimated_angle_deg": estimate})
    return {"spacing_m": 0.04, "sound_speed_m_s": 343.0, "cases": rows}


def gcc_sign_and_silence() -> dict:
    """E04-01: zero-filled three-sample delay, a pulse and a seeded broadband signal."""
    rng = np.random.default_rng(SEED)
    delays = []
    for source in (np.eye(1, 64, 16)[0], rng.standard_normal(64)):
        earlier = np.pad(source, (0, 64))
        later = np.pad(source, (3, 61))
        positive = gcc_phat(later, earlier, 16000, max_tau=4 / 16000)[0]
        negative = gcc_phat(earlier, later, 16000, max_tau=4 / 16000)[0]
        delays.append([float(positive * 16000), float(negative * 16000)])
    try:
        gcc_phat(np.zeros(128), np.zeros(128), 16000)
    except ValueError:
        rejected = True
    else:
        rejected = False
    return {"seed": SEED, "delay_pairs_samples": delays, "silence_rejected": rejected,
            "minimum_spacing_for_3_samples_m": 343 * 3 / 16000}


def coherent_source_rank() -> dict:
    """E04-02: theoretical covariances, not finite-record source-count estimates."""
    vectors = np.exp(1j * np.pi * np.arange(4)[:, None] * np.array([-0.5, 0.5])[None, :])
    independent = vectors @ vectors.conj().T
    coherent = vectors @ np.ones((2, 2)) @ vectors.conj().T
    return {"independent_eigenvalues": np.linalg.eigvalsh(independent).tolist(),
            "coherent_eigenvalues": np.linalg.eigvalsh(coherent).tolist(),
            "physical_source_count": 2}


def spatial_alias() -> dict:
    """E04-03: spacing equals one wavelength, opposite half-sine directions alias."""
    positions = np.column_stack((np.arange(4) * 0.343, np.zeros(4)))
    vectors = plane_wave_steering(positions, [1000.0], np.deg2rad([-30.0, 30.0]))[:, 0, :]
    covariance = np.outer(vectors[0], vectors[0].conj()) + 0.1 * np.eye(4)
    scores = music_spectrum(covariance, vectors, source_count=1)
    return {"steering_max_abs_difference": float(np.max(np.abs(vectors[0] - vectors[1]))),
            "music_scores": scores.tolist(), "denominator_floor": 1e-15}


def mwf_rank_condition() -> dict:
    """E05-01: exact linear MWF versus the rank-one trace identity."""
    rows = []
    for diagonal in ([2.0, 1.0], [2.0, 0.0]):
        target = np.diag(diagonal)
        reference = np.array([1.0, 0.0])
        general = np.linalg.solve(target + np.eye(2), target @ reference)
        rank_one = target @ reference / (1 + np.trace(target))
        rows.append({"target_diagonal": diagonal, "general_weights": general.tolist(),
                     "trace_shortcut_weights": rank_one.tolist()})
    return {"noise_covariance": "identity", "noise_weight": 1.0, "cases": rows}


def mvdr_loading_tradeoff() -> dict:
    """E05-02: target response, WNG and a fixed interferer under relative loading."""
    target = np.ones(2, dtype=complex)
    interferer = np.array([1.0, 1.0j])
    noise = 10 * np.outer(interferer, interferer.conj()) + np.eye(2)
    rows = []
    for loading in (0.0, 1.0):
        weight = mvdr_weights(noise, target, relative_diagonal_loading=loading)
        rows.append({"relative_loading": loading, "weight_real": weight.real.tolist(),
                     "weight_imag": weight.imag.tolist(),
                     "target_response_error": float(abs(np.vdot(weight, target) - 1)),
                     "interference_amplitude": float(abs(np.vdot(weight, interferer))),
                     "wng_linear": float(1 / np.vdot(weight, weight).real)})
    return {"cases": rows, "dsb_wng_linear": 2.0}


def unequal_noise_average() -> dict:
    """E01-03: aligned unit target and independent noise variances 1 and 4."""
    covariance = np.diag([1., 4.])
    equal = np.ones(2) / 2
    inverse_variance = mvdr_weights(covariance, np.ones(2))
    equal_power = float(equal @ covariance @ equal)
    optimal_power = float(np.vdot(inverse_variance, covariance @ inverse_variance).real)
    return {"equal_noise_power": equal_power, "weighted_noise_power": optimal_power,
            "weights": inverse_variance.real.tolist(),
            "gain_over_equal_db": float(10 * np.log10(equal_power / optimal_power))}


def masked_snapshot_rank() -> dict:
    """E02-04: uncentered moments; mask support is not a rank guarantee."""
    spectra = np.array([[1., 0., 1.], [0., 1., 1.]])[:, None, :]
    rows = []
    for weights in (np.array([1., 0., 1.]), np.array([1., 0., 0.])):
        covariance = spatial_covariance(spectra, weights=weights)[0]
        doubled = spatial_covariance(spectra, weights=2 * weights)[0]
        rows.append({"weights": weights.tolist(), "matrix": covariance.real.tolist(),
                     "weight_concentration_count": float(weights.sum()**2 / (weights @ weights)),
                     "rank": int(np.linalg.matrix_rank(covariance)),
                     "rescale_error": float(np.max(np.abs(covariance - doubled)))})
    return {"snapshots": spectra[:, 0].tolist(), "cases": rows}


def recursive_startup() -> dict:
    """E02-05: fixed snapshot isolates zero-initialization weight mass."""
    snapshot = np.array([[1.], [2.]])
    matrix = np.zeros((1, 2, 2), dtype=complex)
    rows = []
    for step in range(1, 4):
        matrix = recursive_covariance(matrix, snapshot, forgetting_factor=.5)
        mass = 1 - .5**step
        rows.append({"step": step, "weight_mass": mass,
                     "raw_matrix": matrix[0].real.tolist(),
                     "normalized_matrix": (matrix[0] / mass).real.tolist()})
    return {"forgetting_factor": .5, "cases": rows}


def azimuth_roundtrip() -> dict:
    """E03-03: +y broadside, positive azimuth toward +x, upper hemisphere."""
    unit = np.array([-.5, -np.sqrt(.5), .5])
    azimuth = np.arctan2(unit[0], unit[1])
    elevation = np.arcsin(unit[2])
    recovered = direction_vector(azimuth, elevation)
    return {"direction": unit.tolist(), "azimuth_deg": float(np.rad2deg(azimuth)),
            "elevation_deg": float(np.rad2deg(elevation)),
            "positive_x_convention_deg": float(np.rad2deg(np.arctan2(unit[1], unit[0]))),
            "recovered_direction": recovered.tolist()}


def near_to_far_error() -> dict:
    """E03-04: reference is centre microphone; compare distance and gain errors."""
    positions = np.array([[-.05, 0.], [0., 0.], [.05, 0.]])
    angle = np.pi / 6
    far_delay = plane_wave_delays(positions, angle, reference=1)
    rows = []
    for radius in (.2, .5, 2., 20.):
        source = radius * direction_vector(angle)
        distances = np.linalg.norm(source - positions, axis=1)
        delays = (distances - distances[1]) / 343
        gain = np.abs(near_field_steering(positions, source, [4000.], reference=1)[0])
        rows.append({"distance_m": radius, "relative_distance_mm": ((distances - distances[1])*1000).tolist(),
                     "delay_rms_error_us": float(np.sqrt(np.mean((delays-far_delay)**2))*1e6),
                     "phase_rms_error_deg_at_4khz": float(np.sqrt(np.mean((delays-far_delay)**2))*4000*360),
                     "relative_amplitudes": gain.tolist()})
    return {"reference_channel_zero_based": 1, "azimuth_deg": 30., "cases": rows}


def coherent_spatial_smoothing() -> dict:
    """E04-04: local two-subarray derivation, not a general smoothing API."""
    vectors = np.exp(1j*np.pi*np.arange(4)[:, None]*np.array([-.5, .5])[None, :])
    coherent = vectors @ np.ones((2, 2)) @ vectors.conj().T
    smoothed = (coherent[:3, :3] + coherent[1:, 1:]) / 2
    return {"original_eigenvalues": np.linalg.eigvalsh(coherent).tolist(),
            "smoothed_matrix_real": smoothed.real.tolist(),
            "smoothed_eigenvalues": np.linalg.eigvalsh(smoothed).tolist(),
            "subarray_count": 2, "subarray_channels": 3,
            "original_aperture_in_spacings": 3, "smoothed_aperture_in_spacings": 2}


def mdl_candidate_scores() -> dict:
    """E04-05: prescribed positive spectrum, not a measured source-count result."""
    eigenvalues = np.array([9., 4., 1.1, .9])
    selected, scores = mdl_source_count(eigenvalues, 100)
    rows = []
    for k in range(eigenvalues.size):
        tail = eigenvalues[k:]
        arithmetic = float(np.mean(tail))
        geometric = float(np.exp(np.mean(np.log(tail))))
        penalty = .5 * k * (2 * eigenvalues.size - k) * np.log(100.)
        rows.append({"candidate": k, "arithmetic_mean": arithmetic,
                     "geometric_mean": geometric, "fit_term": float(scores[k]-penalty),
                     "penalty": float(penalty), "mdl": float(scores[k])})
    equal_count, equal_scores = mdl_source_count(np.ones(4), 100)
    short_count, short_scores = mdl_source_count(eigenvalues, 4)
    scaled_count, scaled_scores = mdl_source_count(eigenvalues * 1e-200, 100)
    rank_deficient_rejected = False
    try:
        mdl_source_count([9., 4., 1., 0.], 100)
    except ValueError:
        rank_deficient_rejected = True
    return {"eigenvalues_descending": eigenvalues.tolist(), "snapshots": 100,
            "cases": rows, "selected_count": selected,
            "equal_spectrum_count": equal_count, "equal_spectrum_scores": equal_scores.tolist(),
            "same_spectrum_n4_count": short_count, "same_spectrum_n4_scores": short_scores.tolist(),
            "scaled_count": scaled_count, "scaled_scores": scaled_scores.tolist(),
            "rank_deficient_rejected": rank_deficient_rejected}


def beam_output_noise() -> dict:
    """E05-03: separate input noise estimate from DSB residual-noise PSD."""
    weight = np.ones(3) / 3
    noise_covariance = .7 * np.eye(3) + .3 * np.ones((3, 3))
    noise_output = float(weight @ noise_covariance @ weight)
    total_output = 3 + noise_output
    input_noise_estimate = 4 - 3.3
    residual_estimate = input_noise_estimate * float(weight @ weight)
    return {"per_channel_noise_true": 1., "per_channel_noise_estimate": input_noise_estimate,
            "true_output_noise": noise_output, "total_output_power": total_output,
            "independent_model_output_noise_estimate": residual_estimate,
            "oracle_gain": 1-noise_output/total_output,
            "independent_model_gain": 1-residual_estimate/total_output}


def complex_constraint_response() -> dict:
    """E05-04: C.H w=f implies the actual response vector is conj(f)."""
    constraints = np.eye(2, dtype=complex)
    desired = np.array([1., 1.j])
    wrong = lcmv_weights(np.eye(2), constraints, desired)
    correct = lcmv_weights(np.eye(2), constraints, desired.conj())
    wrong_response = wrong.conj() @ constraints
    correct_response = correct.conj() @ constraints
    return {"desired_response_real": desired.real.tolist(), "desired_response_imag": desired.imag.tolist(),
            "unconjugated_f_response_imag": wrong_response.imag.tolist(),
            "correct_weight_real": correct.real.tolist(), "correct_weight_imag": correct.imag.tolist(),
            "correct_response_real": correct_response.real.tolist(), "correct_response_imag": correct_response.imag.tolist()}


def run_exercises() -> dict:
    """Return twenty-one JSON-serializable results; no downloads, training or playback."""
    functions = {
        "E01-01": correlated_noise, "E01-02": amplitude_and_power,
        "E02-01": stft_framing, "E02-02": complex_covariance, "E02-03": stft_roundtrip,
        "E03-01": difference_coarray, "E03-02": endfire_error,
        "E04-01": gcc_sign_and_silence, "E04-02": coherent_source_rank, "E04-03": spatial_alias,
        "E05-01": mwf_rank_condition, "E05-02": mvdr_loading_tradeoff,
        "E01-03": unequal_noise_average, "E02-04": masked_snapshot_rank,
        "E02-05": recursive_startup, "E03-03": azimuth_roundtrip,
        "E03-04": near_to_far_error, "E04-04": coherent_spatial_smoothing,
        "E05-03": beam_output_noise, "E05-04": complex_constraint_response,
        "E04-05": mdl_candidate_scores,
    }
    return {identifier: function() for identifier, function in functions.items()}


if __name__ == "__main__":
    print(json.dumps(run_exercises(), indent=2, ensure_ascii=False, allow_nan=False))
