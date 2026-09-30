"""E04-22 known-component, known-angle unitary focusing fixture.

Two coherent sources, no noise or room. Tagged 1/3-kHz components are
reordered using a known permutation; this is not blind STFT separation.
Only the steady-window phasors establish the narrowband focusing identity.
"""
from __future__ import annotations

import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array
from codes.chapters.ch03.core.geometry import plane_wave_delays

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
SAMPLES = 32024
FREQUENCIES = (1000., 3000.)
AMPLITUDE = .08
SCORING_INTERVAL = (2400, 29600)
PERMUTATION = (0, 3, 2, 1)
FILE_NAMES = {"reference": "focus_reference.wav", "delayed_source": "focus_delayed_source.wav",
              "array": "focus_array.wav", "known_focused": "focus_known_focused.wav"}
LIMITS = ("Original mathematical coherent two-source free-field fixture, not speech, a room or a device recording. "
          "Source 2 is source 1 delayed four samples, not statistically independent. "
          "Known analytic frequency components are reordered using known source directions; no blind DOA, "
          "STFT focusing or source separation algorithm is run. Fades introduce sidebands: the permutation "
          "identity is claimed only for steady 1/3-kHz phasors, not the entire finite spectrum. "
          "Three-kHz spatial aliasing remains. Playback/downmixing is not evidence of covariance rank.")


def parameters() -> dict:
    positions = np.column_stack((.1715*np.arange(4), np.zeros(4)))
    return {"exercise_id": "E04-22", "sample_rate_hz": SAMPLE_RATE,
            "source_samples": SOURCE_SAMPLES, "samples_per_channel": SAMPLES,
            "sound_speed_m_s": 343., "positions_m": positions.tolist(),
            "source_angles_deg": [-30., 30.], "source_2_delay_samples": 4,
            "common_propagation_delay_samples": 12,
            "frequencies_hz": list(FREQUENCIES), "source_amplitude_per_frequency": AMPLITUDE,
            "fade_samples": 320, "fade": "squared sine, endpoints included, before all delays",
            "propagation_model": "x_m=D_(12+4m) F + D_(16-4m) F, m=0..3; full zero-extended tail",
            "known_focusing": "keep tagged 1-kHz contribution; permute tagged 3-kHz contribution [0,3,2,1]",
            "channel_order": [0, 1, 2, 3], "scoring_interval_samples": list(SCORING_INTERVAL),
            "samples_in_scoring_window": SCORING_INTERVAL[1]-SCORING_INTERVAL[0],
            "phasor_convention": "cos coefficient minus j*sin coefficient; absolute sample clock",
            "normalized_covariance": "(z/0.08)(z/0.08)^H; equal mean over two frequencies",
            "rank_relative_threshold": 1e-10, "randomness": "none", "noise": "none"}


def generate_signals() -> dict[str, np.ndarray]:
    """Mono source pair and two four-channel observations, all C x 32024."""
    p = parameters()
    delays = plane_wave_delays(np.asarray(p["positions_m"]), np.deg2rad([-30., 30.]))*SAMPLE_RATE
    expected = np.array([4*np.arange(4), -4*np.arange(4)])
    if not np.allclose(delays, expected, rtol=0, atol=1e-12):
        raise ValueError("geometry no longer agrees with the integer propagation fixture")
    n = np.arange(SOURCE_SAMPLES)
    envelope = np.ones(SOURCE_SAMPLES)
    fade = np.sin(np.linspace(0, np.pi/2, 320))**2
    envelope[:320], envelope[-320:] = fade, fade[::-1]
    components = np.array([AMPLITUDE*np.cos(2*np.pi*f*n/SAMPLE_RATE)*envelope for f in FREQUENCIES])
    def delayed(x, lag):
        return np.pad(x, (lag, SAMPLES-SOURCE_SAMPLES-lag))
    contributions = np.array([[delayed(tone, 12+4*m)+delayed(tone, 16-4*m)
                               for m in range(4)] for tone in components])
    return {"reference": delayed(components.sum(axis=0), 0)[None, :],
            "delayed_source": delayed(components.sum(axis=0), 4)[None, :],
            "array": contributions.sum(axis=0),
            "known_focused": contributions[0]+contributions[1, list(PERMUTATION)]}


def complex_pairs(x) -> list:
    z = np.asarray(x)
    return np.stack((z.real, z.imag), axis=-1).tolist()


def measure_signal(samples: np.ndarray) -> dict:
    """Four-column real LS fit; no FFT or DOA API supplies the scoring oracle.

    Phasor covariance here is a single-vector outer product at each frequency,
    not a temporal sample-covariance estimate. Two-frequency pooling is an
    explicitly equal average with denominator 2; amplitude normalization is
    fixed at the known source amplitude, not fitted per file or per frequency.
    """
    x = finite_real_array(samples, "samples")
    if x.shape not in ((1, SAMPLES), (4, SAMPLES)):
        raise ValueError("require one or four finite channels with 32024 samples")
    start, stop = SCORING_INTERVAL
    time = np.arange(start, stop)/SAMPLE_RATE
    design = np.column_stack([g(2*np.pi*f*time) for f in FREQUENCIES for g in (np.cos, np.sin)])
    coefficients = np.linalg.lstsq(design, x[:, start:stop].T, rcond=None)[0]
    phasors = np.array([coefficients[2*k]-1j*coefficients[2*k+1] for k in range(2)])
    normalized = phasors/AMPLITUDE
    covariance = np.array([np.outer(z, z.conj()) for z in normalized])
    pooled = np.mean(covariance, axis=0)
    def spectrum_record(matrix):
        values = np.linalg.eigvalsh(matrix)
        maximum = float(np.max(abs(values)))
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("positive representable phasor energy required")
        return {"covariance_real_imag": complex_pairs(matrix), "eigenvalues": values.tolist(),
                "rank_relative_threshold": 1e-10, "rank": int(np.sum(values > maximum*1e-10))}
    fit = design@coefficients
    return {"scoring_interval_samples": [start, stop], "samples_per_channel": stop-start,
            "channels": len(x), "real_ls_design_columns": 4,
            "phasor_real_imag": complex_pairs(phasors), "normalized_phasor_real_imag": complex_pairs(normalized),
            "normalizing_source_amplitude": AMPLITUDE,
            "per_frequency": [{"frequency_hz": f, "outer_product_denominator": 1,
                               **spectrum_record(c)} for f, c in zip(FREQUENCIES, covariance)],
            "pooled": {"frequency_average_denominator": 2, **spectrum_record(pooled)},
            "time_domain_squared_sum": float(np.sum(x[:, start:stop]**2)),
            "time_domain_sample_denominator": int(x.shape[0]*(stop-start)),
            "two_tone_fit_residual_squared_sum": float(np.sum((x[:, start:stop].T-fit)**2)),
            "raw_three_khz_minus_negative_j_one_khz_max_error": float(np.max(abs(phasors[1]+1j*phasors[0]))),
            "peak": float(np.max(abs(x)))}
