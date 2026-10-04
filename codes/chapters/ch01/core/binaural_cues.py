"""Controlled stereo ITD/ILD cues, without HRTFs or a physical angle model.

Channel 0 is left, channel 1 right. ITD is left minus right arrival;
ILD is right minus left level. All cases share one source and export gain.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
SOURCE_SAMPLES = 32000
OUTPUT_SAMPLES = 32008
WINDOW = (1600, 30400)  # half-open source support; integer tone periods
LAGS = tuple(range(-16, 17))
CASES = {
    "reference": ((0, 0), (1., 1.)),
    "itd_only": ((8, 0), (1., 1.)),
    "ild_only": ((0, 0), (.5, 1.)),
    "consistent": ((8, 0), (.5, 1.)),
    "conflicting": ((8, 0), (1., .5)),
}


def build_cues() -> dict[str, np.ndarray]:
    """Causal zero-filled shifts with every source sample and its tail retained."""
    time = np.arange(SOURCE_SAMPLES) / SAMPLE_RATE
    envelope = np.minimum(1., np.minimum(time / .02, (2. - time) / .02))
    source = sum(.08 * np.sin(2 * np.pi * frequency * time)
                 for frequency in (500, 800, 1300, 2400)) * envelope
    outputs = {}
    for name, (delays, gains) in CASES.items():
        stereo = np.zeros((2, OUTPUT_SAMPLES))
        for channel, (delay, gain) in enumerate(zip(delays, gains)):
            stereo[channel, delay:delay + SOURCE_SAMPLES] = gain * source
        outputs[name] = stereo
    return outputs


def measure_cues(stereo: np.ndarray, delays: tuple[int, int]) -> dict:
    """Separate known-delay power alignment from independent finite-lag search.

Power windows compare identical source support after applying the true delays.
Correlation holds the right receive window fixed; every candidate uses 28800
pairs. Positive lag compares left[n+lag] with right[n], hence left arrives later.
No circular wrap, window-size-dependent denominator or mean removal is used.
"""
    values = np.asarray(stereo)
    if (np.iscomplexobj(values) or values.dtype.kind not in 'biuf'
            or values.shape != (2, OUTPUT_SAMPLES)
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite real stereo cue with 32008 frames")
    # Widen before every square/product: PCM integers must not wrap in their
    # original dtype. Values remain in the caller's amplitude units.
    with np.errstate(over='ignore', invalid='ignore'):
        values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Stereo values must be representable as finite float64")
    if tuple(delays) not in {(0, 0), (8, 0)}:
        raise ValueError("This fixture supports only the documented (0,0) or (8,0) delays")
    start, stop = WINDOW
    left = values[0, start + delays[0]:stop + delays[0]]
    right = values[1, start + delays[1]:stop + delays[1]]
    powers = [float(np.mean(channel**2)) for channel in (left, right)]
    if min(powers) <= 0:
        raise ValueError("Both channels must have nonzero energy")
    correlations, denominators = [], []
    right_receive = values[1, start:stop]
    for lag in LAGS:
        left_receive = values[0, start + lag:stop + lag]
        # Scalar reductions avoid small-vector BLAS thread overhead.
        denominator = float(np.sqrt(np.sum(left_receive**2) * np.sum(right_receive**2)))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid correlation energy denominator")
        correlations.append(float(np.sum(left_receive * right_receive) / denominator))
        denominators.append(denominator)
    estimate = LAGS[int(np.argmax(correlations))]
    truth = delays[0] - delays[1]
    return {
        "power_source_window": [start, stop],
        "power_receive_windows": [[start + delay, stop + delay] for delay in delays],
        "power_window_samples": stop - start,
        "left_mean_square": powers[0], "right_mean_square": powers[1],
        "left_rms": float(np.sqrt(powers[0])), "right_rms": float(np.sqrt(powers[1])),
        "ild_right_minus_left_db": float(10 * np.log10(powers[1] / powers[0])),
        "truth_itd_samples": truth, "truth_itd_seconds": truth / SAMPLE_RATE,
        "estimated_itd_samples": estimate, "estimated_itd_seconds": estimate / SAMPLE_RATE,
        "itd_error_samples": estimate - truth,
        "correlation_right_receive_window": [start, stop],
        "correlation_window_samples": stop - start,
        "correlation_lags_samples": list(LAGS),
        "normalized_linear_correlations": correlations,
        "correlation_energy_denominators": denominators,
        "winning_correlation": correlations[int(np.argmax(correlations))],
    }
