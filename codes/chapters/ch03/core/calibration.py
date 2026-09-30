"""A small, anchored, single-frequency ULA gain/DOA teaching estimator.

This is an original least-squares example, not an implementation of an author's
self-calibration system.  The relative phase of channel 1 must have been measured
independently.  With arbitrary complex gains and no such anchor, a common slope
in all direction sines can be exchanged for a slope in the channel phases.
"""

from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


def ula_prediction(
    angles_deg: np.ndarray,
    gains: np.ndarray,
    source_spectra: np.ndarray,
    *,
    spacing_m: float,
    frequency_hz: float,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Return scenes x frames x channels for a positive-angle, far-field ULA.

    The source direction is measured from broadside (+y) toward +x.  Channel
    ``m`` is at ``x=m*spacing_m`` and has response
    ``g[m]*exp(+1j*2*pi*f*m*d*sin(theta)/c)``.  Source spectra include their
    unknown frame-wise complex phases and amplitudes.
    """
    angles = finite_real_array(angles_deg, "angles_deg")
    gain = np.asarray(gains, dtype=complex)
    source = np.asarray(source_spectra, dtype=complex)
    if (angles.ndim != 1 or gain.ndim != 1 or source.ndim != 2
            or source.shape[0] != angles.size or gain.size < 2
            or not np.all(np.isfinite(angles)) or not np.all(np.isfinite(gain))
            or not np.all(np.isfinite(source))):
        raise ValueError("angles, gains and sources need finite compatible shapes")
    spacing = finite_real_scalar(spacing_m, "spacing_m")
    frequency = finite_real_scalar(frequency_hz, "frequency_hz")
    speed = finite_real_scalar(sound_speed, "sound_speed")
    if spacing <= 0 or frequency <= 0 or speed <= 0:
        raise ValueError("spacing, frequency and sound speed must be positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kd = 2.0 * np.pi * frequency * spacing / speed
        phase = kd * np.sin(np.deg2rad(angles[:, None])) * np.arange(gain.size)[None, :]
    if not np.all(np.isfinite(phase)):
        raise ValueError("ULA phase exceeds supported numerical range")
    steering = np.exp(1j * phase)
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = source[:, :, None] * steering[:, None, :] * gain[None, None, :]
    if not np.all(np.isfinite(prediction)):
        raise ValueError("ULA prediction exceeds supported numerical range")
    return prediction


def _profile_sources(observations: np.ndarray, gains: np.ndarray,
                     angles_deg: np.ndarray, kd: float) -> np.ndarray:
    channels = np.arange(gains.size)
    steering = np.exp(1j * kd * np.sin(np.deg2rad(angles_deg[:, None]))
                      * channels[None, :])
    response = steering * gains[None, :]
    denominator = np.sum(np.abs(response) ** 2, axis=1)
    return np.einsum("qm,qtm->qt", response.conj(), observations) / denominator[:, None]


def _cost(observations: np.ndarray, gains: np.ndarray, angles_deg: np.ndarray,
          sources: np.ndarray, spacing: float, frequency: float, speed: float) -> float:
    residual = observations - ula_prediction(
        angles_deg, gains, sources, spacing_m=spacing,
        frequency_hz=frequency, sound_speed=speed)
    return float(np.sum(np.abs(residual) ** 2))


def _best_angle(observation: np.ndarray, gains: np.ndarray, kd: float,
                low: float, high: float, current: float) -> float:
    """Global half-degree scan plus local golden search of profiled LS score."""
    channels = np.arange(gains.size)
    grid = np.linspace(low, high, max(3, int(np.ceil((high - low) * 2)) + 1))
    response = gains[None, :] * np.exp(
        1j * kd * np.sin(np.deg2rad(grid[:, None])) * channels[None, :])
    scores = np.sum(np.abs(observation @ response.conj().T) ** 2, axis=0)
    scores /= np.sum(np.abs(response) ** 2, axis=1)
    index = int(np.argmax(scores))

    def score(angle: float) -> float:
        vector = gains * np.exp(1j * kd * np.sin(np.deg2rad(angle)) * channels)
        return float(np.sum(np.abs(observation @ vector.conj()) ** 2)
                     / np.sum(np.abs(vector) ** 2))

    if 0 < index < grid.size - 1:
        left, right = float(grid[index - 1]), float(grid[index + 1])
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = right - golden * (right - left), left + golden * (right - left)
        fc, fd = score(c), score(d)
        for _ in range(40):
            if fc < fd:
                left, c, fc = c, d, fd
                d = left + golden * (right - left)
                fd = score(d)
            else:
                right, d, fd = d, c, fc
                c = right - golden * (right - left)
                fc = score(c)
        refined = (left + right) / 2.0
        if score(refined) > scores[index]:
            grid_angle = refined
        else:
            grid_angle = float(grid[index])
    else:
        grid_angle = float(grid[index])
    return grid_angle if score(grid_angle) > score(current) else current


def fit_anchored_ula(
    observations: np.ndarray,
    *,
    spacing_m: float,
    frequency_hz: float,
    measured_relative_phase_rad: float,
    sound_speed: float = 343.0,
    angle_bounds_deg: tuple[float, float] = (-60.0, 60.0),
    initial_angles_deg: np.ndarray | None = None,
    max_iterations: int = 50,
    relative_tolerance: float = 1e-10,
) -> dict:
    """Alternate profiled DOA and complex-gain least squares under two anchors.

    ``observations`` is scenes x frames x channels, one far-field narrowband
    source per scene.  The source spectrum is unknown in every frame.  Fixing
    ``g[0]=1`` only sets a relative scale.  The externally measured phase of
    ``g[1]`` removes the ULA phase-slope gauge; its magnitude remains unknown
    but must be positive.  No true DOA or true gain is supplied to this fit.

    The objective is sum |x[q,t,m]-g[m]*a[m](theta[q])*b[q,t]|^2.  Each DOA
    step profiles out b, scans the bounded angle interval, then refines its
    best grid cell.  Each gain step solves complex LS, with a real-positive
    coefficient on the measured-phase ray for g[1].  A non-converged result is
    returned with ``status='max_iterations'``; invalid input or an impossible
    update raises ValueError rather than silently clipping a direction.
    """
    raw = np.asarray(observations)
    if raw.dtype.kind not in "iufc":
        raise ValueError("observations must be numeric complex spectra")
    x = np.asarray(raw, dtype=complex)
    if (x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2 or x.shape[2] < 3
            or not np.all(np.isfinite(x))):
        raise ValueError("observations need finite scenes x frames x channels")
    spacing = finite_real_scalar(spacing_m, "spacing_m")
    frequency = finite_real_scalar(frequency_hz, "frequency_hz")
    speed = finite_real_scalar(sound_speed, "sound_speed")
    anchor = finite_real_scalar(measured_relative_phase_rad,
                                "measured_relative_phase_rad")
    low = finite_real_scalar(angle_bounds_deg[0], "lower angle bound")
    high = finite_real_scalar(angle_bounds_deg[1], "upper angle bound")
    tolerance = finite_real_scalar(relative_tolerance, "relative_tolerance")
    if (spacing <= 0 or frequency <= 0 or speed <= 0
            or not -90 < low < high < 90 or tolerance <= 0
            or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1):
        raise ValueError("invalid physical parameters, angle interval or stopping rule")
    kd = 2.0 * np.pi * frequency * spacing / speed
    if not np.isfinite(kd):
        raise ValueError("ULA phase exceeds supported numerical range")
    if kd * max(abs(np.sin(np.deg2rad(low))), abs(np.sin(np.deg2rad(high)))) >= np.pi:
        raise ValueError("anchored adjacent pair aliases in this angle interval")
    total_energy = float(np.sum(np.abs(x) ** 2))
    reference_energy = np.sum(np.abs(x[:, :, 0]) ** 2, axis=1)
    if (not np.isfinite(total_energy) or total_energy <= 0
            or np.any(reference_energy <= 1e-10 * total_energy)):
        raise ValueError("reference channel has insufficient energy")

    if initial_angles_deg is None:
        cross = np.sum(x[:, :, 1] * x[:, :, 0].conj(), axis=1)
        if np.any(np.abs(cross) <= 1e-10 * reference_energy):
            raise ValueError("anchored pair has insufficient coherent energy")
        phase = np.angle(np.exp(1j * (np.angle(cross) - anchor)))
        sine = phase / kd
        if np.any(sine < np.sin(np.deg2rad(low))) or np.any(sine > np.sin(np.deg2rad(high))):
            raise ValueError("anchored-pair phase implies a direction outside the stated interval")
        angles = np.rad2deg(np.arcsin(sine))
    else:
        angles = finite_real_array(initial_angles_deg, "initial_angles_deg")
        if (angles.shape != (x.shape[0],) or not np.all(np.isfinite(angles))
                or np.any(angles < low) or np.any(angles > high)):
            raise ValueError("initial angles must be finite and inside the stated interval")
        angles = angles.copy()

    # Use the reference channel only for initialization, not as an exact
    # regressor in the final objective: it may itself contain noise.
    steering0 = np.exp(1j * kd * np.sin(np.deg2rad(angles[:, None]))
                       * np.arange(x.shape[2])[None, :])
    reference_regressors = x[:, :, :1] * steering0[:, None, :]
    initial_numerator = np.sum(reference_regressors.conj() * x, axis=(0, 1))
    initial_denominator = float(np.sum(np.abs(x[:, :, 0]) ** 2))
    gains = initial_numerator / initial_denominator
    gains[0] = 1.0
    initial_anchor_magnitude = float(np.real(gains[1] * np.exp(-1j * anchor)))
    if initial_anchor_magnitude <= 0:
        raise ValueError("measured-phase anchor requires a positive initial gain")
    gains[1] = initial_anchor_magnitude * np.exp(1j * anchor)
    sources = _profile_sources(x, gains, angles, kd)
    history = [_cost(x, gains, angles, sources, spacing, frequency, speed)]
    status = "max_iterations"
    for _ in range(max_iterations):
        for q in range(x.shape[0]):
            angles[q] = _best_angle(x[q], gains, kd, low, high, float(angles[q]))
        sources = _profile_sources(x, gains, angles, kd)
        steering = np.exp(1j * kd * np.sin(np.deg2rad(angles[:, None]))
                          * np.arange(gains.size)[None, :])
        regressors = sources[:, :, None] * steering[:, None, :]
        denominator = np.sum(np.abs(regressors) ** 2, axis=(0, 1))
        if np.any(denominator <= 1e-14 * total_energy):
            raise ValueError("source excitation is insufficient for gain estimation")
        numerator = np.sum(regressors.conj() * x, axis=(0, 1))
        new_gains = numerator / denominator
        new_gains[0] = 1.0
        anchored_magnitude = float(np.real(numerator[1] * np.exp(-1j * anchor))
                                   / denominator[1])
        if anchored_magnitude <= 0:
            raise ValueError("measured-phase anchor requires a positive gain")
        new_gains[1] = anchored_magnitude * np.exp(1j * anchor)
        gains = new_gains
        sources = _profile_sources(x, gains, angles, kd)
        current = _cost(x, gains, angles, sources, spacing, frequency, speed)
        previous = history[-1]
        if not np.isfinite(current) or current > previous + 1e-10 * total_energy:
            raise ValueError("alternating objective increased or became nonfinite")
        history.append(current)
        if previous - current <= tolerance * max(previous, 1e-12 * total_energy):
            status = "converged"
            break
    return {
        "status": status,
        "angles_deg": angles,
        "relative_gains": gains,
        "source_spectra": sources,
        "objective_history": history,
        "iterations": len(history) - 1,
        "measured_relative_phase_rad": anchor,
        "angle_bounds_deg": (low, high),
        "model": "single narrowband far-field source per scene, synchronized ULA, fixed complex channel gains",
    }
