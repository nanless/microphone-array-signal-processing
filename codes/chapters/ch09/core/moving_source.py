"""Continuous free-field propagation from a linearly moving point source.

The receiver time ``t`` and emission time ``u`` obey
``t = u + ||source(u)-microphone|| / c``. No room, diffraction, or microphone
directivity is modeled. A known continuous test waveform avoids interpolation
and makes the retarded-time and Doppler behavior independently checkable.
"""

from __future__ import annotations

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


def retarded_emission_times(
    receiver_times: np.ndarray,
    microphones_xy: np.ndarray,
    *,
    source_start_xy: tuple[float, float],
    source_velocity_xy: tuple[float, float],
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Return ``(microphones, times)`` emission times for fixed microphones.

    For a subsonic, noncolliding trajectory the delay equation has derivative
    at least 1-speed/c > 0, hence a unique root. This bounded Newton solver
    checks finite arithmetic and convergence; it does not promise convergence
    for every representable trajectory. Positions are metres, time seconds.
    """

    t = finite_real_array(receiver_times, "receiver_times")
    m = finite_real_array(microphones_xy, "microphones_xy")
    p0 = finite_real_array(source_start_xy, "source_start_xy")
    v = finite_real_array(source_velocity_xy, "source_velocity_xy")
    if (t.ndim != 1 or t.size == 0 or np.any(np.diff(t) < 0) or m.ndim != 2 or
            m.shape[1] != 2 or m.shape[0] < 2 or p0.shape != (2,) or v.shape != (2,) or
            not all(np.all(np.isfinite(x)) for x in (t, m, p0, v))):
        raise ValueError("finite ordered times, at least two 2-D microphones, and finite 2-D trajectory required")
    sound_speed = finite_real_scalar(sound_speed, "sound_speed")
    if sound_speed <= 0 or np.hypot(*v) >= sound_speed:
        raise ValueError("sound speed must exceed the source speed")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            times = np.broadcast_to(t, (m.shape[0], t.size)).copy()
            for _ in range(12):
                offset = p0[None, None, :] + times[:, :, None] * v[None, None, :] - m[:, None, :]
                distance = np.linalg.norm(offset, axis=2)
                if np.any(distance <= 0):
                    raise ValueError("source crosses a point microphone; 1/r model is singular")
                equation = times + distance / sound_speed - t[None, :]
                slope = 1 + np.sum(offset * v[None, None, :], axis=2) / (sound_speed * distance)
                times -= equation / slope
            offset = p0[None, None, :] + times[:, :, None] * v[None, None, :] - m[:, None, :]
            residual = times + np.linalg.norm(offset, axis=2) / sound_speed - t[None, :]
            if not np.all(np.isfinite(times)) or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1e-10:
                raise ArithmeticError("retarded-time solve did not converge")
    except FloatingPointError as error:
        raise ValueError("retarded-time arithmetic exceeds floating-point range") from error

    return times


def synthetic_source(emission_times: np.ndarray) -> np.ndarray:
    """A deterministic two-harmonic source with a smooth causal onset."""

    u = finite_real_array(emission_times, "emission_times")
    if not np.all(np.isfinite(u)):
        raise ValueError("emission times must be finite")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ramp = np.sin(0.5 * np.pi * np.clip(u / 0.02, 0, 1)) ** 2
            result = np.where(u >= 0, ramp * (np.sin(2 * np.pi * 220 * u) +
                                              0.4 * np.sin(2 * np.pi * 320 * u)), 0.0)
    except FloatingPointError as error:
        raise ValueError("source waveform exceeds floating-point range") from error
    return result


def free_field_array(
    receiver_times: np.ndarray,
    microphones_xy: np.ndarray,
    *,
    source_start_xy: tuple[float, float],
    source_velocity_xy: tuple[float, float],
    sound_speed: float = 343.0,
    reference_distance_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return array waveform and emission times, with amplitude ``r_ref/r``."""

    reference_distance_m = finite_real_scalar(reference_distance_m, "reference_distance_m")
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m must be positive")
    emission = retarded_emission_times(
        receiver_times, microphones_xy, source_start_xy=source_start_xy,
        source_velocity_xy=source_velocity_xy, sound_speed=sound_speed,
    )
    p0 = np.asarray(source_start_xy)
    v = np.asarray(source_velocity_xy)
    m = np.asarray(microphones_xy)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            distance = np.linalg.norm(p0[None, None, :] + emission[:, :, None] * v - m[:, None, :], axis=2)
            waveform = synthetic_source(emission) * (reference_distance_m / distance)
    except FloatingPointError as error:
        raise ValueError("free-field waveform exceeds floating-point range") from error
    if not np.all(np.isfinite(waveform)):
        raise ValueError("free-field waveform is not finite")
    return waveform, emission
