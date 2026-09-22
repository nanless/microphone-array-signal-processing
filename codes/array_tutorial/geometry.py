"""Far-field and near-field steering vectors under the book's conventions."""

from __future__ import annotations

import numpy as np

from .conventions import SPEED_OF_SOUND, validate_frequencies, validate_positions


def direction_vector(azimuth_rad: np.ndarray, elevation_rad: np.ndarray = 0.0) -> np.ndarray:
    """Return array-to-source unit vectors.

    Azimuth zero is broadside (+y), positive azimuth turns towards +x.  The
    returned vector is ``[sin(az), cos(az)]`` in 2-D.  Supplying a non-zero or
    array-valued elevation returns the 3-D counterpart.
    """
    azimuth, elevation = np.broadcast_arrays(
        np.asarray(azimuth_rad, dtype=float), np.asarray(elevation_rad, dtype=float)
    )
    if not np.all(np.isfinite(azimuth)) or not np.all(np.isfinite(elevation)):
        raise ValueError("angles must be finite")
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.cos(elevation) * np.cos(azimuth)
    z = np.sin(elevation)
    if np.all(elevation == 0.0):
        return np.stack((x, y), axis=-1)
    return np.stack((x, y, z), axis=-1)


def plane_wave_delays(
    positions: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    elevation_rad: np.ndarray = 0.0,
    reference: int = 0,
    sound_speed: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Return signed arrival delays relative to one microphone, in seconds."""
    microphones = validate_positions(positions)
    if not 0 <= reference < microphones.shape[0]:
        raise ValueError("reference microphone is out of range")
    if not np.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be positive")
    unit = direction_vector(azimuth_rad, elevation_rad)
    if microphones.shape[1] == 3 and unit.shape[-1] == 2:
        unit = np.concatenate((unit, np.zeros(unit.shape[:-1] + (1,))), axis=-1)
    if unit.shape[-1] != microphones.shape[1]:
        raise ValueError("angle dimension and microphone coordinates do not match")
    relative = microphones - microphones[reference]
    return -np.einsum("...d,md->...m", unit, relative) / sound_speed


def plane_wave_steering(
    positions: np.ndarray,
    frequencies_hz: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    elevation_rad: np.ndarray = 0.0,
    reference: int = 0,
    sound_speed: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Return steering vectors as ``frequency x channels`` or ``angle x frequency x channels``."""
    frequencies = validate_frequencies(frequencies_hz)
    delays = plane_wave_delays(
        positions,
        azimuth_rad,
        elevation_rad=elevation_rad,
        reference=reference,
        sound_speed=sound_speed,
    )
    phase = -2.0j * np.pi * np.einsum("f,...m->...fm", frequencies, delays)
    steering = np.exp(phase)
    if np.ndim(azimuth_rad) == 0 and np.ndim(elevation_rad) == 0:
        return steering.reshape(frequencies.size, -1)
    return steering


def near_field_steering(
    positions: np.ndarray,
    source_position: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    reference: int = 0,
    sound_speed: float = SPEED_OF_SOUND,
    include_amplitude: bool = True,
    minimum_distance: float = 1e-6,
) -> np.ndarray:
    """Return a point-source steering vector relative to one microphone.

    With ``include_amplitude=True``, component ``m`` is scaled by
    ``distance(reference) / distance(m)``.  All reference components are one.
    ``minimum_distance`` is a finite non-negative exclusion radius in metres;
    a source exactly on a microphone is rejected even when that radius is zero.
    """
    microphones = validate_positions(positions)
    source = np.asarray(source_position, dtype=float)
    frequencies = validate_frequencies(frequencies_hz)
    if source.shape != (microphones.shape[1],) or not np.all(np.isfinite(source)):
        raise ValueError("source_position must match the coordinate dimension")
    if not 0 <= reference < microphones.shape[0]:
        raise ValueError("reference microphone is out of range")
    if not np.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be positive")
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be finite and non-negative")
    distances = np.linalg.norm(source - microphones, axis=1)
    if np.any(distances <= minimum_distance):
        raise ValueError("point source is too close to a microphone")
    relative_delay = (distances - distances[reference]) / sound_speed
    steering = np.exp(-2.0j * np.pi * frequencies[:, None] * relative_delay[None, :])
    if include_amplitude:
        steering *= distances[reference] / distances[None, :]
    return steering
