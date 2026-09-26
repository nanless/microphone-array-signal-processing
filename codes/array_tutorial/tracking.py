"""Minimal angle-tracking baselines."""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Map degrees to ``[-180, 180)``."""

    # Reduce before adding 180 so large finite angles cannot overflow.
    wrapped = (finite_real_array(angle, "angle") % 360.0 + 180.0) % 360.0 - 180.0
    return float(wrapped) if wrapped.ndim == 0 else wrapped


class ConstantVelocityKalman:
    """Two-state ``[angle_deg, angular_velocity_deg_per_s]`` Kalman filter.

    ``process_noise`` is already discretized for one prediction interval.  If
    ``dt`` changes, the caller must supply a consistently scaled Q; this small
    class does not infer a continuous white-acceleration density.
    """

    def __init__(self, state: np.ndarray, covariance: np.ndarray, process_noise: np.ndarray):
        self.state = finite_real_array(state, "state").copy()
        self.covariance = finite_real_array(covariance, "covariance").copy()
        self.process_noise = finite_real_array(process_noise, "process_noise").copy()
        if self.state.shape != (2,) or self.covariance.shape != (2, 2) or self.process_noise.shape != (2, 2):
            raise ValueError("state must be (2,), covariance and process_noise must be (2,2)")
        if not all(np.all(np.isfinite(value)) for value in (self.state, self.covariance, self.process_noise)):
            raise ValueError("state and covariance inputs must be finite")
        for name, matrix in (("covariance", self.covariance), ("process_noise", self.process_noise)):
            scale = float(np.max(np.abs(matrix)))
            normalized = matrix / scale if scale else matrix
            if not np.allclose(normalized, normalized.T, rtol=0.0, atol=1e-12):
                raise ValueError(f"{name} must be symmetric")
            symmetric = 0.5 * normalized + 0.5 * normalized.T
            if np.min(np.linalg.eigvalsh(symmetric)) < -1e-12:
                raise ValueError(f"{name} must be positive semidefinite")
            # Accepted roundoff asymmetry must not enter later state updates.
            matrix[:] = 0.5 * matrix + 0.5 * matrix.T

    def predict(self, dt: float) -> np.ndarray:
        dt = finite_real_scalar(dt, "dt")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                state = transition @ self.state
                state[0] = wrap_angle(state[0])
                covariance = transition @ self.covariance @ transition.T + self.process_noise
                covariance = _checked_covariance(covariance)
        except FloatingPointError as error:
            raise ValueError("predicted state or covariance exceeds floating-point range") from error
        if not np.all(np.isfinite(state)):
            raise ValueError("predicted state exceeds floating-point range")
        self.state = state
        self.covariance = covariance
        return self.state.copy()

    def update(self, angle: float, measurement_variance: float) -> np.ndarray:
        angle = finite_real_scalar(angle, "angle")
        measurement_variance = finite_real_scalar(measurement_variance, "measurement_variance")
        if not np.isfinite(angle) or not np.isfinite(measurement_variance) or measurement_variance <= 0:
            raise ValueError("angle must be finite and measurement_variance positive")
        # Reducing the observation before subtraction avoids overflowing two
        # finite angles that represent ordinary directions modulo 360 degrees.
        innovation = wrap_angle(wrap_angle(angle) - self.state[0])
        scale = max(float(np.max(np.abs(self.covariance[:, 0]))), measurement_variance)
        if scale <= 0.0:
            raise ValueError("innovation variance must be positive")
        denominator = self.covariance[0, 0] / scale + measurement_variance / scale
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("innovation variance must be positive and finite")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                gain = (self.covariance[:, 0] / scale) / denominator
                if not np.all(np.isfinite(gain)):
                    raise ValueError("Kalman gain exceeds floating-point range")
                state = self.state + gain * innovation
                state[0] = wrap_angle(state[0])
                residual_map = np.array([[1.0 - gain[0], 0.0], [-gain[1], 1.0]])
                # Joseph form keeps covariance positive semidefinite better
                # than subtracting nearly equal prior and correction matrices.
                covariance = (
                    residual_map @ self.covariance @ residual_map.T
                    + measurement_variance * np.outer(gain, gain)
                )
                covariance = _checked_covariance(covariance)
        except FloatingPointError as error:
            raise ValueError("updated state or covariance exceeds floating-point range") from error
        if not np.all(np.isfinite(state)):
            raise ValueError("updated state exceeds floating-point range")
        self.state = state
        self.covariance = covariance
        return self.state.copy()


def _checked_covariance(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize without doubling large entries; reject invalid covariance."""
    symmetric = 0.5 * matrix + 0.5 * matrix.T
    if not np.all(np.isfinite(symmetric)):
        raise ValueError("tracking covariance exceeds floating-point range")
    scale = float(np.max(np.abs(symmetric)))
    if scale and np.min(np.linalg.eigvalsh(symmetric / scale)) < -1e-10:
        raise ValueError("tracking covariance is not positive semidefinite")
    return symmetric


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return systematic-resampling ancestor indices."""

    weights = finite_real_array(weights, "weights")
    if weights.ndim != 1 or weights.size == 0 or np.any(weights < 0):
        raise ValueError("weights must be a non-empty non-negative vector")
    scale = float(np.max(weights))
    if scale <= 0.0:
        raise ValueError("weights must have a positive sum")
    scaled = weights / scale
    normalized = scaled / np.sum(scaled)
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(weights.size)) / weights.size
    return np.searchsorted(cumulative, positions, side="right")


class CircularParticleFilter:
    """Angle-only SIR particle filter with a Gaussian-plus-uniform likelihood."""

    def __init__(self, particles: np.ndarray):
        self.particles = finite_real_array(particles, "particles").copy()
        if self.particles.ndim != 1 or self.particles.size == 0 or not np.all(np.isfinite(self.particles)):
            raise ValueError("particles must be a non-empty finite vector")
        self.particles = wrap_angle(self.particles)
        self.weights = np.full(self.particles.size, 1.0 / self.particles.size)

    def predict(self, angular_velocity: float, dt: float, process_std: float, rng: np.random.Generator) -> None:
        angular_velocity = finite_real_scalar(angular_velocity, "angular_velocity")
        dt = finite_real_scalar(dt, "dt")
        process_std = finite_real_scalar(process_std, "process_std")
        if not all(np.isfinite(value) for value in (angular_velocity, dt, process_std)) or dt <= 0 or process_std < 0:
            raise ValueError("velocity, dt and process_std must be finite; dt positive and std non-negative")
        self.particles = wrap_angle(
            self.particles + angular_velocity * dt + rng.normal(0.0, process_std, self.particles.size)
        )

    def update(self, observation: float, observation_std: float, *, clutter_probability: float = 0.05) -> None:
        observation = finite_real_scalar(observation, "observation")
        observation_std = finite_real_scalar(observation_std, "observation_std")
        clutter_probability = finite_real_scalar(clutter_probability, "clutter_probability")
        if (
            not np.isfinite(observation)
            or not np.isfinite(observation_std)
            or not np.isfinite(clutter_probability)
            or observation_std <= 0
            or observation_std < np.sqrt(np.finfo(float).tiny)
            or not 0.0 <= clutter_probability < 1.0
        ):
            raise ValueError("invalid observation_std or clutter_probability")
        error = wrap_angle(observation - self.particles)
        absolute_error = np.abs(error)
        if clutter_probability == 0.0:
            minimum = float(np.min(absolute_error))
            log_likelihood = (
                -0.5
                * (absolute_error - minimum)
                * (absolute_error + minimum)
                / (observation_std * observation_std)
            )
        else:
            log_gaussian = (
                -0.5 * (absolute_error / observation_std) ** 2
                - np.log(np.sqrt(2.0 * np.pi) * observation_std)
            )
            log_likelihood = np.logaddexp(
                np.log1p(-clutter_probability) + log_gaussian,
                np.log(clutter_probability / 360.0),
            )
        log_weights = np.log(
            self.weights,
            where=self.weights > 0,
            out=np.full_like(self.weights, -np.inf),
        )
        log_weights += log_likelihood
        maximum = float(np.max(log_weights))
        if not np.isfinite(maximum):
            raise FloatingPointError("particle posterior has no finite support")
        unnormalized = np.exp(log_weights - maximum)
        self.weights = unnormalized / np.sum(unnormalized)

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def estimate(self) -> float:
        radians = np.deg2rad(self.particles)
        vector = np.sum(self.weights * np.exp(1j * radians))
        if abs(vector) <= 1e-12:
            raise ValueError("circular mean is undefined for this symmetric posterior")
        return float(np.rad2deg(np.angle(vector)))

    def resample_if_needed(self, rng: np.random.Generator, threshold: float | None = None) -> bool:
        """Resample when ESS is below a finite threshold in ``[0, N]``.

        Zero disables resampling. Invalid thresholds leave state and RNG intact.
        """
        limit = self.particles.size / 2.0 if threshold is None else threshold
        if (isinstance(limit, (bool, np.bool_))
                or not isinstance(limit, (int, float, np.integer, np.floating))
                or not np.isfinite(limit)
                or not 0 <= limit <= self.particles.size):
            raise ValueError("threshold must be a finite scalar between zero and particle count")
        if self.effective_sample_size >= limit:
            return False
        indices = systematic_resample(self.weights, rng)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.weights.size)
        return True
