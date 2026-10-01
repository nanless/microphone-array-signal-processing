"""Minimal angle-tracking baselines."""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import numpy as np

from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Map degrees to ``[-180, 180)``."""

    # Reduce before adding 180 so large finite angles cannot overflow.
    value = finite_real_array(angle, "angle")
    wrapped = (value % 360.0 + 180.0) % 360.0 - 180.0
    # Addition at the 180-degree scale cannot resolve very small directions.
    wrapped = np.where(np.abs(value) < 1e-12, value, wrapped)
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def _finite_fraction(value, *, positive=False):
    """Convert only the final exact compound result; never call a lost variance zero."""
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("tracking result exceeds floating-point range") from error
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("nonzero tracking result is outside float64 support")
    if positive and value < 0:
        raise ValueError("negative tracking variance")
    return result


def _exceptional(*values):
    nonzero = np.concatenate([np.asarray(v, float).reshape(-1) for v in values])
    nonzero = np.abs(nonzero[nonzero != 0])
    return bool(nonzero.size and (nonzero.min() < 1e-140 or nonzero.max() > 1e140))


def white_acceleration_covariance(dt, density):
    """Integrated white angular acceleration: dt seconds, density deg²/s³.

    Zero interval/density is legal. Ordinary inputs retain the original order;
    exceptional scales accumulate the full monomials exactly before conversion.
    Unrepresentable strictly positive variances are rejected, not physical zero.
    """
    dt = finite_real_scalar(dt, "dt")
    density = finite_real_scalar(density, "density")
    if dt < 0 or density < 0:
        raise ValueError("dt and density must be nonnegative")
    if dt == 0 or density == 0:
        return np.zeros((2, 2))
    if not _exceptional(dt, density):
        try:
            with np.errstate(over="raise", invalid="raise", under="raise"):
                ordinary = density*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
            if np.all(ordinary > 0) and np.all(np.isfinite(ordinary)):
                return ordinary
        except (FloatingPointError, OverflowError):
            pass
    t, q = Fraction(dt), Fraction(density)
    return np.array([[_finite_fraction(q*t**3/3, positive=True), _finite_fraction(q*t*t/2)],
                     [_finite_fraction(q*t*t/2), _finite_fraction(q*t, positive=True)]])


class ConstantVelocityKalman:
    """Two-state ``[angle_deg, angular_velocity_deg_per_s]`` Kalman filter.

    ``process_noise`` is already discretized for one prediction interval.  If
    ``dt`` changes, the caller must supply a consistently scaled Q; this small
    class does not infer a continuous white-acceleration density. Exceptional
    state updates use exact rational compound arithmetic until final conversion.
    Unrepresentable nonzero final entries and PSD normalization loss are rejected
    atomically. This finite teaching interface is not an all-float64 solver.
    """

    def __init__(self, state: np.ndarray, covariance: np.ndarray, process_noise: np.ndarray):
        self.state = finite_real_array(state, "state").copy()
        self.covariance = finite_real_array(covariance, "covariance").copy()
        self.process_noise = finite_real_array(process_noise, "process_noise").copy()
        if self.state.shape != (2,) or self.covariance.shape != (2, 2) or self.process_noise.shape != (2, 2):
            raise ValueError("state must be (2,), covariance and process_noise must be (2,2)")
        if not all(np.all(np.isfinite(value)) for value in (self.state, self.covariance, self.process_noise)):
            raise ValueError("state and covariance inputs must be finite")
        self.state[0] = wrap_angle(self.state[0])
        self.covariance = _checked_covariance(self.covariance, "covariance")
        self.process_noise = _checked_covariance(self.process_noise, "process_noise")

    def predict(self, dt: float) -> np.ndarray:
        dt = finite_real_scalar(dt, "dt")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if _exceptional(dt, self.state, self.covariance, self.process_noise):
                    t = Fraction(dt)
                    x = [Fraction(float(v)) for v in self.state]
                    p = [[Fraction(float(v)) for v in row] for row in self.covariance]
                    q = [[Fraction(float(v)) for v in row] for row in self.process_noise]
                    state = np.array([_finite_fraction(x[0]+t*x[1]), float(x[1])])
                    covariance = np.array([
                        [_finite_fraction(p[0][0]+t*(p[0][1]+p[1][0])+t*t*p[1][1]+q[0][0], positive=True),
                         _finite_fraction(p[0][1]+t*p[1][1]+q[0][1])],
                        [_finite_fraction(p[1][0]+t*p[1][1]+q[1][0]),
                         _finite_fraction(p[1][1]+q[1][1], positive=True)]])
                else:
                    state = transition @ self.state
                    covariance = transition @ self.covariance @ transition.T + self.process_noise
                state[0] = wrap_angle(state[0])
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
                if _exceptional(self.covariance, measurement_variance, innovation):
                    p = [[Fraction(float(v)) for v in row] for row in self.covariance]
                    r = Fraction(measurement_variance)
                    total = p[0][0]+r
                    correction = [p[i][0]*Fraction(float(innovation))/total for i in range(2)]
                    state = np.array([_finite_fraction(Fraction(float(self.state[i]))+correction[i])
                                      for i in range(2)])
                    state[0] = wrap_angle(state[0])
                    # Algebraically Joseph, evaluated without separately rounded terms.
                    covariance = np.array([[_finite_fraction(p[i][j]-p[i][0]*p[0][j]/total,
                                                            positive=i == j)
                                             for j in range(2)] for i in range(2)])
                else:
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


def _checked_covariance(matrix: np.ndarray, name: str = "tracking covariance") -> np.ndarray:
    """Validate a real PSD state matrix, including legitimate singular matrices.

    Reject negative diagonal variances. Only negative eigenvalues within
    32*n*machine_epsilon of the entry scale are projected to zero; this is a
    rounding repair, not an allowance for physical negative uncertainty.
    """
    matrix = finite_real_array(matrix, name)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if np.any(np.diag(matrix) < 0):
        raise ValueError(f"{name} contains a negative variance")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return matrix.copy()
    # A zero variance has an identically zero covariance row/column.
    zero = np.diag(matrix) == 0
    if np.any(matrix[zero] != 0) or np.any(matrix[:, zero] != 0):
        raise ValueError(f"{name} has nonzero covariance with zero variance")
    unit = matrix / scale
    if np.any((matrix != 0) & (unit == 0)):
        raise ValueError(f"{name} normalization loses a nonzero entry")
    tolerance = 32 * matrix.shape[0] * np.finfo(float).eps
    if np.max(np.abs(unit - unit.T)) > tolerance:
        raise ValueError(f"{name} must be symmetric relative to its scale")
    symmetric = matrix.copy() if np.array_equal(matrix, matrix.T) else .5*matrix + .5*matrix.T
    values, vectors = np.linalg.eigh(.5*unit + .5*unit.T)
    if values[0] < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    if values[0] < 0:
        projected = (vectors * np.maximum(values, 0)) @ vectors.T
        with np.errstate(over="ignore", invalid="ignore"):
            symmetric = (.5*projected + .5*projected.T) * scale
        if not np.all(np.isfinite(symmetric)):
            raise ValueError(f"{name} projection exceeds floating-point range")
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
    """Angle-only SIR with persistent log posterior and linear display weights.

    Finite log support survives exp underflow. Explicit set_prior_weights zero
    priors and unselected resampling ancestors really remove support. Gaussian log penalties
    outside float64 support are rejected atomically, not treated as zero density.
    """

    def __init__(self, particles: np.ndarray):
        self.particles = finite_real_array(particles, "particles").copy()
        if self.particles.ndim != 1 or self.particles.size == 0 or not np.all(np.isfinite(self.particles)):
            raise ValueError("particles must be a non-empty finite vector")
        self.particles = wrap_angle(self.particles)
        self.weights = np.full(self.particles.size, 1.0 / self.particles.size)
        self._log_weights = np.log(self.weights)
        self._weight_snapshot = self.weights.copy()
        self.last_resample_ancestor_indices = None
        self.last_resample_removed_indices = []

    def set_prior_weights(self, weights):
        """Explicitly replace the prior, even when its display values are unchanged.

        Accept finite real nonnegative masses of the particle shape and positive
        total mass. Actual input zeros get -inf logs; normalized positive masses
        below the linear display range retain finite logs. Validation and full
        normalization finish before either posterior representation is committed.
        Legacy direct edits whose values differ from the snapshot are detected
        on update; identical in-place writes cannot be detected and are not this
        explicit reset API.
        """
        prior = finite_real_array(weights, "prior weights")
        if prior.shape != self.particles.shape or np.any(prior < 0) or not np.any(prior > 0):
            raise ValueError("prior weights must have the particle shape and positive nonnegative mass")
        logs = np.log(prior, where=prior > 0, out=np.full_like(prior, -np.inf))
        maximum = float(np.max(logs))
        with np.errstate(under="ignore"):
            unnormalized = np.exp(logs-maximum)
            linear = unnormalized/float(np.sum(unnormalized))
        log_weights = (logs-maximum)-math.log(float(np.sum(unnormalized)))
        self.weights = linear
        self._log_weights = log_weights
        self._weight_snapshot = linear.copy()

    def predict(self, angular_velocity: float, dt: float, process_std: float, rng: np.random.Generator) -> None:
        angular_velocity = finite_real_scalar(angular_velocity, "angular_velocity")
        dt = finite_real_scalar(dt, "dt")
        process_std = finite_real_scalar(process_std, "process_std")
        if not all(np.isfinite(value) for value in (angular_velocity, dt, process_std)) or dt <= 0 or process_std < 0:
            raise ValueError("velocity, dt and process_std must be finite; dt positive and std non-negative")
        # Validate deterministic arithmetic before drawing randomness. Work with
        # a private RNG copy so even a stochastic overflow is an atomic failure.
        drift = angular_velocity * dt
        if not math.isfinite(drift):
            raise ValueError("angular displacement exceeds floating-point range")
        local_rng = copy.deepcopy(rng)
        with np.errstate(over="ignore", invalid="ignore"):
            noise = local_rng.normal(0.0, process_std, self.particles.size)
            proposed = self.particles + wrap_angle(drift) + noise
        particles = wrap_angle(proposed)  # validates all finite before committing
        rng.bit_generator.state = copy.deepcopy(local_rng.bit_generator.state)
        self.particles = particles

    def update(self, observation: float, observation_std: float, *, clutter_probability: float = 0.05) -> None:
        observation = finite_real_scalar(observation, "observation")
        observation_std = finite_real_scalar(observation_std, "observation_std")
        clutter_probability = finite_real_scalar(clutter_probability, "clutter_probability")
        if (
            not np.isfinite(observation)
            or not np.isfinite(observation_std)
            or not np.isfinite(clutter_probability)
            or observation_std <= 0
            or not 0.0 <= clutter_probability < 1.0
        ):
            raise ValueError("invalid observation_std or clutter_probability")
        # Legacy value-changing edits are recognized for compatibility. Use
        # set_prior_weights to explicitly reset an unchanged linear display.
        # A zero caused by exp underflow alone does not remove support.
        weights = finite_real_array(self.weights, "weights")
        if weights.shape != self.particles.shape or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("weights must have the particle shape and positive mass")
        manually_changed = not np.array_equal(weights, self._weight_snapshot)
        if manually_changed or not self.weight_underflow_indices:
            prior_logs = np.log(weights, where=weights > 0, out=np.full_like(weights, -np.inf))
        else:
            prior_logs = self._log_weights.copy()
        error = wrap_angle(wrap_angle(observation) - self.particles)
        absolute_error = np.abs(error)
        if clutter_probability == 0.0:
            support = np.isfinite(prior_logs)
            minimum = float(np.min(absolute_error[support]))
            delta = absolute_error - minimum
            log_likelihood = np.full_like(absolute_error, -np.inf)
            closest = support & (delta == 0)
            log_likelihood[closest] = 0.0
            other = support & (delta > 0)
            # log((r²-r_min²)/(2 sigma²)): no sigma² underflow,
            # no 0*inf at the nearest supported particle.
            log_penalty = (np.log(delta[other])
                           + np.log(absolute_error[other] + minimum)
                           - math.log(2.) - 2*math.log(observation_std))
            with np.errstate(over="ignore", under="ignore"):
                log_likelihood[other] = -np.exp(log_penalty)
            if np.any(~np.isfinite(log_likelihood[support])):
                raise ValueError("positive particle log likelihood exceeds float64 support")
        else:
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                log_gaussian = (-0.5 * (absolute_error / observation_std) ** 2
                                - .5*math.log(2*math.pi) - math.log(observation_std))
            with np.errstate(under="ignore"):
                log_likelihood = np.logaddexp(
                    math.log1p(-clutter_probability) + log_gaussian,
                    math.log(clutter_probability) - math.log(360.0),
                )
        with np.errstate(over="ignore", invalid="ignore"):
            log_weights = prior_logs + log_likelihood
        if np.any(np.isfinite(prior_logs) & np.isfinite(log_likelihood) & ~np.isfinite(log_weights)):
            raise ValueError("positive particle log posterior exceeds float64 support")
        maximum = float(np.max(log_weights))
        if not np.isfinite(maximum):
            raise FloatingPointError("particle posterior has no finite support")
        with np.errstate(under="ignore"):
            unnormalized = np.exp(log_weights - maximum)
            weights = unnormalized / np.sum(unnormalized)
        self.weights = weights
        self._log_weights = (log_weights-maximum)-math.log(float(np.sum(unnormalized)))
        self._weight_snapshot = weights.copy()

    @property
    def log_weights(self):
        """Copy of normalized log probabilities; -inf denotes actual removed support."""
        if not np.array_equal(self.weights, self._weight_snapshot):
            weights = finite_real_array(self.weights, "weights")
            if weights.shape != self.particles.shape or np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("invalid explicit particle prior")
            return np.log(weights, where=weights > 0, out=np.full_like(weights, -np.inf))
        return self._log_weights.copy()

    @property
    def weight_underflow_indices(self):
        """Linear display zeros whose stored mathematical log probability is finite."""
        if not np.array_equal(self.weights, self._weight_snapshot):
            return []
        return np.flatnonzero((self.weights == 0) & np.isfinite(self._log_weights)).tolist()

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def estimate(self) -> float:
        radians = np.deg2rad(self.particles)
        vector = np.sum(self.weights * np.exp(1j * radians))
        if abs(vector) <= 1e-12:
            raise ValueError("circular mean is undefined for this symmetric posterior")
        return float(wrap_angle(np.rad2deg(np.angle(vector))))

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
        self.last_resample_ancestor_indices = indices.copy()
        self.last_resample_removed_indices = sorted(set(range(self.particles.size))-set(indices.tolist()))
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.weights.size)
        # Ancestor selection explicitly discards unselected support.
        self._log_weights = np.log(self.weights)
        self._weight_snapshot = self.weights.copy()
        return True
