"""Known 3-D coordinates/rotation and one static vMF update, not a tracker."""
from __future__ import annotations
import math
import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar
from codes.chapters.ch09.core.tracking import _checked_covariance, wrap_angle


def _unit(value, name):
    v = finite_real_array(value, name)
    if v.shape != (3,) or not math.isclose(math.hypot(*v), 1., rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'{name} must be a unit 3-vector')
    return v/math.hypot(*v)


def direction_from_angles(azimuth_deg, elevation_deg):
    """+y azimuth zero, positive azimuth towards +x, elevation towards +z."""
    theta = math.radians(float(wrap_angle(finite_real_scalar(azimuth_deg, 'azimuth_deg'))))
    phi = finite_real_scalar(elevation_deg, 'elevation_deg')
    if not -90 <= phi <= 90:
        raise ValueError('elevation must be in [-90,90] degrees')
    if abs(phi) == 90:
        return np.array([0., 0., math.copysign(1., phi)])
    phi = math.radians(phi)
    return np.array([math.sin(theta)*math.cos(phi), math.cos(theta)*math.cos(phi), math.sin(phi)])


def angles_from_direction(direction):
    """At an exact pole azimuth is None, never an invented atan2(0,0)."""
    x, y, z = _unit(direction, 'direction')
    horizontal = math.hypot(x, y)
    return {'azimuth_deg': None if horizontal == 0 else float(wrap_angle(math.degrees(math.atan2(x, y)))),
            'elevation_deg': math.degrees(math.atan2(z, horizontal)),
            'azimuth_defined': horizontal != 0}


def rotate_direction(direction, array_to_world):
    """Known proper rotation maps source-pointing array vector into world."""
    u = _unit(direction, 'direction')
    rotation = finite_real_array(array_to_world, 'array_to_world')
    if (rotation.shape != (3, 3) or np.any(np.abs(rotation) > 1+1e-12)
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-12, rtol=1e-12)
            or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('array_to_world must be a proper orthogonal rotation')
    return rotation@u


def normalized_direction_covariance(vector, covariance):
    """First-order delta method in ambient coordinates, only a local approximation.

    J=(I-u*u^T)/||v|| removes radial perturbations. This is not an exact
    normalized Gaussian or a concentration-to-angle-variance conversion.
    """
    v = finite_real_array(vector, 'vector')
    if v.shape != (3,):
        raise ValueError('vector must be a 3-vector')
    radius = math.hypot(*v)
    if radius == 0 or not math.isfinite(radius):
        raise ValueError('finite nonzero vector norm is required')
    c = _checked_covariance(covariance, 'covariance')
    if c.shape != (3, 3):
        raise ValueError('covariance must be 3x3')
    u = v/radius
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            j = (np.eye(3)-np.outer(u,u))/radius
            result = j@c@j.T
        except FloatingPointError as error:
            raise ValueError('local covariance exceeds float64 support') from error
    return {'direction': u, 'jacobian': j, 'covariance': result, 'scope': 'first-order local tangent covariance'}


def vmf_log_density(direction, mean_direction, concentration):
    """S² log density with respect to solid angle (sr), not dtheta*dphi."""
    u, mu = _unit(direction, 'direction'), _unit(mean_direction, 'mean_direction')
    k = finite_real_scalar(concentration, 'concentration')
    if k < 0:
        raise ValueError('concentration must be nonnegative')
    dot = min(1., max(-1., float(u@mu)))
    if k < 1e-4:
        return -math.log(4*math.pi)-k*k/6+k**4/180+k*dot
    # Stable even at high concentration and at the density maximum.
    result = math.log(k)-math.log(2*math.pi)-math.log(-math.expm1(-2*k))+k*(dot-1)
    if not math.isfinite(result):
        raise ValueError('log density exceeds float64 support')
    return result


def vmf_resultant_length(concentration):
    """A(k)=coth(k)-1/k, including uniform and stable small/large k."""
    k = finite_real_scalar(concentration, 'concentration')
    if k < 0:
        raise ValueError('concentration must be nonnegative')
    if k < 1e-3:
        return k/3-k**3/45+2*k**5/945
    tail = math.exp(-2*k)
    return 1+2*tail/(-math.expm1(-2*k))-1/k


def vmf_static_update(mean_direction, concentration, observation_direction, observation_concentration):
    """Exact natural-parameter addition for one independent static vMF likelihood.

    Zero resultant is uniform: concentration zero and mean direction None.
    No dynamics, association, rotation uncertainty or parameter estimation.
    """
    mu, y = _unit(mean_direction, 'mean_direction'), _unit(observation_direction, 'observation_direction')
    k = finite_real_scalar(concentration, 'concentration')
    ky = finite_real_scalar(observation_concentration, 'observation_concentration')
    if min(k, ky) < 0:
        raise ValueError('concentrations must be nonnegative')
    scale = max(k, ky)
    if scale == 0:
        return {'concentration': 0., 'mean_direction': None, 'status': 'uniform', 'natural_parameter': np.zeros(3)}
    scaled = (k/scale)*mu+(ky/scale)*y
    length = math.hypot(*scaled)
    new_k = scale*length
    if not math.isfinite(new_k):
        raise ValueError('posterior concentration exceeds float64 support')
    if length == 0:
        return {'concentration': 0., 'mean_direction': None, 'status': 'uniform', 'natural_parameter': np.zeros(3)}
    return {'concentration': new_k, 'mean_direction': scaled/length, 'status': 'concentrated',
            'natural_parameter': scaled*scale}
