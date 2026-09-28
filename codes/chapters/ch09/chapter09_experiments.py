"""Ten deterministic teaching experiments E09-10..19; no file writes on import/run.

These are small arithmetic/controlled-signal examples, not complete EKF, UKF,
JPDA, PHD, GOSPA tracking systems or device evaluations. No upstream imports.
Run: python -m codes.chapters.ch09.chapter09_experiments
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

from itertools import permutations, product
import json
import math
import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar
from codes.chapters.ch09.core.tracking import CircularParticleFilter, ConstantVelocityKalman, wrap_angle
from codes.chapters.ch09.core.tracking_audio import build_fixture


def white_acceleration_covariance(dt, density):
    """Continuous white angular-acceleration density [deg²/s³], not variance."""
    dt = finite_real_scalar(dt, 'dt')
    density = finite_real_scalar(density, 'density')
    if dt < 0 or density < 0:
        raise ValueError('dt and density must be nonnegative')
    try:
        result = density*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    except OverflowError as error:
        raise ValueError('process noise exceeds floating-point range') from error
    if not np.all(np.isfinite(result)):
        raise ValueError('process noise exceeds floating-point range')
    return result


def small_set_distances(truth, estimate, *, cutoff=10., order=1):
    """OSPA and alpha=2 GOSPA on ≤7 scalar circular angles, p=1 or 2.

    Exhaustive enumeration is intentionally small. This scores unordered
    position sets, never identity continuity. Empty/empty is zero; one empty
    set has OSPA=c and GOSPA=(c^p*n/2)^(1/p).
    """
    first = finite_real_array(truth, 'truth')
    second = finite_real_array(estimate, 'estimate')
    cutoff = finite_real_scalar(cutoff, 'cutoff')
    order = finite_real_scalar(order, 'order')
    if first.ndim != 1 or second.ndim != 1 or max(first.size, second.size) > 7:
        raise ValueError('small 1-D sets of at most seven angles are required')
    if not 0 < cutoff <= 180 or order not in (1, 2):
        raise ValueError('this teaching helper supports 0<c<=180 degrees and p=1 or 2')
    first, second = wrap_angle(first), wrap_angle(second)
    if np.unique(first).size != first.size or np.unique(second).size != second.size:
        raise ValueError('sets must not contain duplicate positions modulo 360')
    if first.size > second.size:
        first, second = second, first
    m, n = len(first), len(second)
    if n == 0:
        return {'ospa': 0., 'gospa_alpha2': 0., 'matched_capped_cost_power': 0., 'cardinality_difference': 0}
    best = min(sum(min(cutoff, abs(float(wrap_angle(first[i]-second[j]))))**order
                   for i, j in enumerate(assignment))
               for assignment in permutations(range(n), m))
    unmatched = cutoff**order*(n-m)
    return {'ospa': float(((best+unmatched)/n)**(1/order)),
            'gospa_alpha2': float((best+unmatched/2)**(1/order)),
            'matched_capped_cost_power': float(best), 'cardinality_difference': n-m}


def jpda_anchor():
    """Two tracks, three detections; enumerate all 13 legal events explicitly."""
    predicted, detections, sigma, detection_probability, clutter = [30., 50.], [15., 31., 48.], 3., .9, .01
    events = []
    for assignment in product(range(-1, 3), repeat=2):
        if assignment[0] >= 0 and assignment[0] == assignment[1]:
            continue
        weight = 1.
        for angle, detection in zip(predicted, assignment):
            if detection == -1:
                weight *= 1-detection_probability
            else:
                likelihood = math.exp(-.5*((detections[detection]-angle)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
                weight *= detection_probability*likelihood/clutter
        events.append({'assignment': list(assignment), 'weight': weight})
    normalizer = math.fsum(item['weight'] for item in events)
    marginals = np.zeros((2, 4))  # column0 miss, columns1..3 detections
    for event in events:
        event['probability'] = event['weight']/normalizer
        for i, detection in enumerate(event['assignment']):
            marginals[i, detection+1] += event['probability']
    prior_mean, prior_variance, measurement_variance = 0., 4., 1.
    symmetric_detections, association = np.array([-2., 2.]), np.array([.5, .5])
    gain = prior_variance/(prior_variance+measurement_variance)
    conditional_means = prior_mean+gain*(symmetric_detections-prior_mean)
    conditional_variance = (1-gain)**2*prior_variance+gain**2*measurement_variance
    mixed_mean = association@conditional_means
    mixed_variance = association@(conditional_variance+(conditional_means-mixed_mean)**2)
    return {'predicted_deg': predicted, 'detections_deg': detections,
            'detection_probability': detection_probability, 'clutter_per_deg': clutter,
            'sigma_deg': sigma, 'normalizer': normalizer, 'events': events,
            'marginals_miss_then_detections': marginals,
            'symmetric_update': {'prior_mean': prior_mean, 'prior_variance': prior_variance,
                'measurement_variance': measurement_variance,
                'detections': symmetric_detections, 'association_probabilities': association,
                'conditional_means': conditional_means, 'conditional_variance': conditional_variance,
                'mixture_mean': mixed_mean, 'mixture_variance': mixed_variance,
                'wrong_average_observation_variance': conditional_variance}}



def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_experiments():
    result = {}
    position, prior = np.array([1., 2.]), np.eye(2)*.25
    observation_variance, innovation = .01, .1
    jacobian = np.array([position[1], -position[0]])/(position@position)
    innovation_variance = float(jacobian@prior@jacobian+observation_variance)
    gain = prior@jacobian/innovation_variance
    residual_map = np.eye(2)-np.outer(gain, jacobian)
    result['E09-10'] = {'position_m': position, 'prior_covariance_m2': prior,
        'bearing_rad': math.atan2(*position), 'innovation_rad': innovation,
        'jacobian_rad_per_m': jacobian, 'innovation_variance_rad2': innovation_variance,
        'gain_m_per_rad': gain, 'updated_position_m': position+gain*innovation,
        'posterior_covariance_m2': residual_map@prior@residual_map.T+observation_variance*np.outer(gain, gain)}

    points, wm, wc = np.array([0., 1., -1.]), np.array([0., .5, .5]), np.array([2., .5, .5])
    transformed = points**2
    mean = wm@transformed
    result['E09-11'] = {'alpha': 1., 'beta': 2., 'kappa': 0., 'sigma_points': points,
        'mean_weights': wm, 'covariance_weights': wc, 'transformed': transformed,
        'mean': mean, 'variance': wc@((transformed-mean)**2),
        'state_observation_cross_covariance': wc@(points*(transformed-mean)),
        'boundary': 'squared observation cannot distinguish x from -x; cross covariance and UKF gain are zero'}

    particles = CircularParticleFilter([179., -179.])
    undefined = CircularParticleFilter([90., -90.])
    undefined_angle = None
    status = 'not_checked'
    try:
        undefined_angle = undefined.estimate()
        status = 'unexpected_defined_antipodal_mean'
    except ValueError:
        status = 'undefined_antipodal_mean'
    supported = CircularParticleFilter([0., 10.])
    supported.update(10., .01, clutter_probability=0.)
    first = supported.weights.copy()
    supported.update(0., 1.5e-154, clutter_probability=0.)
    result['E09-12'] = {'particles_deg': [179., -179.], 'linear_mean_deg': 0.,
        'circular_mean_deg': particles.estimate(), 'resultant_length': math.cos(math.radians(1)),
        'antipodal_status': status, 'antipodal_angle_deg': undefined_angle,
        'support_after_first': first, 'support_after_second': supported.weights,
        'support_note': 'zero prior remains zero; the nearest unsupported particle must not set the likelihood shift'}

    result['E09-13'] = jpda_anchor()
    # Enumerate the actual random sets: columns indicate presence at a,b.
    distributions = {
        'A': (np.array([[1., 0.], [0., 1.]]), np.array([.5, .5])),
        'B': (np.array([[0., 0.], [1., 1.]]), np.array([.5, .5])),
    }
    phd_result = {}
    for name, (sets, probabilities) in distributions.items():
        counts = sets.sum(axis=1)
        mean_count = probabilities@counts
        phd_result[f'model_{name}_grid_mass'] = probabilities@sets
        phd_result[f'model_{name}_count_probability_0_1_2'] = [float(probabilities[counts == n].sum()) for n in range(3)]
        phd_result[f'model_{name}_count_variance'] = float(probabilities@((counts-mean_count)**2))
    mass = phd_result['model_A_grid_mass']
    detection_probability = .8
    remaining = (1-detection_probability)*mass
    result['E09-14'] = {**phd_result, 'grid_mass': mass, 'expected_count': float(mass.sum()),
        'detection_probability': detection_probability, 'empty_observation_posterior_mass': remaining,
        'empty_observation_expected_count': float(remaining.sum())}
    result['E09-15'] = {'original': small_set_distances([32], [30, 70]),
        'added_common_correct_target': small_set_distances([32, 100], [30, 70, 100]),
        'both_empty': small_set_distances([], []), 'one_empty': small_set_distances([], [30, 70]),
        'identity_note': 'permutation is optimized independently; equal sets do not establish track identity'}

    initial_mean, initial_variance, process_variance, observation_variance = 0., 1., 1., 1.
    observations = [1., 0.]
    mean, variance = initial_mean, initial_variance
    filtered_means, filtered_variances, predicted_variances = [], [], []
    for observation in observations:
        predicted_variance = variance+process_variance
        gain = predicted_variance/(predicted_variance+observation_variance)
        mean = mean+gain*(observation-mean)
        variance = (1-gain)**2*predicted_variance+gain**2*observation_variance
        predicted_variances.append(predicted_variance)
        filtered_means.append(mean)
        filtered_variances.append(variance)
    m1, m2 = filtered_means
    p1, p2 = filtered_variances
    predicted_p2 = predicted_variances[1]
    smoothing_gain = p1/predicted_p2
    result['E09-16'] = {'initial_mean': initial_mean, 'initial_variance': initial_variance,
        'Q': process_variance, 'R': observation_variance, 'observations': observations,
        'filtered_means': [m1, m2], 'filtered_variances': [p1, p2],
        'smoothing_gain': smoothing_gain, 'smoothed_time1_mean': m1+smoothing_gain*(m2-m1),
        'smoothed_time1_variance': p1+smoothing_gain**2*(p2-predicted_p2),
        'target_time_step': 1, 'latest_used_observation_step': 2, 'extra_observation_delay_steps': 1}

    state = np.array([30., 5.]); covariance = np.array([[4., 1.], [1., 9.]])
    process = np.diag([1., 0.]); transition = np.array([[1., 1.], [0., 1.]])
    tracker = ConstantVelocityKalman(state, covariance, process)
    predicted = tracker.predict(1.)
    predicted_covariance = tracker.covariance.copy()
    nis = (37.-predicted[0])**2/(predicted_covariance[0, 0]+9.)
    tracker.update(37., 9.)
    scale = np.pi/180
    rad_pred = transition@(state*scale)
    rad_cov = transition@(covariance*scale**2)@transition.T+process*scale**2
    rad_innovation, rad_s = 37*scale-rad_pred[0], rad_cov[0, 0]+9*scale**2
    rad_gain = rad_cov[:, 0]/rad_s
    rad_map = np.eye(2)-np.outer(rad_gain, [1., 0.])
    result['E09-17'] = {'initial_state_deg': state, 'initial_covariance_deg_units': covariance,
        'Q_deg_units': process, 'dt_s': 1., 'observation_deg': 37., 'R_deg2': 9.,
        'predicted_state_deg': predicted, 'predicted_covariance_deg_units': predicted_covariance,
        'posterior_state_deg': tracker.state, 'posterior_covariance_deg_units': tracker.covariance,
        'posterior_state_rad': rad_pred+rad_gain*rad_innovation,
        'posterior_covariance_rad_units': rad_map@rad_cov@rad_map.T+9*scale**2*np.outer(rad_gain, rad_gain),
        'nis_degrees': nis, 'nis_radians': rad_innovation**2/rad_s,
        'nis_if_mean_only_converted': rad_innovation**2/(predicted_covariance[0, 0]+9.),
        'scope': 'local chart; radian calculation uses linear equations, not degree-only tracker API; NIS is one observation, not calibrated confidence'}

    q_half = white_acceleration_covariance(.1, 2.)
    f_half = np.array([[1., .1], [0., 1.]])
    decreasing = ConstantVelocityKalman([0., 0.], [[4., -1.], [-1., 1.]], np.zeros((2, 2)))
    decreasing.predict(1.)
    result['E09-18'] = {'density_deg2_s3': 2., 'half_interval_s': .1, 'Q_half': q_half,
        'Q_full': white_acceleration_covariance(.2, 2.),
        'composed_two_halves': f_half@q_half@f_half.T+q_half,
        'wrong_sum_without_state_propagation': 2*q_half,
        'wrong_Q_half_reused_for_full_interval': q_half,
        'negative_cross_prior': [[4., -1.], [-1., 1.]], 'prediction_without_observation': decreasing.covariance}

    _, audio = build_fixture()
    result['E09-19'] = {'audio_directory': 'codes/chapters/ch09/tracking_audio',
        'float_scores': audio['float_analysis']['scores'], 'pcm_scores': audio['pcm_analysis']['scores'],
        'common_export_gain': audio['common_export_gain'], 'config': audio['analysis_config'],
        'first_state_time_s': audio['pcm_analysis']['frames']['state_time_s'][0],
        'first_available_time_s': audio['pcm_analysis']['frames']['available_time_s'][0],
        'missing_state_times_s': [t for t, valid in zip(audio['pcm_analysis']['frames']['state_time_s'],
                                                     audio['pcm_analysis']['frames']['observation_valid']) if not valid]}
    return _plain(result)


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
