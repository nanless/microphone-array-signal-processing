"""Teaching experiments E09-10..26; no writes on import/run.

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
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar
from codes.chapters.ch09.core.tracking import (
    CircularParticleFilter, ConstantVelocityKalman, wrap_angle,
    white_acceleration_covariance,
)
from codes.chapters.ch09.core.tracking_audio import analyze_array, read_pcm16
from codes.chapters.ch09.examples.chapter09_tracking_audio import OUTPUT as TRACKING_OUTPUT, generate
from codes.chapters.ch09.core.imm_teaching import imm_missing_observation_example
from codes.chapters.ch09.examples.tracking_lifecycle_demo import run_demo as lifecycle_demo
from codes.chapters.ch09.core.spherical_tracking import (
    direction_from_angles, angles_from_direction, rotate_direction,
    normalized_direction_covariance, vmf_static_update, vmf_resultant_length,
)


def small_set_distances(truth, estimate, *, cutoff=10., order=1):
    """OSPA and alpha=2 GOSPA on ≤7 scalar circular angles, p=1 or 2.

    Exhaustive enumeration is intentionally small. This scores unordered
    position sets, never identity continuity. Empty/empty is zero; one empty
    set has OSPA=c and GOSPA=(c^p*n/2)^(1/p). Compute roots in cutoff-normalized
    units, so a tiny positive c does not become zero by squaring it. A power
    cost outside float64 is None with an explicit status, never a false zero.
    Reject a lost nonzero normalized distance or final metric: this teaching
    solver does not claim arbitrary finite-scale support.
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
    best_root, best_distances = math.inf, ()
    for assignment in permutations(range(n), m):
        distances = tuple(min(cutoff, abs(float(wrap_angle(first[i]-second[j]))))
                          for i, j in enumerate(assignment))
        normalized = tuple(distance/cutoff for distance in distances)
        if any(d != 0 and unit == 0 for d, unit in zip(distances, normalized)):
            raise ValueError('nonzero normalized set distance exceeds solver support')
        root = math.fsum(normalized) if order == 1 else math.hypot(*normalized)
        if root < best_root:
            best_root, best_distances = root, distances
    # Keep the restored powers in Decimal until the final root. Apart from
    # protecting subnormal cutoffs, this preserves simple exact p=1 anchors
    # such as (2+10)/2=6 without normalized-domain restoration rounding.
    with localcontext() as context:
        context.prec = 80
        exact_power = sum((Decimal.from_float(d)**int(order) for d in best_distances), Decimal(0))
        unmatched = Decimal.from_float(cutoff)**int(order)*(n-m)
        ospa_power = (exact_power+unmatched)/n if n else Decimal(0)
        gospa_power = exact_power+unmatched/2
        ospa = float(ospa_power if order == 1 else ospa_power.sqrt())
        gospa = float(gospa_power if order == 1 else gospa_power.sqrt())
        power = float(exact_power)
    if any(exact != 0 and value == 0 for exact, value in ((ospa_power, ospa), (gospa_power, gospa))):
        raise ValueError('positive set metric exceeds floating-point range')
    status = 'representable'
    if exact_power != 0 and power == 0:
        power, status = None, 'positive_power_underflow'
    return {'ospa': ospa, 'gospa_alpha2': gospa,
            'matched_capped_cost_power': power, 'matched_capped_cost_power_status': status,
            'cardinality_difference': n-m}


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


def actual_tracking_audio(audio_directory=TRACKING_OUTPUT):
    """Strictly check published assets, then decode/analyze the actual WAV.

    generate(check=True) may replay in memory to validate the source contract;
    it never repairs files. The PCM result below is a new analysis of bytes
    read from the checked directory, not that in-memory replay's result.
    """
    directory = Path(audio_directory)
    manifest = generate(directory, check=True)
    array_bytes = (directory/'array_noisy.wav').read_bytes()
    analysis = analyze_array(read_pcm16(array_bytes), export_gain=manifest['common_export_gain'])
    return {'audio_directory': str(directory), 'asset_check': 'strict_read_only_passed',
        'files': manifest['files'], 'source_sha256': manifest['source_sha256'],
        'float_scores': manifest['float_analysis']['scores'], 'pcm_scores': analysis['scores'],
        'common_export_gain': manifest['common_export_gain'], 'config': manifest['analysis_config'],
        'first_state_time_s': analysis['frames']['state_time_s'][0],
        'first_available_time_s': analysis['frames']['available_time_s'][0],
        'missing_state_times_s': [t for t, valid in zip(analysis['frames']['state_time_s'],
                                                     analysis['frames']['observation_valid']) if not valid]}


def run_experiments(*, audio_directory=TRACKING_OUTPUT):
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
    first_log = supported.log_weights
    underflow = supported.weight_underflow_indices
    supported.update(0., .01, clutter_probability=0.)
    literal_zero = CircularParticleFilter([0., 10.])
    literal_zero.set_prior_weights([0., 1.])
    literal_zero.update(0., .01, clutter_probability=0.)
    result['E09-12'] = {'particles_deg': [179., -179.], 'linear_mean_deg': 0.,
        'circular_mean_deg': particles.estimate(), 'resultant_length': math.cos(math.radians(1)),
        'antipodal_status': status, 'antipodal_angle_deg': undefined_angle,
        'support_after_first': first, 'support_after_second': supported.weights,
        'log_weights_after_first': first_log, 'linear_underflow_indices_after_first': underflow,
        'log_weights_after_second': supported.log_weights,
        'explicit_zero_prior_after_update': literal_zero.weights,
        'prior_reset_method': 'set_prior_weights',
        'support_note': 'persistent finite log weights retain positive support; set_prior_weights explicitly resets support, including same-value displayed weights'}

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

    result['E09-19'] = actual_tracking_audio(audio_directory)

    # E20: two measurements of one static scalar state, not two time steps.
    observation_map = np.ones((2, 1))
    correlated_noise = np.array([[1., .9], [.9, 1.]])
    innovation_covariance = observation_map@observation_map.T+correlated_noise
    joint_gain = np.linalg.solve(innovation_covariance, observation_map).T
    independent_covariance = observation_map@observation_map.T+np.eye(2)
    independent_gain = np.linalg.solve(independent_covariance, observation_map).T
    result['E09-20'] = {'prior_mean': 0., 'prior_variance': 1., 'observations': [1., 1.],
        'measurement_noise_covariance': correlated_noise, 'innovation_covariance': innovation_covariance,
        'joint_gain': joint_gain[0], 'joint_mean': float((joint_gain@np.ones(2))[0]),
        'joint_variance': float((1-joint_gain@observation_map)[0, 0]),
        'wrong_independent_mean': float((independent_gain@np.ones(2))[0]),
        'wrong_independent_variance': float((1-independent_gain@observation_map)[0, 0]),
        'perfect_correlation_limit': {'mean': .5, 'variance': .5},
        'scope': 'static state and correlated zero-mean Gaussian noise; rho=1 is a singular limit'}

    initial = np.array([1., 2.]); velocity = np.array([1., 0.]); times = np.arange(3.)
    positions = initial+times[:, None]*velocity
    rows = np.column_stack((positions[:, 1], -positions[:, 0],
                            times*positions[:, 1], -times*positions[:, 0]))
    scale_null = np.r_[initial, velocity]
    result['E09-21'] = {'array_center_m': [0., 0.], 'initial_position_m': initial,
        'velocity_m_s': velocity, 'scale_factor': 2., 'times_s': times,
        'positions_m': positions, 'scaled_positions_m': 2*positions,
        'bearings_deg': np.rad2deg(np.arctan2(positions[:, 0], positions[:, 1])),
        'scaled_bearings_deg': np.rad2deg(np.arctan2(2*positions[:, 0], 2*positions[:, 1])),
        'ranges_m': np.hypot(positions[:, 0], positions[:, 1]),
        'scaled_ranges_m': 2*np.hypot(positions[:, 0], positions[:, 1]),
        'stacked_jacobian_without_positive_row_denominators': rows,
        'stacked_rank': int(np.linalg.matrix_rank(rows)), 'scale_null_direction': scale_null,
        'stacked_null_product': rows@scale_null,
        'scope': 'fixed array, unknown constant Cartesian velocity; instantaneous geometric bearing at source-state time, propagation delay ignored; not equivalence of receiver-clock retarded audio'}

    prior_existence = .8
    cases = []
    for detection_probability in (.9, .1):
        empty_probability = 1-prior_existence*detection_probability
        cases.append({'detection_probability': detection_probability,
            'empty_observation_probability': empty_probability,
            'bernoulli_posterior_existence': prior_existence*(1-detection_probability)/empty_probability,
            'different_poisson_phd_posterior_mass': prior_existence*(1-detection_probability)})
    result['E09-22'] = {'prior_existence': prior_existence, 'cases': cases,
        'scope': 'at most one target, no clutter/birth/survival transition in this update, known constant detection probability; not VAD or identity'}

    # E23: joint conditioning and chronological replay must agree at time 2.
    observation_covariance = np.array([[3., 2.], [2., 4.]])
    final_state_observation_cross = np.array([2., 3.])
    joint_gain = np.linalg.solve(observation_covariance, final_state_observation_cross)
    replay_mean, replay_variance = 0., 1.
    replay = []
    for observation in (2., -1.):
        prior_variance = replay_variance+1
        gain = prior_variance/(prior_variance+1)
        replay_mean += gain*(observation-replay_mean)
        replay_variance = (1-gain)*prior_variance
        replay.append({'mean': replay_mean, 'variance': replay_variance})
    drop_mean, drop_variance = -.75, .75
    wrong_gain = drop_variance/(drop_variance+1)
    result['E09-23'] = {'P0': 1., 'Q': 1., 'R': 1., 'observations_by_measurement_time': [2., -1.],
        'arrival_order_measurement_times': [2, 1], 'final_state_time': 2,
        'joint_observation_covariance': observation_covariance,
        'final_state_observation_cross_covariance': final_state_observation_cross,
        'joint_gain': joint_gain, 'joint_mean': float(joint_gain@np.array([2., -1.])),
        'joint_variance': float(3-joint_gain@final_state_observation_cross),
        'chronological_replay': replay,
        'drop_late_observation': {'mean': drop_mean, 'variance': drop_variance},
        'wrong_current_time_update': {'mean': drop_mean+wrong_gain*(2-drop_mean),
                                      'variance': (1-wrong_gain)*drop_variance},
        'scope': 'scalar Euclidean random walk; chronological replay uses original measurement times once, not a generic out-of-sequence tracker'}
    result['E09-24'] = imm_missing_observation_example()
    result['E09-25'] = lifecycle_demo(audio_directory)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    array_direction = direction_from_angles(30., 0.)
    world_direction = rotate_direction(array_direction, rotation)
    sphere_update = vmf_static_update([0., 1., 0.], 3., [1., 0., 0.], 4.)
    result['E09-26'] = {
        'array_angles_deg': [30., 0.], 'array_direction': array_direction,
        'array_to_world': rotation, 'world_direction': world_direction,
        'world_angles': angles_from_direction(world_direction),
        'north_pole_from_two_azimuths': [direction_from_angles(0., 90.), direction_from_angles(120., 90.)],
        'north_pole_angles': angles_from_direction([0., 0., 1.]),
        'local_normalization': normalized_direction_covariance([0., 2., 0.], np.eye(3)*.04),
        'static_vmf_posterior': sphere_update,
        'posterior_angles': angles_from_direction(sphere_update['mean_direction']),
        'posterior_resultant_length': vmf_resultant_length(5.),
        'uniform_polar_cap_10deg_probability': (1-math.cos(math.radians(10)))/2,
        'equal_theta_phi_grid_density_ratio_at_80deg': 1/math.cos(math.radians(80)),
        'scope': 'known proper rotation and one static independent vMF likelihood; not dynamic FvMFF, pose estimation or a calibrated tangent filter',
    }
    return _plain(result)


if __name__ == '__main__':
    print(json.dumps(run_experiments(), ensure_ascii=False, indent=2, allow_nan=False))
