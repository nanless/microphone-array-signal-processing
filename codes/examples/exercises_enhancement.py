"""Deterministic chapter 6--9 exercises; no files, downloads or work at import.

Run from the root with ``python -m codes.examples.exercises_enhancement``.
All signals are mathematical fixtures, not measured speech-quality benchmarks.
"""

from __future__ import annotations

import json
import numpy as np

from codes.array_tutorial.aec import erle_db, nlms
from codes.array_tutorial.dereverberation import offline_wpe
from codes.array_tutorial.separation import masked_spatial_covariance, pit_permutation, si_sdr
from codes.array_tutorial.tracking import ConstantVelocityKalman, systematic_resample, wrap_angle


def run_exercises() -> dict:
    """Return twenty JSON-safe exercises with stable IDs and stated inputs."""
    results = {}
    x = np.array([1., 0., 0.])
    d = np.array([.8, -.2, .1])
    residual, echo, weights = nlms(x, d, 3, step_size=.5, epsilon=0)
    results['E06-01'] = dict(reference=x.tolist(), microphone=d.tolist(), residual=residual.tolist(), echo_hat=echo.tolist(), weights=weights.tolist())

    delayed = np.array([0., 0., 1., 0.])
    short = nlms(np.array([1., 0., 0., 0.]), delayed, 2, step_size=1, epsilon=0)[0]
    long = nlms(np.array([1., 0., 0., 0.]), delayed, 3, step_size=1, epsilon=0)
    results['E06-02'] = dict(delay_samples=2, short_filter_length=2, short_residual=short.tolist(), long_weights=long[2].tolist(), note='The first delayed impulse is output before its weight is updated.')

    far = np.ones(4)
    mic = np.array([1., 1., 2., 2.])
    mask = np.array([False, False, True, True])
    frozen = nlms(far, mic, 1, step_size=.5, epsilon=0, freeze=mask, initial_weights=np.array([.9]))
    active = nlms(far, mic, 1, step_size=.5, epsilon=0, initial_weights=np.array([.9]))
    results['E06-03'] = dict(freeze_mask=mask.tolist(), frozen_residual=frozen[0].tolist(), frozen_weights=frozen[2].tolist(), active_weights=active[2].tolist(), valid_erle_db=erle_db(mic, frozen[0], double_talk_mask=mask))

    delay, taps, channels, frames, hop_ms = 3, 2, 2, 12, 8
    first = delay + taps - 1
    results['E07-01'] = dict(first_valid_frame=first, valid_frames=frames-first, regression_dimension=channels*taps, first_history_frames=[first-delay-k for k in range(taps)], latest_delay_ms=delay*hop_ms, oldest_delay_ms=(delay+taps-1)*hop_ms)

    predictor = .5 + .25j
    series = np.array([(np.conj(predictor))**t for t in range(12)], dtype=complex)[None, :]
    output = offline_wpe(series, taps=1, delay=1, iterations=1, diagonal_loading=0)
    results['E07-02'] = dict(predictor_real_imag=[predictor.real,predictor.imag], first_output_real_imag=[float(output[0,0].real),float(output[0,0].imag)], maximum_valid_residual=float(np.max(np.abs(output[0,1:]))))

    correlation = np.ones((2, 2))
    loaded = correlation + .1*np.trace(correlation)/2*np.eye(2)
    results['E07-03'] = dict(raw_eigenvalues=np.linalg.eigvalsh(correlation).tolist(), loaded_eigenvalues=np.linalg.eigvalsh(loaded).tolist(), loaded_condition_number=float(np.linalg.cond(loaded)), loaded_solution=np.linalg.solve(loaded, np.ones(2)).tolist())

    target = np.array([1., 0., -1., 0.])
    orthogonal = np.array([0., 1., 0., -1.])
    estimate = target + .1*orthogonal
    results['E08-01'] = dict(reference=target.tolist(), estimate=estimate.tolist(), si_sdr_db=si_sdr(estimate,target), quiet_si_sdr_db=si_sdr(estimate*1e-5,target*1e-5), one_sample_circular_shift_db=si_sdr(np.roll(target,1),target), perfect_db=si_sdr(target,target))

    references = np.stack([target, orthogonal])
    estimates = np.stack([orthogonal+.1*target, target+.1*orthogonal])
    permutation, score = pit_permutation(estimates,references)
    results['E08-02'] = dict(output_to_reference=list(permutation), mean_si_sdr_db=score, identity_fixed_mean_db=float(np.mean([si_sdr(estimates[i],references[i]) for i in range(2)])))

    mixing = np.array([[1., .5],[.5, 1.]])
    near_singular = np.array([[1., .99],[.99, 1.]])
    recovered = np.linalg.solve(mixing,mixing@references)
    perturbation = np.array([.01,-.01])
    results['E08-03'] = dict(mixing_matrix=mixing.tolist(), condition_number=float(np.linalg.cond(mixing)), reconstruction_max_error=float(np.max(np.abs(recovered-references))), microphone_error=perturbation.tolist(), recovered_error=np.linalg.solve(mixing,perturbation).tolist(), near_singular_condition_number=float(np.linalg.cond(near_singular)), near_singular_recovered_error=np.linalg.solve(near_singular,perturbation).tolist(), blind=False)

    kalman = ConstantVelocityKalman([30.,5.], np.diag([4.,1.]),np.diag([.1,.01]))
    history=[]
    for _ in range(2):
        kalman.predict(.1)
        history.append(dict(state=kalman.state.tolist(),covariance=kalman.covariance.tolist()))
    results['E09-01'] = dict(dt_seconds=.1, predictions=history, note='Q is discretized for this 0.1 s interval; no observations are assimilated.')

    circular = ConstantVelocityKalman([179.,0.],np.diag([1.,1.]),np.zeros((2,2)))
    circular.update(-179.,1.)
    results['E09-02'] = dict(raw_innovation_degrees=-358., wrapped_innovation_degrees=wrap_angle(-358.), updated_angle_degrees=float(circular.state[0]), updated_angle_variance=float(circular.covariance[0,0]))

    particle_weights=np.array([.1,.2,.6,.1])
    rng=np.random.default_rng(0)
    first_position=float(np.random.default_rng(0).random()/4)
    ancestors=systematic_resample(particle_weights,rng)
    results['E09-03'] = dict(weights=particle_weights.tolist(), effective_sample_size=float(1/np.sum(particle_weights**2)), default_resampling_threshold=2., threshold_would_resample=bool(1/np.sum(particle_weights**2)<2), demonstration_forces_resampling=True, first_position=first_position, ancestors_zero_based=ancestors.tolist(), seed=0)

    phases = np.array([0., 60., 180.])
    residual_ratio = np.abs(1 - np.exp(1j*np.deg2rad(phases)))**2
    results['E06-04'] = dict(phase_degrees=phases.tolist(), residual_energy_ratio=residual_ratio.tolist(),
                            erle_db=[None, float(-10*np.log10(residual_ratio[1])), float(-10*np.log10(residual_ratio[2]))],
                            zero_phase_note='Exact zero residual has infinite ideal ERLE; null avoids non-finite JSON.')
    echo = np.array([1., 1., -1., -1.]) / 2
    noise = np.array([1., -1., 1., -1.]) / 20
    results['E06-05'] = dict(echo_energy=float(echo@echo), noise_energy=float(noise@noise),
                            cross_inner_product=float(echo@noise),
                            measured_input_output_ratio_db=erle_db(echo+noise, noise),
                            echo_component_residual_energy=0.)

    record_a = np.array([[1., 1., 1., 1.]], dtype=complex)
    record_b = np.array([[1., 1., 1., 2.]], dtype=complex)
    options = dict(taps=1, delay=1, iterations=1, diagonal_loading=0)
    result_a = offline_wpe(record_a, **options)
    result_b = offline_wpe(record_b, **options)
    results['E07-04'] = dict(record_a=record_a.real.tolist(), record_b=record_b.real.tolist(),
                            common_prefix_frames=3, output_a=result_a.real.tolist(), output_b=result_b.real.tolist(),
                            first_valid_difference=float((result_b-result_a)[0,1].real), causal=False)
    correlation = np.array([[2.,0.,1.,0.], [0.,1.,0.,0.], [1.,0.,2.,0.], [0.,0.,0.,1.]])
    steering = np.array([1.,1.,0.,0.])
    inverse_steering = np.linalg.solve(correlation, steering)
    wpd = inverse_steering / (steering@inverse_steering)
    current_only = np.array([1/3,2/3,0.,0.])
    results['E07-05'] = dict(covariance=correlation.tolist(), weights=wpd.tolist(),
                            constraint=float(wpd@steering), objective=float(wpd@correlation@wpd),
                            current_only_objective=float(current_only@correlation@current_only))

    mixing = np.array([[1.,.5],[.5,1.]])
    demixing = np.diag([2.,-3.]) @ np.linalg.inv(mixing)
    separated = demixing @ mixing @ references
    projection_factors = np.linalg.inv(demixing)[0]
    projected = projection_factors[:,None] * separated
    results['E08-04'] = dict(mixing=mixing.tolist(), demixing=demixing.tolist(),
                            projection_factors=projection_factors.tolist(),
                            raw_outputs=separated.tolist(), reference_images=projected.tolist(), blind=False)
    epsilon = 1e-12
    spectrum = np.array([[[2.,4.]]], dtype=complex)
    mask_scales = [1., 2., epsilon/2, epsilon/4]
    covariances = [float(masked_spatial_covariance(spectrum, np.full((1,2), scale), epsilon=epsilon)[0,0,0].real)
                  for scale in mask_scales]
    results['E08-05'] = dict(epsilon=epsilon, equal_mask_weights=mask_scales, floored_covariance=covariances,
                            additive_epsilon_covariance_at_floor=5.)

    dt, density, acceleration_variance = .1, 2., 4.
    continuous = density*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
    impulse = np.array([dt**2/2,dt])
    discrete = acceleration_variance*np.outer(impulse,impulse)
    results['E09-04'] = dict(dt_seconds=dt, continuous_density=density, acceleration_variance=acceleration_variance,
                            continuous_Q=continuous.tolist(), piecewise_constant_Q=discrete.tolist())
    raw_innovation, innovation_variance, gate = -358., 4., 9.
    wrapped = wrap_angle(raw_innovation)
    results['E09-05'] = dict(raw_innovation=raw_innovation, wrapped_innovation=wrapped,
                            raw_squared_mahalanobis=raw_innovation**2/innovation_variance,
                            wrapped_squared_mahalanobis=wrapped**2/innovation_variance,
                            example_gate=gate, raw_accepted=bool(raw_innovation**2/innovation_variance<=gate),
                            wrapped_accepted=bool(wrapped**2/innovation_variance<=gate))
    return results


if __name__ == '__main__':
    print(json.dumps(run_exercises(),ensure_ascii=False,indent=2,allow_nan=False))
