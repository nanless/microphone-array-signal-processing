"""E03-08..14: deterministic geometry, ambiguity and calibration calculations.

Run ``.venv/bin/python -m codes.chapters.ch03.chapter03_experiments``. The extra
E03-07 result is separate from the seven new exercise IDs. No imports write
files. Free-field complex gains and analytic synthetic signals are teaching
models, not measurements, a hardware calibrator or third-party reproductions.
"""
from __future__ import annotations

import json
import numpy as np

from codes.array_tutorial.geometry import plane_wave_delays, plane_wave_steering
from codes.array_tutorial.audio_samples import dma_calibration_case
from codes.chapters.ch03.coarray_covariance_exercise import average_ordered_lags, virtual_toeplitz


def _encode(values):
    a = np.asarray(values)
    return {'real': a.real.tolist(), 'imag': a.imag.tolist()}


def _first_half_power(response, limit):
    """Find first crossing around a symmetric broadside main lobe.

    Fixed 0..limit scan brackets the first root; 60 bisections refine it.
    None denotes no crossing in that explicitly stated angular domain.
    """
    grid = np.linspace(0., limit, 10001)
    values = response(grid)**2
    indices = np.flatnonzero(values <= .5)
    if not indices.size:
        return None
    index = int(indices[0])
    low, high = grid[index-1], grid[index]
    for _ in range(60):
        mid = (low+high)/2
        if response(np.array([mid]))[0]**2 > .5:
            low = mid
        else:
            high = mid
    return float(np.rad2deg(low+high))


def exact_beamwidths():
    """E03-08: first half-power crossings, not a borrowed aperture formula."""
    rows = []
    ring_angles = np.arange(6)*np.pi/3
    layouts = {
        'ula': np.column_stack((.03*np.arange(4), np.zeros(4))),
        'hexagon': .0375*np.column_stack((np.cos(ring_angles), np.sin(ring_angles))),
    }
    for name, positions in layouts.items():
        for frequency in ((1000., 8000.) if name == 'ula' else (1000., 4000.)):
            target = plane_wave_steering(positions, [frequency], 0.)[0]
            def response(angles):
                a = plane_wave_steering(positions, [frequency], angles)[:, 0, :]
                return np.abs(a @ target.conj()/len(positions))
            limit = np.pi/2 if name == 'ula' else np.pi
            rows.append({'layout': name, 'frequency_hz': frequency,
                         'one_sided_domain_limit_deg': float(np.rad2deg(limit)),
                         'exact_hpbw_deg': _first_half_power(response, limit),
                         'domain_edge_amplitude': float(response(np.array([limit]))[0]),
                         'ula_narrow_lobe_approx_deg': float(np.rad2deg(.886*343/frequency/(4*.03))) if name == 'ula' else None})
    grating_checks = []
    frequency, spacing = 8000., .03
    for target_deg in (0., 60.):
        target_rad = np.deg2rad(target_deg)
        target = plane_wave_steering(layouts['ula'], [frequency], target_rad)[0]
        candidates = []
        for integer in (-1, 1):
            sine = np.sin(target_rad) + integer*343/(frequency*spacing)
            visible = bool(abs(sine) <= 1.)
            angle = float(np.arcsin(sine)) if visible else None
            amplitude = None
            if visible:
                other = plane_wave_steering(layouts['ula'], [frequency], angle)[0]
                amplitude = float(abs(np.vdot(target, other))/4)
            candidates.append({'integer_cycle_order': integer, 'sine_argument': float(sine),
                               'visible': visible,
                               'angle_deg': float(np.rad2deg(angle)) if visible else None,
                               'dsb_amplitude': amplitude})
        grating_checks.append({'frequency_hz': frequency, 'target_deg': target_deg,
                               'candidates': candidates})
    return {'sound_speed_m_s': 343., 'target_azimuth_deg': 0.,
            'threshold_power_ratio': .5, 'threshold_db': float(10*np.log10(.5)),
            'cases': rows, 'grating_checks': grating_checks,
            'limits': 'Unit-gain, phase-only free-field DSB. Width is around the selected main lobe; it does not certify alias-free steering.'}


def hexagon_ambiguity():
    """E03-09: construct two exactly equal six-channel relative manifolds."""
    radius = .0375
    ring = radius*np.column_stack((np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)))
    critical = 343/(np.sqrt(3)*radius)
    rows = []
    for f in (critical, 8000.):
        uy = 343/(np.sqrt(3)*radius*f)
        ux = np.sqrt(max(0., 1-uy*uy))
        angles = np.arctan2([ux, ux], [uy, -uy])
        steering = plane_wave_steering(ring, [f], angles)[:, 0, :]
        rows.append({'frequency_hz': f, 'directions_xy': [[float(ux), float(uy)], [float(ux), float(-uy)]],
                     'azimuths_deg': np.rad2deg(angles).tolist(),
                     'relative_steering': _encode(steering),
                     'max_manifold_difference': float(np.max(abs(steering[0]-steering[1])))})
    low = plane_wave_steering(ring, [1000.], [0., np.pi])[:, 0, :]
    return {'positions_m': ring.tolist(), 'reference_channel_zero_based': 0,
            'conservative_check_frequency_hz': 343/(2*radius), 'critical_frequency_hz': critical,
            'cases': rows, 'opposite_directions_difference_at_1khz': float(np.max(abs(low[0]-low[1]))),
            'limits': 'Explicit ambiguous pairs establish existence, not a numerical proof that all other frequencies/directions are identifiable.'}


def planar_mirror():
    """E03-10: a plane loses the sign of the direction's normal component."""
    azimuth = np.arctan2(.3, .4)
    elevation = np.arcsin(np.sqrt(.75))
    positions = np.array([[0,0,0],[.04,0,0],[0,.04,0],[0,0,.04]])
    delays = plane_wave_delays(positions, [azimuth, azimuth], elevation_rad=[elevation, -elevation])
    return {'positions_m': positions.tolist(), 'directions': [[.3,.4,float(np.sqrt(.75))],[.3,.4,float(-np.sqrt(.75))]],
            'relative_delays_us': (delays*1e6).tolist(), 'planar_max_difference_us': float(np.max(abs(delays[0,:3]-delays[1,:3]))*1e6),
            'added_mic_difference_us': float(abs(delays[0,3]-delays[1,3])*1e6),
            'limits': 'The extra microphone distinguishes this mirror pair; it is not a guarantee against all narrowband phase aliases.'}


def coupling_regularization():
    """E03-11: measurement noise is added AFTER the known mixing matrix."""
    c = np.array([[1.,.9],[.9,1.]])
    eta = .01
    inverse = np.linalg.solve(c, np.eye(2))
    regularized = np.linalg.solve(c.T@c+eta*np.eye(2), c.T)
    rows = []
    for name, operator in [('inverse',inverse),('tikhonov',regularized)]:
        transfer = operator@c
        noise_cov = operator@operator.T
        rows.append({'method': name, 'operator': operator.tolist(), 'signal_transfer': transfer.tolist(),
                     'post_mix_noise_output_variances': np.diag(noise_cov).tolist(),
                     'common_mode_signal_gain': float(np.array([1.,1.])@transfer@np.array([1.,1.])/2),
                     'difference_mode_signal_gain': float(np.array([1.,-1.])@transfer@np.array([1.,-1.])/2)})
    return {'mixing_matrix': c.tolist(), 'regularization_eta': eta, 'condition_number_2': float(np.linalg.cond(c)),
            'input_post_mix_noise_covariance': np.eye(2).tolist(), 'cases': rows,
            'limits': 'Dimensionless two-channel linear mixing. Pre-mixing sensor noise has a different transfer. Regularization trades signal bias for lower post-mixing noise; eta is illustrative.'}


def relative_gain_ls(prediction, observation):
    """Fit observation ~= gain*prediction for finite 1-D complex vectors.

    prediction is the known ideal channel spectrum, not an unknown angle.
    Zero reference energy is rejected. A common safe scale leaves the ratio
    unchanged; nonrepresentable gain is rejected rather than returned as NaN.
    """
    z, y = np.asarray(prediction,dtype=complex), np.asarray(observation,dtype=complex)
    if z.ndim != 1 or not z.size or y.shape != z.shape or not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
        raise ValueError('require matching finite nonempty complex vectors')
    scale = max(float(np.max(np.abs(z.real))), float(np.max(np.abs(z.imag))),
                float(np.max(np.abs(y.real))), float(np.max(np.abs(y.imag))))
    if scale == 0:
        raise ValueError('reference energy must be positive')
    zs, ys = z/scale, y/scale
    denominator = np.vdot(zs,zs).real
    if denominator <= 0:
        raise ValueError('reference energy is zero or not representable relative to observations')
    gain = np.vdot(zs,ys)/denominator
    if not np.isfinite(gain):
        raise ValueError('gain exceeds supported numerical range')
    return complex(gain)


def gain_least_squares():
    """E03-12: complex conjugation and finite noisy calibration residuals."""
    z = np.array([1.,2j]); true_gain = 1+.5j
    noise = np.array([.1,-.2j]); y = true_gain*z+noise
    estimate = relative_gain_ls(z,y); residual = y-estimate*z
    return {'ideal_reference': _encode(z), 'observed': _encode(y), 'perturbations': _encode(noise),
            'numerator': _encode(np.vdot(z,y)), 'denominator': float(np.vdot(z,z).real),
            'true_gain': _encode(true_gain), 'estimated_gain': _encode(estimate),
            'residual': _encode(residual), 'residual_squared_sum': float(np.vdot(residual,residual).real),
            'normal_equation_residual': _encode(np.vdot(z,residual)),
            'limits': 'Two prescribed noisy phasors, not statistical accuracy. This assumes known geometry and a direction-independent relative complex gain.'}


def coprime_holes():
    """E03-13: distinguish total lags, aperture and hole-free segment."""
    rows = []
    for name, positions in [('ula',[0,1,2,3,4,5]),('nested',[0,1,2,3,7,11]),('coprime',[0,3,4,6,8,9])]:
        p=np.asarray(positions); lags, counts=np.unique(p[:,None]-p[None,:],return_counts=True)
        lag_set=set(lags.tolist()); end=0
        while end+1 in lag_set and -end-1 in lag_set:
            end+=1
        holes=sorted(set(range(int(lags[0]),int(lags[-1])+1))-lag_set)
        rows.append({'layout':name,'positions_half_wavelength':positions,'physical_aperture_grid_units':int(p.max()-p.min()),
                     'lags':lags.tolist(),'multiplicities':counts.tolist(),'holes':holes,
                     'unique_lag_count':len(lags),'central_contiguous_lag_count':2*end+1,
                     'direct_toeplitz_size':end+1,'ordered_pair_count':int(counts.sum())})
    return {'cases':rows,'limits':'The grid unit is lambda/2 at one reference frequency. Equal microphone count does not mean equal aperture, independent observations or equal DOA performance.'}


def _lag_smoothing(lags, size):
    """Average overlapping ascending-lag slices; do not reverse each slice."""
    z = np.array([lags[lag] for lag in range(1-size, size)], dtype=complex)
    # These ascending-lag slices are the direct Toeplitz matrix's columns,
    # ordered from last to first. The outer-product sum ignores column order.
    return sum(np.outer(z[start:start+size], z[start:start+size].conj())
               for start in range(size))/size


def coarray_companion():
    """E03-07 extension: coherent sources and the fourth-order SS matrix."""
    p=np.array([0,1,3]);x=np.array([1,0,1],complex)
    lags=average_ordered_lags(np.outer(x,x.conj()),p);t=virtual_toeplitz(lags,4)
    ss = _lag_smoothing(lags, 4)
    ideal_lags = {0: 2.1, 1: 1+1j, 2: 0, 3: 1-1j,
                  -1: 1-1j, -2: 0, -3: 1+1j}
    ideal_ss = _lag_smoothing(ideal_lags, 4)
    coherent=np.array([2,1+1j,0]);r=np.outer(coherent,coherent.conj())
    return {'coherent_snapshot':_encode(coherent),'coherent_same_lag_R10_R21':_encode([r[1,0],r[2,1]]),
            'finite_sample_direct_matrix':_encode(t),'finite_sample_ss_matrix':_encode(ss),
            'direct_eigenvalues':np.linalg.eigvalsh(t).tolist(),'ss_eigenvalues':np.linalg.eigvalsh(ss).tolist(),
            'ideal_complex_ss_matrix': _encode(ideal_ss),
            'ideal_complex_ss_eigenvalues': np.linalg.eigvalsh(ideal_ss).tolist(),
            'ss_product_residual_max_abs':float(np.max(abs(ss-t@t.conj().T/4))),
            'ss_units':'fourth power of the physical signal amplitude; not the original power covariance',
            'limits':'PSD follows from outer products. Squaring the spectrum does not repair an invalid source/noise model or create extra recordings.'}


def dma_gain_mismatch():
    """E03-14: use the same analytic source model as the four audio fixtures."""
    case=dma_calibration_case();signals=case['signals'];params=case['parameters']
    rows=[]
    for label,frequency,(start,stop) in [('front',1000.,(2400,10400)),('back',1600.,(15200,23200))]:
        t=np.arange(start,stop)/16000
        amplitudes={}
        for name in ('target','mismatch','corrected'):
            x=signals['dma_calibration_'+name][start:stop]
            amplitudes[name]=float(abs(2*np.mean(x*np.exp(-2j*np.pi*frequency*t))))
        rows.append({'source':label,'frequency_hz':frequency,'scoring_interval_samples':[start,stop],
                     'projected_output_amplitudes':amplitudes})
    return {'parameters':params,'steady_interval_projection':rows,
            'corrected_target_max_abs_difference':float(np.max(abs(signals['dma_calibration_corrected']-signals['dma_calibration_target']))),
            'limits':case['limits']}


def run_exercises():
    return {f'E03-{number:02}':function() for number,function in enumerate(
        (exact_beamwidths,hexagon_ambiguity,planar_mirror,coupling_regularization,gain_least_squares,coprime_holes,dma_gain_mismatch),8)}


if __name__=='__main__':
    print(json.dumps({'exercises':run_exercises(),'E03-07_extension':coarray_companion()},ensure_ascii=False,indent=2,allow_nan=False))
