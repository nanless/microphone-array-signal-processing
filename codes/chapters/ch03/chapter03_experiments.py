"""E03-08..17: deterministic geometry, ambiguity and calibration calculations.

Run ``.venv/bin/python -m codes.chapters.ch03.chapter03_experiments``. The extra
E03-07 result and metadata are separate from the exercise IDs. No imports write
files. Free-field complex gains and analytic synthetic signals are teaching
models, not measurements, a hardware calibrator or third-party reproductions.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json
import hashlib
import io
from pathlib import Path
import wave
import numpy as np

from codes.chapters.ch03.core.geometry import plane_wave_delays, plane_wave_steering
from codes.chapters.ch00.core.audio_samples import dma_calibration_case, read_pcm16
from codes.chapters.ch03.coarray_covariance_exercise import average_ordered_lags, virtual_toeplitz
from codes.chapters.ch03.core.geometry_audio import geometry_parameters

ROOT = Path(__file__).resolve().parents[3]


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
    """E03-14: separate analytic float calculations and actual published PCM."""
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
    manifest = json.loads((ROOT/'codes/chapters/ch00/audio/MANIFEST.json').read_text())
    for path, digest in manifest['generator_inputs'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError('main audio generator source SHA is stale: '+path)
    records, pcm_signals = {}, {}
    for name in signals:
        filename = name+'.wav'
        entry = next((item for item in manifest['files'] if item['file'] == filename), None)
        if entry is None or entry['chapter'] != 'ch03' or entry['group'] != 'dma_calibration':
            raise ValueError('DMA file is missing or assigned to the wrong manifest group')
        blob = (ROOT/'codes/chapters/ch03/audio'/filename).read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError('DMA WAV SHA mismatch: '+filename)
        expected_channels = 2 if name.endswith('_array') else 1
        with wave.open(io.BytesIO(blob), 'rb') as reader:
            if (reader.getframerate(),reader.getnchannels(),reader.getnframes(),reader.getsampwidth(),reader.getcomptype()) != (16000,expected_channels,32000,2,'NONE'):
                raise ValueError('DMA WAV format mismatch: '+filename)
        if (entry['sample_rate_hz'],entry['channels'],entry['samples'],entry['common_export_gain']) != (16000,expected_channels,32000,1.):
            raise ValueError('DMA manifest format or common gain mismatch: '+filename)
        _, pcm = read_pcm16(blob)
        pcm_signals[name] = pcm
        records[filename] = {'sha256': entry['sha256'], 'sample_rate_hz':16000,
                             'channels':expected_channels,'samples_per_channel':32000,
                             'quantization_max_abs_error':float(np.max(abs(pcm-np.atleast_2d(signals[name]))))}
    pcm_rows=[]
    for label,frequency,(start,stop) in [('front',1000.,(2400,10400)),('back',1600.,(15200,23200))]:
        t=np.arange(start,stop)/16000
        basis=np.column_stack((np.cos(2*np.pi*frequency*t),np.sin(2*np.pi*frequency*t)))
        amplitudes={}
        for name,pcm in pcm_signals.items():
            coefficients=np.linalg.lstsq(basis,pcm[:,start:stop].T,rcond=None)[0]
            amplitudes[name]=np.sqrt(np.sum(coefficients**2,axis=0)).tolist()
        pcm_rows.append({'source':label,'frequency_hz':frequency,'scoring_interval_samples':[start,stop],
                         'samples_per_channel':stop-start,'projected_output_amplitudes':amplitudes})
    return {'parameters':params,'steady_interval_projection':rows,
            'float_measurements':{'steady_interval_projection':rows,'sample_counts_per_window':[8000,8000]},
            'pcm_measurements':{'files':records,'steady_interval_projection':pcm_rows,
                                'source_sha256':manifest['generator_inputs'],
                                'corrected_target_max_abs_difference':float(np.max(abs(pcm_signals['dma_calibration_corrected']-pcm_signals['dma_calibration_target'])))},
            'corrected_target_max_abs_difference':float(np.max(abs(signals['dma_calibration_corrected']-signals['dma_calibration_target']))),
            'limits':case['limits']}


def noisy_calibration_reference():
    """E03-15: exact four-outcome error-in-variables bias, not random accuracy."""
    s=np.array([1.,1.,-1.,-1.]);error=np.array([.5,-.5,.5,-.5]);z=s+error;y=2*s
    gain=relative_gain_ls(z,y).real
    residual=y-gain*z
    return {'source':s.tolist(),'reference_error':error.tolist(),'noisy_reference':z.tolist(),
            'observed_channel':y.tolist(),'true_gain':2.,'numerator':float(z@y),
            'denominator':float(z@z),'estimated_gain':gain,'residual':residual.tolist(),
            'residual_squared_sum':float(residual@residual),
            'true_gain_prediction_residual_squared_sum':float(np.sum((y-2*z)**2)),
            'reference_residual_inner_product':float(z@residual),
            'residual_mean_square':float(np.mean(residual**2)),
            'equal_outcome_count':4,'reference_error_mean':float(error.mean()),
            'reference_error_variance':float(np.mean(error**2)),
            'repeated_estimates':[{'repetitions':r,'samples':4*r,
                                   'estimated_gain':relative_gain_ls(np.tile(z,r),np.tile(y,r)).real} for r in (1,10,100)],
            'limits':'Four equal-weight outcomes with source and reference error uncorrelated. Replication leaves the biased population regression unchanged; this is not an experiment estimating random uncertainty.'}


def nearly_planar_geometry():
    """E03-16: full rank does not bound sensitivity or certify a unit vector."""
    u=np.array([.3,.4,np.sqrt(.75)]);rows=[]
    for height in (.04,.004,.0004):
        g=np.diag([.04,.04,height]);tau=-g@u/343
        measured=tau+np.array([0.,0.,1e-6]);raw=np.linalg.solve(g,-343*measured)
        rows.append({'height_m':height,'geometry_matrix_m':g.tolist(),'rank':int(np.linalg.matrix_rank(g)),
                     'singular_values_m':np.linalg.svd(g,compute_uv=False).tolist(),
                     'condition_number_2':float(np.linalg.cond(g)),
                     'true_delays_us':(tau*1e6).tolist(),'measured_delays_us':(measured*1e6).tolist(),
                     'mirror_normal_delay_separation_us':float(2*height*u[2]/343*1e6),
                     'raw_direction':raw.tolist(),'direction_error':(raw-u).tolist(),
                     'raw_direction_norm':float(np.linalg.norm(raw)),
                     'unit_norm_residual':float(np.linalg.norm(raw)-1),
                     'linear_equation_residual_max_m':float(np.max(abs(g@raw+343*measured)))})
    return {'true_direction':u.tolist(),'normal_delay_error_us':1.,'sound_speed_m_s':343.,'cases':rows,
            'limits':'The raw unconstrained linear solution is not silently normalized. Full rank removes a noiseless mirror ambiguity but does not guarantee stability. Unit-vector constrained fitting is an additional model/optimization step.'}


def multi_frequency_ambiguity():
    """E03-17: unknown complex source per frequency; two frequencies can help."""
    params=geometry_parameters();positions=np.asarray(params['positions_m']);angles=np.deg2rad(params['azimuths_deg'])
    candidates=plane_wave_steering(positions,[4000.,8000.],angles);rows=[]
    for index,frequency in enumerate((4000.,8000.)):
        z=candidates[0,index];other=candidates[1,index]
        alpha=np.vdot(other,z)/np.vdot(other,other).real;residual=z-alpha*other
        rows.append({'frequency_hz':frequency,'relative_steering':_encode(candidates[:,index,:]),
                     'max_relative_manifold_difference':float(np.max(abs(z-other))),
                     'unknown_complex_alpha':_encode(alpha),
                     'profile_residual_squared_sum':float(np.vdot(residual,residual).real),
                     'observed_phasor_squared_sum':float(np.vdot(z,z).real),
                     'profile_relative_residual':float(np.vdot(residual,residual).real/np.vdot(z,z).real)})
    # A distinct ULA example prevents claiming that an extra frequency always
    # removes phase aliases: 2k/4k are integer multiples of the same wrapping.
    spacing=.1;sine=.8575;pair=np.arcsin([sine,-sine]);ula=np.column_stack((spacing*np.arange(4),np.zeros(4)))
    counter=[]
    for frequency in (1000.,2000.,4000.):
        v=plane_wave_steering(ula,[frequency],pair)[:,0,:]
        coefficient=np.vdot(v[1],v[0])/4;delta=v[0]-coefficient*v[1]
        counter.append({'frequency_hz':frequency,'adjacent_difference_cycles':frequency*spacing*(2*sine)/343,
                        'max_relative_manifold_difference':float(np.max(abs(v[0]-v[1]))),
                        'profile_relative_residual':float(np.vdot(delta,delta).real/4)})
    from codes.chapters.ch03.examples.generate_geometry_audio import check_assets
    audio_path=ROOT/'codes/chapters/ch03/geometry_audio'
    actual=check_assets(audio_path)['samples']
    return {'ring_parameters':params,'ring_cases':rows,
            'ula_counterexample':{'spacing_m':spacing,'positions_m':ula.tolist(),'angles_deg':np.rad2deg(pair).tolist(),'cases':counter},
            'published_audio_measurements':actual,
            'limits':'The selected ring pair is distinguished at 4k but not 8k. A harmonic frequency pair need not remove an alias. Neither result removes planar mirror ambiguity or establishes noisy localization accuracy. Audio matching never uses the mono reference.'}


def fair_aperture_comparison():
    """Same M, physical aperture, frequency, target, weights and self-noise."""
    rows=[]
    for name,x in [('ula',np.linspace(0,.11,6)),('nested',np.array([0,.01,.02,.03,.07,.11]))]:
        positions=np.column_stack((x,np.zeros(6)))
        def response(theta):
            return np.abs(plane_wave_steering(positions,[4000.],theta)[:,0,:].mean(axis=-1))
        rows.append({'layout':name,'positions_m':positions.tolist(),'physical_aperture_m':.11,
                     'exact_hpbw_deg':_first_half_power(response,np.pi/2),'wng_linear':6.,
                     'wng_db':float(10*np.log10(6.))})
    return {'channels':6,'frequency_hz':4000.,'sound_speed_m_s':343.,'target_azimuth_deg':0.,
            'weights':[1/6]*6,'sensor_model':'synchronized ideal omnidirectional far field',
            'noise_model':'independent equal-power channel self-noise','cases':rows,
            'limits':'No DOA estimator is evaluated. Similar broadside half-power widths do not imply equal sidelobes, all-angle aliasing or source-resolution probability.'}


def run_exercises():
    return {f'E03-{number:02}':function() for number,function in enumerate(
        (exact_beamwidths,hexagon_ambiguity,planar_mirror,coupling_regularization,gain_least_squares,coprime_holes,dma_gain_mismatch,
         noisy_calibration_reference,nearly_planar_geometry,multi_frequency_ambiguity),8)}


if __name__=='__main__':
    print(json.dumps({'exercises':run_exercises(),'E03-07_extension':coarray_companion(),
                      'metadata':{'fair_aperture_comparison':fair_aperture_comparison()}},ensure_ascii=False,indent=2,allow_nan=False))
