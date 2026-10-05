"""Independent algebra, real PCM and strict asset-contract tests for E12."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from codes.chapters.ch02.core.spectral import stft, periodic_hann
from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch12.core.imaging import (
    spherical_steering, conventional_weights, source_power_csm, csm_from_amplitudes,
    scan_power, point_spread_function, damas_gauss_seidel, finite_nnls,
    clean_sc_full_csm, hermitian_real_vector, csm_residual, single_source_csm_fit,
    one_sided_csm_density, integrate_psd, region_power, two_cell_experiment,
    spherical_scan_experiment,
)
from codes.chapters.ch12.core.imaging_audio import (
    make_signals, extract_snapshot_amplitudes, measure_signal, measure_source_pairs,
    SAMPLE_RATE, SAMPLES, POWER_SCALE, FILE_NAMES,
)
from codes.chapters.ch12.examples.generate_imaging_audio import prepare_assets, generate_assets, check_assets
from codes.chapters.ch12.chapter12_exercises import run_experiments


class ImagingAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[1., 1.], [1., np.exp(-2j*np.pi/3)]])
        self.w = self.a/2
        self.p = np.array([[1., .25], [.25, 1.]])

    def test_psf_csm_and_units(self):
        np.testing.assert_allclose(point_spread_function(self.w, self.a), self.p, atol=2e-15)
        r = source_power_csm(self.a, [1., .25])
        expected = np.array([[1.25, .875+1j*np.sqrt(3)/8], [.875-1j*np.sqrt(3)/8, 1.25]])
        np.testing.assert_allclose(r, expected, atol=2e-15)
        np.testing.assert_allclose(scan_power(r, self.w), [17/16, .5], atol=2e-15)
        np.testing.assert_allclose(conventional_weights(self.a), self.w)
        np.testing.assert_allclose(csm_from_amplitudes([[.2, .2]]), [[.02]])
        np.testing.assert_allclose(source_power_csm(self.a, [0., 0.]), np.zeros((2, 2)))

    def test_forward_and_double_pass_are_distinct(self):
        forward = damas_gauss_seidel(self.p, [17/16, .5], iterations=2)
        np.testing.assert_allclose(forward['history'][1], [17/16, 15/64])
        np.testing.assert_allclose(forward['history'][2], [257/256, 255/1024])
        both = damas_gauss_seidel(self.p, [17/16, .5], iterations=1, sweep='forward_backward')
        np.testing.assert_allclose(both['q'], [257/256, 15/64])
        self.assertEqual(both['passes_per_iteration'], 2)
        scaled = damas_gauss_seidel(2*self.p, [17/8, 1.], iterations=2)
        np.testing.assert_array_equal(scaled['history'], forward['history'])
        initial = np.zeros(2)
        damas_gauss_seidel(self.p, [17/16, .5], initial=initial)
        np.testing.assert_array_equal(initial, [0., 0.])

    def test_slow_convergence_and_duplicate_columns(self):
        rho = .99; p = np.array([[1., rho], [rho, 1.]])
        q = damas_gauss_seidel(p, p@np.array([1., .25]), iterations=100)['q']
        self.assertAlmostEqual(q[0]-1, .25*rho**199, places=13)
        duplicate = damas_gauss_seidel(np.ones((2, 2)), [1.25, 1.25], iterations=1)
        np.testing.assert_array_equal(duplicate['q'], [1.25, 0.])

    def test_nnls_is_not_gs_and_has_kkt(self):
        result = finite_nnls(self.p, [1., 0.])
        np.testing.assert_allclose(result['q'], [16/17, 0.], atol=1e-15)
        self.assertAlmostEqual(result['squared_residual'], 1/17)
        self.assertLess(result['scaled_kkt_violation'], 1e-14)
        np.testing.assert_array_equal(damas_gauss_seidel(self.p, [1., 0.])['q'], [1., 0.])
        zero = finite_nnls(np.zeros((2, 2)), [1., 0.])
        np.testing.assert_array_equal(zero['q'], [0., 0.])
        np.testing.assert_allclose(finite_nnls(1e100*self.p, [1e100, 0.])['q'], result['q'])
        with self.assertRaises(ValueError):
            finite_nnls(np.ones((2, 441)), np.ones(2))

    def test_coherent_zero_scan_residual_retains_matrix_error(self):
        case = two_cell_experiment()['cases']['coherent']
        np.testing.assert_allclose(case['b'], [21/16, 3/4])
        np.testing.assert_allclose(case['q_inverse'], [1.2, .45])
        np.testing.assert_allclose(self.p@case['q_inverse'], case['b'])
        self.assertAlmostEqual(case['full_csm_residual']['relative_frobenius'], np.sqrt(.15))
        v = self.a@np.array([1., -.5j]); b = scan_power(np.outer(v, v.conj()), self.w)
        np.testing.assert_allclose(b, [17/16-np.sqrt(3)/4, .5-np.sqrt(3)/4])

    def test_diagonal_removal_can_be_indefinite(self):
        r = source_power_csm(self.a, [1., .25]); removed = r-np.diag(np.diag(r))
        self.assertLess(np.min(np.linalg.eigvalsh(removed)), 0)
        np.testing.assert_allclose(2*scan_power(removed, self.w), [.875, -.25], atol=2e-15)
        with self.assertRaises(ValueError):
            clean_sc_full_csm(removed, self.w)

    def test_exercise_07_unequal_amplitude_dr_compensation(self):
        control = run_experiments()['exercises']['E12-07']['unequal_amplitude_control']
        full = np.asarray(control['full_csm_real_imag'])
        removed = np.asarray(control['diagonal_removed_csm_real_imag'])
        # Independent a=(1,2) outer product, with both retained off-diagonals 2.
        np.testing.assert_array_equal(full[..., 0], [[1., 2.], [2., 4.]])
        np.testing.assert_array_equal(full[..., 1], np.zeros((2, 2)))
        np.testing.assert_array_equal(removed[..., 0], [[0., 2.], [2., 0.]])
        np.testing.assert_allclose(np.asarray(control['weights_real_imag'])[:, 0], [1/5, 2/5])
        self.assertAlmostEqual(control['full_scan'], 1.)
        self.assertAlmostEqual(control['diagonal_removed_scan'], 8/25)
        self.assertEqual(control['microphone_count_compensation'], 2.)
        self.assertAlmostEqual(control['scan_with_microphone_count_compensation'], 16/25)
        self.assertEqual(control['matched_compensation'], 25/8)
        self.assertAlmostEqual(control['scan_with_matched_compensation'], 1.)

    def test_full_csm_clean_sc_author_equations(self):
        r = 2*np.ones((2, 2)); original = r.copy()
        result = clean_sc_full_csm(r, np.ones((2, 1))/2, iterations=2)
        np.testing.assert_allclose([step['allocated_power'] for step in result['steps']], [1.2, .48])
        np.testing.assert_allclose(result['clean_map'], [1.68])
        np.testing.assert_allclose(result['residual_csm'], .32*np.ones((2, 2)))
        np.testing.assert_array_equal(r, original)
        result = clean_sc_full_csm(r, np.ones((2, 1))/2, iterations=20)
        self.assertAlmostEqual(result['clean_map'][0], 2*(1-.4**20), places=14)
        zero = clean_sc_full_csm(np.zeros((2, 2)), self.w)
        self.assertEqual(zero['completed_iterations'], 0)
        two = clean_sc_full_csm(source_power_csm(self.a, [1., .25]), self.w, iterations=2, damping=1.)
        np.testing.assert_allclose(two['clean_map'], [17/16, 9/68], atol=2e-15)
        np.testing.assert_allclose(two['residual_csm'], np.zeros((2, 2)), atol=1e-14)

    def test_frobenius_half_triangle_and_intercept(self):
        a = np.array([1., 2.]); r = np.diag([1., 10.]); template = np.outer(a, a)
        full = hermitian_real_vector(template); half = hermitian_real_vector(template, frobenius=False)
        self.assertAlmostEqual(float(full@full), 25.)
        self.assertAlmostEqual(float(half@half), 21.)
        y = hermitian_real_vector(r, frobenius=False)
        affine = np.linalg.lstsq(np.column_stack([half, np.ones(4)]), y, rcond=None)[0]
        np.testing.assert_allclose(affine, [87/35, -8/5])
        error = np.array([[1., 2+3j], [2-3j, -1.]])
        self.assertAlmostEqual(float(hermitian_real_vector(error)@hermitian_real_vector(error)), 28.)
        self.assertAlmostEqual(float(hermitian_real_vector(error, frobenius=False)@hermitian_real_vector(error, frobenius=False)), 15.)
        self.assertAlmostEqual(single_source_csm_fit(r, a)['q'], 41/25)

    def test_joint_white_noise_and_one_channel_nonidentifiability(self):
        result = single_source_csm_fit([[2.5, 4.], [4., 8.5]], [1., 2.], include_white_noise=True)
        self.assertAlmostEqual(result['q'], 2.)
        self.assertAlmostEqual(result['white_noise_variance'], .5)
        one = single_source_csm_fit([[2.5]], [1.], include_white_noise=True)
        self.assertFalse(one['identifiable']); self.assertIsNone(one['q'])
        self.assertIsNone(one['white_noise_variance'])

    def test_spherical_grid_and_reference_invariance(self):
        result = spherical_scan_experiment()
        self.assertEqual(result['grid_m'].shape, (441, 3))
        np.testing.assert_allclose(result['psf_columns'][[215, 225]], [[1., .25508836620659353], [.25508836620659353, 1.]], atol=1e-14)
        self.assertAlmostEqual(result['local_peaks'][1]['position_error_m'], .03)
        np.testing.assert_allclose(result['local_peaks'][1]['peak_position_m'], [.18, 0., .6])
        self.assertAlmostEqual(result['dirty_grid_sum'], 72.17393775371565, places=11)
        np.testing.assert_allclose(result['reference_change']['q'], np.array([1., .25])*153/73)
        np.testing.assert_allclose(result['reference_change']['R'], result['R'], atol=2e-14)
        np.testing.assert_allclose(result['b'], result['psf_columns']@np.array([1., .25]), atol=1e-14)

    def test_spectrum_endpoints_hann_and_odd_length(self):
        n_fft=24; n=np.arange(n_fft); fs=24000
        for x, expected, window in [(.2*np.cos(2*np.pi*2*n/n_fft), .02, np.ones(n_fft)),
                                    (.2*np.cos(2*np.pi*2*n/n_fft), .02, periodic_hann(n_fft)),
                                    (np.full(n_fft, .2), .04, np.ones(n_fft)),
                                    (.2*(-1.)**n, .04, np.ones(n_fft))]:
            xft=stft(x[None,:],n_fft=n_fft,hop_length=n_fft,window=window,center=False)
            density=one_sided_csm_density(xft,sample_rate_hz=fs,n_fft=n_fft,window_energy=window@window)[:,0,0].real
            self.assertAlmostEqual(float(integrate_psd(density, fs/n_fft)), expected)
        n_fft=5; x=.2*np.cos(2*np.pi*2*np.arange(n_fft)/n_fft)
        xft=stft(x[None,:],n_fft=n_fft,hop_length=n_fft,window=np.ones(n_fft),center=False)
        density=one_sided_csm_density(xft,sample_rate_hz=fs,n_fft=n_fft,window_energy=n_fft)[:,0,0].real
        self.assertAlmostEqual(float(integrate_psd(density,fs/n_fft)), .02)

    def test_band_and_region_units(self):
        self.assertEqual(float(integrate_psd([.01,.02],[100.,200.])),5.)
        self.assertEqual(region_power([1.,.25],[True,True]),1.25)
        self.assertAlmostEqual(region_power([1.,.25],[True,True],cell_area_m2=[.04,.04]),.05)
        self.assertEqual(region_power([1.,.25],[False,False]),0.)

    def test_invalid_inputs_and_floating_range(self):
        calls = [lambda: conventional_weights([[0., 0.]]),
                 lambda: conventional_weights([[True]]),
                 lambda: source_power_csm(self.a, [1.,-.1]),
                 lambda: source_power_csm(self.a, [1.+0j,.25]),
                 lambda: scan_power([[1.,1.],[0.,1.]],self.w),
                 lambda: damas_gauss_seidel(self.p,[1.,0.],iterations=True),
                 lambda: damas_gauss_seidel([[0.]], [1.]),
                 lambda: clean_sc_full_csm(np.eye(2),self.w,damping=0.),
                 lambda: spherical_steering([[0.,0.,0.]],[[0.,0.,0.]],1000.),
                 lambda: spherical_steering([[0.,0.,0.]],[[0.,0.,1.]],True),
                 lambda: one_sided_csm_density(np.ones((1,2,1)),sample_rate_hz=1.,n_fft=4,window_energy=1.),
                 lambda: integrate_psd([.1],[-1.]),
                 lambda: region_power([1.],[1]),
                 lambda: conventional_weights([[1e308]]),
                 lambda: source_power_csm([[1e-200]], [1e-200])]
        for call in calls:
            with self.subTest(call=call),self.assertRaises(ValueError):
                call()


class ImagingAudioTests(unittest.TestCase):
    def test_actual_pcm_csm_and_tail(self):
        signals=make_signals()
        self.assertEqual(set(signals),set(FILE_NAMES))
        decoded={key:read_pcm16(pcm16_bytes(value,SAMPLE_RATE))[1] for key,value in signals.items()}
        for key,x in signals.items():
            self.assertEqual(x.shape[1],48004)
            self.assertEqual(x.shape[0],2 if key.startswith('array_') else 1)
            self.assertLessEqual(float(np.max(abs(x-decoded[key]))),.5/32768+1e-15)
            report=measure_signal(decoded[key],key,pcm=True)
            self.assertEqual(report['scoring_samples_per_channel'],38160)
            self.assertEqual(report['integer_power_sample_denominator'],38160)
            self.assertLess(report['normalized_csm_max_abs_error'],4.58e-4)
        independent=measure_signal(decoded['array_phase_code'],'array_phase_code',pcm=True)
        coherent=measure_signal(decoded['array_coherent'],'array_coherent',pcm=True)
        np.testing.assert_allclose(independent['normalized_inverse_source_mean_square'],[1.,.25],atol=4e-5)
        np.testing.assert_allclose(coherent['normalized_inverse_source_mean_square'],[1.2,.45],atol=6e-5)
        pairs=measure_source_pairs(decoded)
        self.assertLess(pairs['phase_code']['magnitude_squared_coherence'],1e-24)
        self.assertAlmostEqual(pairs['coherent']['magnitude_squared_coherence'],1.)
        # Delayed source tail is present; no wrapped samples appear at the head.
        np.testing.assert_array_equal(signals['source_1'][0,-4:],np.zeros(4))
        self.assertTrue(np.any(signals['array_coherent'][1,-4:] != 0))

    def test_amplitude_guard_and_wrong_pcm(self):
        with self.assertRaises(ValueError):
            extract_snapshot_amplitudes(np.ones((1,SAMPLES-1)))
        with self.assertRaises(ValueError):
            measure_signal(make_signals()['source_1'],'source_1',pcm=True)

    def test_all_eighteen_ids_and_finite_json(self):
        results=run_experiments()
        self.assertEqual(set(results['exercises']),{'E12-'+str(i).zfill(2) for i in range(1,19)})
        json.dumps(results,allow_nan=False)
        self.assertFalse(results['exercises']['E12-12']['sklearn_executed'])
        self.assertAlmostEqual(results['exercises']['E12-12']['affine_math_q'],87/35)


class ImagingAssetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.contents=prepare_assets()

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.directory=Path(self.temporary.name)/'assets'
        self.directory.mkdir()
        for name,blob in self.contents.items():
            (self.directory/name).write_bytes(blob)

    def tearDown(self):
        self.temporary.cleanup()

    def _snapshot(self):
        return {file.name:(hashlib.sha256(file.read_bytes()).hexdigest(),file.stat().st_mtime_ns)
                for file in self.directory.iterdir() if file.is_file() and not file.is_symlink()}

    def test_check_read_only_and_generation(self):
        before=self._snapshot()
        with patch.object(Path,'write_bytes',side_effect=AssertionError('check attempted write')):
            actual=check_assets(self.directory)
        self.assertEqual(actual,self.manifest);self.assertEqual(before,self._snapshot())
        output=Path(self.temporary.name)/'new'
        self.assertEqual(generate_assets(output),self.manifest)
        self.assertEqual(check_assets(output),self.manifest)

    def test_missing_or_extra_members_fail_without_repair(self):
        (self.directory/'extra.txt').write_text('unrelated')
        before=self._snapshot()
        with self.assertRaises(ValueError):check_assets(self.directory)
        with self.assertRaises(ValueError):generate_assets(self.directory)
        self.assertEqual(before,self._snapshot())
        (self.directory/'extra.txt').unlink()
        (self.directory/'source_1.wav').unlink()
        before=self._snapshot()
        with self.assertRaises(ValueError):check_assets(self.directory)
        self.assertEqual(before,self._snapshot())

    def test_symlinks_and_hardlinks_are_rejected(self):
        target=self.directory/'source_1.wav';other=Path(self.temporary.name)/'outside.wav'
        other.write_bytes(target.read_bytes());target.unlink();target.symlink_to(other)
        with self.assertRaises(ValueError):check_assets(self.directory)
        target.unlink();os.link(other,target)
        with self.assertRaises(ValueError):check_assets(self.directory)
        parent=Path(self.temporary.name)/'alias';parent.symlink_to(self.directory,target_is_directory=True)
        with self.assertRaises(ValueError):check_assets(parent)

    def test_manifest_type_duplicate_nonfinite_stale_and_score_tamper(self):
        path=self.directory/'MANIFEST.json'
        original=self.contents['MANIFEST.json']
        for mutation in ['bool','duplicate','nan','sha','score']:
            value=json.loads(original)
            if mutation=='bool':value['samples_per_channel']=True
            elif mutation=='sha':value['source_sha256']['codes/chapters/ch12/core/imaging.py']='0'*64
            elif mutation=='score':value['samples']['array_coherent']['pcm_measurements']['normalized_scan_mean_square'][0]+=1.
            if mutation=='duplicate':blob=b'{"schema_version":1,"schema_version":1}'
            elif mutation=='nan':blob=b'{"schema_version":NaN}'
            else:blob=json.dumps(value,allow_nan=False).encode()
            path.write_bytes(blob);before=self._snapshot()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):check_assets(self.directory)
            self.assertEqual(before,self._snapshot())
        path.write_bytes(original)

    def test_wav_mutation_rejected(self):
        path=self.directory/'array_coherent.wav';blob=bytearray(path.read_bytes());blob[-7]^=1;path.write_bytes(blob)
        before=self._snapshot()
        with self.assertRaises(ValueError):check_assets(self.directory)
        self.assertEqual(before,self._snapshot())


if __name__ == '__main__':
    unittest.main()
