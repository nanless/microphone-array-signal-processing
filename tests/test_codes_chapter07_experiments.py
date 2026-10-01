"""Independent rational, polynomial, DFT and actual PCM anchors for E07-08..21."""
import cmath
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave
from fractions import Fraction as F
import numpy as np
from codes.chapters.ch07 import chapter07_experiments as ex


def z(encoded):
    return np.asarray(encoded['real']) + 1j*np.asarray(encoded['imag'])


class Chapter07Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
        from codes.chapters.ch07.examples import mint_teaching_demo as demo
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)/'repo'
        # An explicit temporary fixture binds actual current source bytes.
        # It never repairs the published manifest or any published WAV.
        original = ex.ROOT/'codes/chapters/ch00/audio/MANIFEST.json'
        manifest = json.loads(original.read_text())
        manifest['generator_inputs'] = {}
        for relative in INPUTS:
            target = cls.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = (ex.ROOT/relative).read_bytes()
            target.write_bytes(blob)
            manifest['generator_inputs'][relative] = hashlib.sha256(blob).hexdigest()
        cls.manifest_path = cls.root/'codes/chapters/ch00/audio/MANIFEST.json'
        cls.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        cls.manifest_path.write_text(json.dumps(manifest))
        audio = cls.root/'codes/chapters/ch07/audio'
        audio.mkdir(parents=True)
        for part in ('target', 'reverberant', 'oracle_inverse', 'output'):
            name = 'wpe_predictable_'+part+'.wav'
            shutil.copyfile(ex.ROOT/'codes/chapters/ch07/audio'/name, audio/name)
        cls.mint = Path(cls.temporary.name)/'mint'
        demo.generate(cls.mint)

    def test_complex_fit_explicit_matrix_and_rational_solution(self):
        result = ex.complex_weighted_fit()
        np.testing.assert_array_equal(z(result['correlation']), [[2,-1j],[1j,3]])
        np.testing.assert_array_equal(z(result['cross']), [3,4j])
        plain = result['absolute_loading_cases']['0']
        np.testing.assert_allclose(z(plain['coefficients']), [1,1j], atol=1e-15)
        np.testing.assert_allclose(z(plain['residual']), [-1,.5j,1], atol=1e-15)
        self.assertAlmostEqual(plain['weighted_residual_cost'], 2.5)
        np.testing.assert_allclose(z(plain['weighted_history_residual_cross']), [0,0], atol=1e-15)
        loaded = result['absolute_loading_cases']['1']
        expected = np.array([8/11,9j/11])
        np.testing.assert_allclose(z(loaded['coefficients']), expected, atol=1e-15)
        np.testing.assert_allclose(z(loaded['weighted_history_residual_cross']), expected, atol=1e-15)
        # Full six-frame observation must generate these histories and targets.
        obs = z(result['observations'])
        np.testing.assert_array_equal(obs[:, [0,1,2]].T, [[1,0],[0,1],[1,1j]])
        np.testing.assert_array_equal(obs[0, [3,4,5]], [0,-.5j,3])

    def test_power_objective_floor_and_smoothing_are_distinct(self):
        r = ex.power_floor_and_smoothing()['cases']
        self.assertAlmostEqual(r['unconstrained']['objective_without_constants'],2+np.log(9))
        self.assertAlmostEqual(r['floor_2']['objective_without_constants'],1.5+np.log(18))
        self.assertAlmostEqual(r['mean_smoothed']['objective_without_constants'],2+2*np.log(5))
        self.assertGreater(r['mean_smoothed']['objective_without_constants'],r['unconstrained']['objective_without_constants'])
        zero = ex.power_floor_and_smoothing()['zero_energy']['objective_without_constants']
        np.testing.assert_allclose(zero, [0,-np.log(10),-2*np.log(10)])

    def test_permutation_preserves_inner_product_only_when_both_move(self):
        r = ex.history_permutation()
        self.assertEqual(r['lag_major_history'],[3,13,2,12,1,11])
        self.assertEqual(r['channel_major_history'],[3,2,1,13,12,11])
        self.assertEqual(z(r['prediction_lag_major']),18-1j)
        self.assertEqual(z(r['prediction_both_permuted']),18-1j)
        self.assertEqual(z(r['prediction_history_only_permuted']),16+11j)

    def test_frame_count_does_not_identify_collinear_parameters(self):
        cases = ex.frame_count_and_rank()['cases']
        np.testing.assert_allclose(cases['repeated']['eigenvalues'],[0,8],atol=1e-15)
        np.testing.assert_allclose(cases['repeated']['minimum_norm_coefficients'],[.75,.75],atol=1e-15)
        np.testing.assert_allclose(cases['repeated']['absolute_loading_1_coefficients'],[2/3,2/3],atol=1e-15)
        np.testing.assert_array_equal(cases['independent']['eigenvalues'],[2,2])
        np.testing.assert_allclose(cases['independent']['minimum_norm_coefficients'],[1,2],atol=1e-15)
        # Null direction changes coefficients but leaves every training value.
        np.testing.assert_array_equal(np.ones((4,2)) @ [1.,-1.],np.zeros(4))

    def test_window_overlap_by_explicit_sample_sets(self):
        r = ex.frame_window_overlap()
        for item, expected in zip(r['cases'], [128,0]):
            a,b = item['nearest_history_support_samples']
            self.assertEqual(len(set(range(512)) & set(range(a,b))),expected)
            self.assertEqual(item['overlap_samples'],expected)
        self.assertEqual([a['oldest_frame_start_offset_samples'] for a in r['cases']],[-896,-1024])

    def test_exponential_statistics_independent_fraction_sum(self):
        r = ex.exponential_statistics()
        # Expand all three weighted terms plus the decayed initial statistics.
        R = F(1,8)*2+F(1,4)*1+F(1,2)*4+1
        cross = F(1,8)+F(1,4)*F(4,5)+F(1,2)*2*F(8,5)+1
        self.assertEqual((R,cross),(F(7,2),F(117,40)))
        self.assertAlmostEqual(r['batch_correlation'],float(R))
        self.assertAlmostEqual(r['batch_cross'],float(cross))
        self.assertAlmostEqual(r['steps'][-1]['posterior_coefficient'],float(F(117,140)))
        np.testing.assert_allclose([s['prior_output'] for s in r['steps']],[.3,.3,.23],atol=1e-15)
        self.assertAlmostEqual(r['two_zero_history_steps_forgetting']['correlation'],7/8)
        self.assertAlmostEqual(r['two_zero_history_steps_forgetting']['cross'],117/160)
        self.assertEqual(r['two_fully_frozen_statistics_steps']['correlation'],3.5)

    def test_mint_polynomial_identity_and_post_path_noise(self):
        a,b = ex.mint_near_common_zero()['cases']
        np.testing.assert_allclose(a['constant_inverse_weights'],[.5,.5],atol=1e-14)
        np.testing.assert_allclose(b['constant_inverse_weights'],[-49,50],rtol=1e-14)
        np.testing.assert_allclose(a['summed_impulse_response'],[1,0],atol=1e-14)
        np.testing.assert_allclose(b['summed_impulse_response'],[1,0],atol=1e-14)
        self.assertAlmostEqual(a['post_path_independent_unit_noise_variance'],.5)
        self.assertAlmostEqual(b['post_path_independent_unit_noise_variance'],4901,places=9)
        self.assertAlmostEqual(b['perturbed_second_response_tap'],-.049,places=12)
        self.assertFalse(ex.mint_near_common_zero()['common_zero_case']['constant_inverse_exists'])

    def test_resource_bytes_not_timing(self):
        a,b = ex.wpe_resource_budget()['cases']
        self.assertEqual((a['channels'],a['taps'],a['history_dimension']),(8,10,80))
        self.assertEqual(a['one_correlation_array_bytes'],26316800)
        self.assertEqual(a['one_predictor_array_bytes'],2631680)
        self.assertEqual(b['one_correlation_array_bytes'],4*a['one_correlation_array_bytes'])
        self.assertEqual(b['one_predictor_array_bytes'],4*a['one_predictor_array_bytes'])
        self.assertEqual(b['dense_factorization_cubic_dimension_proxy'],8*a['dense_factorization_cubic_dimension_proxy'])

    def test_loaded_wpd_missing_term_against_rational_solution(self):
        r = ex.loaded_wpd_factorization()
        np.testing.assert_allclose(r['G'],np.diag([1/3,0]),atol=1e-15)
        np.testing.assert_allclose(r['schur'],np.diag([8/3,2]),atol=1e-15)
        np.testing.assert_allclose(r['residual_covariance'],np.diag([14/9,1]),atol=1e-15)
        np.testing.assert_allclose(r['required_correction'],np.diag([10/9,1]),atol=1e-15)
        np.testing.assert_allclose(r['direct_loaded_filter'],[3/7,4/7,-1/7,0],atol=1e-15)
        self.assertAlmostEqual(r['direct_loaded_objective'],8/7)
        np.testing.assert_allclose(r['omitted_GHG_correction_spatial_filter'],[18/41,23/41],atol=1e-15)
        np.testing.assert_allclose(r['omitted_GHG_correction_history_filter'],[-6/41,0],atol=1e-15)

    def test_stable_ids_and_finite_json(self):
        r = ex.run_experiments(repo_root=self.root, mint_directory=self.mint)
        self.assertEqual(set(r),{f'E07-{i:02d}' for i in range(8,22)})
        self.assertEqual(len(r['E07-17']['files']),4)
        json.dumps(r,allow_nan=False)

    def test_published_predictable_pcm_independent_integer_score(self):
        r = ex.predictable_target_audio(self.root)['published_audio']
        actual = {}
        for part in ('target', 'reverberant', 'oracle_inverse', 'output'):
            name = 'wpe_predictable_'+part
            with wave.open(str(self.root/'codes/chapters/ch07/audio'/(name+'.wav'))) as f:
                actual[name] = struct.unpack('<32000h', f.readframes(32000))
        truth = actual['wpe_predictable_target'][6400:14400]
        D = sum(n*n for n in truth)
        self.assertEqual(D, 61844852000)
        for name, x in actual.items():
            m = r['pcm_measurements'][name]
            self.assertEqual(m['integer_reference_squared_sum'], D)
            self.assertEqual(m['integer_error_squared_sum'], sum((a-b)**2 for a,b in zip(x[6400:14400],truth)))
            self.assertEqual(m['integer_output_reference_cross_sum'], sum(a*b for a,b in zip(x[6400:14400],truth)))
            self.assertEqual(m['integer_tail_squared_sum'], sum(a*a for a in x[16000:32000]))
            self.assertEqual((m['steady_sample_denominator'],m['tail_sample_denominator'],m['pcm_decode_divisor']), (8000,16000,32768))
        self.assertEqual(r['tail_ratio_integer_numerator'],1306726645)
        self.assertEqual(r['tail_ratio_integer_denominator'],19131658897)

    def test_predictable_bad_assets_fail_without_repair(self):
        for case in ('missing', 'tamper', 'zero_reference', 'format', 'source', 'duplicate', 'bool_score'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as t:
                root = Path(t)/'repo'; shutil.copytree(self.root,root)
                mp = root/'codes/chapters/ch00/audio/MANIFEST.json'
                m = json.loads(mp.read_text())
                p = root/'codes/chapters/ch07/audio/wpe_predictable_target.wav'
                if case=='missing': p.unlink()
                elif case=='tamper': p.write_bytes(p.read_bytes()[:-2]+b'\x01\x00')
                elif case in ('zero_reference','format'):
                    with wave.open(str(p),'wb') as f:
                        f.setparams((1,2,8000 if case=='format' else 16000,0,'NONE','not compressed'))
                        f.writeframes(bytes(64000))
                    record = next(v for v in m['files'] if v['file']==p.name)
                    record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
                    mp.write_text(json.dumps(m))
                elif case=='source':
                    relative = next(iter(m['generator_inputs']))
                    (root/relative).write_bytes(b'changed actual source')
                elif case=='duplicate':
                    m['files'].append(next(v for v in m['files'] if v['file']==p.name).copy())
                    mp.write_text(json.dumps(m))
                else:
                    m['groups']['wpe_predictable']['parameters']['pcm_to_pcm_reference_scores']['files']['wpe_predictable_target']['steady_projection_gain']=True
                    mp.write_text(json.dumps(m))
                before={str(v.relative_to(root)):(v.read_bytes(),v.stat().st_mtime_ns) for v in root.rglob('*') if v.is_file()}
                with self.assertRaises(ValueError): ex.predictable_target_audio(root)
                after={str(v.relative_to(root)):(v.read_bytes(),v.stat().st_mtime_ns) for v in root.rglob('*') if v.is_file()}
                self.assertEqual(before,after)

    def test_real_cepstrum_direct_eight_point_sum_and_inverse_prefixes(self):
        r=ex.real_cepstrum_phase_ambiguity()
        spectra=[]
        for h in ([1,.5],[.5,1]):
            H=[sum(v*cmath.exp(-2j*cmath.pi*k*n/8) for n,v in enumerate(h)) for k in range(8)]
            spectra.append(H)
        for key,H in zip(('minimum','maximum'),spectra):
            np.testing.assert_allclose(z(r['spectrum_'+key+'_phase']),H,atol=1e-15)
            expected=[sum(np.log(abs(H[k])**2)*cmath.exp(2j*cmath.pi*k*n/8) for k in range(8)).real/8 for n in range(8)]
            np.testing.assert_allclose(r['real_cepstrum_'+key+'_phase'],expected,atol=1e-15)
        np.testing.assert_allclose([abs(v)**2 for v in spectra[0]],[abs(v)**2 for v in spectra[1]],atol=1e-15)
        self.assertNotAlmostEqual(cmath.phase(spectra[0][2]),cmath.phase(spectra[1][2]))
        for h,key in (([1,.5],'minimum'),([.5,1],'maximum')):
            inverse=r['first_eight_causal_inverse_'+key+'_phase']
            self.assertEqual(h[0]*inverse[0],1)
            self.assertEqual([h[0]*inverse[n]+h[1]*inverse[n-1] for n in range(1,8)],[0]*7)
            self.assertNotEqual(h[1]*inverse[-1],0)  # the truncated FIR still has a tail
        self.assertAlmostEqual(r['real_cepstrum_minimum_phase'][0],np.log(255/256)/4,places=15)

    def test_design_loss_is_not_an_independent_data_rank_loss(self):
        r=ex.design_normal_comparison(); A=z(r['design']); b=z(r['target'])
        # Exact equations: g1+g2=0, epsilon*g2=-epsilon.
        np.testing.assert_array_equal(A@np.array([1,-1]),b)
        self.assertEqual((A.conj().T@A)[1,1],1)  # distinguishable row lost in Gram
        direct,normal=(r['cases'][k] for k in ('design_lstsq','normal'))
        np.testing.assert_allclose(z(direct['coefficients']),[1,-1],atol=1e-14)
        np.testing.assert_allclose(z(normal['coefficients']),[-2.5e-19,-2.5e-19],rtol=1e-14,atol=0)
        self.assertEqual((direct['diagnostic']['rank'],normal['diagnostic']['rank']),(2,1))
        self.assertTrue(normal['diagnostic']['used_lstsq'])
        self.assertEqual(normal['residual_squared_sum'],1e-18)
        self.assertEqual(normal['diagnostic']['condition_number_status'],'infinite')

    def test_constrained_regularizer_fraction_stationarity_and_cost(self):
        r=ex.constrained_regularized_mint(); d=r['regularized']
        a,b,lam=F(1,2),F(49,100),F(1,10000); u1,u2=F(-16),F(17)
        reflected=a*u1+b*u2; norm=u1*u1+u2*u2
        self.assertEqual((reflected,norm,reflected**2+lam*norm),(F(33,100),545,F(817,5000)))
        self.assertEqual(reflected*(a-b)+lam*(u1-u2),0)
        self.assertEqual(d['weights'],[-16,17]); self.assertEqual(d['total_cost'],.1634)
        self.assertEqual(r['exact_inverse']['cost_at_regularization_1e_4'],.4901)


if __name__ == '__main__':
    unittest.main()
