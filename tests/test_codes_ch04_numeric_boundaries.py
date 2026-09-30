"""Analytic DOA/AIC domains and deterministic chapter-4 anchors."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from codes.chapters.ch04.core import doa
from codes.chapters.ch04 import chapter04_experiments as ex, doa_resolution_trials as trials
from codes.chapters.ch00.cross_chapter.exercises_spatial import four_mic_fractional_delay


class NumericBoundaryTests(unittest.TestCase):
    def test_original_nonhermitian_rejected_by_all_four_methods(self):
        raw=np.array([[1.,2.],[0.,1.]])
        calls=(lambda r:doa.bartlett_spectrum(r,[1,1]),lambda r:doa.capon_spectrum(r,[1,1]),
               lambda r:doa.music_spectrum(r,[1,1],source_count=1),
               lambda r:doa.esprit_ula(r,source_count=1,spacing_m=.1715,frequency_hz=1000))
        for f in calls:
            with self.assertRaises(ValueError):f(raw)
            with self.assertRaises(np.linalg.LinAlgError):f(np.diag([-1.,2.]))
        rounded=np.array([[2.,1+1e-12],[1.,2.]])
        self.assertAlmostEqual(doa.bartlett_spectrum(rounded,[1,1])[0],6+1e-12)

    def test_subnormal_covariance_eigenspaces_and_capon_final_power(self):
        # J+I has eigenvalues 4,1,1; [1,1,1] belongs to eigenvalue 4.
        r=(np.ones((3,3))+np.eye(3))*1e-320
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            self.assertAlmostEqual(doa.bartlett_spectrum(r,[1,1,1])[0]/1e-320,12,delta=.003)
            self.assertAlmostEqual(doa.music_spectrum(r,[1,1,1],source_count=1)[0]/1e15,1)
            np.testing.assert_allclose(doa.esprit_ula(r,source_count=1,spacing_m=.1715,frequency_hz=1000),[0],atol=1e-14)
            self.assertAlmostEqual(doa.capon_spectrum(r,[1,1,1])[0]/1e-320,4/3,delta=.002)
        # Zero is valid for Bartlett, but does not identify a MUSIC subspace.
        np.testing.assert_array_equal(doa.bartlett_spectrum(np.zeros((2,2)),[1,1]),[0])

    def test_steering_zero_empty_overflow_and_boolean_count(self):
        for candidate in ([0,0,0],np.empty((0,3))):
            for f in (lambda x:doa.bartlett_spectrum(np.eye(3),x),
                      lambda x:doa.capon_spectrum(np.eye(3),x),
                      lambda x:doa.music_spectrum(np.eye(3),x,source_count=1)):
                with self.assertRaises(ValueError):f(candidate)
        with self.assertRaises(ValueError):doa.music_spectrum(np.eye(3)+np.ones((3,3)),[1e200,0,0],source_count=1)
        with self.assertRaises(ValueError):doa.music_spectrum(np.eye(3),[1,1,1],source_count=True)
        # The diagonal principal vector has exactly zero noise projection.
        with self.assertRaises(ValueError):doa.music_spectrum(np.diag([3.,1.,1.]),[1,0,0],source_count=1,denominator_floor=5e-324)

    def test_phat_subnormal_protection_is_finite_or_explicitly_unsupported(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            r=doa.gcc_phat([1,1],[1,1],16000,epsilon=5e-324)
            self.assertEqual(r[0],0)
            # Four-bin PHAT [1,1,0,1] has IDFT [.75,.25,-.25,.25].
            np.testing.assert_allclose(r[3],[.25,.75,.25],atol=1e-15)
            self.assertTrue(np.all(np.isfinite(r[3])))
            y=np.array([[[1,0]],[[1,0]]],complex)
            np.testing.assert_allclose(doa.srp_phat(y,[1000],[[0,0],[.04,0]],[0],epsilon=5e-324),[.5])

    def test_public_real_scalars_and_trial_inputs(self):
        for bad in (True,1+0j,'1',[1.]):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):doa.gcc_phat([1],[1],bad)
                with self.assertRaises(ValueError):doa.srp_phat(np.ones((2,1,1)),[1000],[[0,0],[.04,0]],[0],epsilon=bad)
                with self.assertRaises(ValueError):doa.esprit_ula(np.ones((3,3)),source_count=1,spacing_m=bad,frequency_hz=1000)
        for bad in (np.array([30+5j]),['30'],[True]):
            with self.assertRaises(ValueError):trials.steering_rows(bad)
        with self.assertRaises(ValueError):trials.two_peaks(np.array([0,1+1j,0]),[-1,0,1])
        with self.assertRaises(ValueError):trials.classify_peaks([], (np.nan,10))
        for snr in (-4000,4000):
            with self.assertRaises(ValueError):trials.run_experiment(trials=1,array_snr_db=snr)

    def test_aic_mdl_hand_log_means_and_scale(self):
        eigenvalues=[9,4,1.2,.8];N=100
        expected_fit=[]
        for k in range(4):
            tail=eigenvalues[k:]
            expected_fit.append(N*len(tail)*(math.log(math.fsum(tail)/len(tail))
                                           -math.fsum(math.log(x) for x in tail)/len(tail)))
        aic=[2*f+2*k*(8-k) for k,f in enumerate(expected_fit)]
        mdl=[f+.5*k*(8-k)*math.log(N) for k,f in enumerate(expected_fit)]
        self.assertEqual(doa.aic_source_count(eigenvalues,N)[0],3)
        self.assertEqual(doa.mdl_source_count(eigenvalues,N)[0],2)
        np.testing.assert_allclose(doa.aic_source_count(eigenvalues,N)[1],aic,atol=1e-12)
        np.testing.assert_allclose(doa.mdl_source_count(eigenvalues,N)[1],mdl,atol=1e-12)
        for scale in (1e-200,1e200):
            np.testing.assert_allclose(doa.aic_source_count(np.array(eigenvalues)*scale,N)[1],aic,atol=1e-10)
        for invalid in ([9,4,1,0],[9,4,1,-1],np.array([9+0j,4,1,1]),['9','4']):
            with self.assertRaises(ValueError):doa.aic_source_count(invalid,N)
        with self.assertRaises(ValueError):doa.aic_source_count(eigenvalues,True)

    def test_nonunitary_noise_and_gls_closed_form(self):
        r=ex.nonunitary_focusing_noise()
        np.testing.assert_array_equal(r['pooled_covariance'],[[4,-1],[-1,4]])
        np.testing.assert_array_equal(r['pooled_noise_covariance'],[[3,-2],[-2,3]])
        np.testing.assert_allclose(r['whitened_eigenvalues'],[1,3],atol=1e-14)
        g=ex.correlated_tdoa_gls()['methods']
        np.testing.assert_allclose(g['ols']['step_m'],[-.08/(3*math.sqrt(2)),.01/(3*math.sqrt(2))],atol=1e-15)
        np.testing.assert_allclose(g['gls']['step_m'],[-.03/math.sqrt(2),0],atol=1e-15)
        self.assertAlmostEqual(g['ols']['mahalanobis_squared'],11/9)
        self.assertAlmostEqual(g['gls']['mahalanobis_squared'],1)
        self.assertGreater(g['gls']['linear_sse_m2'],g['ols']['linear_sse_m2'])

    def test_root_music_factorization_and_unit_circle_quadratic_form(self):
        r=ex.root_music_polynomial();coeff=r['polynomial_coefficients_ascending']
        for angle in (0,.2,1.7,-2.3):
            z=complex(math.cos(angle),math.sin(angle))
            poly=sum(c*z**k for k,c in enumerate(coeff))
            fact=-(z-1)**2*(z*z+4*z+1)/3
            self.assertAlmostEqual(abs(poly-fact),0,places=13)
            # Q=I-aa^H/3 => a(z)^H Q a(z)=3-|1+z+z²|²/3.
            power=3-abs(1+z+z*z)**2/3
            self.assertAlmostEqual((poly/z**2).real,power,places=13)
        self.assertAlmostEqual(r['analytic_roots_real'][2]*r['analytic_roots_real'][3],1)
        self.assertLess(max(r['computed_polynomial_residuals']),1e-11)

    def test_published_e07_exact_integer_power_denominator(self):
        r=four_mic_fractional_delay()
        for name,integer_sum in [('fractional_unaligned.wav',142189211691),('fractional_aligned.wav',2768561806)]:
            row=r['pcm_measurements'][name]
            self.assertEqual(row['integer_residual_squared_sum'],integer_sum)
            self.assertEqual(row['sample_denominator'],31360)
            self.assertEqual(row['mse'],integer_sum/(31360*32768**2))

    def test_published_e18_stale_sources_missing_changed_pcm_and_format_rejected(self):
        for kind in ('source','missing','pcm','format'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                for name in ('doa_ambiguity_tone.wav','doa_ambiguity_broadband.wav'):
                    shutil.copyfile(ex.ROOT/'codes/chapters/ch04/audio'/name,root/name)
                m=json.loads((ex.ROOT/'codes/chapters/ch00/audio/MANIFEST.json').read_text())
                if kind=='source':m['generator_inputs'][next(iter(m['generator_inputs']))]='0'*64
                elif kind=='missing':(root/'doa_ambiguity_tone.wav').unlink()
                else:
                    p=root/'doa_ambiguity_tone.wav';blob=bytearray(p.read_bytes());blob[10000 if kind=='pcm' else 24]^=64;p.write_bytes(blob)
                    if kind=='format':
                        for row in m['files']:
                            if row['file']==p.name:row['sha256']=hashlib.sha256(blob).hexdigest()
                manifest=root/'main.json';manifest.write_text(json.dumps(m))
                with self.assertRaises((ValueError,FileNotFoundError)):
                    ex.audio_phase_ambiguity(audio_root=root,manifest_path=manifest)


if __name__=='__main__':unittest.main()
