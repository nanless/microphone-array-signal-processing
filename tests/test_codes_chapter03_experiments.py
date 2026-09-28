"""Independent scalar identities, rational matrices and PCM checks for ch. 3."""
import json
import math
import unittest
import numpy as np
from codes.chapters.ch03 import chapter03_experiments as ch3
from codes.chapters.ch00.core.audio_samples import dma_calibration_case, prepare_exports, read_pcm16


class Chapter03ExperimentsTest(unittest.TestCase):
    def test_registry_has_seven_serializable_new_exercises(self):
        result=ch3.run_exercises()
        self.assertEqual(set(result),{f'E03-{n:02}' for n in range(8,15)})
        json.dumps(result,allow_nan=False)

    def test_ula_half_power_from_cosine_factorization(self):
        rows=ch3.exact_beamwidths()['cases'];low,high=rows[:2]
        self.assertIsNone(low['exact_hpbw_deg'])
        # Four-term symmetric sum = cos(psi)*cos(psi/2), independent of steering helper.
        p=2*math.pi*1000*.03/343
        self.assertAlmostEqual(low['domain_edge_amplitude'],abs(math.cos(p)*math.cos(p/2)))
        # Half power implies q^3+q^2-1=0, q=cos(psi). Polynomial bisection.
        left,right=0.,1.
        for _ in range(65):
            q=(left+right)/2
            if q**3+q*q-1>0:right=q
            else:left=q
        expected=2*math.degrees(math.asin(math.acos((left+right)/2)*343/(2*math.pi*8000*.03)))
        self.assertAlmostEqual(high['exact_hpbw_deg'],expected,places=10)
        self.assertAlmostEqual(expected,18.7282172241,places=6)
        self.assertGreater(high['exact_hpbw_deg'],high['ula_narrow_lobe_approx_deg'])

    def test_steered_grating_lobe_uses_exact_integer_cycle(self):
        broadside, steered = ch3.exact_beamwidths()['grating_checks']
        self.assertTrue(all(not item['visible'] for item in broadside['candidates']))
        alias = steered['candidates'][0]
        expected = math.degrees(math.asin(math.sqrt(3)/2 - 343/240))
        self.assertTrue(alias['visible'])
        self.assertAlmostEqual(alias['angle_deg'], expected)
        self.assertAlmostEqual(alias['dsb_amplitude'], 1., places=14)
        # Adjacent phase mismatch is exactly -2pi. All four phasors add to four.
        delta_sine = math.sin(math.radians(alias['angle_deg'])) - math.sqrt(3)/2
        self.assertAlmostEqual(8000*.03/343*delta_sine, -1.)
        self.assertFalse(steered['candidates'][1]['visible'])

    def test_ring_half_power_from_opposite_microphone_pairs(self):
        for row in ch3.exact_beamwidths()['cases'][2:]:
            theta=math.radians(row['exact_hpbw_deg']/2);k=2*math.pi*row['frequency_hz']*.0375/343
            dx,dy=math.sin(theta),math.cos(theta)-1
            # Explicit three diametric pairs, not six-channel steering code.
            response=(math.cos(k*dx)+math.cos(k*(dx/2+math.sqrt(3)*dy/2))
                      +math.cos(k*(-dx/2+math.sqrt(3)*dy/2)))/3
            self.assertAlmostEqual(response*response,.5,places=12)
        self.assertAlmostEqual(ch3.exact_beamwidths()['cases'][2]['exact_hpbw_deg'],220.2324,places=3)
        self.assertAlmostEqual(ch3.exact_beamwidths()['cases'][3]['exact_hpbw_deg'],47.30647,places=4)

    def test_hexagon_phase_difference_is_integer_cycles(self):
        result=ch3.hexagon_ambiguity()
        self.assertAlmostEqual(result['critical_frequency_hz'],343/(math.sqrt(3)*.0375))
        for row in result['cases']:
            self.assertLess(row['max_manifold_difference'],1e-13)
            # y positions of the six vertices divided by sqrt(3)r/2 are integers.
            # Wavevector difference = 4pi/(sqrt(3)r) along y -> these integer cycles.
            cycles=[]
            for x,y in result['positions_m']:
                cycles.append(row['frequency_hz']/343*y*(row['directions_xy'][0][1]-row['directions_xy'][1][1]))
            np.testing.assert_allclose(cycles,[0,1,1,0,-1,-1],atol=1e-14)
        self.assertGreater(result['opposite_directions_difference_at_1khz'],.1)

    def test_planar_mirror_uses_zero_z_coordinates(self):
        result=ch3.planar_mirror();delays=np.asarray(result['relative_delays_us'])
        expected=np.array([0,-.012/343,-.016/343,-.04*math.sqrt(.75)/343])*1e6
        np.testing.assert_allclose(delays[0],expected,atol=1e-12)
        np.testing.assert_allclose(delays[1],expected*np.array([1,1,1,-1]),atol=1e-12)
        self.assertEqual(result['planar_max_difference_us'],0)
        self.assertAlmostEqual(result['added_mic_difference_us'],2*.04*math.sqrt(.75)/343*1e6)

    def test_coupling_noise_and_bias_from_two_scalar_eigenmodes(self):
        result=ch3.coupling_regularization();inverse,reg=result['cases']
        self.assertAlmostEqual(result['condition_number_2'],19.)
        # Post-mixing white noise has unit variance in each orthonormal mode.
        inverse_var=.5*(1/1.9**2+1/.1**2)
        reg_var=.5*((1.9/(1.9**2+.01))**2+(.1/(.1**2+.01))**2)
        np.testing.assert_allclose(inverse['post_mix_noise_output_variances'],[inverse_var]*2)
        np.testing.assert_allclose(reg['post_mix_noise_output_variances'],[reg_var]*2)
        self.assertAlmostEqual(reg['common_mode_signal_gain'],361/362)
        self.assertAlmostEqual(reg['difference_mode_signal_gain'],.5)

    def test_gain_ls_hand_conjugate_sum_and_residual(self):
        result=ch3.gain_least_squares()
        self.assertAlmostEqual(result['numerator']['real'],4.7)
        self.assertAlmostEqual(result['numerator']['imag'],2.5)
        self.assertEqual(result['denominator'],5.)
        self.assertAlmostEqual(result['estimated_gain']['real'],.94)
        self.assertAlmostEqual(result['estimated_gain']['imag'],.5)
        np.testing.assert_allclose(result['residual']['real'],[.16,0],atol=1e-15)
        np.testing.assert_allclose(result['residual']['imag'],[0,-.08],atol=1e-15)
        self.assertAlmostEqual(result['residual_squared_sum'],.032)
        for scale in (1e-200,1.,1e200):
            self.assertAlmostEqual(ch3.relative_gain_ls(scale*np.array([1,2j]),scale*np.array([1.1+.5j,-1+1.8j])),.94+.5j)
        for z,y in (([],[]),([0],[1]),([1],[np.nan]),([[1]],[[1]]),([1,2],[1])):
            with self.subTest(z=z),self.assertRaises(ValueError):ch3.relative_gain_ls(z,y)

    def test_lag_holes_and_counts_against_explicit_sets(self):
        ula,nested,coprime=ch3.coprime_holes()['cases']
        self.assertEqual([r['physical_aperture_grid_units'] for r in (ula,nested,coprime)],[5,11,9])
        self.assertEqual(coprime['holes'],[-7,7])
        self.assertEqual(coprime['lags'],[-9,-8,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,8,9])
        self.assertEqual(coprime['central_contiguous_lag_count'],13)
        self.assertEqual(coprime['direct_toeplitz_size'],7)
        for row in (ula,nested,coprime):self.assertEqual(sum(row['multiplicities']),36)

    def test_coarray_smoothing_from_rational_corner_block(self):
        result=ch3.coarray_companion()
        expected=np.eye(4)/9
        expected[0,0]=expected[3,3]=13/36
        expected[0,3]=expected[3,0]=1/3
        np.testing.assert_allclose(result['finite_sample_ss_matrix']['real'],expected,atol=1e-15)
        np.testing.assert_allclose(result['ss_eigenvalues'],[1/36,1/9,1/9,25/36])
        np.testing.assert_allclose(result['coherent_same_lag_R10_R21']['real'],[2,0])
        np.testing.assert_allclose(result['coherent_same_lag_R10_R21']['imag'],[2,0])
        self.assertLess(result['ss_product_residual_max_abs'],1e-15)

    def test_complex_lag_smoothing_preserves_phase_orientation(self):
        lags = {0: 2.1, 1: 1+1j, 2: 0, 3: 1-1j,
                -1: 1-1j, -2: 0, -3: 1+1j}
        actual = ch3._lag_smoothing(lags, 4)
        # Orthogonal unit-power source manifolds of squared norm four imply
        # SS=1.05*(a0*a0^H+a30*a30^H)+.0025I. Write its entries explicitly.
        expected = np.array([[2.1025, 1.05-1.05j, 0, 1.05+1.05j],
                             [1.05+1.05j, 2.1025, 1.05-1.05j, 0],
                             [0, 1.05+1.05j, 2.1025, 1.05-1.05j],
                             [1.05-1.05j, 0, 1.05+1.05j, 2.1025]])
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        np.testing.assert_allclose(np.linalg.eigvalsh(actual), [.0025, .0025, 4.2025, 4.2025], atol=2e-15)
        direct = np.array([[lags[i-j] for j in range(4)] for i in range(4)])
        np.testing.assert_allclose(actual, direct@direct.conj().T/4, atol=1e-15)
        # Reversing INSIDE every slice would conjugate this matrix and fail.
        self.assertAlmostEqual(abs(actual[0,1]-actual[0,1].conjugate()), 2.1)
        companion = ch3.coarray_companion()['ideal_complex_ss_matrix']
        np.testing.assert_allclose(np.array(companion['real'])+1j*np.array(companion['imag']), expected)

    def test_audio_independent_steady_sine_identity_and_null(self):
        case=dma_calibration_case();signals=case['signals'];tau=.01/343
        front=np.arange(2400,10400)/16000-.002
        p=2*math.pi*1000*tau
        expected=.5*np.sin(p)*np.cos(2*np.pi*1000*(front-tau))
        np.testing.assert_allclose(signals['dma_calibration_target'][2400:10400],expected,atol=4e-13)
        back=np.arange(15200,23200)/16000-.002-tau
        expected_back=-.0025*np.sin(2*np.pi*1600*back)
        np.testing.assert_allclose(signals['dma_calibration_mismatch'][15200:23200],expected_back,atol=5e-16)
        np.testing.assert_allclose(signals['dma_calibration_corrected'],signals['dma_calibration_target'],atol=1e-16)
        result=ch3.dma_gain_mismatch();front_row,back_row=result['steady_interval_projection']
        self.assertAlmostEqual(front_row['projected_output_amplitudes']['target'],.5*math.sin(p),places=12)
        self.assertAlmostEqual(back_row['projected_output_amplitudes']['mismatch'],.0025,places=12)
        self.assertLess(back_row['projected_output_amplitudes']['corrected'],1e-16)

    def test_audio_pcm_common_gain_shape_and_quantization(self):
        case=dma_calibration_case();files,groups=prepare_exports({'dma_calibration':case})
        self.assertEqual(len(files),4);self.assertEqual(groups['dma_calibration']['common_export_gain'],1)
        for filename,(blob,info) in files.items():
            rate,data=read_pcm16(blob);original=np.atleast_2d(case['signals'][filename[:-4]])
            self.assertEqual(rate,16000);self.assertEqual(data.shape,original.shape)
            self.assertEqual(data.shape[1],32000)
            self.assertLessEqual(np.max(abs(data-original)),.5/32768+1e-15)
            self.assertLess(np.max(abs(data)),.8)
        # Known correction produces identical PCM to the ideal target output.
        self.assertEqual(files['dma_calibration_corrected.wav'][0],files['dma_calibration_target.wav'][0])


if __name__=='__main__':unittest.main()
