"""Independent closed-form anchors and numerical failure checks for chapter 4."""
import unittest
import numpy as np
from codes.examples import chapter04_experiments as ex
from codes.array_tutorial.doa import esprit_ula, gcc_phat
from codes.array_tutorial.audio_samples import doa_ambiguity_case, prepare_exports, read_pcm16


class Chapter04ExperimentsTests(unittest.TestCase):
    def test_nyquist_recovers_original_grid_and_half_sample(self):
        r = ex.nyquist_interpolation()
        y = np.array(r['correlation'])
        np.testing.assert_allclose(y[::2], [0, 0, 0, 0, 1, 0, 0, 0], atol=2e-15)
        # Dirichlet endpoint splitting: cot(pi/16)/8 at half a sample.
        self.assertAlmostEqual(y[9], 1/(8*np.tan(np.pi/16)), places=14)
        self.assertLess(r['max_error'], 2e-15)

    def test_srp_two_frequency_scalar_cosines(self):
        r = ex.srp_frequency_and_grid()
        np.testing.assert_allclose(r['far_field_scores'], [-.5, np.sqrt(2)/4, 1], atol=1e-14)
        np.testing.assert_allclose(r['grid_peak_m'], [5.567415730337078, 3.838805970149254])
        self.assertAlmostEqual(r['grid_error_m'], .0777867856136663)
        self.assertGreater(r['grid_peak_score'], r['nearest_score'])
        # Independently derive first pair's error and the six scalar scores.
        import math
        p = r['grid_peak_m']; source = [5.5, 3.8]
        mics = [(0, 0), (4, 0), (0, 3), (4, 3)]
        delta = [math.dist(p, m)-math.dist(source, m) for m in mics]
        reference = sum(math.exp(-.5*((delta[i]-delta[j])*6000/343)**2)
                        for i in range(4) for j in range(i+1, 4))/6
        self.assertAlmostEqual(r['grid_peak_score'], reference, places=14)

    def test_phat_order_and_epsilon_boundaries(self):
        r = ex.phat_order_and_floors()
        self.assertEqual((r['framewise_then_average'], r['average_then_normalize']), (0, -1))
        np.testing.assert_allclose(r['add'], [100/101, .5, 1/11, 0])
        np.testing.assert_allclose(r['floor'], [1, 1, .1, 0])
        np.testing.assert_array_equal(r['gate'], [1, 0, 0, 0])

    def test_esprit_common_basis_and_principal_alias(self):
        r = ex.esprit_basis_and_alias()
        psi = np.array(r['mixed_basis_shift_real'])+1j*np.array(r['mixed_basis_shift_imag'])
        np.testing.assert_allclose(psi, [[-1j, -2j], [0, 1j]], atol=1e-14)
        np.testing.assert_allclose(r['estimated_angles_deg'], [-30, 30], atol=1e-12)
        self.assertAlmostEqual(r['principal_alias_deg'], np.rad2deg(np.arcsin(np.sin(np.deg2rad(40))-1)))

    def test_esprit_rejects_uninformative_nonpsd_rank_and_zero_mode(self):
        zero_mode = np.outer([1., 0., 1.], [1., 0., 1.])
        for matrix in [np.zeros((3, 3)), np.diag([-2., -1., 3.]),
                       np.diag([0., 0., 1.]), np.eye(3), zero_mode]:
            with self.subTest(matrix=matrix), self.assertRaises(np.linalg.LinAlgError):
                esprit_ula(matrix, source_count=1, spacing_m=.03, frequency_hz=1000)
        with self.assertRaises(ValueError):
            esprit_ula(np.eye(3), source_count=True, spacing_m=.03, frequency_hz=1000)

    def test_esprit_rejects_overselected_signal_dimension(self):
        a = np.exp(1j*.2*np.arange(4))
        for floor in [0., .1]:
            with self.subTest(floor=floor), self.assertRaises(np.linalg.LinAlgError):
                esprit_ula(np.outer(a, a.conj())+floor*np.eye(4), source_count=2,
                           spacing_m=.03, frequency_hz=1000)

    def test_esprit_extreme_positive_covariance_scaling(self):
        a = np.array([1., 1j, -1., -1j])
        covariance = np.outer(a, a.conj())+.1*np.eye(4)
        for factor in [1e-250, 1., 1e250]:
            with self.subTest(factor=factor):
                actual = esprit_ula(factor*covariance, source_count=1, spacing_m=.1715, frequency_hz=1000)
                np.testing.assert_allclose(actual, [np.pi/6], atol=1e-14)

    def test_gcc_positive_gain_is_not_polarity_invariance(self):
        reference = np.zeros(16); reference[5] = 1
        delayed = np.zeros(16); delayed[7] = 1
        original = gcc_phat(delayed, reference, 16000)
        scaled = gcc_phat(.001*delayed, 1000*reference, 16000)
        self.assertAlmostEqual(original[0], 2/16000)
        self.assertAlmostEqual(scaled[0], original[0])
        self.assertNotAlmostEqual(gcc_phat(-delayed, reference, 16000)[0], original[0])

    def test_gauss_newton_closed_form_normal_equations(self):
        r = ex.gauss_newton_one_step()
        z = np.array([np.sqrt(.85)-np.sqrt(.65), .5-np.sqrt(.65), np.sqrt(.45)-np.sqrt(.65)])
        j = -np.sqrt(2)*np.array([[1., 0.], [0., 1.], [1., 1.]])
        expected = np.array([[2., -1.], [-1., 2.]])/6 @ j.T @ z
        np.testing.assert_allclose(r['observation_range_difference_m'], z)
        np.testing.assert_allclose(r['jacobian'], j)
        np.testing.assert_allclose(r['step_m'], expected)
        self.assertAlmostEqual(r['residual_after_m'], .013927510559068, places=13)

    def test_crlb_exact_sine_square_sum(self):
        r = ex.local_delay_crlb()
        # Eight samples: 0,1/2,1,1/2,0,1/2,1,1/2 sum to four.
        self.assertAlmostEqual(r['fisher_information_per_s2']/(400_000_000*np.pi**2), 4)
        self.assertAlmostEqual(r['std_bound_us'], 25/np.pi, places=12)
        self.assertAlmostEqual(r['variance_bound_s2']*1e12, 625/np.pi**2, places=12)

    def test_audio_actual_pcm_steady_delay_and_quantization(self):
        case = doa_ambiguity_case()
        files, groups = prepare_exports({'doa_ambiguity': case})
        self.assertEqual(groups['doa_ambiguity']['common_export_gain'], 1)
        for filename, (blob, metadata) in files.items():
            fs, pcm = read_pcm16(blob)
            self.assertEqual((fs, pcm.shape), (16000, (2, 32000)))
            np.testing.assert_array_equal(pcm[0, 2:], pcm[1, :-2])
            np.testing.assert_array_equal(pcm[0, :2], [0, 0])
            self.assertLessEqual(metadata['quantization_max_abs_error'], .5/32768)

    def test_audio_scores_geometric_series_and_physical_aliases(self):
        r = ex.audio_phase_ambiguity()
        np.testing.assert_allclose(r['rows'][0]['pcm_scores'], [1, 1], atol=1e-14)
        # Sum exp(j*k*a), k20..384, with a=-pi/64, independently in closed form.
        a = -np.pi/64
        total = np.exp(1j*20*a)*(1-np.exp(1j*365*a))/(1-np.exp(1j*a))
        np.testing.assert_allclose(r['rows'][1]['pcm_scores'], [total.real/365, 1], atol=1e-14)
        self.assertTrue(all(abs(x) < r['physical_max_delay_samples'] for x in [-6, 2]))
        for row in r['rows']:
            np.testing.assert_allclose(row['pcm_scores'], row['float_scores'], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
