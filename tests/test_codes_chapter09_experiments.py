"""Chapter 9 anchors independently derived from fractions/moments/geometry."""
from fractions import Fraction as F
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
from codes.chapters.ch09.chapter09_experiments import (
    actual_tracking_audio, run_experiments, small_set_distances, white_acceleration_covariance,
)
from codes.chapters.ch09.examples.chapter09_tracking_audio import generate


class Chapter09ExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.audio_directory = Path(cls.temporary.name)/'tracking_audio'
        # Only this temporary fixture is generated; published assets are read-only.
        generate(cls.audio_directory)
        cls.results = run_experiments(audio_directory=cls.audio_directory)

    def test_fourteen_stable_ids(self):
        self.assertEqual(set(self.results), {f'E09-{i:02}' for i in range(10, 24)})

    def test_bearing_ekf_geometry_and_fraction_posterior(self):
        item = self.results['E09-10']
        self.assertAlmostEqual(item['bearing_rad'], math.atan(.5))
        np.testing.assert_allclose(item['jacobian_rad_per_m'], [.4, -.2])
        np.testing.assert_allclose(item['gain_m_per_rad'], [5/3, -5/6])
        np.testing.assert_allclose(item['updated_position_m'], [7/6, 23/12])
        np.testing.assert_allclose(item['posterior_covariance_m2'], [[1/12, 1/12], [1/12, 5/24]])
        # Independent central differences for the atan2(x,y) observation.
        position = np.array([1., 2.]); eps = 1e-5
        derivatives = []
        for direction in np.eye(2):
            high, low = position+eps*direction, position-eps*direction
            derivatives.append((math.atan2(*high)-math.atan2(*low))/(2*eps))
        np.testing.assert_allclose(derivatives, [.4, -.2], atol=1e-10)

    def test_ut_matches_gaussian_second_and_fourth_moments(self):
        item = self.results['E09-11']
        # For a standard normal E[X²]=1, E[X⁴]=3; odd moments vanish.
        self.assertEqual(item['mean'], 1.)
        self.assertEqual(item['variance'], 3-1**2)
        self.assertEqual(item['state_observation_cross_covariance'], 0.)
        self.assertEqual(item['covariance_weights'], [2., .5, .5])

    def test_circular_mean_and_supported_particle_boundary(self):
        item = self.results['E09-12']
        self.assertEqual(item['circular_mean_deg'], -180.)
        self.assertAlmostEqual(item['resultant_length'], math.cos(math.pi/180))
        self.assertIsNone(item['antipodal_angle_deg'])
        self.assertEqual(item['antipodal_status'], 'undefined_antipodal_mean')
        self.assertEqual(item['support_after_first'], [0., 1.])
        self.assertEqual(item['linear_underflow_indices_after_first'], [0])
        np.testing.assert_allclose(item['log_weights_after_first'], [-500000., 0.], atol=1e-8)
        np.testing.assert_allclose(item['support_after_second'], [.5, .5], atol=1e-10)
        np.testing.assert_allclose(item['log_weights_after_second'], [-math.log(2)]*2, atol=1e-10)
        self.assertEqual(item['explicit_zero_prior_after_update'], [0., 1.])
        self.assertEqual(item['prior_reset_method'], 'set_prior_weights')

    def test_jpda_normalizer_from_product_minus_collisions(self):
        item = self.results['E09-13']
        likelihood_ratios = [[math.exp(-(z-x)**2/18)/(3*math.sqrt(2*math.pi))/.01
                              for z in [15, 31, 48]] for x in [30, 50]]
        # Independent expansion: multiply all assignments then subtract three
        # forbidden double uses of each detection. No event-loop copy.
        a, b = likelihood_ratios
        normalizer = (.1+.9*sum(a))*(.1+.9*sum(b))-.81*sum(x*y for x, y in zip(a, b))
        self.assertAlmostEqual(item['normalizer'], normalizer, places=11)
        self.assertEqual(len(item['events']), 13)
        self.assertAlmostEqual(sum(row['probability'] for row in item['events']), 1.)
        expected_A31 = .9*a[1]*(.1+.9*(b[0]+b[2]))/normalizer
        self.assertAlmostEqual(item['marginals_miss_then_detections'][0][2], expected_A31)
        self.assertAlmostEqual(item['symmetric_update']['mixture_variance'], float(F(4,5)+F(8,5)**2))
        self.assertGreater(item['symmetric_update']['mixture_variance'], .8)

    def test_phd_does_not_determine_cardinality_distribution(self):
        item = self.results['E09-14']
        for key, variance in [('model_A_count_probability_0_1_2', 0), ('model_B_count_probability_0_1_2', 1)]:
            probabilities = item[key]
            mean = sum(i*p for i, p in enumerate(probabilities))
            self.assertEqual(mean, 1)
            self.assertEqual(sum((i-mean)**2*p for i, p in enumerate(probabilities)), variance)
        np.testing.assert_allclose(item['empty_observation_posterior_mass'], [.1, .1])

    def test_ospa_and_gospa_cardinality_normalization_and_boundaries(self):
        item = self.results['E09-15']
        self.assertEqual(item['original']['ospa'], 6.)
        self.assertEqual(item['original']['gospa_alpha2'], 7.)
        self.assertEqual(item['added_common_correct_target']['ospa'], 4.)
        self.assertEqual(item['added_common_correct_target']['gospa_alpha2'], 7.)
        self.assertEqual(small_set_distances([], [30, 70])['ospa'], 10.)
        self.assertEqual(small_set_distances([179], [-179])['ospa'], 2.)
        self.assertAlmostEqual(small_set_distances([32], [30, 70], order=2)['ospa'], math.sqrt(52))
        self.assertAlmostEqual(small_set_distances([32], [30, 70], order=2)['gospa_alpha2'], math.sqrt(54))
        self.assertEqual(small_set_distances([30, 70], [32]), small_set_distances([32], [30, 70]))
        for first, second, kwargs in [([0, 360], [0], {}), ([0], [1], {'order': .5}), ([0], [1], {'cutoff': 0})]:
            with self.assertRaises(ValueError):
                small_set_distances(first, second, **kwargs)

    def test_fixed_lag_matches_direct_joint_gaussian_conditioning(self):
        item = self.results['E09-16']
        # x1 prior variance2, x2 variance3, cov2; measurement covariance [[3,2],[2,4]].
        cross = np.array([2., 2.]); innovation_covariance = np.array([[3., 2.], [2., 4.]])
        posterior_mean = cross@np.linalg.solve(innovation_covariance, [1., 0.])
        posterior_variance = 2-cross@np.linalg.solve(innovation_covariance, cross)
        self.assertAlmostEqual(posterior_mean, .5)
        self.assertAlmostEqual(posterior_variance, .5)
        self.assertAlmostEqual(item['smoothed_time1_mean'], posterior_mean)
        self.assertAlmostEqual(item['smoothed_time1_variance'], posterior_variance)
        np.testing.assert_allclose(item['filtered_means'], [2/3, 1/4])
        np.testing.assert_allclose(item['filtered_variances'], [2/3, 5/8])
        self.assertEqual(item['extra_observation_delay_steps'], 1)

    def test_units_covariance_and_nis_scale_together(self):
        item = self.results['E09-17']; scale = math.pi/180
        np.testing.assert_allclose(item['posterior_state_deg'], [907/25, 29/5])
        np.testing.assert_allclose(item['posterior_covariance_deg_units'], [[144/25, 18/5], [18/5, 5]])
        np.testing.assert_allclose(np.array(item['posterior_state_rad'])/scale, [907/25, 29/5])
        np.testing.assert_allclose(np.array(item['posterior_covariance_rad_units'])/scale**2, [[144/25, 18/5], [18/5, 5]])
        self.assertAlmostEqual(item['nis_degrees'], 4/25)
        self.assertAlmostEqual(item['nis_radians'], 4/25)
        self.assertAlmostEqual(item['nis_if_mean_only_converted'], 4/25*scale**2)

    def test_continuous_noise_composition_and_negative_cross_term(self):
        item = self.results['E09-18']
        # Integrals of 2*[s²,s;s,1] from s=0 to .2.
        expected = np.array([[2/375, 1/25], [1/25, 2/5]])
        np.testing.assert_allclose(item['Q_full'], expected)
        np.testing.assert_allclose(item['composed_two_halves'], expected)
        self.assertNotEqual(item['wrong_sum_without_state_propagation'][0][0], expected[0, 0])
        np.testing.assert_allclose(item['prediction_without_observation'], [[3., 0.], [0., 1.]])
        np.testing.assert_array_equal(white_acceleration_covariance(0, 2), np.zeros((2, 2)))
        for dt, density in ((-1, 2), (1, -1), (True, 2)):
            with self.assertRaises(ValueError):
                white_acceleration_covariance(dt, density)

    def test_audio_uses_independent_denominators_and_measured_timing(self):
        item = self.results['E09-19']
        self.assertEqual(item['pcm_scores']['valid_observation_count'], 173)
        self.assertEqual(item['pcm_scores']['missing_observation_count'], 24)
        self.assertAlmostEqual(item['first_state_time_s'], 255.5/16000)
        self.assertEqual(item['first_available_time_s'], 512/16000)
        self.assertAlmostEqual(item['first_available_time_s']-item['first_state_time_s'], 256.5/16000)
        self.assertNotEqual(item['pcm_scores']['raw_valid_rmse_deg'], item['float_scores']['raw_valid_rmse_deg'])
        self.assertEqual(item['asset_check'], 'strict_read_only_passed')

    def test_audio_scores_decode_actual_file_not_replay_pcm(self):
        path = self.audio_directory/'array_noisy.wav'
        with wave.open(str(path), 'rb') as reader:
            count, channels = reader.getnframes(), reader.getnchannels()
            data = reader.readframes(count)
        integers = struct.unpack('<'+str(count*channels)+'h', data)
        expected = np.array([[integers[i*channels+j]/32768 for i in range(count)]
                             for j in range(channels)])
        from codes.chapters.ch09.core.tracking_audio import analyze_array
        with patch('codes.chapters.ch09.chapter09_experiments.analyze_array', wraps=analyze_array) as score:
            actual_tracking_audio(self.audio_directory)
        score.assert_called_once()
        np.testing.assert_array_equal(score.call_args.args[0], expected)

    def test_audio_tampering_is_rejected_without_repair(self):
        for member in ('array_noisy.wav', 'MANIFEST.json'):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as root:
                directory = Path(root)/'tracking_audio'
                generate(directory)
                path = directory/member
                changed = bytearray(path.read_bytes()); changed[-1] ^= 1
                path.write_bytes(changed)
                before = {p.name: p.read_bytes() for p in directory.iterdir()}
                with self.assertRaises(ValueError):
                    actual_tracking_audio(directory)
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_small_set_tiny_positive_cutoff_and_power_status(self):
        cutoff = 1e-200
        empty = small_set_distances([], [30], cutoff=cutoff, order=2)
        self.assertEqual(empty['ospa'], cutoff)
        self.assertAlmostEqual(empty['gospa_alpha2']/cutoff, 1/math.sqrt(2))
        near = small_set_distances([0], [cutoff/2], cutoff=cutoff, order=2)
        self.assertEqual(near['ospa'], cutoff/2)
        self.assertEqual(near['gospa_alpha2'], cutoff/2)
        self.assertIsNone(near['matched_capped_cost_power'])
        self.assertEqual(near['matched_capped_cost_power_status'], 'positive_power_underflow')
        exact_zero = small_set_distances([0], [0], cutoff=cutoff, order=2)
        self.assertEqual(exact_zero['matched_capped_cost_power'], 0)
        self.assertEqual(exact_zero['matched_capped_cost_power_status'], 'representable')
        with self.assertRaises(ValueError):
            small_set_distances([0], [math.ulp(0.)], cutoff=180, order=2)

    def test_correlated_observations_fraction_posterior(self):
        item = self.results['E09-20']
        rho = F(9, 10)
        determinant = 4-(1+rho)**2
        gain = (1-rho)/determinant
        self.assertEqual(gain, F(10, 39))
        np.testing.assert_allclose(item['joint_gain'], [float(gain)]*2)
        self.assertAlmostEqual(item['joint_mean'], float(2*gain))
        self.assertAlmostEqual(item['joint_variance'], float(1-2*gain))
        self.assertAlmostEqual(item['wrong_independent_mean'], 2/3)
        self.assertAlmostEqual(item['wrong_independent_variance'], 1/3)
        self.assertGreater(item['joint_variance'], item['wrong_independent_variance'])

    def test_full_time_bearing_scale_ambiguity_and_exact_rank(self):
        item = self.results['E09-21']
        np.testing.assert_array_equal(item['bearings_deg'], item['scaled_bearings_deg'])
        np.testing.assert_allclose(item['ranges_m'], [math.sqrt(5), math.sqrt(8), math.sqrt(13)])
        np.testing.assert_array_equal(item['scaled_ranges_m'], 2*np.array(item['ranges_m']))
        rows = [[2, -1, 0, 0], [2, -2, 2, -2], [2, -3, 4, -6]]
        np.testing.assert_array_equal(item['stacked_jacobian_without_positive_row_denominators'], rows)
        # Exact nonzero 3x3 minor (columns0,1,3) proves rank>=3; one
        # independent null direction proves rank<=3 without float SVD.
        a,b,c = [[F(row[j]) for j in (0,1,3)] for row in rows]
        determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        self.assertEqual(determinant, 4)
        self.assertEqual(item['stacked_rank'], 3)
        self.assertTrue(all(sum(F(v)*u for v,u in zip(row, [1,2,1,0])) == 0 for row in rows))
        np.testing.assert_array_equal(item['stacked_null_product'], [0,0,0])

    def test_empty_observation_bernoulli_differs_from_poisson_phd(self):
        item = self.results['E09-22']
        r = F(4,5)
        for case, detection, posterior in zip(item['cases'], [F(9,10), F(1,10)], [F(2,7), F(18,23)]):
            numerator = r*(1-detection)
            denominator = (1-r)+numerator
            self.assertEqual(numerator/denominator, posterior)
            self.assertAlmostEqual(case['empty_observation_probability'], float(denominator))
            self.assertAlmostEqual(case['bernoulli_posterior_existence'], float(posterior))
            self.assertAlmostEqual(case['different_poisson_phd_posterior_mass'], float(numerator))

    def test_late_observation_joint_conditioning_replay_and_wrong_time(self):
        item = self.results['E09-23']
        # Independent inverse of S=[[3,2],[2,4]], determinant8.
        inverse = [[F(1,2), F(-1,4)], [F(-1,4), F(3,8)]]
        cross = [F(2),F(3)]
        gain = [sum(cross[i]*inverse[i][j] for i in range(2)) for j in range(2)]
        self.assertEqual(gain, [F(1,4), F(5,8)])
        mean = 2*gain[0]-gain[1]
        variance = 3-sum(g*x for g,x in zip(gain,cross))
        self.assertEqual((mean, variance), (F(-1,8), F(5,8)))
        self.assertAlmostEqual(item['joint_mean'], float(mean))
        self.assertAlmostEqual(item['joint_variance'], float(variance))
        self.assertAlmostEqual(item['chronological_replay'][-1]['mean'], float(mean))
        self.assertAlmostEqual(item['chronological_replay'][-1]['variance'], float(variance))
        self.assertEqual(item['drop_late_observation'], {'mean': -.75, 'variance': .75})
        self.assertAlmostEqual(item['wrong_current_time_update']['mean'], 3/7)
        self.assertAlmostEqual(item['wrong_current_time_update']['variance'], 3/7)


if __name__ == '__main__':
    unittest.main()
