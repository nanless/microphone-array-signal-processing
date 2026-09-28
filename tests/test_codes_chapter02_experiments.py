"""Independent hand calculations and physical limits for chapter 2 examples."""
import json
import math
import unittest

import numpy as np

from codes.examples import chapter02_experiments as ch2
from codes.array_tutorial.audio_samples import (room_decay_case, room_decay_components,
                                               prepare_exports, read_pcm16)


class Chapter02ExperimentsTest(unittest.TestCase):
    def test_ids_and_serializable_results(self):
        result = ch2.run_exercises()
        self.assertEqual(set(result), {f'E02-{number:02}' for number in range(9, 16)})
        json.dumps(result, allow_nan=False)

    def test_near_field_hand_geometry_and_endfire_counterexample(self):
        rows = ch2.near_far_errors()['cases']
        near_front, near_side, far_front, far_side = rows
        # Pythagoras gives sqrt(5)/10 and sqrt(401)/10 metres.
        for row, radius, edge in ((near_front, .2, math.sqrt(5)/10),
                                   (far_front, 2., math.sqrt(401)/10)):
            np.testing.assert_allclose(row['ranges_m'], [edge, radius, edge], atol=1e-15)
            self.assertAlmostEqual(row['phase_error_deg'][0], -360000*(edge-radius)/343)
            self.assertAlmostEqual(row['spherical_amplitudes'][0], radius/edge)
        np.testing.assert_allclose(near_side['spherical_amplitudes'], [2/3, 1, 2])
        np.testing.assert_allclose(far_side['spherical_amplitudes'], [20/21, 1, 20/19])
        for row in (near_side, far_side):
            np.testing.assert_allclose(row['phase_error_deg'], [0, 0, 0], atol=1e-10)

    def test_diffuse_integrals_against_orthogonality_and_bessel_series(self):
        result = ch2.diffuse_integration()
        # Sphere integral: every nonzero integer spacing gives sin(q*pi)/(q*pi)=0.
        self.assertAlmostEqual(result['sphere_mean_power'], 1/4, places=12)
        self.assertAlmostEqual(result['sphere_di_db'], 10*math.log10(4), places=12)
        # Independent ring integral: J0(z)=sum((-z*z/4)^k/(k!)^2).
        def bessel_j0(z):
            term, terms = 1., [1.]
            for k in range(1, 80):
                term *= -(z*z/4)/(k*k)
                terms.append(term)
            return math.fsum(terms)
        ring = (4 + 2*sum((4-q)*bessel_j0(math.pi*q) for q in range(1, 4)))/16
        self.assertAlmostEqual(result['ring_mean_power'], ring, places=12)
        self.assertAlmostEqual(result['ring_directivity_db'], -10*math.log10(ring), places=11)
        self.assertEqual(result['wng_linear'], 4)

    def test_edc_rational_sum_time_shift_and_scale_invariance(self):
        expected = np.array([1., 5/21, 5/21, 1/21, 1/21])
        h = np.array([1., 0., .5, 0., .25])
        for scale in (1e-200, 1., 1e200):
            np.testing.assert_allclose(ch2.normalized_edc(scale*h), expected, atol=1e-15)
        np.testing.assert_allclose(ch2.normalized_edc(np.pad(h, (2, 0))),
                                   np.r_[1., 1., expected], atol=1e-15)
        np.testing.assert_array_equal(ch2.normalized_edc([1., 0., 0.]), [1, 0, 0])
        for invalid in ([], [0., 0.], [np.nan], [np.inf], [1j], [[1.]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ch2.normalized_edc(invalid)

    def test_crb_matches_centered_position_information(self):
        rows = ch2.crb_parameter_changes()['cases']
        centered_positions = [.014*(m-2.5) for m in range(6)]
        position_energy = math.fsum(x*x for x in centered_positions)
        for row in rows:
            # Fisher information after projection away from unknown source amplitude.
            information = (2*row['independent_snapshots']*row['linear_snr']
                           *(2*math.pi/.343)**2
                           *math.cos(math.radians(row['azimuth_deg']))**2*position_energy)
            self.assertAlmostEqual(row['variance_bound_rad2'], 1/information, places=15)
        self.assertAlmostEqual(rows[0]['variance_bound_rad2']/rows[1]['variance_bound_rad2'], 10.)
        self.assertAlmostEqual(rows[1]['variance_bound_rad2'], rows[2]['variance_bound_rad2'])
        self.assertAlmostEqual(rows[3]['standard_deviation_bound_deg'],
                               2*rows[0]['standard_deviation_bound_deg'])

    def test_four_point_fft_and_window_support_from_hand_sum(self):
        result = ch2.four_sample_stft()
        np.testing.assert_allclose(result['periodic_hann'], [0, .5, 1, .5], atol=1e-15)
        np.testing.assert_allclose(result['windowed_frame'], [0, 1, 3, 2], atol=1e-15)
        np.testing.assert_allclose(result['rfft_real'], [6, -3, 0], atol=1e-15)
        np.testing.assert_allclose(result['rfft_imag'], [0, 1, 0], atol=1e-15)
        np.testing.assert_allclose(result['window_squared_sum'], [0, .25, 1, .5, 1, .25], atol=1e-15)
        np.testing.assert_allclose(result['synthesis_numerator'], [0, .5, 3, 2, 5, 1.5], atol=1e-15)
        np.testing.assert_allclose(result['local_restored_samples'], [3, 4], atol=1e-15)

    def test_centered_and_uncentered_matrices_and_short_fir(self):
        result = ch2.sample_centering()
        self.assertEqual(result['uncentered_second_moment'], [[1, 1], [1, 2]])
        self.assertEqual(result['centered_divided_by_L'], [[0, 0], [0, 1]])
        self.assertEqual(result['centered_divided_by_L_minus_1'], [[0, 0], [0, 2]])
        self.assertEqual(result['identity_residual_max_abs'], 0)
        fir = ch2.short_fir_convolution()
        self.assertEqual(fir['full_output'], [1, 2, .5, 0, 0, -.5])
        self.assertEqual(fir['direct_energy'], 6)
        self.assertEqual(fir['reflection_energy'], 1.5)
        self.assertEqual(fir['twice_cross_energy'], -2)
        self.assertEqual(fir['total_energy'], 5.5)
        self.assertEqual(fir['cropped_first_four_energy'], 5.25)

    def test_room_filters_have_disjoint_components_and_controlled_energy(self):
        parts = room_decay_components()
        direct = parts['direct_rir']
        self.assertEqual(np.count_nonzero(direct), 1)
        self.assertEqual(direct[192], 1)
        self.assertEqual(len(direct), 19712)
        for name, energy in [('short_drr0', 1.), ('long_drr0', 1.), ('long_drr6', 10**(-.6))]:
            tail = parts['tails'][name]['tail']
            np.testing.assert_array_equal(tail[:512], np.zeros(512))
            self.assertEqual(float(direct @ tail), 0)
            self.assertAlmostEqual(float(tail @ tail), energy, places=13)
        np.testing.assert_allclose(parts['tails']['long_drr6']['tail'],
                                   parts['tails']['long_drr0']['tail'] * 10**(-.3), atol=1e-16)

    def test_audio_fft_convolution_against_direct_time_domain_sums(self):
        parts, case = room_decay_components(), room_decay_case()
        source = parts['source']
        reference = case['signals']['room_decay_dry']
        np.testing.assert_array_equal(reference[192:], source[:-192])
        np.testing.assert_array_equal(reference[:192], np.zeros(192))
        for name, item in parts['tails'].items():
            h = parts['direct_rir'] + item['tail']
            output = case['signals']['room_decay_' + name]
            for n in (2592, 2912, 3500, 9200, 16000, 28000, 30000):
                indices = np.arange(max(0, n-source.size+1), min(n+1, h.size))
                expected = float(h[indices] @ source[n-indices])
                self.assertAlmostEqual(output[n], expected, places=13)
        for row in case['parameters']['conditions']:
            self.assertAlmostEqual(row['total_output_energy'], row['direct_output_energy']
                                   +row['tail_output_energy']+row['output_cross_energy'], places=10)
            self.assertAlmostEqual(row['measured_rir_drr_db'], row['target_rir_drr_db'], places=12)
            self.assertAlmostEqual(row['total_edc_first_minus60_absolute_s']-.012,
                                   row['total_edc_first_minus60_after_direct_s'])

    def test_four_audio_files_preserve_one_gain_and_quantization_bound(self):
        case = room_decay_case()
        files, groups = prepare_exports({'room_decay': case})
        self.assertEqual(set(files), {'room_decay_dry.wav', 'room_decay_short_drr0.wav',
                                     'room_decay_long_drr0.wav', 'room_decay_long_drr6.wav'})
        gain = groups['room_decay']['common_export_gain']
        for filename, (blob, info) in files.items():
            rate, pcm = read_pcm16(blob)
            self.assertEqual(rate, 16000)
            self.assertEqual(pcm.shape, (1, 40000))
            self.assertEqual(info['common_export_gain'], gain)
            np.testing.assert_allclose(pcm[0], gain*case['signals'][filename[:-4]],
                                       rtol=0, atol=.5/32768+1e-15)
            self.assertLessEqual(float(np.max(abs(pcm))), .8+.5/32768)


if __name__ == '__main__':
    unittest.main()
