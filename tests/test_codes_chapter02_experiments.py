"""Independent hand calculations and physical limits for chapter 2 examples."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

import numpy as np

from codes.chapters.ch02 import chapter02_experiments as ch2
from codes.chapters.ch00.core.audio_samples import (room_decay_case, room_decay_components,
                                               prepare_exports, read_pcm16)
from codes.chapters.ch02.core.spectral import istft


class Chapter02ExperimentsTest(unittest.TestCase):
    def test_ids_and_serializable_results(self):
        result = ch2.run_exercises()
        self.assertEqual(set(result), {f'E02-{number:02}' for number in range(9, 19)})
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

    def test_modified_spectra_real_projection_against_trigonometric_sum(self):
        # Direct real Fourier synthesis with odd/even endpoint weighting;
        # no rfft/irfft, stft or tested helper constructs the expected result.
        for nfft in (4, 5):
            spectrum = np.repeat(np.array([2+3j, 1+2j, 4+5j])[None, :, None], 4, axis=2)
            before = spectrum.copy()
            g = [.5-.5*math.cos(2*math.pi*q/nfft) for q in range(nfft)]
            num, den = [0.]*11, [0.]*11
            for frame in range(4):
                for q in range(nfft):
                    value = 2/nfft
                    for k, coeff in enumerate((1+2j, 4+5j), start=1):
                        if nfft % 2 == 0 and k == nfft//2:
                            value += coeff.real*math.cos(math.pi*q)/nfft
                        else:
                            phase = 2*math.pi*k*q/nfft
                            value += 2*(coeff.real*math.cos(phase)-coeff.imag*math.sin(phase))/nfft
                    n = 2*frame+q
                    num[n] += value*g[q]; den[n] += g[q]**2
            pad = nfft//2
            expected = [num[n]/den[n] for n in range(pad, pad+6)]
            np.testing.assert_allclose(istft(spectrum, n_fft=nfft, hop_length=2, length=6)[0], expected, atol=3e-15)
            np.testing.assert_array_equal(spectrum, before)
        rows = ch2.endpoint_projection()['cases']
        self.assertEqual([row['max_abs_output_difference'] == 0 for row in rows], [True, True, True, False])
        self.assertTrue(all(row['input_spectrum_unchanged'] for row in rows))

    def test_source_noise_cross_terms_from_independent_scalar_enumeration(self):
        outcomes = list(itertools.product((-1, 1), repeat=3))
        rows = []
        for s, u, v in outcomes:
            n = [-s/2+u/2+v/math.sqrt(2), -s/2+u/2-v/math.sqrt(2)]
            rows.append((s, n, [s+a for a in n]))
        rnn = [[math.fsum(n[i]*n[j] for _, n, _ in rows)/8 for j in range(2)] for i in range(2)]
        rxx = [[math.fsum(x[i]*x[j] for _, _, x in rows)/8 for j in range(2)] for i in range(2)]
        cross = [math.fsum(s*n[i] for s, n, _ in rows)/8 for i in range(2)]
        result = ch2.correlated_source_noise()
        np.testing.assert_allclose(result['noise_second_moment'], rnn, atol=3e-16)
        np.testing.assert_allclose(result['mixture_second_moment'], rxx, atol=3e-16)
        np.testing.assert_allclose(result['source_noise_cross_row'], cross, atol=1e-16)
        np.testing.assert_allclose(rnn, np.eye(2), atol=3e-16)
        np.testing.assert_allclose(rxx, np.eye(2), atol=3e-16)
        np.testing.assert_allclose(result['cross_contribution'], -np.ones((2, 2)), atol=3e-16)
        np.testing.assert_allclose(result['incorrect_uncorrelated_sum'], [[2, 1], [1, 2]], atol=3e-16)
        np.testing.assert_allclose(result['mixture_eigenvalues'], [1, 1], atol=3e-16)

    def test_coherent_and_noise_window_gains_from_closed_values(self):
        result = ch2.window_calibration()
        self.assertAlmostEqual(result['coherent_gain'], .5)
        self.assertAlmostEqual(result['power_gain'], 3/8)
        self.assertAlmostEqual(result['enbw_bins'], 3/2)
        tone = result['tone']
        for key, expected in (('raw_bin_magnitude', 2), ('naive_amplitude', .5), ('corrected_amplitude', 1),
                              ('original_mean_square', .5), ('windowed_mean_square', 3/16),
                              ('divided_by_U', .5), ('divided_by_G_squared', .75)):
            self.assertAlmostEqual(tone[key], expected)
        noise = result['white_noise_ensemble']
        self.assertEqual(noise['outcomes'], 256)
        np.testing.assert_array_equal(noise['mean'], np.zeros(8))
        np.testing.assert_array_equal(noise['second_moment'], np.eye(8))
        self.assertAlmostEqual(noise['windowed_mean_square'], 3/8)
        self.assertAlmostEqual(noise['divided_by_U'], 1)
        self.assertAlmostEqual(noise['divided_by_G_squared'], 3/2)

    def test_published_room_pcm_readback_and_scalar_energy(self):
        result = ch2.published_room_decay_measurements()
        actual = {}
        for stem, info in result['assets'].items():
            path = ch2.ROOT/info['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info['sha256'])
            with wave.open(str(path), 'rb') as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth()), (16000, 1, 40000, 2))
                actual[stem] = np.frombuffer(wav.readframes(40000), dtype='<i2').astype(float)/32768
        ref = actual['room_decay_dry']
        for row in result['pcm_measurements']['cases']:
            output = actual['room_decay_'+row['name']]
            energy = math.fsum(float(v*v) for v in output)
            self.assertAlmostEqual(row['total_output_energy'], energy, places=12)
            self.assertAlmostEqual(row['total_output_mean_square'], energy/40000, places=15)
            self.assertAlmostEqual(row['total_output_energy'], row['direct_reference_energy']
                                   +row['output_minus_reference_energy']+row['twice_reference_residual_cross_energy'], places=12)
        self.assertEqual(result['energy_interval_samples'], [0, 40000])

    def test_published_room_rejects_forged_pcm_and_stale_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = Path('codes/chapters/ch00/audio/MANIFEST.json')
            manifest = json.loads((ch2.ROOT/manifest_path).read_text())
            paths = set(manifest['generator_inputs']) | {str(manifest_path)}
            paths |= {f"codes/chapters/ch07/audio/{row['file']}" for row in manifest['files'] if row['group'] == 'room_decay'}
            for path in paths:
                target = root/path; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ch2.ROOT/path, target)
            ch2.published_room_decay_measurements(root)
            path = root/'codes/chapters/ch07/audio/room_decay_long_drr0.wav'
            original = path.read_bytes()
            altered = bytearray(original); altered[44+4000*2] ^= 128
            path.write_bytes(altered)
            for row in manifest['files']:
                if row['file'] == path.name:
                    row['sha256'] = hashlib.sha256(altered).hexdigest()
            (root/manifest_path).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'quantization mismatch'):
                ch2.published_room_decay_measurements(root)
            path.write_bytes(original)
            (root/'codes/chapters/ch02/core/spectral.py').write_text('stale source')
            with self.assertRaisesRegex(ValueError, 'source is stale'):
                ch2.published_room_decay_measurements(root)


if __name__ == '__main__':
    unittest.main()
