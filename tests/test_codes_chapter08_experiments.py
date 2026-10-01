"""Analytic/fractional expectations and actual PCM for E08-12..29."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from codes.chapters.ch08.chapter08_experiments import ROOT, css_published_audio, run_experiments


def complex_array(value):
    return np.array(value['real']) + 1j * np.array(value['imag'])


class Chapter08Exercises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_experiments()

    def test_ids_and_finite_json(self):
        self.assertEqual(set(self.r), {f'E08-{i:02}' for i in range(12, 30)})
        json.dumps(self.r, allow_nan=False)

    def test_sequential_complex_rows(self):
        r = self.r['E08-12']
        expected = [[np.sqrt(2/3), -1j/np.sqrt(6)], [0, 1/np.sqrt(2)]]
        np.testing.assert_allclose(complex_array(r['sequential_rows']), expected, atol=1e-15)
        np.testing.assert_allclose(complex_array(r['simultaneous_weighted_gram']), [[1, -.5j], [.5j, 1]], atol=1e-15)

    def test_joint_vs_scalar_regression(self):
        r = self.r['E08-13']
        np.testing.assert_allclose(r['joint_gains'], [1, 2])
        np.testing.assert_allclose(r['individual_gains'], [2, 5/2])
        np.testing.assert_allclose(r['individual_sum'], [2, 5/2, 9/2])
        np.testing.assert_allclose(complex_array(r['complex_application_gain']), [2+1j])

    def test_zero_reference_image(self):
        r = self.r['E08-14']
        self.assertEqual(r['reference0_images'][1], [0, 0, 0, 0])
        self.assertEqual(r['reference1_images'][1], [0, 0, 1, -1])
        self.assertEqual(r['second_reference0_score'], 'undefined_zero_reference_and_estimate')

    def test_mixture_consistency_not_independence(self):
        r = self.r['E08-15']
        self.assertEqual(r['sum_error'], 0)
        self.assertEqual(r['output_rank'], 1)
        np.testing.assert_allclose(r['fixed_si_sdr_db'], [0, 0], atol=1e-14)

    def test_direction_vs_power(self):
        r = self.r['E08-16']
        np.testing.assert_allclose(r['identity_shape_density'], [1, 1])
        np.testing.assert_allclose(r['original_scm'], np.diag([.5, .5]))
        np.testing.assert_allclose(r['radially_changed_scm'], np.diag([.5, 4.5]))

    def test_symmetric_guided_classes(self):
        r = self.r['E08-17']
        np.testing.assert_allclose(r['posterior'], 1/3, atol=1e-15)
        self.assertEqual(r['resets'], 0)
        self.assertEqual(r['beam_diagnostics']['target_relative_eigenvalue_gap'], [0.])

    def test_conditional_prior_counterexample(self):
        r = self.r['E08-18']
        np.testing.assert_allclose(r['posterior'], [[9/10, 1/10], [0, 1], [0, 1]])
        np.testing.assert_allclose(r['mean_posterior_priors'], [.3, .7])
        self.assertAlmostEqual(r['change'], np.log(17/25))
        self.assertLess(r['change'], 0)

    def test_nmf_substep_objectives(self):
        r = self.r['E08-19']
        np.testing.assert_allclose(r['updated_basis'], [[1], [3]])
        np.testing.assert_allclose(r['updated_activation'], [[np.sqrt(2), np.sqrt(2)]])
        np.testing.assert_allclose(r['objectives_initial_basis_activation'], [20, 8+2*np.log(3), 4*np.sqrt(2)+2*np.log(2)+2*np.log(3)])

    def test_underdetermined_covariance(self):
        r = self.r['E08-20']
        np.testing.assert_allclose(r['observation_scm_a'], np.eye(2))
        np.testing.assert_allclose(r['observation_scm_b'], np.eye(2))
        np.testing.assert_allclose(r['first_wiener_image_a'], [.5, 1])
        np.testing.assert_allclose(r['first_wiener_image_b'], [.6, .4])

    def test_congruence_wiener_fractions(self):
        r = self.r['E08-21']
        np.testing.assert_allclose(r['commutator'], [[0, -5], [5, 0]])
        np.testing.assert_allclose(r['microphone_images'], [[11/12, 2/3], [13/12, 1/3]])
        np.testing.assert_allclose(r['sum_images'], [2, 1])

    def test_bounded_real_masks(self):
        r = self.r['E08-22']
        np.testing.assert_allclose(complex_array(r['complex_oracle_mask']), [5, (1+1j)/2])
        np.testing.assert_allclose(r['squared_errors'], [16/25, 1/2])

    def test_overlap_matching_and_ambiguities(self):
        r = self.r['E08-23']
        m = r['audio_parameters']['matching']
        np.testing.assert_allclose(m['absolute_centered_correlation'], [[20/101, 1], [1, 20/101]], atol=1e-13)
        self.assertEqual(m['current_indices_for_previous'], [1, 0])
        for key in ('silence', 'identical_slots'):
            self.assertEqual(r[key]['status'], 'ambiguous')
            self.assertIsNone(r[key]['current_indices_for_previous'])

    def test_published_css_pcm_score_and_sources(self):
        r = self.r['E08-23']['published_audio']
        self.assertEqual(r['score_interval_samples'], [0, 32000])
        self.assertEqual(r['common_export_gain'], 1)
        measured = r['actual_pcm_measurements']
        expected = {'css_overlap_naive': [-3.340758788514897, -3.3405346277825743],
                    'css_overlap_aligned': [19.998887508407464, 19.999652603596694]}
        for stem, values in expected.items():
            scores = measured[stem]['fixed_identity_scores']
            np.testing.assert_allclose([item['si_sdr_db'] for item in scores], values, atol=1e-11)
            self.assertEqual(measured[stem]['samples_per_channel'], 32000)
            for score in scores:
                self.assertIsInstance(score['reference_centered_energy_numerator'], int)
                self.assertGreater(score['residual_energy_ratio_denominator'], 0)
        for name, digest in r['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_css_stale_source_and_corrupt_asset_are_not_repaired(self):
        original = ROOT/'codes/chapters/ch00/audio/MANIFEST.json'
        audio = ROOT/'codes/chapters/ch08/audio'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = json.loads(original.read_text())
            name = 'codes/chapters/ch08/core/css.py'
            manifest['generator_inputs'][name] = '0'*64
            path = directory/'MANIFEST.json'
            path.write_text(json.dumps(manifest))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'source SHA'):
                css_published_audio(manifest_path=path, audio_directory=audio)
            self.assertEqual(path.read_bytes(), before)
            path.write_bytes(original.read_bytes())
            for stem in ('reference', 'mixture', 'naive', 'aligned'):
                filename = f'css_overlap_{stem}.wav'
                (directory/filename).write_bytes((audio/filename).read_bytes())
            corrupted = directory/'css_overlap_aligned.wav'
            blob = bytearray(corrupted.read_bytes())
            blob[-1] ^= 1
            corrupted.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, 'metadata or SHA'):
                css_published_audio(manifest_path=path, audio_directory=directory)
            self.assertEqual(corrupted.read_bytes(), bytes(blob))

    def test_css_nonfinite_boolean_or_incomplete_scores_are_rejected(self):
        original = ROOT/'codes/chapters/ch00/audio/MANIFEST.json'
        audio = ROOT/'codes/chapters/ch08/audio'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'MANIFEST.json'
            for scores in ([float('nan'), 0.], [float('inf'), 0.],
                           [True, 0.], [0.], ['0', 0.]):
                with self.subTest(scores=scores):
                    manifest = json.loads(original.read_text())
                    manifest['groups']['css_overlap']['parameters']['pcm_si_sdr_db'][
                        'css_overlap_naive'] = scores
                    path.write_text(json.dumps(manifest))
                    before = path.read_bytes()
                    with self.assertRaises(ValueError):
                        css_published_audio(manifest_path=path, audio_directory=audio)
                    self.assertEqual(path.read_bytes(), before)

    def test_whitened_four_points_are_dependent(self):
        r = self.r['E08-24']
        np.testing.assert_allclose(r['mixture_covariance'], [[5/4, 1], [1, 5/4]])
        np.testing.assert_allclose(r['whitened_points'], r['source_points'], atol=1e-15)
        np.testing.assert_allclose(r['rotated_covariance'], np.eye(2), atol=1e-15)
        self.assertEqual(r['probability_first_zero'], 1/2)
        self.assertEqual(r['probability_second_zero'], 1/2)
        self.assertEqual(r['probability_both_zero'], 0)
        self.assertEqual(r['independent_product_probability'], 1/4)
        self.assertEqual(np.linalg.matrix_rank(r['rank_one_covariance']), 1)

    def test_three_spd_joint_congruence_obstruction(self):
        r = self.r['E08-25']
        np.testing.assert_array_equal(r['whitened_remaining_commutator'], [[0, -1], [1, 0]])
        np.testing.assert_allclose(r['shape_eigenvalues'], [[1, 1], [1, 2], [1, 3]])
        self.assertFalse(r['joint_congruence_exists_for_all_three'])

    def test_fastmnmf_jacobian_cancels_model_scale(self):
        first, second = self.r['E08-26']['cases']
        expected = 7/12+np.log(12)
        for case in (first, second):
            self.assertAlmostEqual(case['quadratic_term'], 7/12)
            self.assertAlmostEqual(case['full_objective'], expected)
            np.testing.assert_allclose(case['physical_covariance'], [[7, 3], [3, 3]])
        self.assertAlmostEqual(second['jacobian_term'], -2*np.log(4))
        self.assertAlmostEqual(second['objective_without_jacobian']-first['objective_without_jacobian'], 2*np.log(4))

    def test_weighted_projection_kkt_fractions(self):
        r = self.r['E08-27']
        np.testing.assert_array_equal(r['weighted_projection'], [3/2, 5/2])
        np.testing.assert_array_equal(r['equal_projection'], [2, 2])
        np.testing.assert_array_equal(r['kkt_correction_over_variance'], [1/2, 1/2])
        self.assertEqual(r['weighted_cost_at_weighted_projection'], 1/2)
        self.assertAlmostEqual(r['weighted_cost_at_equal_projection'], 2/3)

    def test_mask_representation_reads_published_pcm(self):
        r = self.r['E08-28']['published_audio']
        self.assertEqual(r['sample_rate_hz'], 16000)
        self.assertEqual(r['common_export_gain'], 1)
        self.assertEqual(len(r['files']), 6)
        # Exact integer-energy expectations are independently calculated from
        # the fixed two-tone oracle model, not copied from the float outputs.
        records = r['samples']
        for stem, numerator in [('bounded_real', 166479798100),
                                ('unbounded_real', 73012418550),
                                ('complex_oracle', 0)]:
            measured = records[stem]['pcm_measurements']
            self.assertEqual(measured['integer_error_squared_sum'], numerator)
            self.assertEqual(measured['integer_reference_squared_sum'], 292062085900)
            self.assertEqual(measured['integer_mse_denominator'], 27200*32768**2)

    def test_matched_slots_can_cancel_and_have_gain_error(self):
        r = self.r['E08-29']
        p = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
        first, second = r['cases']
        for case in (first, second):
            self.assertEqual(case['matching']['status'], 'matched')
            self.assertEqual(case['matching']['current_indices_for_previous'], [0, 1])
            np.testing.assert_array_equal(case['signed_centered_correlation'], -np.eye(2))
            np.testing.assert_array_equal(case['corrected_equal_overlap'], p)
        np.testing.assert_array_equal(first['naive_equal_overlap'], np.zeros((2, 4)))
        np.testing.assert_array_equal(first['direct_application_gain'], [-1, -1])
        np.testing.assert_array_equal(second['naive_equal_overlap'], -.5*p)
        np.testing.assert_array_equal(second['polarity_only_overlap'], 1.5*p)
        np.testing.assert_array_equal(second['direct_application_gain'], [-.5, -.5])
