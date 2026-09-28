"""Independent arithmetic, small counterexamples and actual PCM for chapter 11."""
import math
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from codes.array_tutorial.selection import (upper_limit_verdict, pareto_minima,
                                            small_slot_word_errors, token_edit_distance)
from codes.array_tutorial.audio_samples import selection_tradeoff_case, pcm16_bytes
from codes.examples.chapter11_experiments import run_experiments
from codes.examples.engineering_boundary_exercises import paired_sign_test_lower_is_better
from codes.examples.tracking_time_exercises import zero_failure_upper_bound


class Chapter11ExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_experiments()
        cls.audio = selection_tradeoff_case()

    def test_identifiers_and_pareto_analytic_intersection(self):
        self.assertEqual(list(self.results), [f'E11-{i}' for i in range(10, 20)])
        x = self.results['E11-10']
        self.assertEqual(x['feasible'], ['A', 'B', 'C'])
        self.assertEqual(x['pareto'], ['A', 'B'])
        self.assertAlmostEqual(x['weight_crossing'], 4/7)
        self.assertEqual([x['weighted_rankings'][k]['winner'] for k in ['0.5', '0.8']], ['A', 'B'])
        np.testing.assert_allclose(x['weighted_rankings']['0.5']['losses'], [17/30, 7/12, 13/20])

    def test_pareto_equal_candidates_remain_and_invalid_rejected(self):
        self.assertEqual(pareto_minima([[1, 2], [1, 2], [2, 2], [3, 0]]), [0, 1, 3])
        for invalid in ([], [[np.nan]], [[1j]], [[True]]):
            with self.assertRaises(ValueError):
                pareto_minima(invalid)

    def test_selection_keeps_exact_integer_order(self):
        for lower in (2**53, int(np.iinfo(np.int64).max)-1, 10**400):
            with self.subTest(lower=lower):
                self.assertEqual(pareto_minima([[lower], [lower+1]]), [0])
                self.assertEqual(upper_limit_verdict([[lower+1, lower+1]], [lower]), 'fail')
                self.assertEqual(upper_limit_verdict([[lower, lower+1]], [lower]), 'undetermined')
                self.assertEqual(upper_limit_verdict([[lower, lower]], [lower]), 'pass')
                with self.assertRaises(ValueError):
                    upper_limit_verdict([[lower+1, lower]], [lower])
        # Both mixed columns and mixed endpoints preserve the integer operand.
        self.assertEqual(pareto_minima([[float(2**53), 1.5], [2**53+1, 1.5]]), [0])
        self.assertEqual(upper_limit_verdict([[2**53+1, 2**53+1]], [float(2**53)]), 'fail')
        self.assertEqual(upper_limit_verdict([[float(2**53), 2**53+1]], [2**53]), 'undetermined')
        for invalid in (True, 1j, float('nan'), float('inf'), '2'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    pareto_minima([[invalid]])
                with self.assertRaises(ValueError):
                    upper_limit_verdict([[invalid, invalid]], [1])

    def test_composition_reversal_and_fixed_weights(self):
        x = self.results['E11-11']
        np.testing.assert_allclose(x['scene_rates'], [[1/10, 3/10], [2/25, 7/25]])
        np.testing.assert_allclose(x['pooled_rates'], [3/25, 13/50])
        np.testing.assert_allclose(x['fixed_weight_rates'], [1/5, 9/50])

    def test_missing_score_interval_contains_competitor(self):
        x = self.results['E11-12']
        np.testing.assert_allclose(x['complete_mean_bounds'], [81/25, 101/25])
        self.assertFalse(x['ranking_determined'])
        self.assertFalse(x['is_confidence_interval'])

    def test_joint_risk_coverage_uses_bonferroni_not_independence(self):
        x = self.results['E11-13']
        upper = x['simultaneous_upper_each']
        self.assertAlmostEqual((1-upper)**100, 1/60)
        self.assertEqual(x['minimum_zero_failure_trials_each'], 408)
        self.assertGreater(.99**407, 1/60)
        self.assertLessEqual(.99**408, 1/60)

    def test_interval_verdict_boundaries_and_missingness(self):
        self.assertEqual(self.results['E11-14']['verdicts'],
                         {'A': 'pass', 'B': 'undetermined', 'C': 'fail', 'D': 'undetermined'})
        self.assertEqual(upper_limit_verdict([[150, 150]], [150]), 'pass')
        self.assertEqual(upper_limit_verdict([[150, 151]], [150]), 'undetermined')
        self.assertEqual(upper_limit_verdict([None, [2, 3]], [1, 1]), 'fail')
        for bounds, limits in (([[2, 1]], [3]), ([[0, np.inf]], [3]), ([], []), ([None], [])):
            with self.assertRaises(ValueError):
                upper_limit_verdict(bounds, limits)

    def test_interaction_is_computed_from_four_cells(self):
        x = self.results['E11-15']
        self.assertEqual(x['interaction_percentage_points'], -15)
        self.assertEqual(x['best_configuration'], 'AB')
        # At A absent, B costs +3; at A present, B changes errors by -12.
        self.assertEqual((x['errors']['B']-x['errors']['none'], x['errors']['AB']-x['errors']['A']), (3, -12))

    def test_slot_reuse_has_global_permutation_penalty(self):
        x = self.results['E11-16']
        self.assertEqual((x['cp_errors'], x['orc_errors']), (2, 0))
        self.assertEqual(x['orc_assignment_utterance_to_slot'], [0, 1, 0])
        self.assertEqual(x['cp_wer'], 2/3)
        reordered = small_slot_word_errors([['a'], ['b'], ['c']], [0, 1, 2], [['b'], ['a', 'c']])
        self.assertEqual((reordered['cp_errors'], reordered['orc_errors']), (2, 0))
        identity = small_slot_word_errors([['a'], ['b'], ['c']], [0, 1, 0], [['a', 'c'], ['b']])
        self.assertEqual((identity['cp_errors'], identity['orc_errors']), (0, 0))

    def test_edit_distance_and_bounded_domain(self):
        self.assertEqual(token_edit_distance(['a', 'b', 'c'], ['a', 'x', 'c', 'd']), 2)
        self.assertEqual(token_edit_distance([], ['a', 'b']), 2)
        with self.assertRaises(ValueError):
            small_slot_word_errors([['a']]*7, [0]*7, [['a']])
        with self.assertRaises(ValueError):
            small_slot_word_errors([[]], [0], [['a']])

    def test_threshold_cost_direction_and_exposure(self):
        a, b = self.results['E11-17']['rows']
        self.assertEqual([r['misses'] for r in (a, b)], [1, 2])
        self.assertEqual([r['false_events_per_hour'] for r in (a, b)], [.5, 0])
        self.assertEqual([r['cost_2miss_5false'] for r in (a, b)], [7, 4])
        self.assertEqual([r['cost_10miss_1false'] for r in (a, b)], [11, 20])
        self.assertEqual([r['passes_miss_rate_limit'] for r in (a, b)], [True, False])

    def test_same_index_clock_error_by_physical_times(self):
        x = self.results['E11-18']
        expected = [1000*(t-t/(1+80e-6)) for t in [1, 10, 300]]
        np.testing.assert_allclose(x['same_index_exact_ms'], expected, rtol=2e-12)
        self.assertAlmostEqual(x['exact_interval_without_initial_residual_s'], 1.2501)
        interval = x['robust_interval_s']
        self.assertAlmostEqual(interval, 100009/112500)
        self.assertAlmostEqual(20e-6 + interval - interval/(1+90e-6), 100e-6, places=15)

    def test_sign_test_preserves_big_integer_and_mixed_comparisons(self):
        for baseline, candidate in (([2**53+1], [2**53]),
                                    ([np.iinfo(np.int64).max], [np.iinfo(np.int64).max-1]),
                                    ([2**53+1, 1.5], [float(2**53), 1.5]),
                                    ([10**400], [10**400-1])):
            result = paired_sign_test_lower_is_better(baseline, candidate)
            self.assertEqual(result['wins'], 1)
            self.assertEqual(result['one_sided_pvalue'], .5)

    def test_sign_tail_reports_positive_underflow(self):
        x = paired_sign_test_lower_is_better([1]*1074, [0]*1074)
        self.assertEqual(x['one_sided_pvalue'], float.fromhex('0x0.0000000000001p-1022'))
        self.assertFalse(x['pvalue_underflow'])
        y = paired_sign_test_lower_is_better([1]*2000, [0]*2000)
        self.assertEqual(y['one_sided_pvalue'], 0)
        self.assertTrue(y['pvalue_underflow'])
        self.assertEqual(y['status'], 'positive_pvalue_underflow')
        self.assertAlmostEqual(y['log_one_sided_pvalue'], -2000*math.log(2))
        z = paired_sign_test_lower_is_better([1], [1])
        self.assertIsNone(z['log_one_sided_pvalue'])

    def test_zero_failure_extreme_integer_is_controlled(self):
        self.assertGreater(zero_failure_upper_bound(10**300), 0)
        with self.assertRaisesRegex(ValueError, 'range'):
            zero_failure_upper_bound(10**400)

    def test_filter_response_from_boxcar_trigonometric_sum(self):
        for length in (3, 9):
            name = f'selection_fir{length}'
            def response(f):
                return math.sin(length*math.pi*f/16000)/(length*math.sin(math.pi*f/16000))
            expected = abs(response(1500)/response(500))
            self.assertAlmostEqual(self.audio['analytic_response'][name]['amplitude_response']['1500'], expected)
            gain_noise = abs(response(3500)/response(500))
            # Equal 500/1500 reference energies, 500 response exactly one.
            expected_nmse = ((expected-1)**2 + gain_noise**2)/2
            self.assertAlmostEqual(self.audio['float_analysis']['candidates'][name]['aligned_total_nmse'], expected_nmse)

    def test_causal_convolution_startup_by_explicit_sample_sum(self):
        signals = self.audio['signals']
        for length in (3, 9):
            name = f'selection_fir{length}'
            coefficient = self.audio['parameters']['filters'][name]['taps'][0]
            for n in [0, 1, 7, 9, 318, 1600, 31999]:
                expected = coefficient * sum(signals['selection_mixture'][0, n-k]
                                             for k in range(min(length, n+1)))
                self.assertAlmostEqual(signals[name][0, n], expected, places=15)

    def test_pcm_readback_uses_pcm_reference_and_independent_fourier_projection(self):
        # Real WAV bytes are written/read. DFT projection is independent of the
        # generator's seven-column least squares fit (integer periods here).
        with tempfile.TemporaryDirectory() as directory:
            values = {}
            for name, signal in self.audio['signals'].items():
                path = Path(directory)/(name+'.wav')
                path.write_bytes(pcm16_bytes(.8*signal))
                with wave.open(str(path), 'rb') as wav:
                    self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (1, 16000, 32000))
                    values[name] = np.frombuffer(wav.readframes(32000), dtype='<i2').astype(float)/32768
            clean = values['selection_clean'][1600:30400]
            def amplitude(signal, f, start, stop):
                return 2*abs(np.dot(signal[start:stop], np.exp(-2j*np.pi*f*np.arange(start, stop)/16000)))/(stop-start)
            for length in (3, 9):
                name, delay = f'selection_fir{length}', (length-1)//2
                output, start, stop = values[name], 1600+delay, 30400+delay
                actual = self.audio['pcm_analysis']['candidates'][name]
                target = amplitude(output, 1500, start, stop)/amplitude(values['selection_clean'], 1500, 1600, 30400)
                noise = amplitude(output, 3500, start, stop)/amplitude(values['selection_mixture'], 3500, 1600, 30400)
                self.assertAlmostEqual(actual['target_1500_retention'], target, places=12)
                self.assertAlmostEqual(actual['noise_attenuation_db'], -20*math.log10(noise), places=10)
                self.assertAlmostEqual(actual['aligned_total_nmse'], np.dot(output[start:stop]-clean, output[start:stop]-clean)/np.dot(clean, clean), places=14)
            self.assertGreater(self.audio['pcm_analysis']['candidates']['selection_fir3']['target_1500_retention'], .89)
            self.assertLess(self.audio['pcm_analysis']['candidates']['selection_fir9']['target_1500_retention'], .21)


if __name__ == '__main__':
    unittest.main()
