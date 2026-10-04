"""Independent event accounting and rank-one closed-form physical controls."""
import cmath
import math
import unittest
from unittest.mock import patch
import numpy as np
from codes.chapters.ch11.core.selection import score_session_events
from codes.chapters.ch11.core.selection_physics import beamformer_selection_case
from codes.chapters.ch11.chapter11_experiments import scoring_events_anchor


class SelectionEvidenceTests(unittest.TestCase):
    def test_five_real_events_and_success_denominators(self):
        result = scoring_events_anchor()
        self.assertEqual([r['status'] for r in result['rows']],
                         ['scored', 'empty_output', 'missing_hypothesis', 'format_error', 'scoring_exception'])
        self.assertEqual([r['errors'] for r in result['rows']], [1, 2, None, None, None])
        self.assertEqual(result['rows'][3]['exception_type'], 'ValueError')
        self.assertEqual(result['rows'][4]['exception_type'], 'RuntimeError')
        self.assertEqual(result['planned_reference_words'], 10)
        self.assertEqual(result['scored_reference_words'], 4)
        self.assertEqual(result['scored_errors'], 3)
        self.assertEqual(result['session_coverage'], 2/5)
        self.assertEqual(result['reference_word_coverage'], 4/10)
        self.assertEqual(result['successful_subset_wer'], 3/4)
        self.assertIsNone(result['complete_wer'])
        self.assertFalse(result['eligible_for_complete_ranking'])
        self.assertFalse(result['edit_components_inferred'])

    def test_missing_does_not_call_scorer_and_all_failed_has_no_wer(self):
        calls = []
        def score(ref, hyp):
            calls.append((ref, hyp))
            raise ArithmeticError('independent callback failure')
        result = score_session_events([{'session_id': 'a', 'reference': ['a']},
            {'session_id': 'b', 'reference': ['b'], 'hypothesis': ['b']},
            {'session_id': 'c', 'reference': ['c'], 'hypothesis': None}], scorer=score)
        self.assertEqual(calls, [(['b'], ['b'])])
        self.assertEqual(result['failed_sessions'], 3)
        self.assertIsNone(result['successful_subset_wer'])
        self.assertIsNone(result['complete_wer'])

    def test_complete_empty_hypothesis_and_insertions_above_100_percent(self):
        result = score_session_events([{'session_id': 'a', 'reference': ['a'], 'hypothesis': []},
            {'session_id': 'b', 'reference': ['b'], 'hypothesis': ['x', 'y', 'z', 'w']}])
        self.assertEqual(result['scored_errors'], 5)
        self.assertEqual(result['complete_wer'], 5/2)
        self.assertTrue(result['eligible_for_complete_ranking'])
        self.assertEqual(result['session_coverage'], 1)

    def test_bad_plan_is_rejected_before_any_callback(self):
        plans = [[], [{'session_id': 'a', 'reference': []}],
                 [{'session_id': 'a', 'reference': 'a'}],
                 [{'session_id': 'a', 'reference': ['a']}, {'session_id': 'a', 'reference': ['b']}],
                 [{'session_id': True, 'reference': ['a']}]]
        for plan in plans:
            with self.subTest(plan=plan), patch('builtins.print'):
                calls = []
                with self.assertRaises(ValueError):
                    score_session_events(plan, scorer=lambda r, h: calls.append(1))
                self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            score_session_events([{'session_id': 'a', 'reference': ['a']}], scorer=None)

    def test_bad_callback_counts_are_failures_not_success(self):
        for value in [True, -1, .5, math.nan, math.inf, '1']:
            with self.subTest(value=value):
                result = score_session_events([{'session_id': 'a', 'reference': ['a'],
                    'hypothesis': ['a']}], scorer=lambda r, h: value)
                self.assertEqual(result['rows'][0]['status'], 'scoring_exception')
                self.assertIsNone(result['complete_wer'])

    def test_callback_mutation_does_not_change_frozen_denominators(self):
        sessions = [{'session_id': 'a', 'reference': ['a', 'b'], 'hypothesis': ['a']}]
        def score(reference, hypothesis):
            reference.clear()
            hypothesis.append('mutated')
            return 1
        result = score_session_events(sessions, scorer=score)
        self.assertEqual(result['planned_reference_words'], 2)
        self.assertEqual(result['scored_reference_words'], 2)
        self.assertEqual(result['complete_wer'], .5)
        self.assertEqual(sessions[0]['reference'], ['a', 'b'])
        self.assertEqual(sessions[0]['hypothesis'], ['a'])

    def test_physical_case_from_scalar_sherman_morrison(self):
        result = beamformer_selection_case()
        xs = [-.04, 0., .04]
        v = lambda deg: [cmath.exp(2j*math.pi*1000*x*math.sin(math.radians(deg))/343) for x in xs]
        a, b, actual = v(0), v(20), v(5)
        dot = lambda x, y: sum(z.conjugate()*t for z, t in zip(x, y))
        for row, alpha in zip(result['candidates'], [None, 0., .1, 1.]):
            if alpha is None:
                weights = [1/3]*3
            else:
                q = .01+1.01*alpha
                z = [(a[i]-b[i]*dot(b, a)/(q+3))/q for i in range(3)]
                weights = [x/dot(a, z).real for x in z]
            got = [complex(*pair) for pair in row['weights_real_imag']]
            np.testing.assert_allclose(got, weights, rtol=2e-13, atol=2e-13)
            norm = sum(abs(x)**2 for x in weights)
            h = dot(weights, actual)
            noise = abs(dot(weights, b))**2+.01*norm
            diffuse = 0j
            for i, x in enumerate(xs):
                for j, y in enumerate(xs):
                    argument = 2*math.pi*1000*abs(x-y)/343
                    gamma = math.sin(argument)/argument if argument else 1.
                    diffuse += weights[i].conjugate()*gamma*weights[j]
            self.assertAlmostEqual(row['WNG_dB'], -10*math.log10(norm), 11)
            self.assertAlmostEqual(row['DI_dB'], -10*math.log10(diffuse.real), 11)
            self.assertAlmostEqual(row['actual_noise_power'], noise, 12)
            self.assertAlmostEqual(row['actual_NMSE'], noise+abs(h-1)**2, 12)
            self.assertAlmostEqual(row['response_amplitude_error'], abs(h-1), 12)
            self.assertAlmostEqual(complex(*row['nominal_response_real_imag']).real, 1, 12)
            self.assertAlmostEqual(row['nominal_NMSE'], noise, 12)
        self.assertEqual(result['acoustically_eligible'], ['ds', 'mvdr_1'])
        self.assertEqual(result['lowest_actual_NMSE_eligible'], 'mvdr_1')
        self.assertIsNone(result['device_selected'])
        self.assertEqual([r['device_verdict'] for r in result['candidates']],
                         ['undetermined', 'fail', 'fail', 'undetermined'])
        self.assertGreater(result['candidates'][-1]['response_amplitude_error'], .01)
        self.assertLess(result['candidates'][-1]['response_amplitude_error'], .03)

    def test_physical_candidates_really_reuse_unique_solvers(self):
        from codes.chapters.ch05.core.beamforming import mvdr_weights, dsb_weights
        with patch('codes.chapters.ch11.core.selection_physics.mvdr_weights', wraps=mvdr_weights) as mvdr, \
             patch('codes.chapters.ch11.core.selection_physics.dsb_weights', wraps=dsb_weights) as ds:
            result = beamformer_selection_case()
        self.assertEqual(mvdr.call_count, 3)
        self.assertEqual(ds.call_count, 1)
        self.assertEqual([call.kwargs['relative_diagonal_loading'] for call in mvdr.call_args_list], [0., .1, 1.])
        # The covariance actually optimized is not the diffuse model used for DI.
        R = np.asarray(result['noise_covariance_real_imag'])
        R = R[..., 0]+1j*R[..., 1]
        self.assertGreater(np.max(abs(R-np.asarray(result['diffuse_coherence']))), .2)


if __name__ == '__main__':
    unittest.main()
