"""Independent scalar checks and optional original fixed-source DOA execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from codes.chapters.ch04.core import upstream_contracts as contracts

import numpy as np

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch04.examples import audit_upstream_doa as audit

REPORT = audit.ROOT / 'codes/chapters/ch04/reports/upstream_doa.json'
CAN_RUN = (importlib.util.find_spec('scipy') is not None
           and importlib.util.find_spec('pyroomacoustics') is not None
           and all((audit.CACHE / p / '.git').exists() for p in audit.REVISIONS))


def strict_json(text):
    def reject(value):
        raise ValueError(f'nonstandard JSON constant {value}')
    return json.loads(text, parse_constant=reject)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = strict_json(REPORT.read_text())
        self.results = self.report['results']

    def test_provenance_and_scope(self):
        self.assertEqual(self.report['audit_source_sha256'], hashlib.sha256(contracts.historical_bytes(audit.__file__)).hexdigest())
        verify_lock_binding(self.report['lock_sha256'], tuple(audit.REVISIONS), current_lock=audit.LOCK)
        for project, revision in audit.REVISIONS.items():
            self.assertEqual(self.report['before'][project]['head'], revision)
            self.assertEqual(self.report['after'][project], self.report['before'][project])
            self.assertEqual(self.report['before'][project]['status'], '')
        self.assertEqual(self.report['environment']['pyroomacoustics'], '0.10.0')
        self.assertEqual(len(self.report['wheel_python_sources']), 7)
        for source in self.report['wheel_python_sources'].values():
            self.assertTrue(source['same'])
            self.assertEqual(source['cached_sha'], source['wheel_sha'])
        for project, entry in self.report['lock_entries'].items():
            lock_entry = next(p for p in json.loads(audit.LOCK.read_text())['projects']
                              if p['id'] == project)
            self.assertEqual(entry, lock_entry)
        self.assertIn('original _process not run', self.report['scope']['CSSM_WAVES'])
        self.assertIn('real audio accuracy', self.report['scope']['claims_excluded'])

    def test_original_helper_mapping_does_not_silently_discard_failure(self):
        cssm = self.results['cssm_filtered_mapping']
        # Three given diagonal matrices; the retained physical bins are 20/30.
        self.assertEqual(cssm['original_diagonal'], [4 + 9, 1 + 1, 1 + 1])
        self.assertEqual(cssm['independent_expected_diagonal'], [9 + 16, 2, 2])
        self.assertEqual(cssm['frobenius_error'], 12)
        waves = self.results['waves_filtered_mapping']
        np.testing.assert_allclose(waves['original_Z_first_row'],
                                   [3 / np.sqrt(5), 8 / np.sqrt(10)], atol=1e-14, rtol=0)
        np.testing.assert_allclose(waves['independent_expected_first_row'],
                                   [8 / np.sqrt(10), 15 / np.sqrt(17)], atol=1e-14, rtol=0)

    def test_tops_independent_projection_uses_physical_bins(self):
        result = self.results['tops']
        # Known steering and source angle define this result without TOPS or SVD.
        self.assertEqual((result['original_peak_deg'], result['independent_peak_deg']), (49, 30))
        norms = audit.independent_tops_norms(range(360))
        np.testing.assert_allclose(result['independent_projection_norms'], norms,
                                   atol=1e-14, rtol=0)
        self.assertLess(audit.independent_tops_norms([30])[0], 1e-12)
        self.assertGreater(min(result['all_original_singular_values']), .37)
        self.assertGreater(result['max_absolute_norm_difference'], .5)
        self.assertEqual(result['reference_bin'], 20)
        self.assertEqual(result['bins'], [10, 20, 30])

    def test_tops_input_has_exact_known_covariances(self):
        result = self.results['tops']
        x = np.asarray(result['selected_input_real']) + 1j*np.asarray(result['selected_input_imag'])
        geometry = np.asarray(result['geometry_m'])
        # Scalar geometric phase, followed by independent scalar snapshot sum.
        for index, (bin_, power) in enumerate(zip([10, 20, 30], [1, 4, 2])):
            steering = np.exp(2j*np.pi*16000*bin_/256/343
                              * (np.cos(np.pi/6)*geometry[0] + np.sin(np.pi/6)*geometry[1]))
            for first in range(3):
                for second in range(3):
                    actual = sum(x[first, index, t]*x[second, index, t].conjugate()
                                 for t in range(8))/8
                    expected = power*steering[first]*steering[second].conjugate()
                    if first == second:
                        expected += .01
                    self.assertLess(abs(actual-expected), 1e-13)

    def test_near_constructor_and_root_compatibility_boundaries(self):
        self.assertEqual(self.results['srp_near_two_distances']['exception_type'], 'ValueError')
        near = self.results['srp_near_one_distance']
        self.assertEqual(near['candidate_radius'], [.2])
        self.assertEqual(near['grid_cartesian_norms'], [1, 1, 1])
        self.assertEqual(near['mode_vector_near_vs_far_max_difference'], 0)
        self.assertEqual(self.results['root_music_original']['exception_type'], 'AttributeError')
        fixed = self.results['root_music_facade']
        self.assertTrue(fixed['resolved'])
        self.assertAlmostEqual(fixed['angles_deg'][0], 30, delta=1e-6)
        # Quarter-cycle half-wave steering: [1,j,-1,-j,1,j], not upstream output.
        np.testing.assert_allclose(fixed['input_steering_real'], [1, 0, -1, 0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(fixed['input_steering_imag'], [0, 1, 0, -1, 0, 1], atol=1e-14)

    def test_smoothing_preserves_dtype_failure_and_hand_answer(self):
        result = self.results['spatial_smooth']
        self.assertEqual(result['original_forward'], [[2, 0], [0, 2]])
        self.assertEqual(result['original_fb'], [[2, 0], [0, 2]])
        self.assertIn('Cannot cast', self.results['spatial_smooth_integer_input']['message'])

    def test_source_counts_use_independent_penalized_log_formula(self):
        result = self.results['source_counts']
        # Hand E04-20 constants; no use of upstream product/statistic as expected.
        np.testing.assert_allclose(result['independent_AIC'],
                                   [348.8652831986846,160.7938350160402,
                                    32.16439890405105,30], atol=1e-11, rtol=0)
        np.testing.assert_allclose(result['independent_MDL'],
                                   [174.4326415993423,89.51501315897842,
                                    31.713220567954075,34.53877639491069], atol=1e-11, rtol=0)
        np.testing.assert_allclose(result['original_ld'], result['independent_F'],
                                   atol=1e-11, rtol=0)
        self.assertEqual(result['original_AIC_choice'], 3)
        self.assertEqual(result['original_MDL_choice'], 2)
        self.assertEqual(self.results['ld_product_underflow']['classification'], 'positive_infinity')
        self.assertIsNone(self.results['ld_product_underflow']['value'])

    def test_strict_json_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            strict_json('{"failure": NaN}')


@unittest.skipUnless(CAN_RUN, 'fixed cache and isolated SciPy/PRA environment required; no installation')
class OriginalExecutionTests(unittest.TestCase):
    def test_current_fixed_original_methods(self):
        report = audit.build_report()
        self.assertEqual(report['results']['tops']['original_peak_deg'], 49)
        self.assertEqual(report['before'], report['after'])
        self.assertEqual(report['results']['cssm_filtered_mapping']['original_diagonal'], [13, 2, 2])
        self.assertLess(report['results']['tops']['independent_at_true_deg'], 1e-12)


if __name__ == '__main__':
    unittest.main()
