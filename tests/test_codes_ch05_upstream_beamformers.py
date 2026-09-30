"""Independent exact-model checks, source provenance, and optional original calls."""
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
import warnings
from zoneinfo import ZoneInfo

import numpy as np

from codes.chapters.ch05.examples import audit_upstream_beamformers as audit

REPORT = audit.ROOT / 'codes/chapters/ch05/reports/upstream_beamformers.json'
CAN_RUN = (importlib.util.find_spec('scipy') is not None
           and (audit.CACHE / 'pb_bss/.git').exists())


def strict_json(text):
    def reject(constant):
        raise ValueError('Nonstandard JSON constant: '+constant)
    return json.loads(text, parse_constant=reject)


def array(encoded):
    return np.asarray(encoded['real'])+1j*np.asarray(encoded['imag'])


def scalar(encoded):
    return complex(encoded['real'], encoded['imag'])


class SavedReportTests(unittest.TestCase):
    def setUp(self):
        self.report = strict_json(REPORT.read_text())
        self.rows = self.report['results']

    def test_current_tool_lock_and_original_function_identity(self):
        self.assertEqual(self.report['audit_source_sha256'], audit.sha(audit.__file__))
        self.assertEqual(self.report['lock_sha256'], audit.sha(audit.LOCK))
        entry = next(p for p in json.loads(audit.LOCK.read_text())['projects'] if p['id'] == 'pb_bss')
        self.assertEqual(self.report['lock_entry'], entry)
        self.assertEqual(entry['revision'], '10acc347fc9ea21e3d312806a0bd751d0d0af183')
        self.assertEqual(entry['license'], 'MIT')
        self.assertEqual(self.report['before'], self.report['after'])
        self.assertEqual(self.report['before']['head'], entry['revision'])
        self.assertEqual(self.report['before']['status'], '')
        self.assertEqual(self.report['before']['untracked_python'], [])
        self.assertEqual(set(self.report['original_functions']), set(audit.FUNCTIONS))
        self.assertEqual({r['function'] for r in self.rows.values()}, set(audit.FUNCTIONS))
        self.assertEqual(len(self.report['original_functions']), 12)
        self.assertEqual(set(self.report['original_files']), set(audit.SOURCE_SHA))
        self.assertTrue(all(not f['body_modified'] for f in self.report['original_functions'].values()))
        self.assertIn('no compatibility facade', self.report['scope']['dependencies'])
        self.assertIn('SciPy fallback only', self.report['scope']['backend'])
        self.assertIn('No complete pb_bss/ESPnet/Torch/Cython import', self.report['scope']['excluded'])

    def test_rank_one_preserves_reference_image_including_complex_reference(self):
        # Scalar q=|2|²/2+|1+j|²=4; reference1 deliberately tests conjugation.
        b = np.array([2, 1+1j])
        expected = ([.5, .5+.5j], [.25-.25j, .5])
        for reference, weights in enumerate(expected):
            row = self.rows['souden_rank_one_ref'+str(reference)]
            actual = array(row['output'])[0]
            np.testing.assert_allclose(actual, weights, atol=2e-14, rtol=0)
            self.assertAlmostEqual(abs(np.vdot(actual, b)-b[reference]), 0, delta=2e-14)
            self.assertAlmostEqual(abs(np.vdot(actual, b/b[reference])-1), 0, delta=2e-14)
            self.assertAlmostEqual(abs(scalar(row['observables']['physical_target_response'])
                                       -b[reference]), 0, delta=2e-14)
        # Full rank has two independent images; trace output scales the first one.
        np.testing.assert_allclose(array(self.rows['souden_full_rank']['output']),
                                   [[2/3, 0]], atol=2e-14, rtol=0)
        self.assertEqual(self.rows['souden_full_rank']['classification'], 'model_boundary')

    def test_souden_auto_selection_is_distinct_from_merl_failure(self):
        selected, reference = self.rows['souden_auto_reference']['output']
        self.assertEqual(reference, 1)
        np.testing.assert_allclose(array(selected), [[0, .8]], atol=2e-14, rtol=0)
        self.assertEqual(self.rows['optimal_reference_direct']['output'], 1)
        # Independent candidate SNRs are powers 1 and 4, not a sum-and-argmax.
        np.testing.assert_allclose(array(self.rows['merl_reference_1']['output']),
                                   [[.2, 0]], atol=2e-14, rtol=0)
        self.assertEqual(self.rows['merl_reference_1']['classification'], 'reference_selection_failure')
        np.testing.assert_allclose(array(self.rows['merl_reference_4']['output']),
                                   [[.8, 0]], atol=2e-14, rtol=0)

    def test_full_rank_mwf_trace_is_not_general_wiener_solution(self):
        # General scalar Wiener solution: target2 / (target2+noise1).
        np.testing.assert_allclose(array(self.rows['wmwf_full_rank']['output']),
                                   [[1/2, 0]], atol=2e-14, rtol=0)
        self.assertGreater(abs(array(self.rows['wmwf_full_rank']['output'])[0, 0]-2/3), .16)
        np.testing.assert_allclose(array(self.rows['wmwf_rank_one']['output']),
                                   [[2/3, 0]], atol=2e-14, rtol=0)

    def test_ban_unit_norm_does_not_recover_phase_or_target_unit_response(self):
        for name, scale in [('positive', 1), ('positive_double', 1), ('negative', -1), ('phase_j', 1j)]:
            row = self.rows['ban_'+name]
            actual = array(row['output'])[0]
            expected = np.array([scale/np.sqrt(2), scale/np.sqrt(2)])
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
            self.assertAlmostEqual(float(np.vdot(actual, actual).real), 1, delta=2e-14)
            response = sum(v.conjugate() for v in actual)
            self.assertAlmostEqual(abs(response-scale.conjugate()*np.sqrt(2)), 0, delta=2e-14)
            self.assertGreater(abs(response-1), .4)
        self.assertEqual(self.rows['ban_zero']['classification'], 'degenerate_zero_output')
        np.testing.assert_array_equal(array(self.rows['ban_zero']['output']), [[0, 0]])

    def test_gev_known_direction_and_original_singular_failure(self):
        # Rs/Rn per axis: 4/1 and 1/2; no original eigenvalue result supplies expected.
        for id_ in ('gev_public', 'gev_fallback'):
            vector = array(self.rows[id_]['output'])[0]
            self.assertLess(abs(vector[1]), 2e-14)
            self.assertGreater(abs(vector[0]), .5)
            self.assertAlmostEqual((4*abs(vector[0])**2+abs(vector[1])**2)
                                   /(abs(vector[0])**2+2*abs(vector[1])**2), 4, delta=2e-14)
        self.assertEqual(self.rows['gev_singular_noise']['exception']['type'], 'ValueError')

    def test_constraint_response_and_incompatibility_are_both_explicit(self):
        complex_case = array(self.rows['lcmv_complex_response']['output'])[0]
        np.testing.assert_allclose(complex_case.conj(), [1, 1j], atol=2e-14, rtol=0)
        row = self.rows['lcmv_incompatible_constraints']
        weights = array(row['output'])[0]
        self.assertEqual(row['classification'], 'constraint_violation')
        np.testing.assert_allclose(weights, [.25, .25], atol=2e-14, rtol=0)
        actual_constraints = np.array([sum(weights), sum(weights)])
        residual = float(np.linalg.norm(actual_constraints-np.array([1, 0])))
        self.assertAlmostEqual(residual, 1/np.sqrt(2), delta=2e-14)
        self.assertGreater(residual, .7)
        self.assertEqual(self.rows['lcmv_souden_unimplemented']['exception']['type'], 'NotImplementedError')

    def test_singular_solve_is_not_positive_definite_mvdr_optimum(self):
        np.testing.assert_allclose(array(self.rows['stable_solve_regular']['output']),
                                   [[1], [2]], atol=2e-14, rtol=0)
        np.testing.assert_allclose(array(self.rows['stable_solve_singular']['output']),
                                   [[0], [1]], atol=2e-14, rtol=0)
        self.assertEqual(self.rows['stable_solve_vector_rhs']['exception']['type'], 'IndexError')
        weights = array(self.rows['mvdr_singular']['output'])[0]
        self.assertAlmostEqual(sum(weights).real, 1, delta=2e-14)
        self.assertAlmostEqual(abs(weights[1])**2, 1, delta=2e-14)
        # Feasible [1,0] has zero noise under Rn=diag(0,1).
        self.assertEqual(abs(np.array([1, 0])[1])**2, 0)

    def test_masks_retain_dtype_invalid_psd_and_floor_boundaries(self):
        np.testing.assert_array_equal(array(self.rows['scm_float']['output']), np.eye(2)[None]/2)
        self.assertIn('Cannot cast', self.rows['scm_integer']['exception']['message'])
        self.assertEqual(self.rows['scm_boolean']['exception']['type'], 'AttributeError')
        self.assertIn('asfarray', self.rows['scm_boolean']['exception']['message'])
        negative = array(self.rows['scm_negative']['output'])[0]
        self.assertEqual(negative[1, 1].real, -1)
        self.assertEqual(self.rows['scm_negative']['classification'], 'invalid_psd_accepted')
        np.testing.assert_array_equal(array(self.rows['scm_zero']['output']), np.zeros((1, 2, 2)))
        tiny = array(self.rows['scm_tiny']['output'])[0]
        # Relative scaling before comparison keeps a wrong zero distinguishable.
        np.testing.assert_allclose(tiny/1e-290, np.eye(2), atol=2e-14, rtol=0)
        self.assertLess(self.rows['scm_tiny']['absolute_tolerance'], 1e-300)

    def test_run_summary_keeps_failures_as_evidence(self):
        self.assertEqual(self.report['case_count'], 31)
        self.assertEqual(len(self.rows), 31)
        self.assertTrue(all(row['expected_behavior_verified'] for row in self.rows.values()))
        self.assertEqual(self.report['status'], 'expected_behaviors_verified_failures_preserved')
        self.assertEqual(sum(row['exception'] is not None for row in self.rows.values()), 5)
        actual_date = datetime.fromisoformat(self.report['created_utc']).astimezone(
            ZoneInfo('Asia/Shanghai')).date().isoformat()
        self.assertEqual(self.report['verified_date_asia_shanghai'], actual_date)

    def test_strict_json_and_nonfinite_classification(self):
        with self.assertRaises(ValueError):
            strict_json('{"value": NaN}')
        encoded = audit.encode(np.array([np.nan, np.inf, -np.inf]))
        parsed = strict_json(json.dumps(encoded, allow_nan=False))
        self.assertFalse(parsed['finite'])
        self.assertEqual([v['classification'] for v in parsed['real']],
                         ['nan', 'positive_infinity', 'negative_infinity'])

    def test_cached_original_segments_when_available(self):
        if not (audit.CACHE / 'pb_bss/.git').exists():
            self.skipTest('Optional fixed upstream source cache absent')
        for relative, metadata in self.report['original_files'].items():
            path = audit.CACHE / 'pb_bss' / relative
            self.assertEqual(audit.sha(path), metadata['sha256'])
            self.assertEqual(metadata['sha256'], audit.SOURCE_SHA[relative])
            self.assertEqual(audit.git(path.parents[len(Path(relative).parts)-1],
                                       'rev-parse', 'HEAD:'+relative), metadata['git_blob'])
        for name, metadata in self.report['original_functions'].items():
            text = (audit.CACHE / 'pb_bss' / metadata['file']).read_text()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                node = next(n for n in ast.parse(text).body
                            if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(node.lineno, metadata['first_line'])
            self.assertEqual(node.end_lineno, metadata['last_line'])
            self.assertEqual(hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest(),
                             metadata['source_segment_sha256'])


@unittest.skipUnless(CAN_RUN, 'Optional SciPy and locked source cache required; no installation attempted')
class OriginalExecutionTests(unittest.TestCase):
    def test_real_original_methods_without_report_write(self):
        report_before = REPORT.read_bytes()
        report = audit.build_report()
        self.assertEqual(report['status'], 'expected_behaviors_verified_failures_preserved')
        self.assertEqual(report['before'], report['after'])
        self.assertEqual(REPORT.read_bytes(), report_before)
        self.assertEqual(report['case_count'], 31)

    def test_default_cli_does_not_rewrite_saved_report(self):
        report_before = REPORT.read_bytes()
        result = subprocess.run([sys.executable, '-B', '-m',
                                  'codes.chapters.ch05.examples.audit_upstream_beamformers'],
                                 cwd=audit.ROOT, check=True, capture_output=True, text=True)
        summary = strict_json(result.stdout)
        self.assertIsNone(summary['report'])
        self.assertEqual(summary['unexpected_cases'], [])
        self.assertEqual(REPORT.read_bytes(), report_before)


if __name__ == '__main__':
    unittest.main()
