"""Independent tiny objectives and finite publication guards, no source edits."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch12.examples import audit_damas_author_contracts as audit


class AuthorReportGuards(unittest.TestCase):
    def test_preflight_refuses_history_source_cache_and_lock_before_execution(self):
        targets = [audit.upstream.CACHE/'damas-author/new.json', audit.upstream.LOCK,
                   audit.upstream.STATUS, Path(audit.__file__),
                   audit.ROOT/'codes/chapters/ch12/reports/upstream_imaging_contracts.json']
        with patch.object(audit, 'run_audit') as run:
            for path in targets:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    audit.main(['--report', str(path)])
            run.assert_not_called()

    def test_stdout_default_is_read_only(self):
        with patch.object(audit, 'run_audit', return_value={'status': 'original_worker_not_completed'}), \
                patch.object(audit.upstream, 'write_report') as write, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(audit.main([]), 0)
        write.assert_not_called()
        self.assertEqual(audit.upstream.strict_json_loads(out.getvalue())['status'], 'original_worker_not_completed')

    def test_safe_external_target_keeps_real_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve()/'report.json'
            report = {'status': 'original_worker_not_completed', 'worker': {'returncode': 1}}
            with patch.object(audit, 'run_audit', return_value=report), redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(['--report', str(path)]), 0)
            self.assertEqual(audit.upstream.strict_json_loads(path.read_bytes()), report)

    def test_worker_cannot_write_even_a_safe_report(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'worker') as run:
            with self.assertRaisesRegex(ValueError, 'cannot publish'):
                audit.main(['--worker', '--report', str(Path(folder)/'worker.json')])
            run.assert_not_called()

    def test_actual_timeout_retained_with_source_postflight(self):
        identity = {'used_source_identity': 'verified'}
        with patch.object(audit, 'identity', return_value=identity), \
                patch.object(audit.upstream, 'check_unchanged', return_value=identity), \
                patch.object(audit.subprocess, 'run', side_effect=subprocess.TimeoutExpired('fixture', 30,
                    output=b'partial original stdout', stderr=b'partial original stderr')):
            report = audit.run_audit()
        self.assertEqual(report['status'], 'original_worker_not_completed')
        self.assertEqual(report['worker']['execution'], 'original worker timed out')
        self.assertEqual(report['worker']['stdout'], 'partial original stdout')
        self.assertEqual(report['source_identity_after'], identity)


@unittest.skipUnless(importlib.util.find_spec('scipy') and importlib.util.find_spec('matplotlib')
                     and (audit.upstream.CACHE/'damas-author/damas.py').is_file(),
                     'original author calls need existing SciPy/Matplotlib/fixed checkout; no substitutes or installation')
class AuthorOriginalCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.run_audit()
        cls.worker = cls.report['worker']
        if 'cases' not in cls.worker:
            raise AssertionError('real original worker did not complete: ' + repr(cls.worker))
        cls.rows = {r['name']: r for r in cls.worker['cases']}

    def test_real_one_module_identity_and_complete_selection_are_distinct(self):
        before = self.report['source_identity_before']
        after = self.report['source_identity_after']
        self.assertEqual(before['head'], audit.REVISION)
        self.assertEqual(set(before['used_files']), {'LICENSE', 'damas.py'})
        self.assertEqual(before['used_files']['damas.py']['sha256'], audit.SOURCE_SHA)
        self.assertEqual(before['used_files'], after['used_files'])
        self.assertTrue(after['clean_after'])
        self.assertEqual(before['live_complete_selection']['status'], 'source_verified')
        self.assertEqual(self.worker['module_identity']['file'], str(Path(before['checkout'])/'damas.py'))
        self.assertTrue(self.worker['module_identity']['complete_module_bytes_executed'])
        self.assertFalse(self.worker['module_identity']['cached_bytecode_consumed'])
        self.assertEqual(self.report['actual_dependencies_before'], self.report['actual_dependencies_after'])
        for name in ('numpy', 'scipy.linalg', 'matplotlib.pyplot'):
            row = self.worker['external_dependencies'][name]
            self.assertEqual(row['sha256_before'], row['sha256_after'])

    def test_independent_gram_products_and_two_different_objectives(self):
        value = self.worker['known_input']
        d = np.array(value['transfer']['real']) + 1j*np.array(value['transfer']['imag'])
        r = np.array(value['csm']['real']) + 1j*np.array(value['csm']['imag'])
        g = abs(d.conj().T@d)**2
        gd = g - (abs(d)**2).T@(abs(d)**2)
        q = np.array([1, .2])
        np.testing.assert_allclose(self.rows['full_gram_product']['actual'], g@q, rtol=0, atol=audit.ATOL)
        np.testing.assert_allclose(self.rows['removed_diagonal_gram_product']['actual'], gd@q, rtol=0, atol=audit.ATOL)
        for remove, expected_cmf, expected_map in [(False, 1., 16/17), (True, 2/3, 16/15)]:
            matrix = r - np.diag(np.diag(r)) if remove else r
            gram = gd if remove else g
            right = np.real(np.diag(d.conj().T@matrix@d))
            # The second coordinate has a nonnegative KKT gradient at either optimum.
            cmf = np.array([right[0]/gram[0, 0], 0.])
            mapfit = np.array([np.dot(gram[:, 0], right)/np.dot(gram[:, 0], gram[:, 0]), 0.])
            self.assertGreaterEqual((gram@cmf-right)[1], -audit.ATOL)
            self.assertGreaterEqual((gram.T@(gram@mapfit-right))[1], -audit.ATOL)
            suffix = '_dr_lh' if remove else '_lh'
            np.testing.assert_allclose(self.rows['cmf_nnls'+suffix]['actual'], cmf, rtol=0, atol=audit.ATOL)
            np.testing.assert_allclose(self.rows['damas_nnls'+suffix]['actual'], mapfit, rtol=0, atol=audit.ATOL)
            self.assertAlmostEqual(cmf[0], expected_cmf)
            self.assertAlmostEqual(mapfit[0], expected_map)

    def test_zero_original_exception_and_nonunique_duplicate_not_repaired(self):
        self.assertEqual(self.rows['zero_csm_failure']['error_type'], 'NameError')
        self.assertIn("'e'", self.rows['zero_csm_failure']['error_message'])
        duplicate = self.rows['duplicate_columns_conditional_unique_flag']
        self.assertTrue(duplicate['original_unique_flag'])
        d = np.ones((2, 2))
        np.testing.assert_array_equal(d@np.diag(duplicate['actual'])@d.T,
                                      d@np.diag(duplicate['independent_alternative_solution'])@d.T)
        self.assertNotEqual(duplicate['actual'], duplicate['independent_alternative_solution'])

    def test_single_rank_one_cleansc_is_not_a_complete_pipeline(self):
        self.assertEqual(self.rows['single_source_full_cleansc_two_steps']['actual'], [2*(1-.4**2)])
        self.assertEqual(self.worker['counts'], {'bounded_cases': 9, 'numeric_matches': 8,
                                                'original_failures_retained': 1})
        self.assertIn('MATLAB methods', self.worker['unexecuted'])
        self.assertIn('original demos and MAT datasets', self.worker['unexecuted'])


if __name__ == '__main__':
    unittest.main()
