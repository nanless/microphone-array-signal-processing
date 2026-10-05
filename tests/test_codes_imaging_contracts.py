"""Offline numerical oracles, original-source identity and report boundaries."""
from contextlib import redirect_stdout
from fractions import Fraction
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch12.examples import audit_upstream_imaging_contracts as audit
from codes.chapters.ch00.io_contracts import strict_json_loads, write_json_report
from codes.chapters.ch00.upstream.fetch_upstreams import run_git


class ImagingIdentityTests(unittest.TestCase):
    """Small temporary repositories never change the pinned source cache."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve() / 'repo'
        self.repo.mkdir()
        run_git(['init', '-q'], cwd=self.repo)
        run_git(['config', 'user.name', 'Offline fixture'], cwd=self.repo)
        run_git(['config', 'user.email', 'fixture@example.invalid'], cwd=self.repo)
        run_git(['remote', 'add', 'origin', audit.ORIGIN], cwd=self.repo)
        self.file = self.repo / 'LICENSE'
        self.file.write_text('ordinary source identity fixture\n')
        run_git(['add', 'LICENSE'], cwd=self.repo)
        run_git(['commit', '-qm', 'offline source fixture'], cwd=self.repo)
        self.revision = run_git(['rev-parse', 'HEAD'], cwd=self.repo)
        self.files = {'LICENSE': hashlib.sha256(self.file.read_bytes()).hexdigest()}

    def verify(self, **kwargs):
        values = {'revision': self.revision, 'origin': audit.ORIGIN, 'files': self.files}
        values.update(kwargs)
        return audit.verify_checkout(self.repo, **values)

    def test_ordinary_fixed_blob_and_sha(self):
        identity = self.verify()
        row = identity['files']['LICENSE']
        self.assertEqual(row['head_blob'], row['actual_blob'])
        self.assertEqual(row['sha256'], self.files['LICENSE'])
        self.assertTrue(identity['worktree_clean'])

    def test_outer_git_environment_cannot_redirect_identity(self):
        # Every entry would invalidate or redirect a normal inherited git call.
        injection = {'GIT_DIR': '/no/such/gitdir', 'GIT_WORK_TREE': '/no/such/worktree',
                     'GIT_INDEX_FILE': '/no/such/index', 'GIT_OBJECT_DIRECTORY': '/no/such/objects',
                     'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'remote.origin.url',
                     'GIT_CONFIG_VALUE_0': 'https://wrong.example.invalid/repo.git'}
        with patch.dict(os.environ, injection):
            self.assertEqual(self.verify()['head'], self.revision)
            self.assertEqual(self.verify()['origin'], audit.ORIGIN)

    def test_wrong_origin_is_rejected(self):
        run_git(['remote', 'set-url', 'origin', 'https://wrong.example.invalid/repo.git'], cwd=self.repo)
        with self.assertRaisesRegex(ValueError, 'origin mismatch'):
            self.verify()

    def test_wrong_complete_revision_is_rejected(self):
        wrong = ('1' if self.revision[0] != '1' else '2') + self.revision[1:]
        with self.assertRaisesRegex(ValueError, 'HEAD mismatch'):
            self.verify(revision=wrong)
        with self.assertRaisesRegex(ValueError, '40-character'):
            self.verify(revision=self.revision[:12])

    def test_wrong_source_digest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SHA/blob mismatch'):
            self.verify(files={'LICENSE': '0'*64})

    def test_source_paths_cannot_escape_checkout(self):
        for relative in ('../LICENSE', str(self.file)):
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, 'relative checkout'):
                self.verify(files={relative: self.files['LICENSE']})

    def test_untracked_and_tracked_changes_are_rejected(self):
        extra = self.repo / 'untracked.txt'
        extra.write_text('extra')
        with self.assertRaisesRegex(ValueError, 'completely clean'):
            self.verify()
        extra.unlink()
        self.file.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'completely clean'):
            self.verify()

    def test_parent_links_and_hardlinked_sources_are_rejected(self):
        link = self.repo.parent / 'alias'
        link.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            audit.verify_checkout(link, revision=self.revision, origin=audit.ORIGIN, files=self.files)
        hardlink = self.repo.parent / 'license-copy'
        os.link(self.file, hardlink)
        with self.assertRaisesRegex(ValueError, 'singly linked'):
            self.verify()


class ImagingReportBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()

    def test_explicit_external_report_is_strict_json(self):
        target = audit.report_target(self.directory / 'current.json')
        value = {'status': 'audit_completed_with_original_differences', 'matched': False}
        write_json_report(target, value)
        self.assertEqual(strict_json_loads(target.read_bytes()), value)
        self.assertEqual(list(self.directory.iterdir()), [target])

    def test_upstream_old_reports_and_source_destinations_are_protected(self):
        for target in (audit.CACHE / 'acoular' / 'new-report.json',
                       audit.HISTORICAL_REPORT,
                       audit.upstream.STATUS,
                       audit.ROOT / 'codes/chapters/ch09/reports/upstream_tracking_contracts.json',
                       audit.ROOT / 'reviews/old-review.json', Path(audit.__file__), audit.LOCK):
            with self.subTest(path=target), self.assertRaises(ValueError):
                audit.report_target(target)
        self.assertEqual(audit.report_target(audit.CURRENT_REPORT), audit.CURRENT_REPORT)

    def test_links_directories_and_lexical_parent_traversal_are_rejected(self):
        ordinary = self.directory / 'ordinary.json'
        ordinary.write_text('{}')
        symbolic = self.directory / 'symbolic.json'
        symbolic.symlink_to(ordinary)
        hardlink = self.directory / 'hardlink.json'
        os.link(ordinary, hardlink)
        for target in (symbolic, hardlink, self.directory, self.directory / '..' / 'escape.json'):
            with self.subTest(path=target), self.assertRaises(ValueError):
                audit.report_target(target)

    def test_preflight_precedes_original_execution(self):
        with patch.object(audit, 'run_audit') as run:
            for target in (audit.CACHE / 'acoular' / 'bad.json', audit.HISTORICAL_REPORT):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    audit.main(['--report', str(target)])
            run.assert_not_called()

    def test_default_stdout_does_not_write_report(self):
        value = {'status': 'audit_completed_with_original_differences', 'cases': []}
        out = io.StringIO()
        with patch.object(audit, 'run_audit', return_value=value), \
                patch.object(audit, 'write_json_report') as writer, redirect_stdout(out):
            self.assertEqual(audit.main([]), 0)
        writer.assert_not_called()
        self.assertEqual(strict_json_loads(out.getvalue()), value)

    def test_explicit_cli_output_preserves_difference_status(self):
        value = {'status': 'audit_completed_with_original_differences',
                 'cases': [{'matched_independent_expected': False}]}
        target = self.directory / 'current.json'
        with patch.object(audit, 'run_audit', return_value=value), redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(['--report', str(target)]), 0)
        self.assertEqual(strict_json_loads(target.read_bytes()), value)

    def test_nonfinite_report_refused_before_any_publication(self):
        target = self.directory / 'bad.json'
        with self.assertRaises(ValueError):
            write_json_report(target, {'bad': float('nan')})
        self.assertFalse(target.exists())


@unittest.skipUnless((audit.CACHE / 'acoular/acoular/fbeamform.py').is_file(),
                     'optional fixed Acoular checkout absent; no network or substitute algorithm')
class ImagingOriginalMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.run_audit()
        cls.rows = {row['name']: row for row in cls.report['cases']}

    def test_real_source_identity_and_selection_are_separate(self):
        before = self.report['source_identity_before']
        after = self.report['source_identity_after']
        self.assertFalse(before['clean_after'])
        self.assertTrue(after['clean_after'])
        self.assertEqual(before['used_files'], after['used_files'])
        self.assertEqual(before['ignored_members_before'], after['ignored_members_after'])
        self.assertEqual(before['origin'], audit.ORIGIN)
        self.assertEqual(before['head'], audit.REVISION)
        self.assertEqual(before['source_version'], '26.08')
        self.assertTrue(before['required_source_identity_verified'])
        selection = before['acquisition_scope']
        self.assertIn(selection['status'], ('source_verified', 'source_selection_mismatch', 'entrypoints_missing'))
        if selection['status'] == 'source_selection_mismatch':
            self.assertFalse(selection['source_selection_verified'])
        for path, expected in audit.FILES.items():
            identity = before['used_files'][path]
            self.assertEqual(identity['sha256'], expected)
            self.assertEqual(identity['git_blob'], identity['actual_blob'])
        self.assertEqual(before['live_complete_selection'], before['acquisition_scope'])
        lock, status, lock_sha, status_sha = audit.upstream.source_documents()
        self.assertEqual(before['lock_sha256'], lock_sha)
        self.assertEqual(before['status_sha256'], status_sha)
        self.assertEqual(before['recorded_complete_selection'],
                         next(row for row in status['projects'] if row['id'] == 'acoular'))

    def test_psf_dirty_map_and_causal_phase_from_hand_calculation(self):
        np.testing.assert_allclose(self.rows['custom_full_csm_psf']['actual'],
                                   [[1, float(Fraction(1, 4))], [float(Fraction(1, 4)), 1]],
                                   rtol=0, atol=audit.ATOL)
        np.testing.assert_allclose(self.rows['base_full_csm']['actual'], [1.05, .45], rtol=0, atol=audit.ATOL)
        c = self.rows['base_full_csm']['csm']
        np.testing.assert_allclose(c['real'], [[1.2, .9], [.9, 1.2]], rtol=0, atol=audit.ATOL)
        np.testing.assert_allclose(c['imag'], [[0, np.sqrt(3)/10], [-np.sqrt(3)/10, 0]],
                                   rtol=0, atol=audit.ATOL)
        drop = self.rows['base_removed_diagonal']
        np.testing.assert_allclose(drop['raw_before_original_clipping'], [.9, -.3], rtol=0, atol=audit.ATOL)
        self.assertEqual(drop['actual'], [.9, 0.])

    def test_damas_class_replays_dirty_map_initialization(self):
        first = self.rows['damas_original_class_1_sweeps']
        np.testing.assert_allclose(first['actual'], [15/16, 69/320], rtol=0, atol=audit.ATOL)
        self.assertEqual(first['initial_solution'], [1.05, .45])
        np.testing.assert_allclose(self.rows['damas_original_class_20_sweeps']['actual'], [1, .2],
                                   rtol=0, atol=audit.ATOL)

    def test_gs_is_not_residual_nnls(self):
        row = self.rows['damas_gs_is_not_map_residual_nnls']
        self.assertFalse(row['matched_independent_expected'])
        self.assertEqual(row['actual'], [1., 0.])
        self.assertEqual(row['independent_expected'], [16/17, 0.])
        self.assertEqual(row['original_residual_squared'], 1/16)
        self.assertEqual(row['independent_nnls_residual_squared'], 1/17)

    def test_valid_csm_bridge_retains_both_objectives_and_original_case_count(self):
        row, = self.report['additional_valid_csm_controls']
        c = np.array(row['csm']['real']) + 1j * np.array(row['csm']['imag'])
        self.assertGreaterEqual(np.linalg.eigvalsh(c)[0], -audit.ATOL)
        a = np.array(row['transfer']['real']) + 1j * np.array(row['transfer']['imag'])
        w = a / np.sum(abs(a)**2, axis=0)
        b = np.real(np.diag(w.conj().T @ c @ w))
        np.testing.assert_allclose(b, [1, 0], atol=audit.ATOL, rtol=0)
        np.testing.assert_allclose(row['dirty_map'], b, atol=audit.ATOL, rtol=0)
        np.testing.assert_allclose(row['actual'], [1, 0], atol=audit.ATOL, rtol=0)
        self.assertAlmostEqual(row['covariance_residual_squared'], float(Fraction(28, 9)), places=13)
        self.assertAlmostEqual(row['scan_nnls_covariance_residual_squared'],
                               float(Fraction(8128, 2601)), places=13)
        self.assertGreater(row['scan_nnls_covariance_residual_squared'], row['covariance_residual_squared'])
        self.assertEqual(self.report['counts']['limited_numerical_cases'], 21)

    def test_actual_dependencies_unchanged_and_exactly_current(self):
        self.assertEqual(self.report['actual_dependencies_before'], self.report['actual_dependencies_after'])
        self.assertEqual(self.report['direct_sources'], audit.upstream.dependencies(audit.__file__))
        self.assertTrue(self.report['actual_dependencies_unchanged'])

    def test_changed_local_dependency_is_rejected(self):
        actual = audit.upstream.dependencies(audit.__file__)
        changed = dict(actual)
        changed['codes/chapters/ch00/io_contracts.py'] = '0' * 64
        with patch.object(audit.upstream, 'dependencies', side_effect=[actual, changed]):
            with self.assertRaisesRegex(ValueError, 'dependencies changed'):
                audit.run_audit()

    def test_full_cleansc_difference_not_hidden_by_tool_completion(self):
        row = self.rows['cleansc_full_csm_20']
        self.assertFalse(row['matched_independent_expected'])
        self.assertEqual(row['classification'], 'observed_original_behavior_difference')
        self.assertTrue(row['matched_prediction_of_original_behavior'])
        self.assertAlmostEqual(row['actual'][0], 4.812502861022949, places=10)
        self.assertAlmostEqual(row['independent_expected'][0], 2*(1-.4**20), places=14)
        for n in (1, 2, 4, 20):
            other = self.rows['cleansc_removed_diagonal_' + str(n)]
            self.assertTrue(other['matched_independent_expected'])
        self.assertFalse(self.report['status'] == 'all_algorithms_passed')

    def test_cmf_independent_pair_order_and_half_triangle_target(self):
        row = self.rows['cmf_dictionary_full_csm']
        self.assertEqual(row['real_pairs'], [[0, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 2]])
        self.assertEqual(row['imaginary_pairs'], [[0, 1], [0, 2], [1, 2]])
        np.testing.assert_allclose(row['actual'], [[2.5], [2], [2.5], [3.5], [4], [8.5], [-.5], [0], [-.5]],
                                   rtol=0, atol=audit.ATOL)
        difference = self.rows['cmf_half_triangle_vs_full_frobenius']
        self.assertFalse(difference['matched_independent_expected'])
        self.assertAlmostEqual(difference['actual'][0], 41/21, places=14)
        self.assertEqual(difference['independent_expected'], [41/25])

    def test_estimator_is_unexecuted_and_intercept_is_independent(self):
        estimator = self.report['unexecuted_estimator']
        self.assertEqual(estimator['execution'], 'not_run')
        self.assertEqual(estimator['factory_static'], 'LinearRegression(positive=True)')
        self.assertFalse(estimator['fit_intercept_argument_explicit'])
        control = estimator['independent_conditional_intercept_control']
        self.assertIn('not original estimator', control['execution'])
        self.assertAlmostEqual(control['power'], 87/35, places=14)
        self.assertAlmostEqual(control['intercept'], -8/5, places=14)

    def test_original_spectral_normalization_boundaries(self):
        interior = self.rows['power_spectra_integer_cosine']
        self.assertEqual(interior['actual'], [.5])
        partial = self.rows['power_spectra_nonintegral_block_count']
        self.assertEqual(partial['actual_complete_blocks'], 1)
        self.assertEqual(partial['original_num_blocks'], 25/16)
        self.assertEqual(partial['actual'], [.32])
        self.assertFalse(partial['matched_independent_expected'])
        for name in ('power_spectra_dc_endpoint', 'power_spectra_nyquist_endpoint'):
            self.assertEqual(self.rows[name]['actual'], [2.])
            self.assertEqual(self.rows[name]['independent_expected'], [1.])
            self.assertFalse(self.rows[name]['matched_independent_expected'])

    def test_scope_and_counts_do_not_claim_full_package(self):
        scope = self.report['execution_scope']
        self.assertFalse(scope['package_imported'])
        self.assertFalse(scope['source_math_bodies_changed'])
        self.assertTrue(scope['traits_construction_replaced_by_protocol_objects'])
        self.assertTrue(scope['numpy_fft_replaces_scipy_fft'])
        self.assertTrue(scope['custom_grid_driver_replaces_original_beamformerFreq_and_guvectorize_dispatch'])
        self.assertEqual(self.report['counts'], {'limited_numerical_cases': 21,
                         'matched_independent_expected': 13,
                         'observed_original_behavior_differences': 8,
                         'unexecuted_estimators': 1})
        self.assertEqual(len(self.rows), 21)
        self.assertEqual(len(self.report['original_definitions']), 18)
        self.assertTrue(all(r['mathematical_body_unchanged'] for r in self.report['original_definitions']))
        self.assertEqual(self.report['direct_sources']['codes/chapters/ch12/examples/audit_upstream_imaging_contracts.py'],
                         hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        strict_json_loads(json.dumps(self.report, allow_nan=False))


class ImagingHistoricalReportTests(unittest.TestCase):
    """Preserve real historical bytes and their real historical tool binding."""
    revision = 'ba622a6a2999cc30e15efe3e77a15c30d99cb51d'

    def blob(self, relative):
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        env['GIT_NO_LAZY_FETCH'] = '1'
        return subprocess.check_output(['git', 'show', self.revision + ':' + relative],
                                       cwd=audit.ROOT, env=env)

    def test_original_report_and_bound_tool_are_real_preedit_git_bytes(self):
        relative = 'codes/chapters/ch14/reports/upstream_imaging_contracts.json'
        old = self.blob(relative)
        self.assertEqual(audit.HISTORICAL_REPORT.read_bytes(), old)
        report = strict_json_loads(old)
        tool = 'codes/chapters/ch14/examples/audit_upstream_imaging_contracts.py'
        self.assertEqual(hashlib.sha256(self.blob(tool)).hexdigest(), report['direct_sources'][tool])
        for relative, expected in report['direct_sources'].items():
            self.assertEqual(hashlib.sha256(self.blob(relative)).hexdigest(), expected)
        self.assertEqual(report['counts']['limited_numerical_cases'], 21)
        self.assertEqual(report['counts']['observed_original_behavior_differences'], 8)
        self.assertNotEqual(audit.HISTORICAL_REPORT, audit.CURRENT_REPORT)


if __name__ == '__main__':
    unittest.main()
