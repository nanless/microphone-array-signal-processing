"""Offline source negatives and actual bounded contracts; no upstream writes."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from codes.chapters.ch00.io_contracts import write_json_report
from codes.chapters.ch00.upstream.fetch_upstreams import run_git
from codes.chapters.ch13.examples import audit_upstream_distributed_contracts as audit


class OfflineSharedSourceGuards(unittest.TestCase):
    """Temporary Git fixture exercises real shared contracts, not original methods."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.repo = self.directory / 'fixture'
        self.repo.mkdir()
        run_git(['init', '-q'], cwd=self.repo)
        run_git(['config', 'user.name', 'Offline fixture'], cwd=self.repo)
        run_git(['config', 'user.email', 'fixture@example.invalid'], cwd=self.repo)
        self.origin = 'https://example.invalid/fixture.git'
        run_git(['remote', 'add', 'origin', self.origin], cwd=self.repo)
        self.license = self.repo / 'LICENSE'
        self.license.write_text('Offline MIT identity fixture.\n')
        (self.repo / '.gitignore').write_text('ignored.txt\n')
        run_git(['add', 'LICENSE', '.gitignore'], cwd=self.repo)
        run_git(['commit', '-qm', 'offline fixture'], cwd=self.repo)
        self.revision = run_git(['rev-parse', 'HEAD'], cwd=self.repo)
        self.project = {'id': 'danse-wola', 'revision': self.revision,
                        'url': self.origin, 'license': 'MIT', 'entrypoints': ['LICENSE']}
        self.lock = self.directory / 'lock.json'
        self.status = self.directory / 'status.json'
        write_json_report(self.lock, {'schema_version': 1, 'projects': [self.project]})
        self.state = {'id': 'danse-wola', 'revision': self.revision, 'status': 'source_verified',
                      'source_selection_verified': True, 'missing_entrypoints': [],
                      'observed_sparse_patterns': [], 'expected_sparse_patterns': [],
                      'requested_source_paths': ['repository'], 'asset_policy': 'existing_checkout_preserved'}
        self.write_status()
        registry = {'danse-wola': (self.origin, self.revision, 'MIT', 'LICENSE',
                                  hashlib.sha256(self.license.read_bytes()).hexdigest())}
        self.guard = patch.object(audit.upstream, 'PROJECTS', registry)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def write_status(self):
        write_json_report(self.status, {'schema_version': 1,
                          'lock_sha256': hashlib.sha256(self.lock.read_bytes()).hexdigest(),
                          'projects': [self.state]})

    def verify(self):
        return audit.upstream.verify_project('danse-wola', self.repo, relatives=('LICENSE',),
                                           lock_path=self.lock, status_path=self.status)

    def test_actual_origin_blob_license_and_prepost(self):
        row = self.verify()
        self.assertEqual(row['used_files']['LICENSE']['git_blob'], row['used_files']['LICENSE']['actual_blob'])
        self.assertTrue(audit.upstream.check_unchanged(row)['clean_after'])

    def test_injected_git_environment_is_filtered(self):
        with patch.dict(os.environ, {'GIT_DIR': '/missing/override', 'GIT_INDEX_FILE': '/missing/index',
                                    'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'remote.origin.url',
                                    'GIT_CONFIG_VALUE_0': 'https://wrong.invalid/override'}):
            self.assertEqual(self.verify()['origin'], self.origin)

    def test_origin_and_head_changes_rejected(self):
        run_git(['remote', 'set-url', 'origin', 'https://wrong.invalid/fixture.git'], cwd=self.repo)
        with self.assertRaisesRegex(ValueError, 'origin or HEAD'):
            self.verify()
        run_git(['remote', 'set-url', 'origin', self.origin], cwd=self.repo)
        run_git(['commit', '--allow-empty', '-qm', 'different HEAD'], cwd=self.repo)
        with self.assertRaisesRegex(ValueError, 'origin or HEAD'):
            self.verify()

    def test_own_root_and_symlink_guard(self):
        inner = self.repo / 'inner'
        inner.mkdir()
        with self.assertRaises(FileNotFoundError):
            audit.upstream.verify_project('danse-wola', inner, lock_path=self.lock, status_path=self.status)
        link = self.directory / 'alias'
        link.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            audit.upstream.verify_project('danse-wola', link, lock_path=self.lock, status_path=self.status)

    def test_dirty_and_nonignored_untracked_rejected(self):
        self.license.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'clean'):
            self.verify()
        run_git(['checkout', '--', 'LICENSE'], cwd=self.repo)
        (self.repo / 'extra').write_text('untracked')
        with self.assertRaisesRegex(ValueError, 'clean'):
            self.verify()

    def test_ignored_member_change_rejected_after_use(self):
        before = self.verify()
        (self.repo / 'ignored.txt').write_text('not a tracked source')
        with self.assertRaisesRegex(ValueError, 'Ignored source membership'):
            audit.upstream.check_unchanged(before)

    def test_current_status_requires_exact_lock_bytes(self):
        document = json.loads(self.status.read_text())
        document['lock_sha256'] = '0' * 64
        write_json_report(self.status, document)
        with self.assertRaisesRegex(ValueError, 'bind current lock'):
            self.verify()

    def test_strict_json_and_duplicate_ids_rejected(self):
        self.status.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):
            self.verify()
        self.write_status()
        write_json_report(self.lock, {'schema_version': 1, 'projects': [self.project, self.project]})
        self.write_status()
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.verify()

    def test_fixed_license_sha_rejected_even_when_git_clean(self):
        self.license.write_text('different license')
        run_git(['add', 'LICENSE'], cwd=self.repo)
        run_git(['commit', '-qm', 'changed license'], cwd=self.repo)
        self.revision = run_git(['rev-parse', 'HEAD'], cwd=self.repo)
        self.project['revision'] = self.revision
        self.state['revision'] = self.revision
        write_json_report(self.lock, {'schema_version': 1, 'projects': [self.project]})
        self.write_status()
        original = audit.upstream.PROJECTS['danse-wola']
        audit.upstream.PROJECTS['danse-wola'] = (original[0], self.revision, *original[2:])
        with self.assertRaisesRegex(ValueError, 'license differs'):
            self.verify()


class CurrentActualContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are required current source contracts; missing caches are an
        # explicit error here, never a fake passing original implementation.
        cls.report = audit.run_audit()

    def test_counts_remain_original_16_and_static_14_independent_5(self):
        self.assertEqual(self.report['counts'], {'original_case_invocations': 16,
                         'matched_independent_expected': 13,
                         'observed_original_behavior_differences': 1,
                         'unobserved_peaks_with_numeric_output': 2})
        self.assertEqual(len(self.report['matlab_static_contracts']), 14)
        self.assertEqual(len(self.report['independent_matlab_controls']), 5)

    def test_direct_closure_is_actual_and_unchanged(self):
        expected = audit.actual_dependencies()
        self.assertEqual(self.report['direct_sources'], expected)
        self.assertEqual(self.report['actual_dependencies_before'], expected)
        self.assertEqual(self.report['actual_dependencies_after'], expected)
        self.assertEqual(len(expected), 6)
        self.assertIn('codes/chapters/ch00/core/source_history.py', expected)
        self.assertIn('codes/chapters/ch04/core/upstream_contracts.py', expected)

    def test_current_lock_status_and_original_blobs_separate(self):
        report = self.report
        self.assertEqual(report['source_lock_sha256'], audit.upstream.sha(audit.LOCK))
        self.assertEqual(report['source_status_sha256'], audit.upstream.sha(audit.upstream.STATUS))
        for name, after in report['source_identity_after']['projects'].items():
            self.assertTrue(after['clean_after'])
            self.assertEqual(after['recorded_complete_selection']['status'], 'source_verified')
            self.assertEqual(after['live_complete_selection']['status'], 'source_verified')
            for record in after['used_files'].values():
                self.assertEqual(record['git_blob'], record['actual_blob'])
            self.assertEqual(after['ignored_members_before'], after['ignored_members_after'])

    def test_actual_numpy_entry_has_limited_claim(self):
        self.assertEqual(self.report['external_namespace_before'], audit.numpy_entry_identity())
        self.assertEqual(self.report['external_namespace_after'], audit.numpy_entry_identity())
        self.assertIn('not verified', self.report['external_namespace_before']['scope'])

    def test_matlab_not_probed_is_not_runtime_absence(self):
        scope = self.report['execution_scope']
        self.assertFalse(scope['matlab_runtime_probed'])
        self.assertFalse(scope['matlab_executed'])
        self.assertIn('not probed', scope['matlab_status'])
        self.assertNotIn('unavailable', scope['matlab_status'])

    def test_dwacd_is_static_and_independent_not_original_call(self):
        row = self.report['additional_dwacd_static_activity_control']
        self.assertFalse(row['static_contract']['original_dwacd_executed'])
        self.assertEqual(row['static_contract']['assignments']['activity_seg_delayed']['end_line'], 349)
        control = row['independent_control']
        self.assertEqual(control['original_slice_length'], 16384)
        self.assertEqual(control['matching_signal_slice_length'], 8192)
        self.assertEqual(control['threshold'], 6144)
        self.assertEqual(control['original_slice_counts'], [8192]*4)
        self.assertEqual(control['matching_signal_slice_counts'], [8192,8192,0,8192])
        self.assertTrue(control['original_slice_gate'])
        self.assertFalse(control['matching_signal_slice_gate'])
        self.assertFalse(self.report['execution_scope']['dwacd_executed'])

    def test_static_original_slice_change_fails(self):
        source = audit.CACHE / 'paderwasn/paderwasn/synchronization/sro_estimation.py'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'changed.py'
            path.write_text(source.read_text().replace(
                'activity_sig[start_delayed+shift:start+shift+self.seg_len]',
                'activity_sig[start_delayed+shift:start_delayed+shift+self.seg_len]'))
            with self.assertRaisesRegex(ValueError, 'static contract changed'):
                audit.dwacd_static_gate_control(path)

    def test_local_dependency_change_rejected_after_calls(self):
        actual = audit.actual_dependencies()
        changed = dict(actual)
        changed['codes/chapters/ch00/io_contracts.py'] = '0' * 64
        with patch.object(audit, 'actual_dependencies', side_effect=[actual, changed]):
            with self.assertRaisesRegex(ValueError, 'dependencies or historical provenance changed'):
                audit.run_audit()

    def test_external_namespace_change_rejected(self):
        actual = audit.numpy_entry_identity()
        changed = {**actual, 'sha256': '0' * 64}
        with patch.object(audit, 'numpy_entry_identity', side_effect=[actual, changed]):
            with self.assertRaisesRegex(ValueError, 'dependencies or historical provenance changed'):
                audit.run_audit()


class HistoricalAndWriteGuards(unittest.TestCase):
    def test_real_historical_git_bytes_have_separate_commits(self):
        original = audit.HISTORICAL_REPORT.read_bytes()
        mtime = audit.HISTORICAL_REPORT.stat().st_mtime_ns
        evidence = audit.historical_provenance()
        self.assertEqual(evidence['report_sha256'], '8a8ef88ae9660ecb2fe8ea99d726652ae9aa44f91790e3f1f02d4d736ee8da4b')
        bindings = evidence['direct_source_git_bindings']
        helper = 'codes/chapters/ch14/examples/audit_upstream_imaging_contracts.py'
        self.assertEqual(bindings[helper]['commit'], 'a215b4630c0c21a8744cf27436a2c9ffa9c00053')
        self.assertNotEqual(bindings[helper]['sha256'], audit.actual_dependencies()['codes/chapters/ch12/examples/audit_upstream_imaging_contracts.py'])
        self.assertEqual(evidence['source_lock_binding']['report_sha256'],
                         'e3478006c7dbc6cec442bf6bccc4df9eca946d7d353b661e947596dd8b87608a')
        self.assertTrue(evidence['source_lock_binding']['historical'])
        for path, row in bindings.items():
            if path != helper:
                self.assertEqual(row['commit'], '6c1f1448dc0efdeb2ea964b9a6323643406ab209')
        self.assertEqual(audit.HISTORICAL_REPORT.read_bytes(), original)
        self.assertEqual(audit.HISTORICAL_REPORT.stat().st_mtime_ns, mtime)

    def test_all_protected_targets_fail_before_any_original_call(self):
        targets = [audit.HISTORICAL_REPORT, audit.upstream.STATUS, audit.LOCK,
                   audit.source_history.SNAPSHOT_ROOT / 'forbidden.json',
                   audit.CACHE / 'paderwasn/forbidden.json',
                   audit.ROOT / 'codes/chapters/ch13/reports/unregistered.json']
        with patch.object(audit, 'run_audit') as run:
            for target in targets:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    audit.main(['--report', str(target)])
            run.assert_not_called()

    def test_postflight_rejects_replaced_output_link_without_touching_victim(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            target, victim = directory / 'report.json', directory / 'victim.json'
            victim.write_text('preserved')
            def fake_run(cache):
                target.symlink_to(victim)
                return {'finite': True}
            with patch.object(audit, 'run_audit', side_effect=fake_run), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'symbolic link'):
                    audit.main(['--report', str(target)])
            self.assertEqual(victim.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()
