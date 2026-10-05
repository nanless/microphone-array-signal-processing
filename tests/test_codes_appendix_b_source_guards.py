"""TAC current/history, write-preflight and actual source identity controls.

Fixtures are local synthetic Git repositories; none executes an author network.
Source-history is a test dependency only, not an audit execution dependency.
"""
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.appendix_b.examples import audit_tac_contracts as audit
from codes.chapters.ch00.core import source_history
from tests import test_codes_tac_contracts as fixtures


class TACGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.folder = Path(self.temp.name)
        self.cache = self.folder / 'cache'
        self.cache.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    @contextmanager
    def fixture(self):
        helper = fixtures.TACSourceAndPathTests()
        helper.setUp()
        try:
            checkout, lock, revision, files, git = helper.make_fixture()
            with helper.fixture_context(lock, revision, files):
                yield helper, checkout, lock, git
        finally:
            helper.tearDown()

    def test_repo_whitelist_and_protected_paths_before_execution(self):
        forbidden = (audit.HISTORICAL_REPORT, audit.ROOT / 'AGENTS.md', Path(audit.__file__),
                     audit.ROOT / 'chapters/15_appendix-guide.md', audit.LOCK, audit.STATUS,
                     audit.ROOT / 'codes/chapters/ch00/source_snapshots/new.json',
                     audit.ROOT / 'reviews/new.json', audit.CACHE / 'new.json',
                     self.cache / 'new.json', audit.CURRENT_REPORT.with_name('tac_contracts_current2.json'))
        for path in forbidden:
            with self.subTest(path=path), patch.object(audit, 'run_audit') as run, \
                    patch.object(audit, 'write_json_report') as write, self.assertRaises(ValueError):
                audit.main(['--cache', str(self.cache), '--report', str(path)])
            run.assert_not_called()
            write.assert_not_called()
        self.assertEqual(audit.report_target(audit.CURRENT_REPORT, self.cache), audit.CURRENT_REPORT)

    def test_default_stdout_and_external_near_match_name(self):
        report = {'scope': 'mock CLI, not author execution', 'counts': {'original_runtime_calls': 0}}
        before = sorted(self.folder.iterdir())
        stream = io.StringIO()
        with patch.object(audit, 'run_audit', return_value=report), \
                patch.object(audit, 'write_report') as write, redirect_stdout(stream):
            audit.main(['--cache', str(self.cache)])
        write.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue()), report)
        self.assertEqual(sorted(self.folder.iterdir()), before)
        # Matching a historical basename outside the repository is not itself forbidden.
        target = self.folder / 'tac_contracts.json'
        audit.write_report(target, report, self.cache)
        self.assertEqual(json.loads(target.read_text()), report)

    def test_hardlink_and_fifo_members_rejected(self):
        victim = self.folder / 'victim'; victim.write_text('preserve')
        hard = self.folder / 'hard'; os.link(victim, hard)
        fifo = self.folder / 'fifo'; os.mkfifo(fifo)
        for path in (hard, fifo):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.write_report(path, {'x': 1}, self.cache)
        self.assertEqual(victim.read_text(), 'preserve')

    def test_late_link_rejected_after_execution(self):
        target = self.folder / 'report.json'; victim = self.folder / 'victim.json'
        victim.write_text('preserve')
        def run(cache):
            target.symlink_to(victim)
            return {'scope': 'mock'}
        with patch.object(audit, 'run_audit', side_effect=run), self.assertRaises(ValueError):
            audit.main(['--cache', str(self.cache), '--report', str(target)])
        self.assertEqual(victim.read_text(), 'preserve')

    def test_replacement_rechecks_parent_member_after_temporary_write(self):
        target = self.folder / 'report.json'; target.write_text('old')
        victim = self.folder / 'victim.json'; victim.write_text('preserve')
        original_fdopen = os.fdopen
        @contextmanager
        def write_then_link(*args, **kwargs):
            with original_fdopen(*args, **kwargs) as handle:
                yield handle
            target.unlink(); target.symlink_to(victim)
        with patch('codes.chapters.ch00.io_contracts.os.fdopen', side_effect=write_then_link), \
                patch('codes.chapters.ch00.io_contracts.os.replace') as replace, self.assertRaises(ValueError):
            audit.write_report(target, {'x': 1}, self.cache)
        replace.assert_not_called()
        self.assertEqual(victim.read_text(), 'preserve')
        self.assertEqual(list(self.folder.glob('.report-*')), [])

    def test_status_stale_lock_binding(self):
        with self.fixture() as (_, _, lock, _):
            status = lock.with_name('status.json')
            doc = json.loads(status.read_text()); doc['lock_sha256'] = '0' * 64
            status.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'bind current lock'):
                audit.verify_sources(lock.parent / 'cache')

    def test_status_duplicate_schema_and_type_not_coerced(self):
        with self.fixture() as (_, _, lock, _):
            status = lock.with_name('status.json'); original = json.loads(status.read_text())
            variants = []
            d = copy.deepcopy(original); d['schema_version'] = True; variants.append(d)
            d = copy.deepcopy(original); d['projects'] *= 2; variants.append(d)
            d = copy.deepcopy(original); d['projects'][0]['source_selection_verified'] = 1; variants.append(d)
            for doc in variants:
                status.write_text(json.dumps(doc))
                with self.subTest(doc=doc), self.assertRaises(ValueError):
                    audit.verify_sources(lock.parent / 'cache')

    def test_current_selection_mismatch_is_separate_from_used_blob_identity(self):
        with self.fixture() as (helper, checkout, lock, _):
            sparse = checkout / '.git/info/sparse-checkout'
            sparse.write_text(sparse.read_text().replace('!**/*.wav\n', ''))
            helper.refresh_status(lock)
            result = audit.verify_sources(helper.cache)
            identity = result['source_identity']
            self.assertEqual(identity['used_source_identity'], 'verified')
            for name in ('recorded_complete_selection', 'live_complete_selection'):
                self.assertEqual(identity[name]['status'], 'source_selection_mismatch')
                self.assertIs(identity[name]['source_selection_verified'], False)
            self.assertEqual(len(identity['used_files']), 4)

    def test_modified_blob_hidden_from_git_status_is_rejected(self):
        with self.fixture() as (helper, checkout, _, git):
            git('update-index', '--assume-unchanged', 'utility/models.py')
            (checkout / 'utility/models.py').write_text('# altered despite empty status')
            self.assertEqual(git('status', '--porcelain'), '')
            with self.assertRaisesRegex(ValueError, 'differs from fixed Git blob'):
                audit.verify_sources(helper.cache)

    def test_origin_change_after_static_read_is_rejected(self):
        with self.fixture() as (helper, _, _, git):
            def changed(*texts):
                git('remote', 'set-url', 'origin', 'https://example.invalid/changed.git')
                return []
            with patch.object(audit, 'structure_contracts', side_effect=changed), \
                    self.assertRaisesRegex(ValueError, 'changed during'):
                audit.run_audit(helper.cache)

    def test_direct_dependency_change_fails_closed(self):
        before = audit.actual_dependencies(); after = dict(before)
        after[next(iter(after))] = '0' * 64
        with patch.object(audit, 'actual_dependencies', side_effect=[before, after]), \
                self.assertRaisesRegex(ValueError, 'direct source'):
            audit.run_audit()

    def test_upstream_dirty_after_static_read_fails_closed(self):
        with self.fixture() as (helper, checkout, _, _):
            def changed(*texts):
                (checkout / 'utility/models.py').write_text('# changed after read')
                return []
            with patch.object(audit, 'structure_contracts', side_effect=changed), \
                    self.assertRaisesRegex(ValueError, 'changed during'):
                audit.run_audit(helper.cache)

    def test_status_bytes_changed_after_static_read_fails_closed(self):
        with self.fixture() as (helper, _, lock, _):
            def changed(*texts):
                status = lock.with_name('status.json')
                status.write_text(status.read_text() + '\n')
                return []
            with patch.object(audit, 'structure_contracts', side_effect=changed), \
                    self.assertRaisesRegex(ValueError, 'recorded selection changed'):
                audit.run_audit(helper.cache)

    def test_ignored_membership_change_is_detected(self):
        with self.fixture() as (helper, checkout, _, _):
            (checkout / '.git/info/exclude').write_text('*.ignored\n')
            def changed(*texts):
                (checkout / 'late.ignored').write_text('ignored but not invisible')
                return []
            with patch.object(audit, 'structure_contracts', side_effect=changed), \
                    self.assertRaisesRegex(ValueError, 'Ignored source membership'):
                audit.run_audit(helper.cache)

    def test_actual_four_direct_sources_and_current_115_status(self):
        report = audit.run_audit()
        self.assertEqual(report['direct_sources_before'], report['direct_sources_after'])
        self.assertEqual(len(report['direct_sources_before']), 4)
        self.assertNotIn('codes/chapters/ch00/core/source_history.py', report['direct_sources_before'])
        for relative, digest in report['direct_sources_before'].items():
            self.assertEqual(hashlib.sha256((audit.ROOT / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual(report['source_lock_project_count'], 115)
        self.assertEqual(report['source_status_sha256'], audit.digest(audit.STATUS.read_bytes()))
        self.assertFalse(report['source_identity_before']['clean_after'])
        self.assertTrue(report['source_identity_after']['clean_after'])
        self.assertEqual(report['counts'], {'static_contracts': 10, 'original_runtime_calls': 0})

    def test_historical_report_actual_git_tool_and_original_ten_contracts(self):
        payload = audit.HISTORICAL_REPORT.read_bytes()
        mtime = audit.HISTORICAL_REPORT.stat().st_mtime_ns
        self.assertEqual(audit.digest(payload), 'a55fa1f2e77788fb21e4ddb9051dc61e2c078e64954bf38e571a092f681cbbda')
        old = audit.strict_loads(payload)
        relative = Path(audit.__file__).resolve().relative_to(audit.ROOT).as_posix()
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        tool = subprocess.check_output(['git', '-C', str(audit.ROOT), 'show',
            '2237e021630b62c372082b7247e4f8d9b54ee674:' + relative], env=env, timeout=20)
        self.assertEqual(audit.digest(tool), old['tool_sha256'])
        self.assertEqual(old['tool_sha256'], '17795245c1af54ead26355010397f4e83a452d21124674e97f3875f8cd8cb422')
        binding = source_history.verify_lock_binding(old['source_lock_sha256'], ['tac'])
        self.assertTrue(binding['historical'])
        self.assertEqual(len(json.loads(Path(binding['snapshot_path']).read_text())['projects']), 100)
        current = audit.run_audit()
        self.assertEqual(current['contracts'], old['contracts'])
        self.assertEqual(current['counts'], old['counts'])
        self.assertEqual(audit.HISTORICAL_REPORT.read_bytes(), payload)
        self.assertEqual(audit.HISTORICAL_REPORT.stat().st_mtime_ns, mtime)


if __name__ == '__main__':
    unittest.main()
