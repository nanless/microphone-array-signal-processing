"""Meaningful negative controls for source bytes, history and report boundaries."""
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.appendix_a.examples import audit_upstream_solver_contracts as audit
from codes.chapters.ch00.core import source_history
from tests import test_codes_upstream_solver_contracts as fixtures


class SolverGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.folder = Path(self.temp.name)
        self.cache = self.folder / 'cache'
        self.cache.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_all_in_repo_targets_except_new_current_rejected_before_calls(self):
        forbidden = (audit.HISTORICAL_REPORT, audit.ROOT / 'AGENTS.md',
                     audit.ROOT / 'chapters/12_appendix-symbols-math.md',
                     Path(audit.__file__), audit.LOCK, audit.STATUS,
                     audit.ROOT / 'codes/chapters/ch00/source_snapshots/fake.json',
                     audit.ROOT / 'reviews/forbidden.json', audit.CACHE / 'forbidden.json')
        for target in forbidden:
            with self.subTest(target=target), patch.object(audit, 'run_audit') as run, \
                    patch.object(audit, 'write_json_report') as write, self.assertRaises(ValueError):
                audit.main(['--cache', str(self.cache), '--report', str(target)])
            run.assert_not_called()
            write.assert_not_called()
        # Only preflight; never create or change a formal current report.
        self.assertEqual(audit.report_target(audit.CURRENT_REPORT, self.cache), audit.CURRENT_REPORT)

    def test_external_current_report_roundtrip_and_default_stdout_no_writer(self):
        payload = {'finite': 1, 'scope': 'mock CLI output, no original execution'}
        before = sorted(self.folder.iterdir())
        stream = io.StringIO()
        with patch.object(audit, 'run_audit', return_value=payload) as run, \
                patch.object(audit, 'write_report') as write, redirect_stdout(stream):
            audit.main(['--cache', str(self.cache)])
        run.assert_called_once_with(self.cache)
        write.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue()), payload)
        self.assertEqual(sorted(self.folder.iterdir()), before)
        target = self.folder / 'current.json'
        with patch.object(audit, 'run_audit', return_value=payload):
            audit.main(['--cache', str(self.cache), '--report', str(target)])
        self.assertEqual(json.loads(target.read_text()), payload)

    def test_late_target_change_is_rejected_and_old_file_unchanged(self):
        target, victim = self.folder / 'current.json', self.folder / 'victim.json'
        victim.write_text('preserve me')
        def change_during_run(cache):
            target.symlink_to(victim)
            return {'x': 1}
        with patch.object(audit, 'run_audit', side_effect=change_during_run), self.assertRaises(ValueError):
            audit.main(['--cache', str(self.cache), '--report', str(target)])
        self.assertEqual(victim.read_text(), 'preserve me')
        self.assertTrue(target.is_symlink())

    def test_replacement_preflight_after_json_serialization(self):
        target, victim = self.folder / 'current.json', self.folder / 'victim.json'
        target.write_text('old')
        victim.write_text('safe')
        real_dumps = json.dumps
        def serialize_then_replace_with_link(*args, **kwargs):
            result = real_dumps(*args, **kwargs)
            target.unlink()
            target.symlink_to(victim)
            return result
        with patch('codes.chapters.ch00.io_contracts.json.dumps', side_effect=serialize_then_replace_with_link), \
                self.assertRaises(ValueError):
            audit.write_report(target, {'x': 1}, self.cache)
        self.assertEqual(victim.read_text(), 'safe')
        self.assertEqual(list(self.folder.glob('.report-*')), [])

    def test_hard_link_report_member_rejected(self):
        victim = self.folder / 'victim.json'
        victim.write_text('safe')
        target = self.folder / 'linked.json'
        os.link(victim, target)
        with self.assertRaises(ValueError):
            audit.write_report(target, {'x': 1}, self.cache)
        self.assertEqual(victim.read_text(), 'safe')

    def source_fixture(self):
        path = self.cache / 'pb_bss/math/solve.py'
        path.parent.mkdir(parents=True)
        source = b'import math\nORIGINAL = math.sqrt(9)\nFILE = __file__\n'
        path.write_bytes(source)
        return path, source

    def test_real_valid_timestamp_pyc_is_ignored_without_deletion(self):
        path, source = self.source_fixture()
        cached = Path(importlib.util.cache_from_source(str(path)))
        cached.parent.mkdir()
        sentinel = compile('BYTECODE_SENTINEL = "unverified_cache_loaded"', str(path), 'exec')
        payload = (importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, int(path.stat().st_mtime), len(source))
                   + marshal.dumps(sentinel))
        cached.write_bytes(payload)
        cached_mtime = cached.stat().st_mtime_ns
        # Independently establish that this is a genuinely loadable valid pyc,
        # not merely a file with a .pyc suffix.
        spec = importlib.util.spec_from_file_location('_negative_cached', path)
        poisoned = importlib.util.module_from_spec(spec)
        old = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(poisoned)
        finally:
            sys.dont_write_bytecode = old
        self.assertEqual(poisoned.BYTECODE_SENTINEL, 'unverified_cache_loaded')
        with patch.dict(audit.FILES, {'pb_bss/math/solve.py': audit.digest(source)}):
            original = audit.load_original(self.cache)
        self.assertEqual(original.ORIGINAL, 3)
        self.assertFalse(hasattr(original, 'BYTECODE_SENTINEL'))
        self.assertEqual(original.FILE, str(path))
        self.assertEqual(original.__spec__.origin, str(path))
        self.assertEqual(original.__package__, '')
        self.assertIsNone(original.__cached__)
        self.assertEqual(path.read_bytes(), source)
        self.assertEqual(cached.read_bytes(), payload)
        self.assertEqual(cached.stat().st_mtime_ns, cached_mtime)

    def test_changed_source_bytes_fail_before_execution(self):
        path, source = self.source_fixture()
        path.write_bytes(b'raise RuntimeError("must not execute")\n')
        with patch.dict(audit.FILES, {'pb_bss/math/solve.py': audit.digest(source)}), \
                self.assertRaisesRegex(ValueError, 'source bytes changed'):
            audit.load_original(self.cache)

    def test_dependencies_are_four_actual_files_and_numpy_scope_is_finite(self):
        rows = audit.actual_dependencies()
        self.assertEqual(set(rows), {
            'codes/chapters/appendix_a/examples/audit_upstream_solver_contracts.py',
            'codes/chapters/ch04/core/upstream_contracts.py',
            'codes/chapters/ch00/io_contracts.py',
            'codes/chapters/ch00/upstream/fetch_upstreams.py'})
        for relative, digest in rows.items():
            self.assertEqual(digest, hashlib.sha256((audit.ROOT / relative).read_bytes()).hexdigest())
        for record in audit.numpy_identities().values():
            self.assertEqual(record['sha256'], hashlib.sha256(Path(record['path']).read_bytes()).hexdigest())
            self.assertIn('not whole package or native LAPACK', record['scope'])

    def run_with_mock_original(self, dependencies=None, numpy_ids=None):
        # This fake source is used only to test fail-closed sequencing, never to
        # produce source-verified numerical results or a written report.
        with patch.object(audit, 'verify_sources', return_value={'checkout': str(self.cache), 'source_identity': {}}), \
                patch.object(audit, 'load_original') as load, \
                patch.object(audit, 'solver_cases', return_value=[]), \
                patch.object(audit, 'numpy_cases', return_value=[]), \
                patch.object(audit, 'additional_solver_cases', return_value=[]), \
                patch.object(audit.upstream, 'check_unchanged', return_value={}) as post:
            patches = []
            if dependencies is not None:
                patches.append(patch.object(audit, 'actual_dependencies', side_effect=dependencies))
            if numpy_ids is not None:
                patches.append(patch.object(audit, 'numpy_identities', side_effect=numpy_ids))
            from contextlib import ExitStack
            with ExitStack() as stack:
                for change in patches:
                    stack.enter_context(change)
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    audit.run_audit(self.cache)
            load.assert_called_once()
            post.assert_called_once()

    def test_direct_source_change_during_call_fails_closed(self):
        before = audit.actual_dependencies()
        after = dict(before)
        after[next(iter(after))] = '0' * 64
        self.run_with_mock_original(dependencies=[before, after])

    def test_numpy_wrapper_change_during_call_fails_closed(self):
        before = audit.numpy_identities()
        after = copy.deepcopy(before)
        after['linalg_lstsq_wrapper']['sha256'] = '0' * 64
        self.run_with_mock_original(numpy_ids=[before, after])

    def test_preflight_failure_does_not_load_original(self):
        with patch.object(audit, 'verify_sources', side_effect=ValueError('bad preflight')), \
                patch.object(audit, 'load_original') as load, self.assertRaisesRegex(ValueError, 'bad preflight'):
            audit.run_audit(self.cache)
        load.assert_not_called()

    def test_postflight_failure_does_not_return_success_report(self):
        with patch.object(audit, 'verify_sources', return_value={'checkout': str(self.cache), 'source_identity': {}}), \
                patch.object(audit, 'load_original'), patch.object(audit, 'solver_cases', return_value=[]), \
                patch.object(audit, 'numpy_cases', return_value=[]), \
                patch.object(audit, 'additional_solver_cases', return_value=[]), \
                patch.object(audit.upstream, 'check_unchanged', side_effect=ValueError('postflight changed')), \
                self.assertRaisesRegex(ValueError, 'postflight changed'):
            audit.run_audit(self.cache)


class SourceFixtureGuards(unittest.TestCase):
    """Additional realistic Git/status controls reuse the fixture construction."""
    setUp = fixtures.PathAndSourceTests.setUp
    tearDown = fixtures.PathAndSourceTests.tearDown
    make_fixture = fixtures.PathAndSourceTests.make_fixture
    fixture_identity = fixtures.PathAndSourceTests.fixture_identity

    def test_wrong_head_rejected(self):
        checkout, lock, rev, files, git = self.make_fixture()
        (checkout / 'extra').write_text('a new commit')
        git('add', '.')
        git('-c', 'commit.gpgsign=false', 'commit', '-m', 'other')
        with self.fixture_identity(lock, rev, files), self.assertRaisesRegex(ValueError, 'origin or HEAD'):
            audit.verify_sources(self.cache)

    def test_status_wrong_lock_binding_rejected(self):
        _, lock, rev, files, _ = self.make_fixture()
        record = json.loads(self.status.read_text())
        record['lock_sha256'] = '0' * 64
        self.status.write_text(json.dumps(record))
        with self.fixture_identity(lock, rev, files), self.assertRaisesRegex(ValueError, 'bind current lock'):
            audit.verify_sources(self.cache)

    def test_real_source_blob_change_rejected_even_if_expected_sha_changed(self):
        checkout, lock, rev, files, _ = self.make_fixture()
        source = checkout / 'pb_bss/math/solve.py'
        source.write_text('# changed original\n')
        files['pb_bss/math/solve.py'] = audit.digest(source.read_bytes())
        with self.fixture_identity(lock, rev, files), self.assertRaisesRegex(ValueError, 'entirely clean'):
            audit.verify_sources(self.cache)

    def test_non_root_subfolder_cannot_borrow_outer_git_identity(self):
        checkout, lock, rev, files, _ = self.make_fixture()
        nested = checkout / 'nested'
        nested.mkdir()
        with self.fixture_identity(lock, rev, files), self.assertRaises((FileNotFoundError, ValueError)):
            audit.upstream.verify_project('pb_bss', nested, relatives=tuple(files),
                                          lock_path=lock, status_path=self.status)

    def test_true_selection_mismatch_is_separate_from_used_blob_identity(self):
        checkout, lock, rev, files, _ = self.make_fixture()
        entry = json.loads(lock.read_text())
        entry['projects'][0]['source_paths'] = ['pb_bss/math']
        lock.write_text(json.dumps(entry))
        record = json.loads(self.status.read_text())
        record['lock_sha256'] = audit.digest(lock.read_bytes())
        record['projects'][0]['status'] = 'source_selection_mismatch'
        self.status.write_text(json.dumps(record))
        with self.fixture_identity(lock, rev, files):
            identity = audit.verify_sources(self.cache)['source_identity']
        self.assertEqual(identity['used_source_identity'], 'verified')
        self.assertFalse(identity['live_complete_selection']['source_selection_verified'])
        self.assertEqual(identity['live_complete_selection']['status'], 'source_selection_mismatch')

    def test_ignored_membership_change_rejected_after_calls(self):
        checkout, lock, rev, files, git = self.make_fixture()
        (checkout / '.gitignore').write_text('*.pyc\n')
        git('add', '.gitignore')
        git('-c', 'commit.gpgsign=false', 'commit', '-m', 'ignore bytecode')
        rev = git('rev-parse', 'HEAD')
        document = json.loads(lock.read_text())
        document['projects'][0]['revision'] = rev
        lock.write_text(json.dumps(document))
        record = json.loads(self.status.read_text())
        record['projects'][0]['revision'] = rev
        record['lock_sha256'] = audit.digest(lock.read_bytes())
        self.status.write_text(json.dumps(record))
        with self.fixture_identity(lock, rev, files):
            identity = audit.verify_sources(self.cache)['source_identity']
            (checkout / 'new.pyc').write_bytes(b'ignored fixture')
            with self.assertRaisesRegex(ValueError, 'Ignored source membership changed'):
                audit.upstream.check_unchanged(identity)


class HistoricalSolverTests(unittest.TestCase):
    def test_historical_report_binds_actual_old_git_tool_and_registered_lock(self):
        raw = audit.HISTORICAL_REPORT.read_bytes()
        self.assertEqual(audit.digest(raw), 'a12c768cacc7d6c1256a4c758cfc18c25b1ed00ba9fef74a7c7350388a27c7ab')
        old = audit.strict_loads(raw)
        environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                           GIT_NO_LAZY_FETCH='1', GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0')
        original = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false',
            'show', '241513c8f35d952747f16ccbb40cf85a5363b5ae:codes/chapters/appendix_a/examples/audit_upstream_solver_contracts.py'],
            cwd=audit.ROOT, env=environment, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=20).stdout
        self.assertEqual(audit.digest(original), old['tool_sha256'])
        self.assertNotEqual(audit.digest(Path(audit.__file__).read_bytes()), old['tool_sha256'])
        bound = source_history.verify_lock_binding(old['source_lock_sha256'], ['pb_bss'])
        saved = audit.strict_loads(Path(bound['snapshot_path']).read_bytes())
        self.assertEqual(len(saved['projects']), 100)
        self.assertTrue(bound['historical'])
        self.assertNotIn('source_status_sha256', old)
        self.assertEqual(old['counts'], {'original_function_cases': 4, 'independent_numpy_cases': 2,
                                       'original_matches': 3, 'original_known_dtype_failures': 1})
        self.assertEqual({r['path']: r['sha256'] for r in old['sources']}, audit.FILES)


if __name__ == '__main__':
    unittest.main()
