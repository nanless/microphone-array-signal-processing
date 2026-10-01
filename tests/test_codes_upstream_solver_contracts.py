"""Independent expectations and offline source/output boundary checks."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from codes.chapters.appendix_a.examples import audit_upstream_solver_contracts as audit


class SolverContractTests(unittest.TestCase):
    def test_numpy_cutoff_closed_forms(self):
        rows = audit.numpy_cases()
        self.assertEqual([r['rank'] for r in rows], [1, 2])
        np.testing.assert_allclose(rows[0]['output'], [1, 0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(rows[1]['output'], [1, 1], rtol=0, atol=1e-14)
        self.assertEqual([r['returned_residual_array'] for r in rows], [[], []])
        self.assertEqual(rows[0]['explicit_residual_norm'], 1e-8)
        self.assertEqual(rows[1]['explicit_residual_norm'], 0)

    @unittest.skipUnless((audit.CACHE / 'pb_bss/pb_bss/math/solve.py').is_file(),
                         'optional fixed pb_bss checkout is absent; no download')
    def test_actual_original_module(self):
        report = audit.run_audit()
        rows = {r['name']: r for r in report['cases']}
        expected = np.array([[float(Fraction(1, 10)), float(Fraction(1, 5))]] * 2)
        np.testing.assert_allclose(rows['singular_float_rhs']['output'], expected, rtol=0, atol=1e-14)
        self.assertEqual(rows['singular_integer_rhs']['output'], [[0, 0], [0, 0]])
        self.assertFalse(rows['singular_integer_rhs']['matched_expected'])
        self.assertEqual(rows['singular_integer_rhs']['classification'], 'observed_integer_fallback_truncation')
        self.assertEqual(rows['nonsingular_integer_rhs']['output'], [[1., 0.], [0., 1.]])
        psd = rows['psd_nullspace_ls']
        self.assertEqual(psd['output'], [[0.], [1.]])
        self.assertEqual(psd['independent_mvdr_model']['optimal_noise_power'], 0)
        self.assertEqual(psd['independent_mvdr_model']['normalized_ls_noise_power'], 1)
        self.assertEqual(report['counts'], {'original_function_cases': 4, 'independent_numpy_cases': 2,
                                          'original_matches': 3, 'original_known_dtype_failures': 1})
        self.assertEqual(report['tool_sha256'], hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        self.assertTrue(report['before_clean'] and report['after_clean'])
        self.assertFalse(report['execution']['ast_extraction'])
        self.assertEqual(report['execution']['substitutes'], [])
        audit.strict_loads(json.dumps(report, allow_nan=False))

    def test_json_invalid_and_near_valid(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.strict_loads(text)
        self.assertEqual(audit.strict_loads('{"x":1e-99,"X":2,"nested":{"x":3}}')['X'], 2)

    def test_bad_result_shape_and_nonfinite(self):
        for value in ([1], [[float('nan')]], [[float('inf')]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.compare(value, [[0]])

    def test_false_known_failure_not_hidden(self):
        class WrongModule:
            @staticmethod
            def stable_solve(a, b):
                return np.zeros_like(b)
        with self.assertRaisesRegex(ValueError, 'behavior changed'):
            audit.solver_cases(WrongModule)


class PathAndSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.folder = Path(self.temp.name)
        self.cache = self.folder / 'cache'
        self.cache.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_report_and_no_sidefiles(self):
        target = self.folder / 'report.json'
        target.write_text('old')
        audit.write_report(target, {'value': 1}, self.cache)
        self.assertEqual(json.loads(target.read_text()), {'value': 1})
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ['cache', 'report.json'])

    def test_reject_nan_before_write(self):
        target = self.folder / 'report.json'
        target.write_text('old')
        with self.assertRaises(ValueError):
            audit.write_report(target, {'value': float('nan')}, self.cache)
        self.assertEqual(target.read_text(), 'old')

    def test_parent_and_member_symlink(self):
        other = self.folder / 'other'
        other.mkdir()
        (self.folder / 'linked').symlink_to(other, target_is_directory=True)
        target = other / 'report.json'
        target.write_text('old')
        (self.folder / 'member').symlink_to(target)
        for path in (self.folder / 'linked/report.json', self.folder / 'member',
                     self.folder / 'linked/../report.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.write_report(path, {'x': 1}, self.cache)
        self.assertEqual(target.read_text(), 'old')

    def test_cache_directory_and_missing_parent(self):
        for path in (self.cache / 'report.json', self.folder / 'missing/report.json', self.folder):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.write_report(path, {'x': 1}, self.cache)

    def test_cli_preflight_does_not_run(self):
        with patch.object(audit, 'run_audit') as run, self.assertRaises(ValueError):
            audit.main(['--cache', str(self.cache), '--report', str(self.cache / 'report.json')])
        run.assert_not_called()

    def make_fixture(self):
        checkout = self.cache / 'pb_bss'
        checkout.mkdir()
        (checkout / 'pb_bss/math').mkdir(parents=True)
        files = {'LICENSE': b'MIT fixture\n', 'pb_bss/math/solve.py': b'# identity fixture only; not original numerical execution\n'}
        for relative, content in files.items():
            (checkout / relative).write_bytes(content)
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        def git(*args):
            return subprocess.check_output(['git', '-C', str(checkout), *args], env=env, stderr=subprocess.PIPE).decode().strip()
        git('init')
        git('config', 'user.email', 'fixture@example.invalid')
        git('config', 'user.name', 'Offline fixture')
        git('remote', 'add', 'origin', audit.ORIGIN)
        git('add', '.')
        git('-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture')
        revision = git('rev-parse', 'HEAD')
        lock = self.folder / 'lock.json'
        entry = {'id': 'pb_bss', 'url': audit.ORIGIN, 'revision': revision,
                 'license': 'MIT', 'entrypoints': list(files)}
        lock.write_text(json.dumps({'projects': [entry]}))
        hashes = {p: audit.digest(v) for p, v in files.items()}
        return checkout, lock, revision, hashes, git

    def test_fixture_identity_and_git_environment(self):
        checkout, lock, rev, files, _ = self.make_fixture()
        with patch.object(audit, 'LOCK', lock), patch.object(audit, 'REVISION', rev), patch.object(audit, 'FILES', files), \
                patch.dict(os.environ, {'GIT_DIR': '/nonexistent', 'GIT_WORK_TREE': '/wrong', 'GIT_CONFIG_COUNT': '1'}):
            record = audit.verify_sources(self.cache)
        self.assertEqual(record['checkout'], str(checkout))
        self.assertEqual(record['source_lock_project_count'], 1)
        self.assertTrue(record['clean'])
        self.assertEqual(len(record['sources']), 2)

    def test_dirty_fixture_rejected(self):
        checkout, lock, rev, files, _ = self.make_fixture()
        (checkout / 'extra').write_text('untracked')
        with patch.object(audit, 'LOCK', lock), patch.object(audit, 'REVISION', rev), patch.object(audit, 'FILES', files):
            with self.assertRaisesRegex(ValueError, 'not clean'):
                audit.verify_sources(self.cache)

    def test_wrong_hash_rejected(self):
        _, lock, rev, files, _ = self.make_fixture()
        files['LICENSE'] = '0' * 64
        with patch.object(audit, 'LOCK', lock), patch.object(audit, 'REVISION', rev), patch.object(audit, 'FILES', files):
            with self.assertRaisesRegex(ValueError, 'digest/blob'):
                audit.verify_sources(self.cache)

    def test_wrong_origin_rejected(self):
        _, lock, rev, files, git = self.make_fixture()
        git('remote', 'set-url', 'origin', 'https://example.invalid/wrong.git')
        with patch.object(audit, 'LOCK', lock), patch.object(audit, 'REVISION', rev), patch.object(audit, 'FILES', files):
            with self.assertRaisesRegex(ValueError, 'HEAD or origin'):
                audit.verify_sources(self.cache)

    def test_wrong_lock_and_duplicate_entry_rejected(self):
        _, lock, rev, files, _ = self.make_fixture()
        document = json.loads(lock.read_text())
        for entries in ([dict(document['projects'][0], license='unknown')], document['projects'] * 2):
            lock.write_text(json.dumps({'projects': entries}))
            with patch.object(audit, 'LOCK', lock), patch.object(audit, 'REVISION', rev), patch.object(audit, 'FILES', files):
                with self.assertRaises(ValueError):
                    audit.verify_sources(self.cache)


if __name__ == '__main__':
    unittest.main()
