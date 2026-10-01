"""Offline TAC source identity, static evidence and safe report boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.appendix_b.examples import audit_tac_contracts as audit


class TACStructureTests(unittest.TestCase):
    @unittest.skipUnless((audit.CACHE / 'tac/utility/models.py').is_file(),
                         'optional fixed author TAC source absent; no network acquisition')
    def test_actual_static_source_and_scope(self):
        report = audit.run_audit()
        self.assertEqual(report['status'], 'passed_static_contracts')
        self.assertEqual([r['name'] for r in report['contracts']], [
            'shared_three_layer_groups', 'shared_layer_application', 'fixed_channel_mean',
            'valid_prefix_mean', 'concatenation', 'residual', 'noncausal_intra_and_group_norm',
            'single_stage_tac_module', 'reference_channel_zero', 'final_valid_channel_mean'])
        self.assertEqual(report['counts'], {'static_contracts': 10, 'original_runtime_calls': 0})
        self.assertEqual(report['execution']['original_functions_called'], [])
        self.assertEqual(report['execution']['original_modules_imported'], [])
        self.assertFalse(report['execution']['ast_extraction_for_execution'])
        self.assertFalse(report['environment']['torch_imported_by_audit'])
        self.assertTrue(report['before_clean'] and report['after_clean'])
        self.assertEqual(report['tool_sha256'], hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        self.assertEqual(len(report['sources']), 4)
        self.assertFalse(report['license']['upstream_code_redistributed'])
        self.assertFalse(report['license']['weights_or_audio_obtained'])
        for row in report['contracts']:
            self.assertEqual(row['evidence_level'], 'static original AST only')
            self.assertLessEqual(row['line_start'], row['line_end'])
            self.assertEqual(len(row['method_source_sha256']), 64)

    def test_ast_not_text_or_comment_match(self):
        import ast
        source = '# output + ch_output\nvalue = output - ch_output\n'
        self.assertFalse(audit.contains(ast.parse(source), 'output + ch_output'))
        self.assertTrue(audit.contains(ast.parse(source), 'output - ch_output'))
        self.assertTrue(audit.contains(ast.parse('x = output+(ch_output)'), 'output + ch_output'))

    @unittest.skipUnless((audit.CACHE / 'tac/utility/models.py').is_file(),
                         'optional fixed author TAC source absent; no network acquisition')
    def test_targeted_ast_mutations_rejected_in_memory(self):
        models = (audit.CACHE / 'tac/utility/models.py').read_text()
        fasnet = (audit.CACHE / 'tac/FaSNet.py').read_text()
        mutations = [
            (models.replace('ch_output.mean(2)', 'ch_output.mean(3)'), fasnet, 'fixed_channel_mean'),
            (models.replace('torch.cat([ch_output, ch_mean], 2)', 'torch.cat([ch_output, ch_mean], 1)'), fasnet, 'concatenation'),
            (models.replace('output + ch_output', 'output - ch_output'), fasnet, 'residual'),
            (models, fasnet.replace('ref_seg = all_seg[:,0]', 'ref_seg = all_seg[:,1]'), 'reference_channel_zero'),
        ]
        for model_text, fa_text, name in mutations:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                audit.structure_contracts(model_text, fa_text)

    def test_missing_or_duplicate_method(self):
        import ast
        for code in ('class Model: pass', 'class Model:\n def forward(self): pass\n def forward(self): pass'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                audit.method(ast.parse(code), 'Model', 'forward')

    def test_json_invalid_and_near_valid(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.strict_loads(text)
        self.assertEqual(audit.strict_loads('{"x":1e-99,"X":2,"nested":{"x":3}}')['X'], 2)


class TACSourceAndPathTests(unittest.TestCase):
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
        audit.write_report(target, {'static': True}, self.cache)
        self.assertEqual(json.loads(target.read_text()), {'static': True})
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ['cache', 'report.json'])

    def test_reject_nonfinite_before_write(self):
        target = self.folder / 'report.json'
        target.write_text('old')
        with self.assertRaises(ValueError):
            audit.write_report(target, {'x': float('nan')}, self.cache)
        self.assertEqual(target.read_text(), 'old')

    def test_parent_member_and_lexical_symlink(self):
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

    def test_cli_preflight_rejects_before_audit(self):
        with patch.object(audit, 'run_audit') as run, self.assertRaises(ValueError):
            audit.main(['--cache', str(self.cache), '--report', str(self.cache / 'report.json')])
        run.assert_not_called()

    def make_fixture(self):
        checkout = self.cache / 'tac'
        checkout.mkdir()
        (checkout / 'utility').mkdir()
        files = {p: b'# Synthetic identity fixture, not author TAC implementation\n' for p in audit.FILES}
        files['README.md'] = (b'creativecommons.org/licenses/by-nc-sa/3.0/us/\n'
                              b'Attribution-NonCommercial-ShareAlike 3.0 United States License\n')
        for relative, content in files.items():
            (checkout / relative).write_bytes(content)
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        def git(*args):
            return subprocess.check_output(['git', '-C', str(checkout), *args], env=env,
                                           stderr=subprocess.PIPE).decode().strip()
        git('init')
        git('config', 'user.email', 'fixture@example.invalid')
        git('config', 'user.name', 'Offline fixture')
        git('remote', 'add', 'origin', audit.ORIGIN)
        git('add', '.')
        git('-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture')
        revision = git('rev-parse', 'HEAD')
        lock = self.folder / 'lock.json'
        entry = {'id': 'tac', 'url': audit.ORIGIN, 'revision': revision, 'license': audit.LICENSE,
                 'source_paths': list(files), 'entrypoints': ['README.md', 'utility/models.py', 'FaSNet.py']}
        lock.write_text(json.dumps({'projects': [entry]}))
        hashes = {p: audit.digest(v) for p, v in files.items()}
        return checkout, lock, revision, hashes, git

    def fixture_context(self, lock, revision, files):
        return patch.multiple(audit, LOCK=lock, REVISION=revision, FILES=files)

    def test_fixture_identity_and_git_environment(self):
        checkout, lock, revision, files, _ = self.make_fixture()
        with self.fixture_context(lock, revision, files), patch.dict(os.environ, {
                'GIT_DIR': '/nonexistent', 'GIT_WORK_TREE': '/wrong', 'GIT_CONFIG_COUNT': '1'}):
            result = audit.verify_sources(self.cache)
        self.assertEqual(result['checkout'], str(checkout))
        self.assertEqual(len(result['sources']), 4)
        self.assertTrue(result['clean'])

    def test_dirty_source_rejected(self):
        checkout, lock, revision, files, _ = self.make_fixture()
        (checkout / 'extra').write_text('untracked')
        with self.fixture_context(lock, revision, files), self.assertRaisesRegex(ValueError, 'not clean'):
            audit.verify_sources(self.cache)

    def test_wrong_hash_rejected(self):
        _, lock, revision, files, _ = self.make_fixture()
        files['README.md'] = '0' * 64
        with self.fixture_context(lock, revision, files), self.assertRaisesRegex(ValueError, 'digest/blob'):
            audit.verify_sources(self.cache)

    def test_wrong_origin_rejected(self):
        _, lock, revision, files, git = self.make_fixture()
        git('remote', 'set-url', 'origin', 'https://example.invalid/wrong.git')
        with self.fixture_context(lock, revision, files), self.assertRaisesRegex(ValueError, 'HEAD or origin'):
            audit.verify_sources(self.cache)

    def test_lock_identity_and_duplicate_rejected(self):
        _, lock, revision, files, _ = self.make_fixture()
        doc = json.loads(lock.read_text())
        variants = ([dict(doc['projects'][0], license='unknown')], doc['projects'] * 2,
                    [dict(doc['projects'][0], source_paths=['README.md'])])
        for entries in variants:
            lock.write_text(json.dumps({'projects': entries}))
            with self.subTest(entries=entries), self.fixture_context(lock, revision, files), self.assertRaises(ValueError):
                audit.verify_sources(self.cache)

    def test_clean_but_extra_selected_file_rejected(self):
        checkout, lock, _, files, git = self.make_fixture()
        (checkout / 'extra.py').write_text('# not in selected scope')
        git('add', 'extra.py')
        git('-c', 'commit.gpgsign=false', 'commit', '-m', 'extra')
        revision = git('rev-parse', 'HEAD')
        doc = json.loads(lock.read_text())
        doc['projects'][0]['revision'] = revision
        lock.write_text(json.dumps(doc))
        with self.fixture_context(lock, revision, files), self.assertRaisesRegex(ValueError, 'four literal'):
            audit.verify_sources(self.cache)


if __name__ == '__main__':
    unittest.main()
