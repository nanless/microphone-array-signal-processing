"""Independent fixed-Git fixtures and write-before-execution negative controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch00.upstream.fetch_upstreams import inspect_project
from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch04.examples import (
    audit_upstream_doa as doa, audit_said_compression as said,
    reproduce_sbl_reference as sbl, reproduce_doatools_esprit as esprit,
    reproduce_smpphat_reference as smp, reproduce_smpphat_portable_overlay as overlay,
)


TOOLS = (doa, said, sbl, esprit, smp, overlay)
OLD_NAMES = ('upstream_doa.json', 'said_compression.json', 'sbl_reference.json',
             'doatools_esprit_reference.json', 'smpphat_reference.json', 'smpphat_portable_overlay.json')


class WriteBoundaryTests(unittest.TestCase):
    def test_all_original_reports_and_scripts_bind_real_pre_edit_git_objects(self):
        for tool, name in zip(TOOLS, OLD_NAMES):
            report_path = contracts.ROOT / 'codes/chapters/ch04/reports' / name
            with self.subTest(report=name):
                self.assertEqual(report_path.read_bytes(), contracts.historical_bytes(report_path))
                original_script = contracts.historical_bytes(tool.__file__)
                compile(original_script, tool.__file__, 'exec')
                report = contracts.strict_json_loads(report_path.read_bytes())
                digest = hashlib.sha256(original_script).hexdigest()
                if 'audit_source_sha256' in report:
                    self.assertEqual(report['audit_source_sha256'], digest)
                elif 'provenance' in report:
                    key = 'runner_sha256' if tool is smp else 'harness_sha256'
                    self.assertEqual(report['provenance'][key], digest)

    def test_only_two_designated_current_paths_are_writable_inside_repository(self):
        for tool in (doa, said):
            self.assertEqual(contracts.report_target(tool.CURRENT_REPORT, tool.CURRENT_REPORT), tool.CURRENT_REPORT)
        for name in (*OLD_NAMES, '../../ch00/SOURCES.lock.json'):
            with self.assertRaises(ValueError):
                contracts.report_target(contracts.ROOT/'codes/chapters/ch04/reports'/name, doa.CURRENT_REPORT)
        with self.assertRaises(ValueError):
            contracts.work_target(contracts.ROOT/'new-build')
        with self.assertRaises(ValueError):
            contracts.work_target(contracts.ROOT.parent)

    def test_external_ordinary_outputs_are_valid_but_links_and_source_overlap_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'new'/'report.json'
            contracts.write_report(target, {'case': 7})
            self.assertEqual(json.loads(target.read_text()), {'case': 7})
            alias = root/'alias'
            alias.symlink_to(target.parent, target_is_directory=True)
            for bad in (alias/'report.json', root/'leaf'):
                if bad == root/'leaf':
                    bad.symlink_to(target)
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    contracts.report_target(bad)
            original = root/'source'
            original.mkdir()
            with self.assertRaises(ValueError):
                contracts.report_target(original/'source.py', protected=(original,))
            for bad in (original, root):
                with self.assertRaises(ValueError):
                    contracts.work_target(bad, (original,))

    def test_every_cli_rejects_historical_output_before_original_execution(self):
        for tool, name in zip(TOOLS, OLD_NAMES):
            option = '--report' if tool in (doa, said) else '--output'
            argv = ['tool', option, str(contracts.ROOT/'codes/chapters/ch04/reports'/name)]
            if tool is overlay:
                argv.extend(['--fftw-prefix', '/private/tmp/not-an-installed-library'])
            function = 'build_report' if tool in (doa, said) else 'run' if tool is overlay else 'run_experiment'
            with self.subTest(tool=tool.__name__), patch('sys.argv', argv), \
                    patch.object(tool, function) as execute, self.assertRaises(ValueError):
                tool.main()
            execute.assert_not_called()

    def test_work_directory_rejected_before_compilation(self):
        argv = ['tool', '--work-dir', str(contracts.ROOT/'new-build')]
        with patch('sys.argv', argv), patch.object(smp, 'run_experiment') as execute, self.assertRaises(ValueError):
            smp.main()
        execute.assert_not_called()

    def test_fftw_prefix_and_generated_case_paths_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root/'prefix'
            (prefix/'include').mkdir(parents=True)
            (prefix/'lib').mkdir()
            (prefix/'include/fftw3.h').write_text('artificial header, never compiled')
            (prefix/'lib/libfftw3f.a').write_bytes(b'artificial library, never linked')
            _, info = smp.validate_fftw_prefix(prefix)
            self.assertEqual(info['source'], 'provided_prefix')
            self.assertEqual(info['version'], 'not verified from an archive')
            alias = root/'prefix-alias'
            alias.symlink_to(prefix, target_is_directory=True)
            with self.assertRaises(ValueError):
                smp.validate_fftw_prefix(alias)
            target = root/'reference.f32'
            target.write_bytes(b'preserve these exact bytes')
            output = root/'output.f32'
            output.symlink_to(target)
            with self.assertRaises(ValueError):
                smp._write_float32(output, [1.0])
            self.assertEqual(target.read_bytes(), b'preserve these exact bytes')

    def test_fftw_cached_manifest_requires_strict_json_before_any_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root/'fftw-prefix-verified'
            (prefix/'include').mkdir(parents=True)
            (prefix/'lib').mkdir()
            (prefix/'include/fftw3.h').write_text('artificial header')
            (prefix/'lib/libfftw3f.a').write_bytes(b'artificial archive')
            (root/'fftw-prefix-verified-manifest.json').write_text('{"version":1,"version":1}')
            with self.assertRaises(ValueError):
                smp.ensure_fftw(root, download=False)


class FixedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.checkout = self.base/'fixture'
        self.checkout.mkdir()
        contracts.git(self.checkout, 'init', '-q')
        contracts.git(self.checkout, 'config', 'user.name', 'Independent Fixture')
        contracts.git(self.checkout, 'config', 'user.email', 'fixture@example.org')
        self.url = 'https://example.org/author/fixture.git'
        contracts.git(self.checkout, 'remote', 'add', 'origin', self.url)
        (self.checkout/'LICENSE').write_text('Explicit fixture license\n')
        (self.checkout/'original.py').write_text('def original(x):\n    return x + 2\n')
        contracts.git(self.checkout, 'add', 'LICENSE', 'original.py')
        contracts.git(self.checkout, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Fixed fixture')
        revision = contracts.git(self.checkout, 'rev-parse', 'HEAD')
        self.entry = {'id': 'fixture', 'url': self.url, 'revision': revision, 'license': 'MIT',
                      'fetch_enabled': True, 'entrypoints': ['LICENSE', 'original.py']}
        self.lock = self.base/'lock.json'
        self.status = self.base/'status.json'
        self.policy = (self.url, revision, 'MIT', 'LICENSE', contracts.sha(self.checkout/'LICENSE'))
        self.patchers = [patch.dict(contracts.PROJECTS, {'fixture': self.policy}),
                         patch.object(contracts, 'CACHE', self.base),
                         patch.object(contracts, 'LOCK', self.lock), patch.object(contracts, 'STATUS', self.status)]
        for patcher in self.patchers:
            patcher.start()
        self.save_documents()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def save_documents(self):
        self.lock.write_text(json.dumps({'schema_version': 1, 'projects': [self.entry]}))
        record = inspect_project(self.entry, self.base)
        self.status.write_text(json.dumps({'schema_version': 1, 'lock_sha256': contracts.sha(self.lock),
                                          'projects': [record]}))

    def verify(self):
        return contracts.verify_project('fixture', relatives=('original.py',))

    def test_true_fixed_blob_and_independent_source_bytes(self):
        identity = self.verify()
        payload = b'def original(x):\n    return x + 2\n'
        self.assertEqual(identity['used_files']['original.py']['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(identity['used_files']['original.py']['git_blob'],
                         hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest())
        self.assertTrue(contracts.check_unchanged(identity)['clean_after'])

    def test_complete_selection_mismatch_is_not_upgraded_by_valid_used_file(self):
        (self.checkout/'.git/info/sparse-checkout').write_text('/*\n!**/*.wav\n')
        self.save_documents()
        identity = self.verify()
        self.assertEqual(identity['used_source_identity'], 'verified')
        self.assertEqual(identity['live_complete_selection']['status'], 'source_selection_mismatch')
        self.assertFalse(identity['recorded_complete_selection']['source_selection_verified'])

    def test_inherited_git_route_and_configuration_cannot_redirect_identity(self):
        with patch.dict(os.environ, {'GIT_DIR': str(self.base/'absent.git'),
                                     'GIT_WORK_TREE': str(self.base),
                                     'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.worktree',
                                     'GIT_CONFIG_VALUE_0': str(self.base)}):
            self.assertEqual(self.verify()['head'], self.entry['revision'])

    def test_wrong_origin_dirty_source_and_untracked_file_are_rejected(self):
        contracts.git(self.checkout, 'remote', 'set-url', 'origin', 'https://example.org/impostor.git')
        with self.assertRaises(ValueError):
            self.verify()

        contracts.git(self.checkout, 'remote', 'set-url', 'origin', self.url)
        (self.checkout/'untracked.txt').write_text('must not be ignored')
        with self.assertRaises(ValueError):
            self.verify()
        (self.checkout/'untracked.txt').unlink()
        (self.checkout/'original.py').write_text('def original(x):\n    return x - 2\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrong_head_and_false_top_level_are_rejected(self):
        original_git = contracts.git

        def wrong_root(checkout, *arguments):
            if arguments == ('rev-parse', '--show-toplevel'):
                return str(self.base)
            return original_git(checkout, *arguments)

        with patch.object(contracts, 'git', side_effect=wrong_root), self.assertRaises(ValueError):
            self.verify()
        contracts.git(self.checkout, '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'Wrong HEAD')
        with self.assertRaises(ValueError):
            self.verify()

    def test_changed_file_and_lock_after_execution_are_rejected(self):
        identity = self.verify()
        (self.checkout/'original.py').write_text('changed source\n')
        with self.assertRaises(ValueError):
            contracts.check_unchanged(identity)
        (self.checkout/'original.py').write_text('def original(x):\n    return x + 2\n')
        self.status.write_text(self.status.read_text()+'\n')
        with self.assertRaises(ValueError):
            contracts.check_unchanged(identity)

    def test_duplicate_keys_ids_false_success_and_stale_digest_are_rejected(self):
        original = self.status.read_text()
        malformed = ('{"schema_version":1,"schema_version":1,"projects":[]}',
                     original.replace('"schema_version": 1', '"schema_version": true'),
                     original.replace(contracts.sha(self.lock), '0'*64))
        for text in malformed:
            self.status.write_text(text)
            with self.subTest(text=text[:70]), self.assertRaises(ValueError):
                self.verify()
        record = json.loads(original)
        record['projects'].append(copy.deepcopy(record['projects'][0]))
        self.status.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.verify()
        record = json.loads(original)
        record['projects'][0]['source_selection_verified'] = 1
        self.status.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.verify()

    def test_symlink_checkout_or_original_file_and_fake_license_are_rejected(self):
        alias = self.base/'checkout-alias'
        alias.symlink_to(self.checkout, target_is_directory=True)
        with self.assertRaises(ValueError):
            contracts.verify_project('fixture', alias)
        identity = self.verify()
        original = self.checkout/'original.py'
        original.unlink()
        original.symlink_to(self.checkout/'LICENSE')
        with self.assertRaises(ValueError):
            contracts.record_files(identity, ('original.py',))
        original.unlink()
        original.write_text('def original(x):\n    return x + 2\n')
        with patch.dict(contracts.PROJECTS, {'fixture': (*self.policy[:-1], '0'*64)}):
            with self.assertRaises(ValueError):
                self.verify()


class CurrentReportTests(unittest.TestCase):
    def test_new_reports_bind_current_real_dependencies_and_separate_selection(self):
        for tool in (doa, said):
            report = contracts.strict_json_loads(tool.CURRENT_REPORT.read_bytes())
            self.assertEqual(report['audit_source_sha256'], contracts.sha(tool.__file__))
            self.assertEqual(report['lock_sha256'], contracts.sha(contracts.LOCK))
            self.assertEqual(report['report_source_sha256'], contracts.dependencies(tool.__file__))
            identities = (report['source_contracts'].values() if tool is doa
                          else (report['source_contract'],))
            for identity in identities:
                self.assertEqual(identity['status_sha256'], contracts.sha(contracts.STATUS))
                self.assertEqual(identity['used_source_identity'], 'verified')
                self.assertTrue(identity['clean_before'])
                self.assertTrue(identity['clean_after'])
                self.assertEqual(identity['ignored_members_before'], identity['ignored_members_after'])
                self.assertEqual(identity['origin'], contracts.PROJECTS[identity['project']][0])
                expected = 'source_selection_mismatch' if identity['project'] == 'doatools' else 'source_verified'
                self.assertEqual(identity['live_complete_selection']['status'], expected)
                self.assertEqual(identity['recorded_complete_selection']['status'], expected)
                for record in identity['used_files'].values():
                    self.assertEqual(record['actual_blob'], record['git_blob'])
                    self.assertEqual(len(record['sha256']), 64)

    def test_new_reports_keep_original_failures_and_exact_artificial_scope(self):
        report = contracts.strict_json_loads(doa.CURRENT_REPORT.read_bytes())
        self.assertEqual(report['results']['tops']['original_peak_deg'], 49)
        self.assertEqual(report['results']['tops']['independent_peak_deg'], 30)
        self.assertEqual(report['results']['cssm_filtered_mapping']['original_diagonal'], [4+9, 2, 2])
        self.assertEqual(report['results']['root_music_original']['exception_type'], 'AttributeError')
        self.assertEqual(report['results']['ld_product_underflow']['classification'], 'positive_infinity')
        self.assertIsNone(report['results']['ld_product_underflow']['value'])
        self.assertIn('original _process not run', report['scope']['CSSM_WAVES'])
        report = contracts.strict_json_loads(said.CURRENT_REPORT.read_bytes())
        self.assertEqual(report['results']['high_confidence_output']['segmentation'], [[[0,0,1],[2,0,.5]]])
        self.assertFalse(report['results']['failures']['one_byte_limit']['published_output_exists'])
        self.assertIn('Torch/model execution', report['scope']['claims_excluded'])


if __name__ == '__main__':
    unittest.main()
