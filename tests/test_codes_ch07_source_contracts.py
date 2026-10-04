"""WPE source and report boundaries; history is bound to actual original Git bytes."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch07.examples import (
    audit_upstream_wpe_contracts as audit, audit_wpe_upstream_interfaces as interfaces,
    wpe_silence_boundary as silence, compare_online_wpe_reference as online,
)

HISTORICAL_REVISION = '34d494dd8074169a92f79bf90b9ff2bbcf25220f'
TOOLS = (audit, interfaces, silence)
REPORTS = contracts.ROOT/'codes/chapters/ch07/reports'
OLD_NAMES = ('upstream_wpe_contracts.json', 'wpe_upstream_interfaces.json',
             'chapter07_online_wpe_silence.json')


def old_bytes(path):
    spec = HISTORICAL_REVISION+':'+str(Path(path).resolve().relative_to(contracts.ROOT))
    payload = (contracts.git(contracts.ROOT, 'show', spec)+'\n').encode()
    if len(payload) != int(contracts.git(contracts.ROOT, 'cat-file', '-s', spec)):
        raise ValueError('Historical source must be byte-exact')
    return payload


class HistoricalBoundaries(unittest.TestCase):
    def test_original_reports_and_tool_identities_are_preserved(self):
        for name in OLD_NAMES:
            path = REPORTS/name
            self.assertEqual(path.read_bytes(), old_bytes(path))
        for tool, name, field in zip(TOOLS[:2], OLD_NAMES[:2], ('audit_source_sha256', 'harness_sha256')):
            source = old_bytes(tool.__file__)
            compile(source, tool.__file__, 'exec')
            report = contracts.strict_json_loads((REPORTS/name).read_bytes())
            self.assertEqual(report[field], hashlib.sha256(source).hexdigest())
        report = contracts.strict_json_loads((REPORTS/OLD_NAMES[2]).read_bytes())
        for path, digest in report['generator_inputs'].items():
            self.assertEqual(hashlib.sha256(old_bytes(contracts.ROOT/path)).hexdigest(), digest)

    def test_forbidden_report_destinations_fail_before_any_calculation(self):
        targets = [*(REPORTS/n for n in OLD_NAMES), contracts.LOCK, contracts.STATUS,
                   Path(audit.__file__), contracts.CACHE/'nara_wpe/new-report.json',
                   contracts.ROOT/'codes/chapters/ch00/source_snapshots/new.json']
        for tool in TOOLS:
            operation = 'run_report' if tool is silence else 'run_audit'
            for target in [*targets, *(t.CURRENT_REPORT for t in TOOLS if t is not tool)]:
                with self.subTest(tool=tool.__name__, target=target), \
                        patch('sys.argv', ['audit', '--report', str(target)]), \
                        patch.object(tool, operation) as calculate, self.assertRaises(ValueError):
                    tool.main()
                calculate.assert_not_called()

    def test_symlinks_and_custom_source_overlap_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/'source'
            source.mkdir()
            original = source/'original'
            original.write_bytes(b'preserve')
            (base/'alias').symlink_to(source, target_is_directory=True)
            (base/'leaf').symlink_to(original)
            for tool in TOOLS:
                operation = 'run_report' if tool is silence else 'run_audit'
                for target in (base/'alias/new.json', base/'leaf'):
                    with patch('sys.argv', ['audit', '--report', str(target)]), \
                            patch.object(tool, operation) as calculate, self.assertRaises(ValueError):
                        tool.main()
                    calculate.assert_not_called()
            with patch('sys.argv', ['audit', '--source-root', str(source), '--report', str(source/'new.json')]), \
                    patch.object(interfaces, 'run_audit') as calculate, self.assertRaises(ValueError):
                interfaces.main()
            calculate.assert_not_called()
            self.assertEqual(original.read_bytes(), b'preserve')

    def test_silence_external_installed_source_is_protected_before_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'installed_original'
            source.mkdir()
            target = source/'wpe.py'
            target.write_bytes(b'preserve installed source')
            with patch.object(silence, 'installed_source_roots', return_value=(source,)), \
                    patch('sys.argv', ['silence', '--upstream', '--report', str(target)]), \
                    patch.object(silence, 'run_report') as calculate, self.assertRaises(ValueError):
                silence.main()
            calculate.assert_not_called()
            self.assertEqual(target.read_bytes(), b'preserve installed source')

    def test_stdout_and_explicit_external_reports_preserve_history(self):
        original = {REPORTS/n: ((REPORTS/n).read_bytes(), (REPORTS/n).stat().st_mtime_ns) for n in OLD_NAMES}
        fixture = {'status': 'verified_fixture_only'}
        for tool in TOOLS:
            operation = 'run_report' if tool is silence else 'run_audit'
            with patch('sys.argv', ['audit']), patch.object(tool, operation, return_value=fixture), \
                    patch.object(contracts, 'write_report') as write, contextlib.redirect_stdout(io.StringIO()) as out:
                tool.main()
                write.assert_not_called()
                self.assertEqual(contracts.strict_json_loads(out.getvalue()), fixture)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)/'ordinary/report.json'
                with patch('sys.argv', ['audit', '--report', str(target)]), patch.object(tool, operation, return_value=fixture):
                    tool.main()
                self.assertEqual(contracts.strict_json_loads(target.read_bytes()), fixture)
        for path, data in original.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), data)

    def test_json_document_duplicates_nonfinite_and_wrong_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, status = Path(directory)/'lock.json', Path(directory)/'status.json'
            status.write_text('{"schema_version":1,"projects":[]}')
            for payload in ('{"schema_version":1,"schema_version":1,"projects":[]}',
                            '{"schema_version":true,"projects":[]}',
                            '{"schema_version":1,"projects":[],"x":NaN}',
                            '{"schema_version":1,"projects":[],"x":1e999}'):
                lock.write_text(payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    contracts.source_documents(lock, status)


class OriginalLoadingBoundaries(unittest.TestCase):
    def test_installed_source_is_rejected_before_module_execution(self):
        original_root = contracts.CACHE/'nara_wpe'
        if not (original_root/'LICENSE').exists():
            self.skipTest('Optional fixed NARA source is absent; no download')
        identity = {'used_files': {
            'nara_wpe/wpe.py': {'sha256': online.NARA_WPE_MODULE_SHA256},
            'nara_wpe/__init__.py': {'sha256': online.NARA_INIT_SHA256}}}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base/'nara_wpe'
            package.mkdir()
            info = base/'nara_wpe-0.0.11.dist-info'
            info.mkdir()
            paths = {'wpe.py': original_root/'nara_wpe/wpe.py',
                     '__init__.py': original_root/'nara_wpe/__init__.py'}
            for name, path in paths.items():
                (package/name).write_bytes(path.read_bytes())
            (info/'LICENSE').write_bytes((original_root/'LICENSE').read_bytes())
            distribution = SimpleNamespace(version='0.0.11', files=[Path('nara_wpe-0.0.11.dist-info/LICENSE')],
                                           locate_file=lambda path: base/path)
            spec = SimpleNamespace(submodule_search_locations=[str(package)])
            for target in (package/'wpe.py', package/'__init__.py', info/'LICENSE'):
                content = target.read_bytes()
                target.write_text('raise RuntimeError("must never execute altered original")\n')
                with self.subTest(target=target), patch.object(contracts, 'verify_project', return_value=identity), \
                        patch.object(online.importlib.metadata, 'distribution', return_value=distribution), \
                        patch.object(online.importlib.util, 'find_spec', return_value=spec), \
                        patch.object(online.importlib.util, 'spec_from_file_location') as load, \
                        self.assertRaisesRegex(RuntimeError, 'Installed'):
                    online._load_locked_module()
                load.assert_not_called()
                target.write_bytes(content)

    @unittest.skipUnless((contracts.CACHE/'nara_wpe/.git').exists(), 'Fixed source checkout required')
    def test_wrong_official_origin_is_rejected_before_original_loader(self):
        original_git = contracts.git
        def changed_git(directory, *args):
            return 'https://invalid.example/nara.git' if args == ('remote', 'get-url', 'origin') else original_git(directory, *args)
        with patch.object(contracts, 'git', side_effect=changed_git), \
                patch.object(online.importlib.util, 'spec_from_file_location') as load, \
                self.assertRaisesRegex(ValueError, 'origin or HEAD'):
            online._load_locked_module()
        load.assert_not_called()

    @unittest.skipUnless((contracts.CACHE/'nara_wpe/.git').exists(), 'Fixed source checkout required')
    def test_post_call_origin_and_file_changes_are_not_accepted(self):
        identity = contracts.verify_project('nara_wpe', relatives=['nara_wpe/wpe.py'])
        original_git = contracts.git
        def changed_git(directory, *args):
            return 'https://invalid.example/nara.git' if args == ('remote', 'get-url', 'origin') else original_git(directory, *args)
        with patch.object(contracts, 'git', side_effect=changed_git), self.assertRaisesRegex(ValueError, 'changed during'):
            contracts.check_unchanged(identity)
        original_sha = contracts.sha
        with patch.object(contracts, 'sha', side_effect=lambda path: '0'*64 if Path(path) == contracts.LOCK else original_sha(path)), \
                self.assertRaisesRegex(ValueError, 'lock or recorded selection changed'):
            contracts.check_unchanged(identity)

    def test_installed_bytes_changed_after_original_calls_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wpe.py'
            path.write_bytes(b'original fixed file')
            installed = {'files': {'wpe.py': {'path': str(path), 'sha256': contracts.sha(path)}}, 'clean_after': False}
            path.write_bytes(b'changed after original call')
            with patch.object(contracts, 'check_unchanged'), self.assertRaisesRegex(RuntimeError, 'changed during execution'):
                online._check_locked_module({}, installed)
            self.assertFalse(installed['clean_after'])

    @unittest.skipUnless((contracts.CACHE/'nara_wpe/.git').exists(), 'Fixed source checkout required')
    def test_changed_original_blob_after_execution_is_rejected(self):
        identity = contracts.verify_project('nara_wpe', relatives=['nara_wpe/wpe.py'])
        original_git = contracts.git
        def changed_git(directory, *args):
            return '0'*40 if args and args[0] == 'hash-object' else original_git(directory, *args)
        with patch.object(contracts, 'git', side_effect=changed_git), self.assertRaisesRegex(ValueError, 'differs from fixed Git blob'):
            contracts.check_unchanged(identity)


@unittest.skipUnless(all(tool.CURRENT_REPORT.exists() for tool in TOOLS), 'Current reports await actual frozen-source execution')
class CurrentReports(unittest.TestCase):
    def assert_identity(self, identity):
        lock, status, lock_sha, status_sha = contracts.source_documents()
        project = identity['project']
        self.assertEqual(identity['lock_sha256'], lock_sha)
        self.assertEqual(identity['status_sha256'], status_sha)
        self.assertEqual(identity['lock_entry'], next(p for p in lock['projects'] if p['id'] == project))
        self.assertEqual(identity['recorded_complete_selection'], next(p for p in status['projects'] if p['id'] == project))
        self.assertEqual(identity['live_complete_selection']['status'], identity['recorded_complete_selection']['status'])
        self.assertEqual(identity['used_source_identity'], 'verified')
        self.assertTrue(identity['clean_before'] and identity['clean_after'])
        self.assertEqual(identity['ignored_members_before'], identity['ignored_members_after'])
        url, revision, license_name, license_path, license_sha = contracts.PROJECTS[project]
        self.assertEqual((identity['origin'], identity['head'], identity['lock_entry']['license']), (url, revision, license_name))
        self.assertEqual(identity['used_files'][license_path]['sha256'], license_sha)
        for relative, metadata in identity['used_files'].items():
            payload = contracts.ordinary_file(Path(identity['checkout'])/relative).read_bytes()
            self.assertEqual(metadata['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(metadata['bytes'], len(payload))
            self.assertEqual(metadata['git_blob'], hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest())
            self.assertEqual(metadata['actual_blob'], metadata['git_blob'])

    def test_current_audits_bind_dependencies_and_separate_selection(self):
        for tool in TOOLS[:2]:
            report = contracts.strict_json_loads(tool.CURRENT_REPORT.read_bytes())
            self.assertEqual(report['schema_version'], 2)
            self.assertEqual(report['actual_dependency_sha256'], contracts.dependencies(Path(tool.__file__)))
            for identity in report['source_identities'].values():
                self.assert_identity(identity)
        report = contracts.strict_json_loads(interfaces.CURRENT_REPORT.read_bytes())
        espnet = report['source_identities']['espnet']
        self.assertEqual(espnet['live_complete_selection']['status'], 'source_selection_mismatch')
        self.assertFalse(espnet['live_complete_selection']['source_selection_verified'])
        self.assertEqual(report['source_identities']['metaaf']['used_files']['zoo/LICENSE']['sha256'],
                         interfaces.SOURCES['metaaf']['files']['zoo/LICENSE'])
        self.assertFalse(report['espnet']['framework_or_network_executed'])
        self.assertFalse(report['metaaf_nara_wrapper']['jax_or_metaaf_package_executed'])
        self.assertFalse(report['metaaf_nara_wrapper']['real_stft_or_istft_executed'])
        self.assertEqual(report['nara']['get_power_context_0_2']['exception_type'], 'ValueError')
        report = contracts.strict_json_loads(audit.CURRENT_REPORT.read_bytes())
        self.assertEqual(len(report['nemo']['facts']), 12)
        self.assertFalse(report['nemo']['framework_executed'])

    def test_current_silence_preserves_original_failure_and_installed_identity(self):
        report = contracts.strict_json_loads(silence.CURRENT_REPORT.read_bytes())
        expected = contracts.dependencies(Path(silence.__file__),
            (Path(online.__file__), contracts.ROOT/'codes/chapters/ch02/core/conventions.py'))
        self.assertEqual(report['actual_dependency_sha256'], expected)
        self.assertEqual(len(report['records']), 4)
        for record in report['records']:
            self.assertEqual(record['first_nonfinite_state_frame'], 1024 if record['alpha'] == .5 else 13838)
            self.assertEqual(record['first_nonfinite_output_resume_frame'], 2)
            if record['implementation'] == 'upstream':
                self.assert_identity(record['source_identity'])
                self.assertTrue(record['installed_identity']['clean_after'])
                for row in record['installed_identity']['files'].values():
                    self.assertEqual(contracts.sha(row['path']), row['sha256'])
            else:
                self.assertIsNone(record['source_identity'])


if __name__ == '__main__':
    unittest.main()
