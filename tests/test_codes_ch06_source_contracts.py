"""Bounded AEC source/report/output contracts; no network or production binaries."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch06.examples import (
    audit_aec_upstream_interfaces as interfaces, audit_upstream_apa as apa,
    aec3_offline_compare as aec3, aec_real_pair_experiment as pair,
    aec_same_input_truth as truth,
)

TOOLS = (interfaces, apa)
OLD_NAMES = ('aec_upstream_interfaces.json', 'upstream_apa.json')
REPORTS = contracts.ROOT/'codes/chapters/ch06/reports'
HISTORICAL_REVISION = '621d727a63475e3ee8ad2a29d518889c42e92571'


def old_bytes(path):
    spec = HISTORICAL_REVISION+':'+str(Path(path).resolve().relative_to(contracts.ROOT))
    payload = (contracts.git(contracts.ROOT, 'show', spec)+'\n').encode()
    if len(payload) != int(contracts.git(contracts.ROOT, 'cat-file', '-s', spec)):
        raise ValueError('Historical object is not byte-exact')
    return payload


class AECSourceContracts(unittest.TestCase):
    def test_original_reports_bind_original_tool_bytes(self):
        for tool, name, field in zip(TOOLS, OLD_NAMES, ('script_sha256', 'audit_source_sha256')):
            path = REPORTS/name
            self.assertEqual(path.read_bytes(), old_bytes(path))
            original = old_bytes(tool.__file__)
            compile(original, tool.__file__, 'exec')
            self.assertEqual(contracts.strict_json_loads(path.read_bytes())[field],
                             hashlib.sha256(original).hexdigest())

    def test_forbidden_report_paths_stop_before_original_execution(self):
        forbidden = [*(REPORTS/n for n in OLD_NAMES), contracts.LOCK, contracts.STATUS,
                     contracts.CACHE/'pyaec/report.json', Path(interfaces.__file__),
                     contracts.ROOT/'codes/chapters/ch00/source_snapshots/new.json']
        for tool in TOOLS:
            for target in [*forbidden, *(t.CURRENT_REPORT for t in TOOLS if t is not tool)]:
                with self.subTest(tool=tool.__name__, target=target), \
                        patch('sys.argv', ['audit', '--report', str(target)]), \
                        patch.object(tool, 'run_audit') as execute, self.assertRaises(ValueError):
                    tool.main()
                execute.assert_not_called()

    def test_symlink_and_external_source_overlap_stop_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/'source'
            source.mkdir()
            original = source/'original'
            original.write_bytes(b'preserve')
            (base/'alias').symlink_to(source, target_is_directory=True)
            (base/'leaf').symlink_to(original)
            for tool in TOOLS:
                for target in (base/'alias/new.json', base/'leaf'):
                    with patch('sys.argv', ['audit', '--report', str(target)]), \
                            patch.object(tool, 'run_audit') as execute, self.assertRaises(ValueError):
                        tool.main()
                    execute.assert_not_called()
            with patch('sys.argv', ['audit', '--download-root', str(source), '--report', str(source/'new.json')]), \
                    patch.object(interfaces, 'run_audit') as execute, self.assertRaises(ValueError):
                interfaces.main()
            execute.assert_not_called()
            self.assertEqual(original.read_bytes(), b'preserve')

    def test_default_stdout_does_not_rewrite_history(self):
        snapshots = {REPORTS/n: ((REPORTS/n).read_bytes(), (REPORTS/n).stat().st_mtime_ns) for n in OLD_NAMES}
        fixture = {'status': 'expected_behaviors_verified_warnings_preserved', 'fixture_only': True}
        for tool in TOOLS:
            with patch('sys.argv', ['audit']), patch.object(tool, 'run_audit', return_value=fixture), \
                    patch.object(contracts, 'write_report') as write, contextlib.redirect_stdout(io.StringIO()) as out:
                tool.main()
                write.assert_not_called()
                self.assertEqual(contracts.strict_json_loads(out.getvalue()), fixture)
        for path, (payload, mtime) in snapshots.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (payload, mtime))

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

    def test_current_reports_bind_actual_dependencies_and_preserve_scope(self):
        for tool in TOOLS:
            report = contracts.strict_json_loads(tool.CURRENT_REPORT.read_bytes())
            self.assertEqual(report['actual_dependency_sha256'], contracts.dependencies(Path(tool.__file__)))
            self.assertEqual(report['schema_version'], 2)
            identities = report.get('source_identities') or {'pyaec': report['source_identity']}
            for identity in identities.values():
                self.assert_identity(identity)
        report = contracts.strict_json_loads(interfaces.CURRENT_REPORT.read_bytes())
        for name in ('fdkf', 'pfdkf'):
            self.assertEqual(report['results']['pyaec'][name]['status'], 'numpy_complex_removed')
            self.assertFalse(report['results']['pyaec'][name]['output_computed'])
        self.assertFalse(report['results']['dtln']['real_neural_inference'])
        self.assertFalse(report['results']['dtln']['real_audio_write'])
        self.assertEqual([r['samples_outside_unit_range'] for r in report['results']['dtln']['rows']], [640, 0])
        report = contracts.strict_json_loads(apa.CURRENT_REPORT.read_bytes())
        self.assertEqual(report['case_count'], 4)
        self.assertTrue(all(row['expected_behavior_verified'] for row in report['results'].values()))
        self.assertEqual({w['category'] for w in report['results']['complex_discard']['warnings']}, {'ComplexWarning'})


class ExternalOutputContracts(unittest.TestCase):
    def test_aec3_all_existing_members_and_links_fail_before_loader_or_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binary = base/'binary'
            binary.write_bytes(b'fixture, never executed')
            binary.chmod(0o700)
            out = base/'out'
            out.mkdir()
            original = out/'doubletalk_aec3_linear.wav'
            original.write_bytes(b'preserve output')
            for target in (out, contracts.CACHE/'new-test-output', contracts.ROOT/'new-test-output'):
                with patch.object(aec3, 'load_pinned_pair') as load, patch.object(aec3.subprocess, 'run') as execute, self.assertRaises((ValueError, FileExistsError)):
                    aec3.run(binary, target)
                load.assert_not_called()
                execute.assert_not_called()
            alias = base/'alias'
            alias.symlink_to(out, target_is_directory=True)
            with self.assertRaises(ValueError):
                aec3.preflight_members(alias, ['new.wav'])
            self.assertEqual(original.read_bytes(), b'preserve output')

    def test_speex_optional_output_is_checked_before_computation(self):
        for target in (contracts.CACHE/'output.wav', contracts.LOCK, Path(pair.__file__)):
            with patch('sys.argv', ['pair', '--speex-library', '/private/tmp/absent', '--output-wav', str(target)]), \
                    patch.object(pair, 'run') as execute, self.assertRaises(ValueError):
                pair.main()
            execute.assert_not_called()

    def test_exclusive_wav_write_and_original_input_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/'input'
            source.mkdir()
            with self.assertRaises(ValueError):
                pair.output_destination(source/'x.wav', protected=(source,))
            output = base/'new.wav'
            pcm = np.array([100, -100], dtype=np.int16)
            pair.write_output_wav(output, pcm)
            np.testing.assert_array_equal(aec3.read_wav(output, 2), pcm)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                pair.write_output_wav(output, pcm)
            alias = base/'alias.wav'
            alias.symlink_to(output)
            with self.assertRaises(ValueError):
                pair.write_output_wav(alias, pcm)
            self.assertEqual(output.read_bytes(), original)

    def test_shared_input_aec3_preserves_existing_final_before_external_call(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            out = base/'out'
            out.mkdir()
            final = out/'base_final.wav'
            final.write_bytes(b'preserve')
            with patch.object(truth.subprocess, 'run') as execute, self.assertRaises(FileExistsError):
                truth._run_aec3(base/'binary', np.zeros(160, dtype=np.int16), np.zeros(160, dtype=np.int16), out, 'base')
            execute.assert_not_called()
            self.assertEqual(final.read_bytes(), b'preserve')

    def test_build_manifest_rejects_nonfinite_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'build.json'
            for payload in ('{"schema_version":1,"schema_version":1}', '{"unused":NaN}', '{"unused":1e999}'):
                path.write_text(payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    truth._load_build_manifest(path)


if __name__ == '__main__':
    unittest.main()
