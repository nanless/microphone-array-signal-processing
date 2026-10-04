"""Chapter-5 write-before-execution controls and independently read report contracts.

These tests never execute the installed package, MATLAB or firmware. Shared Git,
strict-JSON and selection negative fixtures live in the chapter-4 contract tests.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch05.examples import (
    audit_upstream_beamformers as upstream,
    audit_beamformer_reference as reference,
    audit_sof_tdfb_design as sof,
)

TOOLS = (upstream, reference, sof)
EXECUTORS = ('build_report', 'run_experiment', 'run_audit')
OLD_NAMES = ('upstream_beamformers.json', 'beamformer_reference_audit.json', 'sof_tdfb_design_audit.json')
REPORTS = contracts.ROOT / 'codes/chapters/ch05/reports'
HISTORICAL_REVISION = '747ec3fec96ef20c7c128cc4291fd7bb9ee36c34'


def old_bytes(path):
    spec = HISTORICAL_REVISION + ':' + str(Path(path).resolve().relative_to(contracts.ROOT))
    payload = (contracts.git(contracts.ROOT, 'show', spec) + '\n').encode()
    if len(payload) != int(contracts.git(contracts.ROOT, 'cat-file', '-s', spec)):
        raise ValueError('Historical Git object is not byte-exact')
    return payload


class WriteBoundaryTests(unittest.TestCase):
    def test_original_reports_are_exact_original_git_bytes(self):
        for tool, name in zip(TOOLS, OLD_NAMES):
            with self.subTest(report=name):
                path = REPORTS / name
                self.assertEqual(path.read_bytes(), old_bytes(path))
                original = old_bytes(tool.__file__)
                compile(original, str(tool.__file__), 'exec')
                report = contracts.strict_json_loads(path.read_bytes())
                digest = report.get('audit_source_sha256') or report['provenance']['harness_sha256']
                self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_all_forbidden_targets_are_rejected_before_original_call(self):
        forbidden = [*(REPORTS / n for n in OLD_NAMES), contracts.LOCK, contracts.STATUS,
                     contracts.ROOT/'codes/chapters/ch05/examples/audit_sof_tdfb_design.py',
                     contracts.ROOT/'codes/chapters/ch00/source_snapshots/unauthorized.json',
                     contracts.CACHE/'sof/new-report.json']
        for tool, execute in zip(TOOLS, EXECUTORS):
            for target in [*forbidden, *(other.CURRENT_REPORT for other in TOOLS if other is not tool)]:
                with self.subTest(tool=tool.__name__, target=target), \
                        patch('sys.argv', ['audit', '--report', str(target)]), \
                        patch.object(tool, execute) as call, self.assertRaises(ValueError):
                    tool.main()
                call.assert_not_called()

    def test_symlink_output_and_parent_rejected_before_original_call(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            ordinary = base/'ordinary'
            ordinary.mkdir()
            original = ordinary/'result.json'
            original.write_bytes(b'preserve original bytes')
            (base/'alias').symlink_to(ordinary, target_is_directory=True)
            (base/'leaf').symlink_to(original)
            for tool, execute in zip(TOOLS, EXECUTORS):
                for target in (base/'alias/result.json', base/'leaf'):
                    with self.subTest(tool=tool.__name__, target=target), \
                            patch('sys.argv', ['audit', '--report', str(target)]), \
                            patch.object(tool, execute) as call, self.assertRaises(ValueError):
                        tool.main()
                    call.assert_not_called()
            self.assertEqual(original.read_bytes(), b'preserve original bytes')

    def test_external_sof_source_is_protected_before_static_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'original'
            source.mkdir()
            with patch('sys.argv', ['audit', '--source-dir', str(source), '--report', str(source/'report.json')]), \
                    patch.object(sof, 'run_audit') as call, self.assertRaises(ValueError):
                sof.main()
            call.assert_not_called()

    def test_default_stdout_and_explicit_external_write_are_separate(self):
        result = {'status': 'diagnostic_fixture', 'pb_bss': {'status': 'diagnostic_fixture'},
                  'pyroomacoustics': {'status': 'diagnostic_fixture'}}
        snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in REPORTS.glob('*.json')}
        for tool, execute in zip(TOOLS, EXECUTORS):
            with self.subTest(tool=tool.__name__), patch('sys.argv', ['audit']), \
                    patch.object(tool, execute, return_value=result), \
                    patch.object(contracts, 'write_report') as write, contextlib.redirect_stdout(io.StringIO()) as out:
                tool.main()
                write.assert_not_called()
                self.assertEqual(json.loads(out.getvalue()), result)
        with tempfile.TemporaryDirectory() as directory:
            for tool, execute in zip(TOOLS, EXECUTORS):
                target = Path(directory)/(tool.__name__.split('.')[-1]+'.json')
                with patch('sys.argv', ['audit', '--report', str(target)]), \
                        patch.object(tool, execute, return_value=result), contextlib.redirect_stdout(io.StringIO()):
                    tool.main()
                self.assertEqual(contracts.strict_json_loads(target.read_bytes()), result)
        for p, (payload, mtime) in snapshots.items():
            self.assertEqual(p.read_bytes(), payload)
            self.assertEqual(p.stat().st_mtime_ns, mtime)

    def test_installed_package_is_not_imported_before_python_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory)/'pyroomacoustics'
            installed.mkdir()
            (installed/'__init__.py').write_text('raise RuntimeError("must not execute")\n')
            specification = type('Spec', (), {'submodule_search_locations': [str(installed)]})()
            with patch.object(reference.importlib.util, 'find_spec', return_value=specification), \
                    self.assertRaises(ValueError):
                reference.run_pra(Path(directory)/'missing-fixed-source')


class CurrentReportTests(unittest.TestCase):
    def read(self, tool):
        report = contracts.strict_json_loads(tool.CURRENT_REPORT.read_bytes())
        self.assertEqual(report['report_source_sha256'], contracts.dependencies(Path(tool.__file__)))
        return report

    def assert_identity(self, identity):
        lock, status, lock_sha, status_sha = contracts.source_documents()
        self.assertEqual(identity['lock_sha256'], lock_sha)
        self.assertEqual(identity['status_sha256'], status_sha)
        entry = next(r for r in lock['projects'] if r['id'] == identity['project'])
        recorded = next(r for r in status['projects'] if r['id'] == identity['project'])
        self.assertEqual(identity['lock_entry'], entry)
        self.assertEqual(identity['recorded_complete_selection'], recorded)
        self.assertEqual(identity['live_complete_selection']['status'], recorded['status'])
        self.assertEqual(identity['used_source_identity'], 'verified')
        self.assertTrue(identity['clean_before'])
        self.assertTrue(identity['clean_after'])
        self.assertEqual(identity['ignored_members_before'], identity['ignored_members_after'])
        url, revision, license_name, license_path, license_sha = contracts.PROJECTS[identity['project']]
        self.assertEqual((identity['origin'], identity['head'], entry['license']), (url, revision, license_name))
        self.assertEqual(identity['used_files'][license_path]['sha256'], license_sha)
        for relative, metadata in identity['used_files'].items():
            path = Path(identity['checkout'])/relative
            payload = contracts.ordinary_file(path).read_bytes()
            self.assertEqual(metadata['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(metadata['bytes'], len(payload))
            # Blob ID is independently calculated from ordinary file bytes.
            header = b'blob ' + str(len(payload)).encode() + b'\0'
            self.assertEqual(metadata['git_blob'], hashlib.sha1(header+payload).hexdigest())

    def test_current_original_asts_and_phase_axis_counterexample(self):
        report = self.read(upstream)
        self.assert_identity(report['source_identity'])
        self.assertEqual(report['case_count'], 33)
        self.assertEqual(len(report['results']), 33)
        self.assertEqual(len(report['original_functions']), 13)
        self.assertEqual(set(report['original_functions']), set(upstream.FUNCTIONS))
        self.assertTrue(all(r['expected_behavior_verified'] for r in report['results'].values()))
        self.assertEqual(report['status'], 'expected_behaviors_verified_failures_preserved')
        for name, metadata in report['original_functions'].items():
            text = (Path(report['source_identity']['checkout'])/metadata['file']).read_text()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertFalse(metadata['body_modified'])
            self.assertEqual(metadata['first_line'], node.lineno)
            self.assertEqual(metadata['last_line'], node.end_lineno)
            self.assertEqual(metadata['source_segment_sha256'], hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest())
        rows = report['results']
        np.testing.assert_array_equal(rows['phase_correction_2d']['output']['real'], np.ones((3, 2)))
        np.testing.assert_array_equal(rows['phase_correction_3d']['output']['real'], [[[1, 1], [1, 1], [-1, -1]]])
        self.assertEqual(rows['phase_correction_3d']['classification'], 'phase_axis_failure')
        np.testing.assert_array_equal(rows['phase_correction_3d']['observables']['weight_norm_squared_by_frequency']['real'], [[2, 2, 2]])
        np.testing.assert_array_equal(rows['phase_correction_3d']['observables']['target_response_by_frequency']['real'], [[2, 2, -2]])
        self.assertEqual(sum(r['classification'] == 'expected_exception' for r in rows.values()), 5)

    def test_current_reference_preserves_package_and_method_failures(self):
        report = self.read(reference)
        for identity in report['source_identities'].values():
            self.assert_identity(identity)
        self.assertEqual(report['status'], 'original_failures_preserved')
        pb, pra = report['pb_bss'], report['pyroomacoustics']
        self.assertEqual(pb['package_import_probe']['status'], 'failed')
        self.assertIn('paderbox', pb['package_import_probe']['message'])
        self.assertEqual([r['reference_selection_correct'] for r in pb['results']], [False, True])
        self.assertEqual(pra['exception']['type'], 'TypeError')
        self.assertEqual(pra['exception']['traceback'][-1]['line'], 1375)
        self.assertFalse(pra['filters_computed'])
        self.assertTrue(pra['installed_sources_unchanged_after'])
        for relative, digest in pra['installed_python_preflight_sha256'].items():
            self.assertEqual(digest, report['source_identities']['pyroomacoustics']['used_files'][relative]['sha256'])
        self.assertIn('no fixed-Git binary', pra['installed_artifact_scope'])
        for relative, metadata in pra['installed_artifact_sha256'].items():
            self.assertEqual(metadata['sha256'], contracts.sha(Path(pra['installed_path'])/relative))

    def test_current_sof_static_boundaries_and_defaults(self):
        report = self.read(sof)
        self.assert_identity(report['source_identity'])
        scope = report['execution_scope']
        for field in ('matlab_or_octave_design', 'fir_filter_generation', 'firmware_or_hardware', 'upstream_modified'):
            self.assertFalse(scope[field])
        self.assertTrue(scope['python_scalar_examples'])
        self.assertEqual(report['mathematical_examples'], sof.independent_examples())
        self.assertEqual([r['line'] for r in report['findings']], [123, 272, 412])
        self.assertEqual(report['findings'][-1]['execution'], 'static_source_only; no MATLAB, audio generation or firmware execution')
        self.assertEqual(report['fixed_source_defaults']['mu_db'], -40)
        self.assertEqual(report['mathematical_examples']['loading_conversion']['fixed_source_10_power_mu_db_over_20'], .01)


if __name__ == '__main__':
    unittest.main()
