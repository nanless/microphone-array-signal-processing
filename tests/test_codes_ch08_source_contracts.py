"""Finite source/write boundaries; original reports bind original Git bytes."""
import contextlib
import hashlib
import io
import importlib
import os
import py_compile
import sys
from types import SimpleNamespace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch08.examples import (
    audit_separation_upstream_interfaces as interfaces,
    audit_upstream_separation_contracts as audit,
    streamfm_source_audit as stream,
    tflocoformer_source_audit as tf,
)

TOOLS = (interfaces, audit, stream, tf)
REPORTS = contracts.ROOT/'codes/chapters/ch08/reports'
OLD = ('separation_upstream_interfaces.json', 'upstream_separation_contracts.json',
       'streamfm_source_audit.json', 'tflocoformer_source_audit.json')
REVISION = '7b80fab1d1012f104cdf441423218bf0898ce918'


def old_bytes(path):
    return subprocess.check_output(['git', '-C', str(contracts.ROOT), 'show',
        REVISION+':'+str(Path(path).resolve().relative_to(contracts.ROOT))])


class HistoricalAndWriteBoundaries(unittest.TestCase):
    def test_four_original_reports_and_actual_original_tool_bindings(self):
        for tool, name in zip(TOOLS, OLD):
            path = REPORTS/name
            self.assertEqual(path.read_bytes(), old_bytes(path))
            original = old_bytes(tool.__file__)
            compile(original, tool.__file__, 'exec')
            report = contracts.strict_json_loads(path.read_bytes())
            field = 'harness_sha256' if tool is interfaces else 'tool_sha256' if tool is audit else None
            if field:
                self.assertEqual(report[field], hashlib.sha256(original).hexdigest())
        legacy = contracts.strict_json_loads((REPORTS/OLD[1]).read_bytes())
        self.assertEqual(legacy['legacy_helper_sha256'], hashlib.sha256(old_bytes(interfaces.__file__)).hexdigest())

    def test_forbidden_targets_are_rejected_before_execution(self):
        for tool in TOOLS:
            calculate = {interfaces:'run_audit', audit:'build_report', stream:'audit_file', tf:'audit'}[tool]
            option = '--output' if tool is tf else '--report'
            targets = [*(REPORTS/name for name in OLD), contracts.LOCK, contracts.STATUS,
                       Path(tool.__file__), contracts.CACHE/'new.json',
                       contracts.ROOT/'codes/chapters/ch00/source_snapshots/new.json',
                       *(t.CURRENT_REPORT for t in TOOLS if t is not tool)]
            for target in targets:
                with self.subTest(tool=tool.__name__, target=target), \
                    patch('sys.argv', ['audit', option, str(target)]), \
                    patch.object(tool, calculate) as run, self.assertRaises(ValueError):
                    tool.main()
                run.assert_not_called()

    def test_custom_source_overlap_symlink_and_parent_symlink_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'; source.mkdir()
            real = root/'real'; real.mkdir()
            (root/'linked').symlink_to(real, target_is_directory=True)
            (root/'report.json').symlink_to(real/'report.json')
            for tool in TOOLS:
                option = '--output' if tool is tf else '--report'
                source_option = '--source-root' if tool in (interfaces, audit) else '--source'
                source_path = source/'sgmse/backbones/streaming_unet.py' if tool is stream else source
                for target in (source/'report.json', root/'linked/report.json', root/'report.json'):
                    with self.subTest(tool=tool.__name__, target=target), \
                        patch('sys.argv', ['audit', source_option, str(source_path), option, str(target)]), \
                        self.assertRaises(ValueError):
                        tool.main()

    def test_stdout_default_does_not_write(self):
        for tool in TOOLS:
            calculate = {interfaces:'run_audit', audit:'build_report', stream:'audit_file', tf:'audit'}[tool]
            with patch('sys.argv', ['audit']), patch.object(tool, calculate, return_value={'scope':'fixture'}), \
                patch.object(contracts, 'write_report') as write, contextlib.redirect_stdout(io.StringIO()) as out:
                tool.main()
            self.assertEqual(contracts.strict_json_loads(out.getvalue()), {'scope':'fixture'})
            write.assert_not_called()

    def test_original_package_import_bypasses_existing_valid_pyc_without_writing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root/'micarray_original_fixture'; package.mkdir()
            source = package/'__init__.py'
            source.write_text("VALUE = 'poison'\n")
            compiled = Path(py_compile.compile(str(source), doraise=True))
            stat = source.stat()
            source.write_text("VALUE = 'actual'\n")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            cached = compiled.read_bytes()
            sys.path.insert(0, str(root))
            try:
                importlib.invalidate_caches()
                normal = importlib.import_module('micarray_original_fixture')
                self.assertEqual(normal.VALUE, 'poison')
                del sys.modules['micarray_original_fixture']
                with interfaces.source_imports('micarray_original_fixture'):
                    actual = importlib.import_module('micarray_original_fixture')
                self.assertEqual(actual.VALUE, 'actual')
                self.assertEqual(actual.__micarray_original_source_sha256__, contracts.sha(source))
                self.assertEqual(compiled.read_bytes(), cached)
            finally:
                sys.path.remove(str(root))
                sys.modules.pop('micarray_original_fixture', None)

    def test_preloaded_unverified_module_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'__init__.py'; source.write_text('VALUE=1\n')
            spec = SimpleNamespace(origin=str(source), submodule_search_locations=[str(root)])
            for filename in (str(root/'outside'/'other.py'), str(source)):
                module = SimpleNamespace(__file__=filename)
                with patch.dict(sys.modules, {'micarray_preload_fixture':module}), \
                    patch.object(interfaces.importlib.machinery.PathFinder, 'find_spec', return_value=spec), \
                    self.assertRaisesRegex(ValueError, 'Preloaded'):
                    interfaces.package_preflight('micarray_preload_fixture', root)

    def test_current_dependency_hashes_are_real_and_scope_not_upgraded(self):
        for tool in TOOLS:
            report = contracts.strict_json_loads(tool.CURRENT_REPORT.read_bytes())
            for path, expected in report['actual_dependency_sha256'].items():
                self.assertEqual(contracts.sha(contracts.ROOT/path), expected)
            identities = report.get('source_identities', {'single':report.get('source_identity')})
            for identity in identities.values():
                self.assertTrue(identity['clean_before'])
                self.assertTrue(identity['clean_after'])
                self.assertEqual(identity['used_source_identity'], 'verified')
                for name, record in identity['used_files'].items():
                    self.assertEqual(record['git_blob'], record['actual_blob'])
                    self.assertEqual(contracts.sha(Path(identity['checkout'])/name), record['sha256'])
            if tool in (stream, tf):
                self.assertNotIn('inference_result', report)
        r = contracts.strict_json_loads(audit.CURRENT_REPORT.read_bytes())
        self.assertEqual(r['source_identities']['arraydps']['live_complete_selection']['status'],
                         'source_selection_mismatch')
        self.assertEqual(r['executed']['pra_original_methods']['auxiva_eigen_initialization']['exception_type'], 'ValueError')
        self.assertGreater(r['executed']['pra_original_methods']['ilrma']['max_output_filter_difference'], 3.)


if __name__ == '__main__':
    unittest.main()
