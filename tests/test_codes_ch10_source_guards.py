"""Report target preflight, immutable historical evidence and scope controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch10.examples import audit_industrial_contracts as arithmetic
from codes.chapters.ch10.examples import audit_industrial_upstream_interfaces as interfaces
from codes.chapters.ch10.examples import run_industrial_interfaces as native
from codes.chapters.ch10.examples import run_stk_delay_probe as stk

ROOT = contracts.ROOT
HISTORY = 'bde483bcc429a553aeaf8830ca3687ab46831db0'
OLD = {
    'industrial_contracts.json': 'd39a1d280d92e54fc3ba24523769518666e382afb2bd8becf24469f15cd6cfc7',
    'industrial_upstream_interfaces.json': '2666e9b2905a9712aed5c2d89e3bcf1fff0f9467fc90fe8b1c4638abb511cb87',
    'industrial_interfaces.json': 'b2477e300b8f12753c2430d249817802dd8b936378091ee0cb76e1ae809d6d8a',
    'stk_delay.json': 'ac071f6ff910dea839e3782746406d650af6464ab840fb08fab1214369b95284',
}


def original(path):
    environment = native.clean_environment()
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_NO_LAZY_FETCH='1')
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', HISTORY + ':' + path], env=environment)


class ReportGuards(unittest.TestCase):
    def test_original_four_reports_are_byte_exact(self):
        for name, expected in OLD.items():
            relative = 'codes/chapters/ch10/reports/' + name
            payload = (ROOT / relative).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)
            self.assertEqual(payload, original(relative))

    def test_real_historical_tool_bindings(self):
        for filename, tool, field in (
                ('industrial_contracts.json', 'audit_industrial_contracts.py', 'tool_sha256'),
                ('industrial_upstream_interfaces.json', 'audit_industrial_upstream_interfaces.py', 'harness_sha256'),
                ('stk_delay.json', 'run_stk_delay_probe.py', 'runner_sha256')):
            report = json.loads((ROOT / 'codes/chapters/ch10/reports' / filename).read_text())
            payload = original('codes/chapters/ch10/examples/' + tool)
            self.assertEqual(report[field], hashlib.sha256(payload).hexdigest())

    def test_all_four_cli_targets_rejected_before_execution(self):
        # These commands would execute original methods / compile native
        # libraries if the target were not rejected first.
        for module in (arithmetic, interfaces, native, stk):
            for target in (ROOT/'AGENTS.md', ROOT/'codes/chapters/ch10/reports/industrial_contracts.json'):
                before = target.read_bytes()
                result = subprocess.run([sys.executable, str(Path(module.__file__)), '--report', str(target)],
                    env=native.clean_environment() | {'PYTHONDONTWRITEBYTECODE': '1'},
                    capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('designated new current report', result.stderr)
                self.assertEqual(target.read_bytes(), before)

    def test_native_function_target_guard_precedes_load(self):
        with patch.object(native, 'load_projects') as load:
            with self.assertRaises(ValueError):
                native.run(ROOT/'AGENTS.md')
            load.assert_not_called()

    def test_ordinary_external_report_and_forbidden_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root/'source'
            cache.mkdir()
            target = root/'result.json'
            contracts.write_report(target, {'value': 3}, arithmetic.CURRENT, (cache,))
            self.assertEqual(json.loads(target.read_text()), {'value': 3})
            link = root/'link'
            link.symlink_to(root, target_is_directory=True)
            for bad in (link/'result.json', cache/'result.json', root/'absent'/'..'/'result.json'):
                with self.assertRaises(ValueError):
                    contracts.report_target(bad, arithmetic.CURRENT, (cache,))
            self.assertEqual(json.loads(target.read_text()), {'value': 3})

    def test_original_package_pollution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            foreign = root/'foreign.py'
            foreign.write_text('')
            identity = {'checkout': str(root/'upstream'), 'used_files': {}}
            with patch.dict(sys.modules, {'pystoi': SimpleNamespace(__file__=str(foreign))}):
                with self.assertRaisesRegex(ValueError, 'foreign preloaded'):
                    interfaces.loaded_pystoi_modules(identity)

    def test_clock_counter_is_host_callback_not_model_frame_time(self):
        result = interfaces.callback_clock_example()
        self.assertEqual([r['first_eligible_callback'] for r in result['rows']], [1002, 1001])
        self.assertAlmostEqual(result['rows'][0]['input_audio_seconds_by_host_block']['128'], 2.672)
        self.assertAlmostEqual(result['rows'][1]['input_audio_seconds_by_host_block']['480'], 10.01)
        self.assertIn('not Rust', interfaces.callback_clock_example.__doc__)

    def test_clean_build_environment(self):
        with patch.dict('os.environ', {'GIT_DIR': '/foreign', 'CFLAGS': '-I/foreign',
                'DYLD_LIBRARY_PATH': '/foreign', 'CMAKE_PREFIX_PATH': '/foreign', 'PYTHONPATH': '/foreign'}):
            environment = native.clean_environment()
            self.assertTrue(all(x not in environment for x in
                ('GIT_DIR', 'CFLAGS', 'DYLD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH', 'PYTHONPATH')))

    def test_failed_run_postcheck_keeps_primary_exception(self):
        error = RuntimeError('original compiler failure')
        identity = {'project': 'fixture'}
        with patch.object(native.upstream, 'check_unchanged', side_effect=ValueError('changed source')) as check:
            native.verify_failure_sources({'fixture': identity}, error)
        check.assert_called_once_with(identity)
        self.assertEqual(str(error), 'original compiler failure')
        self.assertIn('changed source', error.__notes__[0])

    def test_dependency_closure_rejects_unpreflighted_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp).resolve()
            source = work/'upstream'
            source.mkdir()
            header = source/'foreign.h'
            header.write_text('')
            (work/'unit.d').write_text('unit.o: ' + str(header) + '\n')
            with self.assertRaisesRegex(ValueError, 'not preflighted'):
                native.compiled_dependencies(work, {'fixture': {'checkout': str(source), 'used_files': {}}})

    def test_current_reports_actual_dependency_and_history_separation(self):
        modules = (arithmetic, interfaces, native, stk)
        if any(not module.CURRENT.exists() for module in modules):
            self.skipTest("ROOT has not yet generated all four frozen current reports")
        for module in modules:
            current = module.CURRENT
            report = contracts.strict_json_loads(current.read_bytes())
            for name, digest in report['actual_dependencies_sha256'].items():
                self.assertEqual(contracts.sha(ROOT/name), digest)
            self.assertNotIn(current.name, OLD)


if __name__ == '__main__':
    unittest.main()
