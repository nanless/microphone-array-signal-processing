"""Independent write-preflight, provenance, and fixed original failure controls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch04.core import upstream_contracts as contracts
from codes.chapters.ch09.examples import audit_tracking_upstream_interfaces as interfaces
from codes.chapters.ch09.examples import audit_upstream_tracking_contracts as tracking


class ReportPreflight(unittest.TestCase):
    def test_rejected_target_precedes_every_original_method_call(self):
        for tool, option, old in (
            (interfaces, '--output', 'tracking_upstream_interfaces.json'),
            (tracking, '--report', 'upstream_tracking_contracts.json')):
            with tempfile.TemporaryDirectory(prefix='masp-ch09-target-') as directory:
                base = Path(directory)
                ordinary = base / 'ordinary'; ordinary.mkdir()
                (base / 'parent-link').symlink_to(ordinary, target_is_directory=True)
                (base / 'member-link.json').symlink_to(ordinary / 'other.json')
                (base / 'directory.json').mkdir()
                targets = [tool.CURRENT.with_name(old), Path(tool.__file__),
                    tool.CACHE / 'blocked.json', base / 'parent-link/a.json',
                    base / 'member-link.json', base / 'directory.json', base / 'non-json.txt']
                for target in targets:
                    with self.subTest(tool=tool.__name__, target=target):
                        with patch.object(sys, 'argv', ['audit', option, str(target)]), \
                             patch.object(tool, 'run') as runner:
                            with self.assertRaises((ValueError, OSError)):
                                tool.main()
                            runner.assert_not_called()

    def test_default_stdout_never_opens_a_report(self):
        for tool in (interfaces, tracking):
            with self.subTest(tool=tool.__name__), patch.object(sys, 'argv', ['audit']), \
                 patch.object(tool, 'run', return_value={'only': 'stdout'}), \
                 patch.object(contracts, 'write_report') as writer, \
                 patch('builtins.print') as output:
                tool.main()
                writer.assert_not_called()
                self.assertEqual(json.loads(output.call_args.args[0]), {'only': 'stdout'})

    def test_final_guard_rejects_target_changed_to_symlink(self):
        for tool, option in ((interfaces, '--output'), (tracking, '--report')):
            with tempfile.TemporaryDirectory(prefix='masp-ch09-second-guard-') as directory:
                target = Path(directory) / 'result.json'
                victim = Path(directory) / 'victim'; victim.write_bytes(b'original')
                def substituted_execution(root):
                    target.symlink_to(victim)
                    return {'controlled': True}
                with patch.object(sys, 'argv', ['audit', option, str(target)]), \
                     patch.object(tool, 'run', side_effect=substituted_execution):
                    with self.assertRaises((ValueError, OSError)):
                        tool.main()
                self.assertEqual(victim.read_bytes(), b'original')

    def test_foreign_preimported_filterpy_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='masp-ch09-foreign-') as directory:
            source = Path(directory) / 'foreign.py'; source.write_text('pass\n')
            fake = type('Module', (), {'__file__': str(source)})()
            with patch.dict(sys.modules, {'filterpy': fake}):
                with self.assertRaisesRegex(ValueError, 'foreign FilterPy'):
                    interfaces.loaded_filterpy_modules({'checkout': directory + '/author', 'used_files': {}})


class CurrentReportOracles(unittest.TestCase):
    def test_real_dependencies_and_separate_complete_selection(self):
        for tool, name, extras in ((interfaces, 'tracking_upstream_interfaces_current.json', ()),
                                  (tracking, 'upstream_tracking_contracts_current.json', (tracking.INTERFACE,))):
            report = json.loads(tool.CURRENT.read_text())
            self.assertEqual(report['actual_dependencies_sha256'],
                             contracts.dependencies(Path(tool.__file__), extras))
            self.assertEqual(report['source_lock_sha256'], contracts.sha(contracts.LOCK))
            self.assertEqual(report['source_status_sha256'], contracts.sha(contracts.STATUS))
            for name, identity in report['source_identities'].items():
                self.assertIs(identity['clean_before'], True)
                self.assertIs(identity['clean_after'], True)
                expected = 'source_verified' if name == 'odas' else 'source_selection_mismatch'
                self.assertEqual(identity['recorded_complete_selection']['status'], expected)
                self.assertEqual(identity['live_complete_selection']['status'], expected)
                self.assertEqual(identity['used_source_identity'], 'verified')
                self.assertEqual(identity['origin'], contracts.PROJECTS[name][0])
                self.assertEqual(identity['head'], contracts.PROJECTS[name][1])
                for record in identity['used_files'].values():
                    self.assertEqual(record['git_blob'], record['actual_blob'])

    def test_original_none_failure_is_partial_mutation_and_modules_are_bound(self):
        report = json.loads(interfaces.CURRENT.read_text())
        actual = report['filterpy']
        failure = actual['ekf_predict_update_none']
        self.assertEqual(failure['exception']['type'], 'TypeError')
        self.assertEqual(failure['after']['mean'], 1)
        self.assertEqual(failure['after']['variance'], 1)
        self.assertEqual(failure['after']['innovation_variance'], 2**2*4+1)
        self.assertAlmostEqual(failure['after']['gain'], 4*2/17)
        self.assertNotEqual(failure['before']['innovation_variance'], failure['after']['innovation_variance'])
        self.assertEqual(actual['imported_module_count'], len(actual['imported_modules']))
        self.assertEqual(actual['imported_module_count'], 33)
        for relative in actual['imported_modules'].values():
            self.assertIn(relative, actual['source_identity']['used_files'])
        self.assertFalse(report['stonesoup_reducer']['stonesoup_package_executed'])
        self.assertTrue(report['stonesoup_reducer']['package_import_probe'].startswith('not_run:'))
        for current, expected_sha in (
            ('tracking_upstream_interfaces.json', '1b839022a0c55373db49cb0dc1883ef3e6fd7faf5aafdc89aea004e8ac4f0f4f'),
            ('upstream_tracking_contracts.json', '979193d9e3a9ab883d69882f49a81d25f4bdc0c3ee67d3625ff4b746c69443cc')):
            self.assertEqual(hashlib.sha256(interfaces.CURRENT.with_name(current).read_bytes()).hexdigest(), expected_sha)


if __name__ == '__main__':
    unittest.main()
