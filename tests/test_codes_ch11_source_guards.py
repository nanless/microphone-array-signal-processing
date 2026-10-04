"""Report destinations, import isolation and actual current source identity."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from codes.chapters.ch11.examples import audit_meeting_kernel_contracts as native
from codes.chapters.ch11.examples import audit_meeting_scoring_interfaces as scoring


class MeetingSourceGuards(unittest.TestCase):
    def test_all_repository_targets_except_current_rejected_before_execution(self):
        for module, option, runner, old in [
            (native, '--report', 'run_audit', 'meeting_kernel_contracts.json'),
            (scoring, '--output', 'run', 'meeting_scoring_interfaces.json'),
        ]:
            for target in [module.CURRENT.with_name(old), Path(module.__file__),
                           native.LOCK, native.CACHE/'meeteval'/'README.md']:
                with self.subTest(target=target), patch.object(module, runner) as run:
                    with self.assertRaises(ValueError):
                        module.main([option, str(target)])
                    run.assert_not_called()

    def test_scoring_output_parent_and_leaf_symlinks_rejected_before_worker(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            folder = Path(directory)
            sentinel = folder/'sentinel'; sentinel.write_bytes(b'preserve')
            link = folder/'link'; link.symlink_to(sentinel)
            parent = folder/'parent'; parent.symlink_to(folder, target_is_directory=True)
            for target in [link, parent/'new.json', parent/'..'/'new.json']:
                with self.subTest(target=target), patch.object(scoring, 'run') as run:
                    with self.assertRaises(ValueError):
                        scoring.main(['--output', str(target)])
                    run.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'preserve')
            self.assertFalse((folder/'new.json').exists())

    def test_ordinary_missing_report_parents_are_created_by_shared_writer(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            target = Path(directory)/'missing'/'report.json'
            with patch.object(scoring, 'run', return_value={'limited': True}):
                scoring.main(['--output', str(target)])
            self.assertEqual(json.loads(target.read_bytes()), {'limited': True})

    def test_scoring_default_is_stdout_without_report_writes(self):
        with patch.object(scoring, 'run', return_value={'limited': True}), \
             patch.object(scoring.upstream, 'write_report') as write, \
             patch('sys.stdout', new_callable=io.StringIO) as stdout:
            scoring.main([])
        self.assertEqual(json.loads(stdout.getvalue()), {'limited': True})
        write.assert_not_called()

    def test_preloaded_author_module_rejected_before_source_execution(self):
        for function in [scoring.run, scoring._worker]:
            with patch.dict(sys.modules, {'meeteval': types.ModuleType('meeteval')}), \
                 patch.object(scoring, 'verify_sources') as verify:
                with self.assertRaisesRegex(ValueError, 'preloaded'):
                    function(native.CACHE)
                verify.assert_not_called()

    def test_environment_removes_search_and_build_injection(self):
        values = {key: 'poison' for key in ['GIT_DIR', 'CPATH', 'CPLUS_INCLUDE_PATH',
                  'CFLAGS', 'LDFLAGS', 'PYTHONPATH', 'DYLD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH']}
        with patch.dict(scoring.os.environ, values):
            environment = scoring.clean_environment()
        self.assertTrue(set(values).isdisjoint(environment))

    def test_unverified_author_bytecode_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            root = Path(directory)
            package = root/'meeteval'/'meeteval'
            package.mkdir(parents=True)
            for suffix in ['.pyc', '.so', '.pyd']:
                path = package/('injected'+suffix); path.write_bytes(b'not a verified original blob')
                with patch.object(scoring.upstream, 'verify_project') as verify:
                    with self.assertRaisesRegex(ValueError, 'unverified'):
                        scoring.verify_sources(root)
                    verify.assert_not_called()
                path.unlink()


class CurrentMeetingReportIdentity(unittest.TestCase):
    def reports(self):
        return [(native, json.loads(native.CURRENT.read_bytes())),
                (scoring, json.loads(scoring.CURRENT.read_bytes()))]

    def test_actual_tool_dependencies_and_full_selection_are_independent(self):
        for module, report in self.reports():
            self.assertEqual(report['schema_version'], 2)
            for relative, expected in report['actual_dependencies'].items():
                self.assertEqual(hashlib.sha256((native.ROOT/relative).read_bytes()).hexdigest(), expected)
            identities = report.get('source_identities', {'meeteval': report.get('source_identity')})
            for identity in identities.values():
                self.assertTrue(identity['clean_before'] and identity['clean_after'])
                self.assertEqual(identity['used_source_identity'], 'verified')
                for path, row in identity['used_files'].items():
                    self.assertEqual(hashlib.sha256((Path(identity['checkout'])/path).read_bytes()).hexdigest(), row['sha256'])
                    self.assertEqual(row['git_blob'], row['actual_blob'])
            self.assertEqual(identities['meeteval']['live_complete_selection']['status'], 'source_selection_mismatch')
            self.assertFalse(identities['meeteval']['live_complete_selection']['source_selection_verified'])

    def test_actual_compile_inputs_and_imported_modules_are_subsets(self):
        native_report = json.loads(native.CURRENT.read_bytes())
        original = native_report['compiled_dependencies']['original_compile_inputs']
        self.assertEqual(set(original), {'meeteval/meeteval/wer/matching/levenshtein.h'})
        self.assertEqual(len(native_report['cases']), 13)
        self.assertEqual([r['timed'] for r in native_report['cases'][9:]], [0, 2, 2, 0])
        report = json.loads(scoring.CURRENT.read_bytes())
        used = report['source_identities']['meeteval']['used_files']
        self.assertGreater(len(report['actual_imported_original_modules']), 1)
        for row in report['actual_imported_original_modules'].values():
            self.assertEqual(row['sha256'], used[row['path']]['sha256'])
        self.assertEqual(report['package']['cp_word_error_rate']['exception_type'], 'ModuleNotFoundError')
        self.assertEqual(report['documentation']['di_cp_documentation_call']['exception_type'], 'NameError')
        self.assertFalse(report['chime']['challenge_pipeline_executed'])

    def test_original_timing_helpers_match_independent_character_lengths(self):
        timing = json.loads(scoring.CURRENT.read_bytes())['word_timing']
        self.assertEqual([(x['start_time'], x['end_time']) for x in timing['reference']], [(0, 3), (3, 4)])
        self.assertEqual([(x['start_time'], x['end_time']) for x in timing['hypothesis']], [(1.5, 1.5), (3.5, 3.5)])
        self.assertTrue(timing['matched_independent_expected'])
        self.assertIn('no complete tcpWER', timing['execution'])


if __name__ == '__main__':
    unittest.main()
