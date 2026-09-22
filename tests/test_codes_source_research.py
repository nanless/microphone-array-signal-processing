"""Offline consistency checks for the expanded source research inventory."""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceResearchTests(unittest.TestCase):
    def test_report_binds_exact_lock_and_project_set(self):
        lock = ROOT / 'codes/SOURCES.lock.json'
        projects = json.loads(lock.read_text())['projects']
        report = json.loads((ROOT / 'codes/SOURCE_STATUS.json').read_text())
        self.assertEqual(report['lock_sha256'], hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual({p['id']: p['revision'] for p in projects},
                         {p['id']: p['revision'] for p in report['projects']})

    def test_new_sources_have_license_entries_and_research_coverage(self):
        projects = {p['id']: p for p in json.loads(
            (ROOT / 'codes/SOURCES.lock.json').read_text())['projects']}
        expected = {
            'sbl': 'LICENSE', 'robustsbl': 'LICENSE', 'btk20': 'LICENSE',
            'smpphat': 'LICENSE', 'libsoxr': 'LICENCE', 'libebur128': 'COPYING',
            'pystoi': 'LICENSE', 'visqol': 'LICENSE', 'libsndfile': 'COPYING',
            'lib-xcore-math': 'LICENSE.rst', 'e2e-ad-aec': 'LICENSE',
            'integrated-aec-nr': 'LICENSE.md', 'nbss': 'LICENSE',
        }
        coverage = (ROOT / 'codes/COVERAGE.md').read_text()
        for project_id, license_file in expected.items():
            with self.subTest(project=project_id):
                self.assertIn(license_file, projects[project_id]['entrypoints'])
                self.assertIn(f'`{project_id}`', coverage)

    def test_asset_heavy_sources_use_explicit_source_subsets(self):
        projects = {p['id']: p for p in json.loads(
            (ROOT / 'codes/SOURCES.lock.json').read_text())['projects']}
        for project_id, excluded in {
            'integrated-aec-nr': {'Audio'}, 'e2e-ad-aec': {'data'},
            'visqol': {'model', 'testdata'}, 'pystoi': {'tests'},
        }.items():
            with self.subTest(project=project_id):
                paths = set(projects[project_id]['source_paths'])
                self.assertTrue(paths)
                self.assertFalse(paths & excluded)


if __name__ == '__main__':
    unittest.main()
