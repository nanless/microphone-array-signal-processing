"""Real source fixtures for Figure 21 transitive provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import quality_check as quality


class WPEFigureProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = ('scripts/make_figures.py',
                      'codes/chapters/ch07/core/dereverberation.py',
                      'codes/chapters/ch02/core/conventions.py')
        for index, name in enumerate(self.paths):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'actual fixture {index}\n'.encode())
        self.inputs = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                       for name in self.paths}
        self.metadata = {'GeneratorInputs': json.dumps(self.inputs)}
        self.report = {'generator_inputs': self.inputs.copy(),
                       'source_sha256': self.inputs[self.paths[0]]}
        self.patcher = patch.object(quality, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_current_sources_pass_and_unrelated_file_does_not_affect(self):
        (self.root / 'unrelated.py').write_bytes(b'other source')
        self.assertEqual(quality.wpe_figure_provenance_issues(self.metadata, self.report), [])

    def test_changed_core_rejects_both_image_and_report(self):
        (self.root / self.paths[1]).write_bytes(b'changed actual numerical kernel')
        self.assertEqual(len(quality.wpe_figure_provenance_issues(self.metadata, self.report)), 2)

    def test_missing_extra_and_corrupt_maps_are_rejected(self):
        for inputs in ({}, {**self.inputs, 'unrelated.py': '0' * 64},
                       {self.paths[0]: self.inputs[self.paths[0]]}):
            with self.subTest(inputs=inputs):
                self.assertTrue(quality.wpe_figure_provenance_issues(
                    {'GeneratorInputs': json.dumps(inputs)},
                    {**self.report, 'generator_inputs': inputs}))
        for metadata in ({}, {'GeneratorInputs': 'not JSON'}, {'GeneratorInputs': None}):
            self.assertTrue(quality.wpe_figure_provenance_issues(metadata, self.report))
        self.assertTrue(quality.wpe_figure_provenance_issues(self.metadata, None))


if __name__ == '__main__':
    unittest.main()
