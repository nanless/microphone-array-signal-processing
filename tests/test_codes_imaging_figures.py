"""Independent figure gates: genuine numeric controls and adversarial reports.

Temporary fixture source hashes replace provenance only. The gate computes its
numeric oracles itself and never imports the drawing/teaching implementations.
No PNG, publication page, original report or original source is rewritten.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import quality_check as gate
from codes.chapters.ch00.io_contracts import strict_json_loads


class ImagingFigureGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sources = {}
        for relative in gate.IMAGING_FIGURE_SOURCES:
            path = self.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('ordinary immutable fixture for '+relative+'\n').encode())
            self.sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.reports = {}
        for number, filename in gate.IMAGING_FIGURE_REPORTS.items():
            # Actual recorded data is the subject under test; numerical expected
            # values come from independent algebra in the gate, not this report.
            report = strict_json_loads((gate.ROOT/'codes/chapters/ch14/reports'/filename).read_bytes())
            report['source_sha256'] = self.sources.copy()
            self.reports[number] = report
            self.write(number)

    def path(self, number):
        return self.root/'codes/chapters/ch14/reports'/gate.IMAGING_FIGURE_REPORTS[number]

    def write(self, number):
        path = self.path(number)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.reports[number], allow_nan=False))

    def run_gate(self):
        errors = []
        with patch.object(gate, 'ROOT', self.root):
            gate.check_imaging_figures(errors)
        return errors

    def snapshot(self):
        return {str(p.relative_to(self.root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in self.root.rglob('*') if p.is_file() and not p.is_symlink()}

    def test_genuine_numeric_reports_pass_without_writing(self):
        before = self.snapshot()
        with patch.object(Path, 'write_bytes', side_effect=AssertionError('read-only gate wrote bytes')), \
                patch.object(Path, 'write_text', side_effect=AssertionError('read-only gate wrote text')):
            self.assertEqual(self.run_gate(), [])
        self.assertEqual(before, self.snapshot())

    def test_each_real_source_is_bound(self):
        for relative in gate.IMAGING_FIGURE_SOURCES:
            path = self.root/relative
            before = path.read_bytes()
            with self.subTest(source=relative):
                path.write_bytes(before+b'changed\n')
                errors = self.run_gate()
                self.assertEqual(len(errors), 3)
                self.assertTrue(all('真实源摘要过期' in error for error in errors))
                path.write_bytes(before)

    def test_forged_source_set_and_boolean_schema_are_rejected(self):
        original = copy.deepcopy(self.reports[67])
        self.reports[67]['source_sha256']['fake.py'] = '0'*64
        self.write(67)
        self.assertEqual(len(self.run_gate()), 1)
        self.reports[67] = copy.deepcopy(original)
        self.reports[67]['schema_version'] = True
        self.write(67)
        self.assertEqual(len(self.run_gate()), 1)

    def test_two_cell_psf_and_causal_phase_tampering_fail(self):
        original = copy.deepcopy(self.reports[67])
        self.reports[67]['results']['two_cell']['P'][0][1] = .2
        self.write(67)
        self.assertIn('两格PSF', self.run_gate()[0])
        self.reports[67] = original
        self.reports[67]['results']['two_cell']['A']['imag'][1][1] *= -1
        self.write(67)
        self.assertIn('负相位', self.run_gate()[0])

    def test_spherical_grid_and_local_peak_are_independently_checked(self):
        original = copy.deepcopy(self.reports[67])
        self.reports[67]['results']['spherical_scan']['b'][225] += .01
        self.write(67)
        self.assertIn('完整二维扫描', self.run_gate()[0])
        self.reports[67] = original
        self.reports[67]['results']['spherical_scan']['local_peaks'][1]['peak_grid_index'] = 225
        self.write(67)
        self.assertIn('局部峰格索引', self.run_gate()[0])

    def test_coherent_zero_csm_residual_cannot_fake_success(self):
        self.reports[68]['results']['two_cell']['cases']['coherent']['full_csm_residual']['relative_frobenius'] = 0.
        self.write(68)
        errors = self.run_gate()
        self.assertEqual(len(errors), 1)
        self.assertIn('完整CSM残差', errors[0])

    def test_iteration_order_and_work_per_round_are_checked(self):
        original = copy.deepcopy(self.reports[68])
        self.reports[68]['results']['forward']['history'][1][0] = 1.
        self.write(68)
        self.assertIn('逐轮状态', self.run_gate()[0])
        self.reports[68] = original
        self.reports[68]['results']['forward_backward']['passes_per_iteration'] = 1
        self.write(68)
        self.assertIn('每轮扫描次数', self.run_gate()[0])

    def test_known_original_overallocation_is_not_the_teaching_oracle(self):
        self.reports[68]['results']['rank_one_cumulative'][-1] = 4.812502861022949
        self.write(68)
        self.assertIn('CLEAN累计分配', self.run_gate()[0])

    def test_fit_target_and_db_aggregation_tampering_fail(self):
        original = copy.deepcopy(self.reports[69])
        self.reports[69]['results']['different_csm_objective_optima'][1] = 41/25
        self.write(69)
        self.assertIn('拟合目标', self.run_gate()[0])
        self.reports[69] = original
        self.reports[69]['results']['region_db_relative_to_one'][0] = -3.01029995664
        self.write(69)
        self.assertIn('汇总后的dB', self.run_gate()[0])

    def test_nearby_legal_controls_are_not_false_positives(self):
        # Valid negative phase parts, negative dB and a negative intercept are
        # physically/analytically permitted here, unlike nonfinite metadata.
        self.assertLess(self.reports[67]['results']['two_cell']['A']['imag'][1][1], 0)
        self.assertLess(self.reports[69]['results']['intercept_prediction'], 0)
        self.assertLess(self.reports[69]['results']['region_db_relative_to_one'][2], 0)
        self.assertEqual(self.run_gate(), [])

    def test_numeric_strings_bool_nonfinite_and_duplicate_keys_fail(self):
        original = copy.deepcopy(self.reports[69])
        for value in ('1.64', True):
            with self.subTest(value=value):
                self.reports[69] = copy.deepcopy(original)
                self.reports[69]['results']['different_csm_objective_optima'][0] = value
                self.write(69)
                self.assertIn('非数字或布尔', self.run_gate()[0])
        for text in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}', '{"x":1e999}'):
            with self.subTest(text=text):
                self.path(69).write_text(text)
                self.assertEqual(len(self.run_gate()), 1)

    def test_missing_linked_and_hardlinked_reports_are_not_repaired(self):
        path = self.path(69)
        backup = self.root/'outside-report.json'
        backup.write_bytes(path.read_bytes())
        path.unlink()
        self.assertEqual(len(self.run_gate()), 1)
        self.assertFalse(path.exists())
        path.symlink_to(backup)
        self.assertIn('symbolic link', self.run_gate()[0])
        path.unlink()
        os.link(backup, path)
        self.assertIn('singly linked', self.run_gate()[0])


if __name__ == '__main__':
    unittest.main()
