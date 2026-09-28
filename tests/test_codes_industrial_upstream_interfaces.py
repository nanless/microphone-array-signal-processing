"""Offline report oracles: no downloaded source, SciPy or network required."""
import json
from pathlib import Path
import unittest

from codes.chapters.ch10.examples import audit_industrial_upstream_interfaces as audit

ROOT = Path(__file__).resolve().parents[1]


class IndustrialUpstreamReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'codes/chapters/ch10/reports/industrial_upstream_interfaces.json').read_text(),
                                parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

    def test_binding_and_scope(self):
        r = self.report
        self.assertEqual(r['harness_sha256'], audit.sha256(audit.__file__))
        self.assertEqual(r['source_config_sha256'], audit.binding_sha256())
        self.assertEqual(r['sources'], audit.SOURCES)
        self.assertEqual(r['config'], audit.CONFIG)
        self.assertEqual(r['pystoi']['execution'], 'original pystoi package functions')
        self.assertFalse(r['deepfilternet']['rust_plugin_executed'])
        self.assertIn('not upstream execution', r['deepfilternet']['queue_example']['execution'])

    def test_super_short_and_warning_are_different_failures(self):
        records = self.report['pystoi']['records']
        for r in records:
            if r['input'] == 'zeros_256':
                self.assertEqual(r['classification'], 'exception')
                self.assertEqual(r['exception_type'], 'AxisError')
                self.assertIsNone(r['value'])
                self.assertEqual(r['warnings'], [])
            elif r['input'] == 'zeros_1024':
                self.assertEqual(r['classification'], 'warning_sentinel')
                self.assertEqual(r['value'], 1e-5)
                self.assertEqual(r['warnings'][0]['category'], 'RuntimeWarning')

    def test_zero_energy_is_not_valid_intelligibility(self):
        records = [r for r in self.report['pystoi']['records'] if r['input'] == 'zeros_10000']
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]['value'], 0.)
        # Fixed, independently retained outputs from the first read-only call.
        self.assertAlmostEqual(records[1]['value'], 0.00856518978523145, places=12)
        self.assertAlmostEqual(records[2]['value'], -0.00020999388836695469, places=12)
        for r in records:
            self.assertTrue(r['finite'])
            self.assertFalse(r['reference_has_energy'])
            self.assertEqual(r['warnings'], [])
            self.assertEqual(r['classification'], 'finite_zero_energy_invalid')

    def test_nonzero_self_correlation_identity(self):
        records = [r for r in self.report['pystoi']['records'] if r['input'] == 'random_self']
        self.assertEqual(len(records), 3)
        # Independent correlation identity: identical nonconstant normalized
        # vectors have inner product one. This is not a speech-quality test.
        for r in records:
            self.assertAlmostEqual(r['value'], 1., places=12)
            self.assertTrue(r['reference_has_energy'])
            self.assertEqual(r['classification'], 'finite_self_comparison')
            self.assertEqual(r['warnings'], [])

    def test_channel_axis_and_frame_axis_are_not_interchangeable(self):
        r = self.report['deepfilternet']['queue_example']
        # Two channel queues, one pop each: 960-1=959; a whole hop is 480.
        self.assertEqual(r['queue_lengths_after'], [959, 959])
        self.assertEqual(r['actual_removed_per_channel'], [1, 1])
        self.assertEqual(r['first_remaining_sample_indices'], [1, 1])
        self.assertEqual(r['proc_delay_after'], 480)
        self.assertEqual(r['intended_frame_drop_lengths'], [480, 480])
        self.assertEqual(audit.queue_semantics_example(), r)
        self.assertEqual(self.report['deepfilternet']['evidence_lines']['metadata_frame_decrement'], [479])


if __name__ == '__main__':
    unittest.main()
