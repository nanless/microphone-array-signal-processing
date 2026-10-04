"""Independent small-map expectations for unchanged SAID compression module."""
import hashlib
import json
from pathlib import Path
import unittest

from codes.chapters.ch04.core import upstream_contracts as contracts

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch04.examples import audit_said_compression as audit

REPORT = audit.ROOT / 'codes/chapters/ch04/reports/said_compression.json'


def strict_json(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


class CompressionReportTests(unittest.TestCase):
    def setUp(self):
        self.report = strict_json(REPORT.read_text())
        self.result = self.report['results']

    def test_actual_source_and_lock_binding(self):
        self.assertEqual(self.report['audit_source_sha256'], hashlib.sha256(contracts.historical_bytes(audit.__file__)).hexdigest())
        verify_lock_binding(self.report['lock_sha256'], (audit.PROJECT,), current_lock=audit.LOCK)
        self.assertEqual(self.report['before'], self.report['after'])
        self.assertEqual(self.report['before']['head'], audit.REVISION)
        self.assertEqual(self.report['before']['status'], '')
        entry = next(p for p in json.loads(audit.LOCK.read_text())['projects'] if p['id'] == audit.PROJECT)
        self.assertEqual(entry, self.report['lock_entry'])
        self.assertEqual(len(self.report['sources']), 5)
        self.assertIn('Torch/model execution', self.report['scope']['claims_excluded'])

    def test_relative_floor_and_identity_are_independent_hand_answer(self):
        output = self.result['high_confidence_output']
        self.assertEqual(output, {'id': 7, 'image_id': 11, 'category_id': 1, 'score': .9,
                                  'extra': 'preserved',
                                  'segmentation': [[[0,0,1],[2,0,.5]]]})
        self.assertEqual(self.result['high_confidence_input']['segmentation'][0][-1], [12,0,0])

    def test_boundary_seeds_and_seam_wrap_by_scalar_arithmetic(self):
        # Explicit +/-2 and four diagonal offsets around (359,0), energy .12.
        points = self.result['low_confidence_output']['segmentation'][0]
        self.assertEqual(points[0], [359,0,1])
        actual = sorted(tuple(point) for point in points[1:])
        expected = []
        for dx,dy in [(2,0),(-2,0),(0,2),(0,-2),
                      (1.4142,1.4142),(1.4142,-1.4142),
                      (-1.4142,1.4142),(-1.4142,-1.4142)]:
            expected.append((round((359+dx)%360,3), round(max(0,min(179,dy)),3), .12))
        self.assertEqual(actual, sorted(expected))
        self.assertEqual(self.result['wrapped_coordinates'], [[1,0],[359,179]])
        self.assertEqual(self.result['wrapped_distance_squared'], [2**2])

    def test_detection_retention_and_actual_serialization_size(self):
        directory = self.result['directory']
        self.assertEqual([a['id'] for a in directory['output_payload']['annotations']], [7,8])
        encoded = (json.dumps(directory['output_payload'], ensure_ascii=False,
                              allow_nan=False, separators=(',',':'))+'\n').encode()
        self.assertEqual(len(encoded), directory['actual_output_bytes'])
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), directory['actual_output_sha256'])
        self.assertEqual(directory['file_record']['output_sha256'], directory['actual_output_sha256'])
        self.assertEqual(directory['file_record']['input_sha256'], directory['actual_input_sha256'])
        self.assertEqual(directory['maximum_file_bytes'], 20_000_000)
        self.assertEqual(directory['file_record']['input_detection_count'], 2)
        self.assertEqual(directory['file_record']['output_detection_count'], 2)
        # Adding low-confidence boundary support can enlarge a tiny input.
        self.assertGreater(directory['actual_output_bytes'], directory['file_record']['input_bytes'])

    def test_failed_size_limit_does_not_publish_partial_directory(self):
        failures = self.result['failures']
        self.assertEqual(failures['zero_energy']['exception_type'], 'ValueError')
        self.assertEqual(failures['nonfinite_energy']['exception_type'], 'ValueError')
        limited = failures['one_byte_limit']
        self.assertEqual(limited['exception_type'], 'RuntimeError')
        self.assertIn('strict-32', limited['message'])
        self.assertFalse(limited['published_output_exists'])
        self.assertEqual(limited['temporary_leftovers'], [])
        with self.assertRaises(ValueError):
            strict_json('{"x":NaN}')


@unittest.skipUnless((audit.CACHE/'.git').exists(), 'fixed SAID cache missing; no network or model installation')
class OriginalExecutionTests(unittest.TestCase):
    def test_current_original_compression_module(self):
        report = audit.build_report()
        self.assertEqual(report['before'], report['after'])
        self.assertEqual(report['results']['high_confidence_output']['segmentation'], [[[0,0,1.],[2,0,.5]]])
        self.assertFalse(report['results']['failures']['one_byte_limit']['published_output_exists'])


if __name__ == '__main__':
    unittest.main()
