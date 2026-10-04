"""Independent rational cases and optional original APA execution; no network."""
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch06.examples import audit_upstream_apa as audit

REPORT = audit.ROOT / 'codes/chapters/ch06/reports/upstream_apa.json'


def strict_json(text):
    def reject(constant):
        raise ValueError(constant)
    return json.loads(text, parse_constant=reject)


class APAReportTests(unittest.TestCase):
    def setUp(self):
        self.report = strict_json(REPORT.read_text())
        self.rows = self.report['results']

    def test_provenance_and_current_source(self):
        relative = str(Path(audit.__file__).resolve().relative_to(audit.ROOT))
        spec = '621d727a63475e3ee8ad2a29d518889c42e92571:' + relative
        original = (audit.contracts.git(audit.ROOT, 'show', spec) + '\n').encode()
        self.assertEqual(len(original), int(audit.contracts.git(audit.ROOT, 'cat-file', '-s', spec)))
        self.assertEqual(self.report['audit_source_sha256'], hashlib.sha256(original).hexdigest())
        verify_lock_binding(self.report['lock_sha256'], ('pyaec',), current_lock=audit.LOCK)
        entry = next(p for p in json.loads(audit.LOCK.read_text())['projects'] if p['id'] == 'pyaec')
        self.assertEqual(self.report['lock_entry'], entry)
        self.assertEqual(entry['revision'], audit.REVISION)
        self.assertEqual(entry['license'], 'Apache-2.0')
        self.assertIn(audit.SOURCE, entry['entrypoints'])
        self.assertEqual(self.report['before'], self.report['after'])
        self.assertEqual(self.report['before']['status'], '')
        self.assertEqual(self.report['before']['untracked_python'], [])
        self.assertEqual(self.report['original_files'][audit.SOURCE]['git_blob'],
                         '38080efa28eecb85dd3417a09f370931db9e9d4c')
        self.assertEqual({k:v['sha256'] for k,v in self.report['original_files'].items()},
                         audit.SOURCE_SHA)
        self.assertEqual(self.report['scope']['execution'], 'original_complete_single_file_module_and_function_call')
        self.assertIn('no AST', self.report['scope']['compatibility'])
        self.assertIn('not Ozeki/Umeda', self.report['scope']['identity'])
        date = datetime.fromisoformat(self.report['created_utc']).astimezone(
            ZoneInfo('Asia/Shanghai')).date().isoformat()
        self.assertEqual(self.report['verified_date_asia_shanghai'], date)

    def test_zero_reference_is_exact_but_tail_is_absent(self):
        row = self.rows['zero_reference']
        np.testing.assert_array_equal(row['residual']['real'], [1,2,3,4])
        self.assertEqual(row['unprocessed_tail_indices'], [4,5])
        self.assertEqual(row['classification'], 'zero_reference_tail_not_processed')
        self.assertEqual(row['warnings'], [])

    def test_scalar_order_one_exact_closed_products(self):
        # Explicit independent closed products; no audit expectation helper.
        expected = [Fraction(2), Fraction(4,101), Fraction(6,101*401),
                    Fraction(8,101*401*901)]
        np.testing.assert_allclose(self.rows['order_one']['residual']['real'],
                                   [float(v) for v in expected], atol=2e-14, rtol=0)
        self.assertGreater(self.rows['order_one']['residual']['real'][1], 0)
        self.assertEqual(self.rows['order_one']['unprocessed_tail_indices'], [4])

    def test_two_observations_hand_solution(self):
        # w1(second coefficient)=2040300/1040401 from a 2x2 adjugate.
        expected = [1, 203/101, 40502/1040401]
        np.testing.assert_allclose(self.rows['order_two']['residual']['real'],
                                   expected, atol=2e-14, rtol=0)
        self.assertEqual(self.rows['order_two']['unprocessed_tail_indices'], [3,4])

    def test_complex_warning_and_real_discard_not_valid_complex_apa(self):
        row = self.rows['complex_discard']
        self.assertEqual(row['classification'], 'complex_input_discarded')
        np.testing.assert_array_equal(row['residual']['real'], [1,1,1,1])
        self.assertTrue(row['warnings'])
        self.assertEqual(len(row['warnings']), 8)
        self.assertEqual({w['category'] for w in row['warnings']}, {'ComplexWarning'})
        self.assertEqual({w['line'] for w in row['warnings']}, {30,34})
        self.assertTrue(all('discards the imaginary part' in w['message'] for w in row['warnings']))

    def test_strict_json_and_summary(self):
        self.assertEqual(set(self.rows), {'zero_reference','order_one','order_two','complex_discard'})
        self.assertEqual(self.report['case_count'], 4)
        self.assertTrue(all(row['expected_behavior_verified'] for row in self.rows.values()))
        self.assertTrue(all(row['exception'] is None for row in self.rows.values()))
        self.assertEqual(self.report['status'], 'expected_behaviors_verified_warnings_preserved')
        for row in self.rows.values():
            digest = hashlib.sha256(json.dumps(row['inputs'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            self.assertEqual(row['input_sha256'], digest)
        with self.assertRaises(ValueError):
            strict_json('{"value":NaN}')
        encoded = audit.encode(np.array([np.nan,np.inf,-np.inf]))
        self.assertEqual([v['classification'] for v in encoded['real']],
                         ['nan','positive_infinity','negative_infinity'])
        strict_json(json.dumps(encoded, allow_nan=False))

    @unittest.skipUnless((audit.CACHE/'.git').exists(), 'Fixed upstream cache absent; saved report checks remain offline')
    def test_real_original_call_and_no_checkout_writes(self):
        current = audit.run_audit()
        self.assertEqual(current['before'], current['after'])
        self.assertEqual(current['results'], self.rows)
        for path, expected in audit.SOURCE_SHA.items():
            self.assertEqual(audit.sha(audit.CACHE/path), expected)

    def test_unavailable_checkout_fails_without_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                audit.run_audit(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
