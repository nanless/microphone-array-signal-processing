"""Independent integer/rational WPE contracts and static source negative cases."""
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch07.examples import audit_upstream_wpe_contracts as audit
from tests.test_codes_ch07_source_contracts import old_bytes
import hashlib

REPORT = audit.ROOT/'codes/chapters/ch07/reports/upstream_wpe_contracts.json'
NEMO = audit.CACHE/'nemo_wpe/nemo/collections/audio'


def strict_json(text):
    def reject(value):
        raise ValueError(value)
    return json.loads(text, parse_constant=reject)


class BookCalculationTests(unittest.TestCase):
    def setUp(self):
        self.result = audit.independent_numpy_examples()

    def test_amplitude_and_power_masks_are_different(self):
        row = self.result['mask_example']
        self.assertEqual(row['masked_power'], 1)
        self.assertEqual(row['power_mask_interpretation'], 2)
        self.assertEqual(row['unmasked_power'], 4)
        self.assertIn('not_nemo_execution', self.result['execution_kind'])

    def test_first_unit_power_solve_has_explicit_integer_sums(self):
        row = self.result['iteration_example']
        # Sum u*y = 2+6+12 = 20; sum u*u = 1+4+9 = 14.
        self.assertAlmostEqual(row['first_coefficient'], 20/14)
        np.testing.assert_allclose(row['first_output'], [1,4/7,1/7,-2/7], atol=1e-14, rtol=0)
        self.assertEqual(row['initial_power'], [1,1,1,1])
        self.assertEqual(row['eps'], 0)

    def test_second_solve_uses_independent_rational_expectations(self):
        row = self.result['iteration_example']
        self.assertAlmostEqual(row['fixed_observation_coefficient'], float(Fraction(146,101)))
        self.assertAlmostEqual(row['residual_observation_coefficient'], float(Fraction(28,103)))
        np.testing.assert_allclose(row['fixed_observation_output'],
                                  [1,56/101,11/101,-34/101], atol=1e-14, rtol=0)
        np.testing.assert_allclose(row['residual_observation_output'],
                                  [1,216/721,-9/721,-234/721], atol=1e-14, rtol=0)
        self.assertGreater(np.linalg.norm(np.asarray(row['fixed_observation_output'])-
                                         row['residual_observation_output']), .2)
        self.assertTrue(self.result['expected_behaviors_verified'])

    def test_strict_json_rejects_nonfinite_numbers(self):
        strict_json(json.dumps(self.result, allow_nan=False))
        for token in ('NaN','Infinity','-Infinity'):
            with self.assertRaises(ValueError):
                strict_json('{"x":'+token+'}')


@unittest.skipUnless((NEMO/'modules/masking.py').exists(), 'Fixed NeMo source cache absent; no download is attempted')
class StaticSourceTests(unittest.TestCase):
    def setUp(self):
        self.mask = (NEMO/'modules/masking.py').read_text()
        self.multi = (NEMO/'parts/submodules/multichannel.py').read_text()
        self.model = (NEMO/'models/enhancement.py').read_text()

    def test_fixed_original_ast_has_twelve_bounded_facts(self):
        row = audit.inspect_nemo(self.mask, self.multi, self.model)
        self.assertEqual(len(row['facts']), 12)
        self.assertTrue(row['all_facts_verified'])
        self.assertFalse(row['framework_executed'])
        self.assertEqual(row['snippets']['mask_forward']['start_line'], 1028)

    def test_power_mask_mutation_is_detected(self):
        changed = self.mask.replace('power = magnitude**2', 'power = magnitude', 1)
        row = audit.inspect_nemo(changed, self.multi, self.model)
        self.assertFalse(row['facts']['power_is_magnitude_squared'])

    def test_fixed_input_mutation_is_detected(self):
        changed = self.mask.replace('self.filter(input=output,', 'self.filter(input=input,', 1)
        row = audit.inspect_nemo(changed, self.multi, self.model)
        self.assertFalse(row['facts']['previous_output_is_next_regression_input'])

    def test_loading_and_length_mutations_are_detected(self):
        changed = self.multi.replace('self.diag_reg * torch.diagonal', 'self.diag_reg / 2 * torch.diagonal', 1)
        row = audit.inspect_nemo(self.mask, changed, self.model)
        self.assertFalse(row['facts']['trace_loading_without_dimension_division'])
        changed = self.multi.replace('weight = weight.masked_fill(length_mask, 0.0)', 'weight = weight', 1)
        row = audit.inspect_nemo(self.mask, changed, self.model)
        self.assertFalse(row['facts']['length_masks_weights_before_correlations'])

    def test_unrelated_function_cannot_prove_selected_method(self):
        changed = self.mask.replace('power = magnitude**2', 'power = magnitude', 1)
        changed += '\ndef unrelated():\n    power = magnitude ** 2\n'
        row = audit.inspect_nemo(changed, self.multi, self.model)
        self.assertFalse(row['facts']['power_is_magnitude_squared'])


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.lock = {'projects': [{'id': p, 'revision': s['revision'], 'license': s['license']}
                                  for p,s in audit.SOURCES.items()]}
        self.status = {'lock_sha256': 'selected_lock', 'projects': [
            {'id': p, 'revision': s['revision'], 'status': 'source_verified',
             'missing_entrypoints': [], 'source_selection_verified': True}
            for p,s in audit.SOURCES.items()]}

    def test_current_source_verified_is_not_method_execution(self):
        entries, states = audit.verify_acquisition(self.lock, self.status, 'selected_lock')
        self.assertEqual(set(entries), {'nara_wpe','nemo_wpe'})
        self.assertNotIn('execution', states['nemo_wpe'])

    def test_stale_status_and_failed_acquisition_rejected(self):
        with self.assertRaises(RuntimeError):
            audit.verify_acquisition(self.lock, self.status, 'another_lock')
        self.status['projects'][1]['status'] = 'failed'
        with self.assertRaises(RuntimeError):
            audit.verify_acquisition(self.lock, self.status, 'selected_lock')

    def test_revision_and_ambiguous_entries_rejected(self):
        self.lock['projects'][0]['revision'] = 'other_commit'
        with self.assertRaises(RuntimeError):
            audit.verify_acquisition(self.lock, self.status, 'selected_lock')
        self.lock['projects'][0]['revision'] = audit.SOURCES['nara_wpe']['revision']
        self.lock['projects'].append(self.lock['projects'][1])
        with self.assertRaises(RuntimeError):
            audit.verify_acquisition(self.lock, self.status, 'selected_lock')

    def test_missing_checkout_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                audit.checkout_state(directory, audit.SOURCES['nara_wpe']['revision'])
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless((audit.CACHE/'nara_wpe/.git').exists() and (audit.CACHE/'nemo_wpe/.git').exists(),
                         'Existing fixed source checkouts required')
    def test_changed_required_source_hash_is_rejected_before_execution(self):
        altered = json.loads(json.dumps(audit.SOURCES))
        altered['nara_wpe']['files']['nara_wpe/wpe.py'] = '0'*64
        with patch.object(audit, 'SOURCES', altered), self.assertRaisesRegex(RuntimeError, 'hash/blob differs'):
            audit.run_audit()


@unittest.skipUnless(REPORT.exists(), 'Historical report unavailable')
class SavedReportTests(unittest.TestCase):
    def setUp(self):
        self.report = strict_json(REPORT.read_text())

    def test_report_binds_real_sources_and_scope(self):
        r = self.report
        self.assertEqual(r['audit_source_sha256'], hashlib.sha256(old_bytes(audit.__file__)).hexdigest())
        binding = verify_lock_binding(r['lock_sha256'], tuple(audit.SOURCES), current_lock=audit.LOCK)
        self.assertEqual(r['lock_entries'], binding['records'])
        self.assertEqual(r['before'], r['after'])
        for project, spec in audit.SOURCES.items():
            self.assertEqual(r['lock_entries'][project]['revision'], spec['revision'])
            self.assertEqual(r['before'][project]['tracked_status'], '')
            self.assertEqual(r['before'][project]['untracked_python'], [])
            self.assertEqual({p:v['sha256'] for p,v in r['original_files'][project].items()}, spec['files'])
        self.assertFalse(r['nemo']['framework_executed'])
        self.assertTrue(r['nemo']['all_facts_verified'])
        self.assertEqual(r['nara']['warnings'], [])
        self.assertEqual(r['verified_date_asia_shanghai'], datetime.fromisoformat(r['created_utc']).astimezone(
            ZoneInfo('Asia/Shanghai')).date().isoformat())

    def test_labeled_history_and_outputs_have_independent_expectations(self):
        r = self.report['nara']
        self.assertEqual(r['offline_window'], [50,500,40,400])
        self.assertEqual(r['state_window'], [[20,30,200,300]])
        self.assertEqual(r['stateless_window'], [[40,30,400,300]])
        self.assertEqual(r['stateless_output'], [[-2440,600]])
        self.assertEqual(r['class_step_output'], [[-1820,600]])
        self.assertEqual(r['class_prediction'], r['class_step_output'])
        self.assertEqual(r['zero_input']['output'], [[0,0]])
        np.testing.assert_array_equal(r['zero_input']['inverse_covariance'], 2*np.eye(4)[None])
        self.assertEqual(r['wrong_frame_shape_exception']['type'], 'AssertionError')
        self.assertTrue(r['expected_behaviors_verified'])
        self.assertEqual(r['input_sha256'], audit.digest(r['inputs']))

    @unittest.skipUnless((audit.CACHE/'nara_wpe/.git').exists() and (audit.CACHE/'nemo_wpe/.git').exists(),
                         'Existing fixed checkouts required; saved report remains checkable offline')
    def test_real_original_methods_reproduce_report_without_source_writes(self):
        current = audit.run_audit()
        self.assertEqual(current['nara'], self.report['nara'])
        self.assertEqual(current['nemo'], self.report['nemo'])
        self.assertEqual(current['book_examples'], self.report['book_examples'])
        self.assertEqual(current['before'], current['after'])

    def test_default_main_prints_and_only_explicit_report_writes(self):
        # CLI behavior isolated from upstream results; real source execution
        # is covered independently above.
        with patch.object(audit, 'run_audit', return_value={'status': 'verified_fixture'}), \
             patch('sys.argv', ['audit']), patch('builtins.print') as output:
            self.assertEqual(audit.main(), 0)
            output.assert_called_once()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'report.json'
            with patch.object(audit, 'run_audit', return_value={'status': 'verified_fixture'}), \
                 patch('sys.argv', ['audit', '--report', str(target)]):
                self.assertEqual(audit.main(), 0)
            self.assertEqual(strict_json(target.read_text()), {'status': 'verified_fixture'})


if __name__ == '__main__':
    unittest.main()
