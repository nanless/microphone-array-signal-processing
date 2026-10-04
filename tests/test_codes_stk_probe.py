"""Offline report validation; native compilation is an explicit experiment."""
import unittest
from codes.chapters.ch10.examples.run_stk_delay_probe import validate_measurements, run_probe, ROOT
import shutil


class STKProbeTest(unittest.TestCase):
    def fixture(self):
        return dict(analytic_max_abs_error=0., zero_delay_max_abs_error=0.,
                    one_delay_max_abs_error=0., chunk37_max_abs_error=0.,
                    reset_chunk37_max_abs_difference=.1)

    def test_known_fir_and_state_check_pass(self):
        validate_measurements(self.fixture())

    def test_disagreement_nonfinite_and_missing_evidence_fail(self):
        for key, value in [('analytic_max_abs_error', .01), ('one_delay_max_abs_error', float('nan')),
                           ('reset_chunk37_max_abs_difference', 0.)]:
            data = self.fixture(); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_measurements(data)
        data = self.fixture(); del data['chunk37_max_abs_error']
        with self.assertRaises(ValueError):
            validate_measurements(data)

    def test_boolean_json_values_are_not_measurements(self):
        for key in self.fixture():
            data = self.fixture()
            data[key] = key.startswith('reset_')
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_measurements(data)


class STKActualSourceTest(unittest.TestCase):
    @unittest.skipUnless((ROOT/'codes/chapters/ch00/upstream/_downloads/stk/src/DelayL.cpp').is_file()
        and shutil.which('c++'), 'optional fixed STK checkout and local C++ compiler required')
    def test_three_actual_translation_units_and_unchanged_identity(self):
        report = run_probe()
        self.assertTrue(report['source_identity']['clean_before'])
        self.assertTrue(report['source_identity']['clean_after'])
        dependencies = report['compiled_dependencies']
        self.assertEqual(len(dependencies['dependency_files']), 3)
        for name in ('src/DelayL.cpp', 'src/Stk.cpp', 'include/DelayL.h', 'include/Filter.h', 'include/Stk.h'):
            row = dependencies['original_compile_inputs']['stk/' + name]
            self.assertEqual(row['git_blob'], row['actual_blob'])
        validate_measurements(report['measurements'])


if __name__ == '__main__':
    unittest.main()
