"""Offline report validation; native compilation is an explicit experiment."""
import unittest
from codes.examples.run_stk_delay_probe import validate_measurements


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


if __name__ == '__main__':
    unittest.main()
