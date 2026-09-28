"""Independent binary exponent boundary; no optional upstream dependency."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from codes.chapters.ch07.examples.wpe_silence_boundary import silence_probe,classify_scalar
from codes.chapters.ch07.examples.compare_online_wpe_reference import NumpyOnlineWPE011,NARA_WPE_MODULE_SHA256


class WPESilenceBoundary(unittest.TestCase):
    def test_zero_history_inverse_covariance_exact_powers_of_two(self):
        state = NumpyOnlineWPE011(taps=1,delay=1,alpha=.5,frequency_bins=1,channels=1)
        for n in range(1,11):
            np.testing.assert_array_equal(state.step_frame(np.zeros((1,1),complex)),[[0]])
            np.testing.assert_array_equal(state.inverse_covariance,[[[2**n]]])
            np.testing.assert_array_equal(state.filter_taps,[[[0]]])

    def test_binary64_overflow_at_1024_and_resumption_failure(self):
        r = silence_probe(.5,max_frames=1100)
        self.assertEqual(r['first_nonfinite_state_frame'],1024)
        self.assertEqual(r['last_pre_step_covariance']['real'],float(2**1023))
        self.assertFalse(r['final_zero_frame_covariance']['finite'])
        self.assertEqual(r['final_zero_frame_output']['real'],0)
        self.assertEqual(r['resumed_unit_frame_outputs'][0]['real'],1)
        self.assertEqual(r['first_nonfinite_output_resume_frame'],2)
        json.dumps(r,allow_nan=False)

    def test_no_forgetting_stays_finite_and_bad_parameters_rejected(self):
        r = silence_probe(1,max_frames=20)
        self.assertIsNone(r['first_nonfinite_state_frame'])
        self.assertEqual(r['final_zero_frame_covariance']['real'],1.)
        for value in (True,0,-1,1.01,1j,[.5],np.inf):
            with self.assertRaises(ValueError):silence_probe(value,max_frames=1)
        for value in (False,0,1.5):
            with self.assertRaises(ValueError):silence_probe(.5,max_frames=value)
        self.assertEqual(classify_scalar(np.nan)['value_class'],'nan')
        self.assertEqual(classify_scalar(np.inf)['value_class'],'infinity')

    def test_committed_report_binds_generator_and_locked_source(self):
        root = Path(__file__).resolve().parents[1]
        r = json.loads((root/'codes/chapters/ch07/reports/chapter07_online_wpe_silence.json').read_text())
        self.assertEqual(r['source']['installed_module_sha256'],NARA_WPE_MODULE_SHA256)
        self.assertEqual(r['source']['upstream_execution'],'executed_after_hash_verification')
        for path,digest in r['generator_inputs'].items():
            self.assertEqual(hashlib.sha256((root/path).read_bytes()).hexdigest(),digest)
        self.assertEqual(len(r['records']),4)
        for record in r['records']:
            self.assertEqual(record['first_nonfinite_state_frame'],1024 if record['alpha']==.5 else 13838)
        json.dumps(r,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
