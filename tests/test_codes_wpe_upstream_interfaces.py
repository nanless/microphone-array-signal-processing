"""Offline report checks with independent rational regressions; no upstream needed."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from codes.chapters.ch07.examples import audit_wpe_upstream_interfaces as audit
from tests.test_codes_ch07_source_contracts import old_bytes

ROOT = Path(__file__).resolve().parents[1]


class WPEInterfaceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'codes/chapters/ch07/reports/wpe_upstream_interfaces.json').read_text())

    def test_report_bound_to_harness_sources_inputs(self):
        r = self.report
        self.assertEqual(r['harness_sha256'], hashlib.sha256(old_bytes(audit.__file__)).hexdigest())
        self.assertEqual(r['source_config_sha256'], audit.binding_sha256())
        self.assertEqual(r['config'], audit.CONFIG)
        self.assertEqual(r['sources'], audit.SOURCES)
        self.assertFalse(r['environment']['bytecode_writes'])

    def test_floor_counterexample_against_rational_regression(self):
        # Solve sum w*x*(y-g*x)=0 using exact fractions, independently of NARA.
        y = [Q(x) for x in [1, 3, 2, 1, 5, 2]]
        weights = [1 / v**2 for v in y[1:]]
        g = sum(w*x*z for w,x,z in zip(weights,y[:-1],y[1:])) / sum(w*x*x for w,x in zip(weights,y[:-1]))
        self.assertEqual(g, Q(2940,5693))
        uniform = sum(x*z for x,z in zip(y[:-1],y[1:])) / sum(x*x for x in y[:-1])
        self.assertEqual(uniform,Q(13,20))
        for method, coefficient in [('wpe_v6',uniform),('wpe_v7',uniform),('wpe_v8',g)]:
            expected = [float(y[0])] + [float(z-coefficient*x) for x,z in zip(y[:-1],y[1:])]
            got = self.report['nara']['variants'][method]
            np.testing.assert_allclose(np.array(got['output_real'])[1,0]/1e-8, expected,rtol=2e-14,atol=2e-14)
            self.assertTrue(got['finite'])
            self.assertEqual(np.count_nonzero(got['output_imag']),0)
        self.assertTrue(self.report['nara']['default_alias_is_v7'])

    def test_inverse_power_floor_has_global_scope(self):
        # max input power=16, floor=16e-10; all six weak-frequency powers below it.
        np.testing.assert_array_equal(self.report['nara']['inverse_power_batched'][1], [625000000.]*6)
        p = np.array([1,3,2,1,5,2],dtype=float)**2*1e-16
        np.testing.assert_allclose(self.report['nara']['inverse_power_frequency1_alone'],1/p)

    def test_helper_smooths_frequency_and_reports_tuple_failure(self):
        expected = [Q(i*i+(i+6)**2,2) for i in range(1,7)]
        self.assertEqual(self.report['nara']['get_power_context1'], [[float(v) for v in expected]]*2)
        self.assertEqual(self.report['nara']['get_power_context_0_2']['exception_type'],'ValueError')

    def test_ast_check_distinguishes_read_controls_and_discarded_returns(self):
        dnn = '''class Model:
 def forward(self):
  return wpe_one_iteration(x,p,taps=self.taps,diagonal_loading=self.diagonal_loading)
'''
        masks = '''def f(mask):
 mask = mask.masked_fill(pad,0)
 mask.masked_fill_(pad,0)
 mask.masked_fill(pad,0)
'''
        low = '''def get_filter_matrix_conj(r,eps=1e-8):
 return solve(r)
'''
        r = audit.inspect_espnet(dnn,masks,low)
        self.assertEqual(r['public_controls_not_read_in_forward'],['diag_eps','use_torch_solver'])
        self.assertEqual(r['discarded_masked_fill_lines'],[4])
        self.assertEqual(r['inverse_call_lines'],[])
        self.assertEqual(r['solver_defaults'],{'eps':'1e-08'})

    def test_real_static_results_are_not_framework_claims(self):
        r=self.report['espnet']
        self.assertEqual(r['execution_kind'],'static_ast_only')
        self.assertFalse(r['framework_or_network_executed'])
        self.assertEqual(r['public_controls_not_read_in_forward'],['diagonal_loading','diag_eps','use_torch_solver'])
        self.assertEqual(r['discarded_masked_fill_lines'],[82])
        self.assertEqual(r['wpe_call_keywords'],['delay','inverse_power','taps'])
        self.assertEqual(r['solver_defaults'],{'eps':'1e-10'})

    def test_metaaf_window_acceptance_is_scoped_to_transform_stubs(self):
        r = self.report['metaaf_nara_wrapper']
        self.assertFalse(r['real_stft_or_istft_executed'])
        self.assertFalse(r['jax_or_metaaf_package_executed'])
        self.assertEqual(r['status'],'window_input_supported_by_actual_state_implementation')
        self.assertEqual(r['max_difference_from_original_single_frame_calls'],0.)
        got = np.array(r['captured_spectra_real'])
        source = np.arange(1.,73.).reshape(8,9,1)
        np.testing.assert_array_equal(got[:4],source[:4])
        self.assertGreater(float(np.max(np.abs(got[4:]-source[4:]))),1.)
        self.assertTrue(r['finite'])

    def test_function_extraction_does_not_execute_module_import_or_main(self):
        text = 'raise RuntimeError("do not execute module")\ndef f(x):\n return helper(x)+1\n'
        f = audit.extract_function(text,'f',{'helper':lambda x:x*2})
        self.assertEqual(f(3),7)


if __name__ == '__main__':
    unittest.main()
