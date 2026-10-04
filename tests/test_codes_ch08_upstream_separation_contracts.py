"""Independent fixed contract expectations; no downloads or optional frameworks."""
import hashlib
import json
import hashlib
import subprocess
from pathlib import Path
import tempfile
import unittest

import numpy as np

from codes.chapters.ch00.core.source_history import verify_lock_binding
from codes.chapters.ch08.examples import audit_upstream_separation_contracts as audit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'codes/chapters/ch08/reports/upstream_separation_contracts.json'


def decode(record):
    return np.array(record['real']) + 1j*np.array(record['imag'])


class SeparationContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def reject(value):
            raise ValueError(value)
        cls.r = json.loads(REPORT.read_text(), parse_constant=reject)

    def test_final_tool_lock_legacy_and_fixed_sources_are_bound(self):
        r = self.r
        self.assertEqual(r['tool_sha256'], hashlib.sha256(subprocess.check_output(['git', '-C', str(ROOT), 'show', '7b80fab:'+str(Path(audit.__file__).relative_to(ROOT))])).hexdigest())
        verify_lock_binding(r['source_lock_sha256'], tuple(audit.SOURCES), current_lock=audit.LOCK)
        self.assertEqual(r['legacy_helper_sha256'], hashlib.sha256(subprocess.check_output(['git', '-C', str(ROOT), 'show', '7b80fab:'+str(audit.LEGACY.relative_to(ROOT))])).hexdigest())
        self.assertEqual(r['sources'], audit.SOURCES)
        self.assertEqual(r['source_contract_sha256'], audit.digest(audit.SOURCES))
        for name, spec in audit.SOURCES.items():
            row = r['verification'][name]
            self.assertEqual(row['before'], row['after'])
            self.assertEqual(row['before']['head'], spec['revision'])
            self.assertEqual(row['before']['tracked_status'], '')
            self.assertEqual(row['before']['untracked_python'], [])
            self.assertEqual(set(row['files']), set(spec['files']))
            for filename, digest in spec['files'].items():
                self.assertEqual(row['files'][filename]['sha256'], digest)
                self.assertEqual(row['files'][filename]['git_blob_sha256'], digest)
                self.assertRegex(row['files'][filename]['git_blob_id'], r'^[0-9a-f]{40}$')

    def test_arraydps_actual_entrypoint_truth_and_parameter_layers(self):
        a = self.r['static']['arraydps']
        self.assertEqual(a['imports'][0]['module'], 'src.sampler_spatial_v1_reverb_iva_8kHz')
        self.assertIn('separate', a['sampler_symbols'])
        self.assertNotIn('separate', a['generic_sampler_symbols'])
        self.assertTrue(any('sources[:, 0, :]' in s for call in a['truth_metric_calls']
                            for s in call['arguments']))
        self.assertTrue(any('max_snr' in s['test'] and 'snr_stop' in s['test']
                            for s in a['stopping_conditions']))
        p = a['parameter_layers']
        self.assertEqual(a['paper_configuration']['hop_length'], 64)
        self.assertEqual(p['class_defaults']['hop_length'], 128)
        self.assertEqual(p['CLI_defaults']['hop_length'], 128)
        self.assertEqual(p['README_recipe']['hop_length'], '128')
        self.assertEqual([p['CLI_defaults'][k] for k in ['max_trials','max_trials2','max_trials3']], [2,3,4])
        self.assertEqual([p['README_recipe'][k] for k in ['max_trials','max_trials2','max_trials3']], ['5','5','5'])
        self.assertFalse(a['torch_weights_sampling_or_SDR_function_executed'])

    def test_gss_power_SCM_independent_scalar_outer_products(self):
        rows = self.r['executed']['gss_original_power_SCM']['cases']
        self.assertEqual([r['case'] for r in rows],
                         ['power','direction','double_amplitude','below_mask_floor','empty_mask'])
        for row in rows:
            x, mask = decode(row['observation'])[0], np.array(row['mask'])[0]
            den = max(float(sum(mask)), 1e-10)
            oracle = np.array([[sum(float(mask[t])*complex(x[d,t])*complex(x[e,t]).conjugate()
                                     for t in range(3))/den for e in range(2)] for d in range(2)])
            np.testing.assert_allclose(decode(row['observed'])[0], oracle, atol=1e-12, rtol=0)
            self.assertTrue(row['mask_unchanged'])
        np.testing.assert_array_equal(decode(rows[0]['observed'])[0], [[2.5,0],[0,0]])
        np.testing.assert_array_equal(decode(rows[1]['observed'])[0], [[1,0],[0,0]])
        np.testing.assert_array_equal(decode(rows[2]['observed'])[0], [[10,0],[0,0]])
        self.assertAlmostEqual(decode(rows[3]['observed'])[0,0,0].real, .05)

    def test_nemo_is_static_and_calculation_is_not_original_execution(self):
        n = self.r['static']['nemo_gss']
        self.assertEqual(n['forward_arguments'], ['self','input','activity'])
        self.assertEqual(n['returned_expressions'], ['gamma'])
        self.assertEqual(n['component_count_assignment'], ['num_outputs = activity.size(1)'])
        self.assertFalse(n['background_component_added_by_this_class'])
        self.assertFalse(n['torch_or_NeMo_executed'])
        c = self.r['book_calculations']['nemo_mask_arithmetic']
        self.assertIn('not_upstream_execution', c['execution_kind'])
        # Rational weights 3/10 and 1/2, total 4/5: no Torch calculation used.
        np.testing.assert_allclose(c['cases'][0]['observed_book_calculation'],
                                   [0, .3/(.8+1e-8), .5/(.8+1e-8)], rtol=1e-14)
        self.assertLess(c['cases'][0]['sum'], 1.)
        self.assertEqual(c['cases'][1]['observed_book_calculation'], [0,0,0])
        self.assertEqual(c['cases'][1]['active_support_oracle'], [0,.375,.625])

    def test_demo_main_pipeline_and_method_identities_are_separate(self):
        s = self.r['static']
        demo = s['notsofar_independent_demo']
        self.assertIn('device', demo['constructor_arguments'])
        self.assertNotIn('device_id', demo['constructor_arguments'])
        self.assertIn('device_id', demo['calls'][0]['keywords'])
        self.assertFalse(demo['main_css_pipeline_executed_or_implied_failed'])
        self.assertEqual(s['mamba_identity']['Mamba_TasNet_paper'], 'https://arxiv.org/abs/2407.09732')
        self.assertEqual(s['mamba_identity']['Dual_Path_Mamba_paper'], 'https://arxiv.org/abs/2403.18257')
        calls = s['gss_observation_chain']['ordered_relevant_calls']
        self.assertEqual([c['source'].split('(')[0] for c in calls],
                         ['self.wpe_block','self.gss_block','self.bf_block'])
        self.assertTrue(calls[-1]['source'].startswith('self.bf_block(Obs[:, :, st:en]'))

    def test_actual_pra_methods_and_failed_optional_import_not_relabelled(self):
        e = self.r['executed']
        p = e['pra_original_methods']
        self.assertEqual(p['auxiva_eigen_initialization']['exception_type'], 'ValueError')
        self.assertEqual(p['package_version'], '0.10.0')
        x = decode(p['input'])
        for row in p['fastmnmf'].values():
            # Independently sum the three Wiener source images.
            z = decode(row['output'])
            np.testing.assert_allclose(sum(z[:,:,:,k] for k in range(3)), x.transpose(2,0,1), atol=3e-14)
            np.testing.assert_array_equal(decode(row['caller_W0_after']), np.tile(np.eye(2),(3,1,1)))
        a = self.r['attempted']['ssspy_cacgmm_package']
        self.assertFalse(a['dependencies_installed'])
        self.assertFalse(a['source_patched'])
        if a['status'] == 'failed':
            self.assertEqual(a['exception_type'], 'ModuleNotFoundError')
            self.assertEqual(a['message'], "No module named 'packaging'")
        else:
            self.assertEqual(a['status'], 'executed')
            self.assertEqual(a['n_iter_for_separation_quality'], 0)
            self.assertLess(a['zero_iterations']['mask_reference_error'], 1e-12)
            np.testing.assert_allclose(a['shape_step']['observed_real'],
                [[[[1.5,0],[0,.5]]],[[[.5,0],[0,1.5]]]], atol=1e-12)
        self.assertFalse(e['gss_original_power_SCM']['CuPy_GSS_package_or_beamformer_chain_executed'])

    def test_contract_helpers_and_missing_checkout_reject(self):
        fixture='class A:\n    def f(self, n=7):\n        return n\n'
        self.assertEqual(audit.defaults(audit.method(fixture,'A.f')), {'n':7})
        info=audit.snippet(fixture,'A.f')
        self.assertEqual(info['line'],2)
        self.assertEqual(info['segment_sha256'],hashlib.sha256(b'def f(self, n=7):\n        return n').hexdigest())
        self.assertEqual(audit.cli_defaults("parser.add_argument('--n',default=3)"), {'n':3})
        self.assertEqual(audit.readme_recipe('python separate.py \\\n --n 8\n```'), {'n':'8'})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                audit.checkout_state(Path(d),'a'*40)
        with self.assertRaises(ValueError):
            audit.digest({'bad':float('nan')})


if __name__ == '__main__':
    unittest.main()
