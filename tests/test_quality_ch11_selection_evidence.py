"""Publication controls reject mismatched physics, loss and missing-device claims."""
import copy
from pathlib import Path
import tempfile
import unittest
from scripts import make_selection_figures as figure
from scripts.quality_check import _check_selection_figure_report


class SelectionPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.report = figure.generate_figure(root/'figure.png', root/'report.json')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_current_report_passes_independent_scalar_oracle(self):
        _check_selection_figure_report(self.report)

    def test_source_rebinding_and_extra_source_are_rejected(self):
        for kind in ('missing', 'extra', 'stale', 'script', 'bool_schema'):
            r = copy.deepcopy(self.report)
            if kind == 'missing': r['source_sha256'].pop('codes/chapters/ch11/core/selection.py')
            elif kind == 'extra': r['source_sha256']['other.py'] = '0'*64
            elif kind == 'stale': r['source_sha256']['codes/chapters/ch11/core/selection.py'] = '0'*64
            elif kind == 'script': r['script_sha256'] = '0'*64
            else: r['schema_version'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _check_selection_figure_report(r)

    def test_design_loading_cannot_be_counted_as_actual_noise(self):
        r = copy.deepcopy(self.report)
        r['numerical_case']['candidates'][3]['actual_noise_power'] += 1.01
        with self.assertRaises(ValueError): _check_selection_figure_report(r)

    def test_separate_diffuse_model_and_propagation_sign_cannot_be_relabelled(self):
        for kind in ('diffuse', 'covariance', 'steering', 'weight'):
            r = copy.deepcopy(self.report); c = r['numerical_case']
            if kind == 'diffuse': c['diffuse_coherence'][0][1] += .1
            elif kind == 'covariance': c['noise_covariance_real_imag'][0][1][1] *= -1
            elif kind == 'steering': c['steering_vectors_real_imag']['actual'][0][1] *= -1
            else: c['candidates'][3]['weights_real_imag'][0][1] *= -1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _check_selection_figure_report(r)

    def test_squaring_amplitude_limit_and_changing_plot_metric_are_rejected(self):
        for kind in ('threshold', 'plot', 'nonfinite'):
            r = copy.deepcopy(self.report)
            if kind == 'threshold': r['numerical_case']['parameters']['response_amplitude_error_upper_limit'] = .03**2
            elif kind == 'plot': r['plot_data']['actual_NMSE'][1] = .075
            else: r['numerical_case']['candidates'][0]['WNG_dB'] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _check_selection_figure_report(r)

    def test_missing_latency_cannot_be_reported_as_success_or_selection(self):
        for kind in ('measured', 'verdict', 'selection'):
            r = copy.deepcopy(self.report); c = r['numerical_case']
            if kind == 'measured': c['candidates'][3]['measured_latency_ms'] = 0.
            elif kind == 'verdict': c['candidates'][3]['device_verdict'] = 'pass'
            else: c['device_selected'] = 'mvdr_1'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _check_selection_figure_report(r)

    def test_contradictory_model_and_execution_claims_are_rejected(self):
        for kind in ('scope', 'DI_scope', 'decision_scope', 'limits'):
            r = copy.deepcopy(self.report)
            if kind == 'scope': r['numerical_case']['scope'] = 'blind DOA/device benchmark verified'
            elif kind == 'DI_scope': r['plot_data']['DI_scope'] = 'DI computed from actual R_n'
            elif kind == 'decision_scope': r['plot_data']['decision_scope'] = 'all device specifications passed'
            else: r['limits'] = 'measured hardware and ASR results'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _check_selection_figure_report(r)


if __name__ == '__main__':
    unittest.main()
