"""Figure61 provenance and geometry are checked independently of plotting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import quality_check as quality

ROOT = Path(__file__).resolve().parents[1]


class TrackingInformationReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)/'report.json'
        self.original = json.loads((ROOT/'codes/chapters/ch09/reports/figure61_tracking_information.json').read_text())

    def check(self, report):
        self.path.write_text(json.dumps(report))
        quality._check_tracking_information_report(self.path)

    def test_actual_report_and_original_png_have_current_same_script(self):
        self.check(self.original)
        self.assertEqual(quality.png_provenance_issues(ROOT/'figures/fig61_tracking_information.png',
                         ROOT/'scripts/make_figures.py'), [])

    def test_wrong_source_or_receiver_time_scope_is_rejected(self):
        for field, value in (('script_sha256', '0'*64), ('scope', 'receiver-clock audio equivalence')):
            with self.subTest(field=field):
                report = copy.deepcopy(self.original); report[field] = value
                with self.assertRaises(ValueError): self.check(report)

    def test_false_variance_boolean_sampling_and_nan_are_rejected(self):
        for field, value in (('posterior_variance', .1), ('rho', False), ('posterior_mean', float('nan'))):
            with self.subTest(field=field):
                report = copy.deepcopy(self.original); report['same_scalar_state'][field][37] = value
                with self.assertRaises(ValueError): self.check(report)

    def test_wrong_bearing_or_null_direction_is_rejected(self):
        for field, index, value in (('bearing_deg', 50, 46), ('scale_null_direction', 0, 2)):
            with self.subTest(field=field):
                report = copy.deepcopy(self.original); report[field][index] = value
                with self.assertRaises(ValueError): self.check(report)
