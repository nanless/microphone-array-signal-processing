"""Figure62 must retain exact recurrence and distinguish response from its bound."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import quality_check as quality


class EngineeringFigureReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root/'scripts').mkdir()
        script = root/'scripts/make_figures.py'
        script.write_text('# fixed source fixture\n')
        patcher = patch.object(quality, 'ROOT', root)
        patcher.start(); self.addCleanup(patcher.stop)
        self.path = root/'report.json'
        fields = ['indicator', 'speech_probability', 'retention', 'soft_noise',
                  'ungated_noise', 'hard_noise']
        rows = [(1,'1/2','9/10','9/5','13/5','1'),
                (1,'3/4','19/20','54/25','97/25','1'),
                (0,'3/8','7/8','603/200','613/125','13/5')]
        self.report = {
            'schema_version': 1,
            'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
            'soft_update': {'power': [9,9,9], 'initial_noise': 1,
                'initial_probability': 0, 'alpha_p': '1/2', 'alpha_d': '4/5',
                'rows': [dict(zip(fields, row)) for row in rows]},
            'finite_tail': {'sample_rate_hz': 10, 'not_rt20': 'only 11.76dB',
                'time_s': [0,.1,.2,.3], 'squared_impulse': [1,.5,.25,.125],
                'reverse_energy': [15/8,7/8,3/8,1/8],
                'finite_db': [10*math.log10(x/15) for x in [15,7,3,1]],
                'infinite_db': [-10*n*math.log10(2) for n in range(4)],
                'infinite_t60_s': .6/math.log10(2),
                'four_point_endpoint_extrapolation_s': 1.8/math.log10(15)},
            'nonpreemptive': {'B_start_ms': -1, 'B_finish_ms': 14,
                'A_release_ms': 0, 'A_deadline_ms': 10, 'A_start_ms': 14,
                'A_finish_ms': 16, 'A_response_ms': 16,
                'blocking_supremum_plus_service_ms': 17, 'utilization': .35}}

    def write(self, report):
        self.path.write_text(json.dumps(report, allow_nan=False))
        return self.path.read_bytes(), self.path.stat().st_mtime_ns

    def test_independent_report_is_checked_without_writes(self):
        before = self.write(self.report)
        quality._check_engineering_limits_report(self.path)
        self.assertEqual(before, (self.path.read_bytes(), self.path.stat().st_mtime_ns))

    def test_wrong_fractions_tail_response_types_and_source_fail(self):
        changes = [
            lambda r: r.update(schema_version=True),
            lambda r: r.update(script_sha256='0'*64),
            lambda r: r['soft_update'].update(initial_probability=False),
            lambda r: r['soft_update']['rows'][0].update(indicator=True),
            lambda r: r['soft_update']['rows'][2].update(soft_noise='613/125'),
            lambda r: r['finite_tail']['reverse_energy'].__setitem__(3, .25),
            lambda r: r['finite_tail']['finite_db'].pop(),
            lambda r: r['finite_tail'].update(infinite_t60_s=True),
            lambda r: r['finite_tail'].update(not_rt20=''),
            lambda r: r['nonpreemptive'].update(A_response_ms=17),
            lambda r: r['nonpreemptive'].update(A_release_ms=False)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = copy.deepcopy(self.report); change(report)
                before = self.write(report)
                with self.assertRaises(ValueError):
                    quality._check_engineering_limits_report(self.path)
                self.assertEqual(before, (self.path.read_bytes(), self.path.stat().st_mtime_ns))


if __name__ == '__main__':
    unittest.main()
