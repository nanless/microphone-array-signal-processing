"""Independent field semantics and range checks; no device timing claim."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from codes.chapters.ch10.core.engineering import TELEMETRY_FIELDS, validate_telemetry

ROOT = Path(__file__).resolve().parents[1]


def record():
    return dict(timestamp_ns=0, frame_index=0, sample_rate_hz=16000,
                queue_depth=0, xrun_count=0, dropped_samples=0,
                clipping_fraction=0., sro_ppm=0., rtf=91/110,
                vad_active=False, deadline_miss=False, agc_gain=1.,
                model_version='known-scope')


class TelemetryContractTests(unittest.TestCase):
    def test_optional_single_frame_does_not_replace_aggregate(self):
        # 10ms/1ms then 100ms/90ms: sums give 91/110, mean ratios .5.
        aggregate = (1+90)/(10+100)
        mean_ratios = ((1/10)+(90/100))/2
        self.assertEqual(aggregate, 91/110)
        self.assertEqual(mean_ratios, .5)
        self.assertNotEqual(aggregate, mean_ratios)
        self.assertEqual(validate_telemetry(record()), [])
        self.assertEqual(validate_telemetry(record() | {'frame_service_rtf': .9}), [])
        self.assertEqual(validate_telemetry(record() | {'frame_service_rtf': 0}), [])
        self.assertEqual(validate_telemetry(record() | {'frame_service_rtf': np.float64(.9)}), [])

    def test_schema_preserves_required_fields_and_distinct_semantics(self):
        schema = json.loads((ROOT/'codes/chapters/ch10/engineering/telemetry_schema.json').read_text())
        self.assertEqual(set(schema['required']), set(TELEMETRY_FIELDS))
        self.assertNotIn('frame_service_rtf', schema['required'])
        self.assertIn('Cumulative', schema['properties']['rtf']['description'])
        field = schema['properties']['frame_service_rtf']
        self.assertEqual((field['type'], field['minimum']), ('number', 0))
        self.assertIn('distinct from cumulative', field['description'])

    def test_optional_field_rejects_bool_string_complex_and_objects(self):
        for value in (True, np.bool_(False), '0.9', .9+0j, object(), None):
            with self.subTest(value=repr(value)):
                self.assertIn('wrong type for frame_service_rtf',
                              validate_telemetry(record() | {'frame_service_rtf': value}))

    def test_optional_field_rejects_negative_nonfinite_and_huge_integer(self):
        for value in (math.inf, -math.inf, math.nan, 10**400):
            with self.subTest(value=repr(value)):
                self.assertIn('frame_service_rtf must be finite',
                              validate_telemetry(record() | {'frame_service_rtf': value}))
        self.assertIn('frame_service_rtf must be non-negative',
                      validate_telemetry(record() | {'frame_service_rtf': -.1}))


if __name__ == '__main__':
    unittest.main()
