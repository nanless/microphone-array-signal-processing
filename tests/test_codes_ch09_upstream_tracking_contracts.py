"""Independent small-model expectations; no download and no upstream writes."""
from fractions import Fraction
import copy
import hashlib
import json
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch09.examples import audit_upstream_tracking_contracts as audit


@unittest.skipUnless(all((audit.CACHE / name).is_dir() for name in audit.REVISIONS),
                     "optional fixed upstream caches missing; no downloads")
class FixedTrackingContracts(unittest.TestCase):
    def test_original_q_against_integral_and_outer_product(self):
        audit.verify_sources()
        result = audit.python_contracts(audit.CACHE)["filterpy_q"]
        h, q = Fraction(1, 2), Fraction(2)
        # Continuous integral of [t,1][t,1]^T vs one constant acceleration
        # sample propagated by [h²/2,h]. These independent models differ.
        expected_continuous = [[q*h**3/3, q*h**2/2], [q*h**2/2, q*h]]
        impulse = [h*h/2, h]
        expected_discrete = [[q*x*y for y in impulse] for x in impulse]
        np.testing.assert_allclose(result["continuous"], np.array(expected_continuous, float), rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["discrete"], np.array(expected_discrete, float), rtol=0, atol=1e-15)
        self.assertNotEqual(result["continuous"], result["discrete"])

    def test_jpda_one_to_one_includes_two_misses(self):
        result = audit.python_contracts(audit.CACHE)["jpda_isvalid"]
        valid = {tuple(r["assignment"]) for r in result["events"] if r["valid"]}
        expected = {(-1,-1),(-1,0),(-1,1),(-1,2),(0,-1),(1,-1),(2,-1),
                    (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)}
        self.assertEqual(valid, expected)
        self.assertEqual(result["valid_count"], 13)

    @unittest.skipUnless(shutil.which("cc"), "optional compiler unavailable")
    def test_native_mean_covariance_and_delayed_state_contracts(self):
        before = audit.verify_sources()
        result = audit.c_contracts(audit.CACHE)
        odas = dict(zip(result["odas"]["output_order"], result["odas"]["actual"]))
        # FPF^T on the x/v block: [[1+h²,h],[h,1]]; only velocity gets .2².
        for key, expected in {"F03": .5, "Q00": 0, "Q33": .04, "R00": .01,
                              "direction_x": 1, "velocity_x": 0,
                              "P00": 1.25, "P03": .5, "P33": 1.04}.items():
            self.assertAlmostEqual(odas[key], expected, delta=1e-6)
        self.assertNotAlmostEqual(odas["direction_x"], 1.5)
        saf = result["saf"]["actual"]
        self.assertEqual(saf["empty1"], [1,0,9,1,101])
        self.assertEqual(saf["empty2"], [2,0,9,1,101])
        self.assertEqual(saf["resume"], [0,3,1,12,3])
        self.assertEqual(saf["twoobs"], [0,4,3,13,1,0])
        self.assertEqual(saf["copy"][:4], [404]*4)
        self.assertEqual(saf["copy"][4:8], [.25]*4)
        self.assertIn("original prediction/update", result["saf"]["not_run"])
        self.assertEqual(audit.verify_sources(), before)

    def test_wrong_digest_rejected_before_native_execution(self):
        wrong = copy.deepcopy(audit.FILES)
        wrong["odas"]["LICENSE"] = "0"*64
        with patch.object(audit, "FILES", wrong):
            with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
                audit.verify_sources()


class ReadOnlyContracts(unittest.TestCase):
    def test_source_function_fragment_is_exact(self):
        if not (audit.CACHE / "spatial-audio-framework").is_dir():
            self.skipTest("optional SAF cache missing")
        source = audit.CACHE / "spatial-audio-framework/framework/modules/saf_tracker/saf_tracker.c"
        fragment = audit.extract_c_step(source)
        self.assertIn(fragment, source.read_text())
        self.assertIn("//resampstr", fragment)
        self.assertIn("pData->incrementTime++", fragment)
        self.assertEqual(audit.digest(fragment.encode()), hashlib.sha256(fragment.encode()).hexdigest())

    def test_strict_json_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            audit.stable_digest({"not_a_valid_result": float("nan")})


if __name__ == "__main__":
    unittest.main()
