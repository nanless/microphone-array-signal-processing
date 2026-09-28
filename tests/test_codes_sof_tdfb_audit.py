"""Independent mathematical expectations; no MATLAB, network or SOF execution."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from codes.chapters.ch05.examples import audit_sof_tdfb_design as audit


class SofTdfbAuditTests(unittest.TestCase):
    def test_spherical_coherence_matches_direction_integral(self):
        result = audit.independent_examples()["coherence"]
        # Isotropic 3-D directions have a uniform direction cosine on [-1, 1].
        cosine = np.linspace(-1., 1., 100001)
        wave_number = 2 * np.pi * 1000 / 343
        integral = np.trapezoid(np.exp(1j * wave_number * .05 * cosine), cosine) / 2
        self.assertAlmostEqual(integral.real, result["spherical_coherence_sin_phase_over_phase"], places=10)
        self.assertAlmostEqual(integral.imag, 0., places=14)
        self.assertAlmostEqual(result["fixed_source_expression_with_standard_sinc"], .09073897733712175, places=14)
        self.assertGreater(abs(integral.real - result["fixed_source_expression_with_standard_sinc"]), .7)

    def test_wng_denominator_must_follow_each_weight(self):
        fixture = audit.independent_examples()["wng_loop_fixture"]
        self.assertEqual(fixture["correct_wng_linear"], [2., 1.])
        self.assertEqual(fixture["stale_denominator_wng_linear"], [1., 1.])
        # The last frequency happens to agree; one passing point cannot validate the loop.
        self.assertEqual(fixture["white_noise_denominators"], [.5, 1.])

    def test_report_is_current_and_keeps_execution_boundary(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "codes/chapters/ch05/reports/sof_tdfb_design_audit.json").read_text())
        self.assertEqual(report["provenance"]["revision"], audit.REVISION)
        self.assertEqual(report["provenance"]["source_sha256"], audit.SOURCE_HASHES)
        self.assertEqual(report["provenance"]["harness_sha256"], hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        self.assertEqual(report["mathematical_examples"], audit.independent_examples())
        scope = report["execution_scope"]
        self.assertTrue(scope["source_verification"])
        self.assertFalse(scope["matlab_or_octave_design"])
        self.assertFalse(scope["firmware_or_hardware"])
        self.assertFalse(scope["upstream_modified"])

    def test_missing_source_is_not_silently_treated_as_verified(self):
        with self.assertRaises(FileNotFoundError):
            audit.verify_source(Path("/private/tmp/no-such-sof-audit-checkout"))


if __name__ == "__main__":
    unittest.main()
