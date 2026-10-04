"""Offline checks: no upstream imports, downloads or optional dependencies."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from codes.chapters.ch05.examples import audit_beamformer_reference as audit
from codes.chapters.ch04.core import upstream_contracts as contracts


def historical_script():
    relative = str(Path(audit.__file__).resolve().relative_to(contracts.ROOT))
    spec = '747ec3fec96ef20c7c128cc4291fd7bb9ee36c34:' + relative
    payload = (contracts.git(contracts.ROOT, 'show', spec) + '\n').encode()
    if len(payload) != int(contracts.git(contracts.ROOT, 'cat-file', '-s', spec)):
        raise ValueError('Historical tool bytes are not exact')
    return payload


class BeamformerReferenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((audit.ROOT / "chapters/ch05/reports/beamformer_reference_audit.json").read_text())

    def test_provenance_binds_exact_source_configuration_and_script(self):
        p = self.report["provenance"]
        self.assertEqual(p["sources"], audit.SOURCES)
        self.assertEqual(p["harness_sha256"], hashlib.sha256(historical_script()).hexdigest())
        self.assertEqual(p["configuration_sha256"], audit.configuration_sha256())
        self.assertEqual(self.report["configuration"], audit.CONFIG)
        self.assertFalse(p["upstream_modified"])
        self.assertEqual(self.report["status"], "original_failures_preserved")

    def test_reference_selection_against_fixed_scalar_answers(self):
        # For Rn=I, Rs diagonal, candidate SNR equals the selected diagonal.
        # Fixed answers are independent of both the AST code and report.
        for diagonal, channel, weights in (([1., 4.], 1, [0., .8]),
                                            ([4., 1.], 0, [.8, 0.]),
                                            ([2., 2.], 0, [.5, 0.])):
            result = audit.independent_diagonal_reference(diagonal)
            self.assertEqual(result["selected_channel"], channel)
            self.assertEqual(result["weights"], weights)
            self.assertEqual(result["candidate_output_snr_linear"], diagonal)
        for invalid in ([], [0., 1.], [-1., 4.], [float("nan"), 1.], [1., 2., 3.]):
            with self.assertRaises(ValueError):
                audit.independent_diagonal_reference(invalid)

    def test_saved_ast_calls_preserve_asymmetric_failure_and_control(self):
        p = self.report["pb_bss"]
        self.assertEqual(p["execution_kind"], "unmodified_function_ast_extraction")
        self.assertFalse(p["upstream_function_modified"])
        self.assertEqual(p["status"], "failed_reference_selection")
        self.assertFalse(p["package_import_probe"]["algorithm_called"])
        self.assertEqual(p["package_import_probe"]["status"], "failed")
        self.assertIn("paderbox", p["package_import_probe"]["message"])
        self.assertEqual(len(p["results"]), 2)
        self.assertEqual(p["function_sha256"],
                         "30c4f08002dcf2695e2b8d251854ffc8d78e5bfb4d0c2e9b8927862486a79997")
        for row, weights, correct in zip(p["results"], ([[.2, 0.]], [[.8, 0.]]), (False, True)):
            np.testing.assert_allclose(row["actual_weights"], weights, rtol=0, atol=1e-15)
            self.assertEqual(row["reference_selection_correct"], correct)
            diagonal = row["target_diagonal"]
            self.assertEqual(row["analytic_reference"], audit.independent_diagonal_reference(diagonal))

    def test_saved_original_package_failure_is_bound_to_bad_statement(self):
        p = self.report["pyroomacoustics"]
        self.assertEqual(p["execution_kind"], "installed_original_package_method_call")
        self.assertEqual(p["package_version"], "0.10.0")
        self.assertEqual(p["status"], "failed_float_slice_index")
        self.assertFalse(p["filters_computed"])
        self.assertFalse(p["upstream_modified"])
        self.assertEqual(p["exception"]["type"], "TypeError")
        self.assertIn("slice indices", p["exception"]["message"])
        frame = p["exception"]["traceback"][-1]
        self.assertEqual((frame["file"], frame["line"], frame["function"]),
                         ("beamforming.py", 1375, "rake_distortionless_filters"))
        self.assertIn("H[:, L:]", frame["statement"])
        for name, value in p["installed_source_sha256"].items():
            self.assertEqual(value, audit.SOURCES["pyroomacoustics"]["files"]["pyroomacoustics/" + name])
        # Python 3 division produces a float even when the integer quotient is exact.
        with self.assertRaises(TypeError):
            np.zeros((2, 8))[:, 8 / 2:]

    def test_missing_source_cannot_silently_download_or_use_substitute(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                audit.run_experiment(Path(directory))


if __name__ == "__main__":
    unittest.main()
