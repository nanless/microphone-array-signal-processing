"""Offline arithmetic/report checks; no upstream checkout or optional package."""
import json
from pathlib import Path
import unittest

import numpy as np

from codes.chapters.ch08.examples import audit_separation_upstream_interfaces as audit

ROOT = Path(__file__).resolve().parents[1]


def complex_array(record):
    return np.array(record["real"]) + 1j*np.array(record["imag"])


class SeparationUpstreamReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "codes/chapters/ch08/reports/separation_upstream_interfaces.json").read_text(),
                               parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

    def test_report_is_bound_to_harness_source_and_config(self):
        r = self.report
        self.assertEqual(r["harness_sha256"], audit.sha256(audit.__file__))
        self.assertEqual(r["source_config_sha256"], audit.binding_sha256())
        self.assertEqual(r["sources"], audit.SOURCES)
        self.assertEqual(r["config"], audit.CONFIG)
        self.assertFalse(r["gss_arithmetic"]["cupy_or_gss_package_executed"])
        self.assertFalse(r["static"]["neural_frameworks_executed"])

    def test_input_recipe_and_auxiva_failure(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal((9,3,2)) + 1j*rng.standard_normal((9,3,2))
        x[:,:,1] *= 3
        np.testing.assert_array_equal(complex_array(self.report["pra"]["input"]), x)
        self.assertEqual(self.report["pra"]["auxiva_eigen_initialization"]["exception_type"], "ValueError")

    def test_ilrma_linear_map_and_independent_unit_power_oracle(self):
        p = self.report["pra"]
        x = complex_array(p["input"])
        r = p["ilrma"]
        before = complex_array(r["filters_before_normalization"])
        w = complex_array(r["returned_filters"])
        y = complex_array(r["returned_output"])
        # Independent scalar dot products, not the audit's einsum path.
        pre = np.empty_like(y)
        actual = np.empty_like(y)
        for t in range(9):
            for f in range(3):
                for s in range(2):
                    pre[t,f,s] = sum(before[f,s,c]*x[t,f,c] for c in range(2))
                    actual[t,f,s] = sum(w[f,s,c]*x[t,f,c] for c in range(2))
        np.testing.assert_allclose(y, pre, atol=2e-14)
        self.assertGreater(np.max(abs(y-actual)), 3.)
        scale = 1 / np.sqrt(np.mean(abs(pre)**2, axis=(0,1)))
        np.testing.assert_allclose(r["normalization_scale"], scale, rtol=1e-14)
        np.testing.assert_allclose(w, before*scale[None,None,:], rtol=1e-14)
        # Scaling each source, rather than each input channel, has unit mean power.
        np.testing.assert_allclose(np.mean(abs(pre*scale)**2, axis=(0,1)), [1,1], atol=1e-14)

    def test_fastmnmf_mixture_consistency_and_mutable_initialization(self):
        p = self.report["pra"]
        x = complex_array(p["input"])
        for r in p["fastmnmf"].values():
            z = complex_array(r["output"])
            self.assertEqual(z.shape, (2,9,3,3))
            np.testing.assert_allclose(z[...,0]+z[...,1]+z[...,2], x.transpose(2,0,1), atol=3e-14)
            np.testing.assert_array_equal(complex_array(r["caller_W0_after"]), np.tile(np.eye(2),(3,1,1)))
            self.assertEqual(r["max_caller_W0_change"], 1.)

    def test_trinicon_known_filter_tail(self):
        expected_lengths = {16:16, 17:19, 19:21, 23:25}
        for r in self.report["pra"]["trinicon"]:
            n = r["input_length"]
            wave = np.arange(2*n).reshape(2,n)*.001
            # Unit impulse at lag two; convolution does not use upstream code.
            expected = np.stack([np.convolve(a, [0.,0.,1.,0.])[:n] for a in wave])
            np.testing.assert_array_equal(r["expected_same_length_delayed_input"], expected)
            self.assertEqual(r["output_length"], expected_lengths[n])
            z = np.array(r["output"])
            bad = np.flatnonzero(np.any(abs(z[:,:n]-expected)>1e-14, axis=0)).tolist()
            self.assertEqual(bad, [16] if n == 17 else [])
            if n == 17:
                self.assertAlmostEqual(expected[0,16], .014)
                self.assertEqual(z[0,16], 0.)

    def test_gss_clipping_and_masked_log_domain_reference(self):
        r = self.report["gss_arithmetic"]
        eps = 1e-10
        np.testing.assert_array_equal(r["results"]["clip"]["affiliation"], [eps,eps,1-eps])
        self.assertGreater(r["results"]["clip"]["sum"], 1.)
        np.testing.assert_array_equal(r["results"]["underflow"]["affiliation"], [0,0,0])
        # Mask before selecting the stabilizing maximum: active log weights stay finite.
        log_weight = np.log([.3,.5])
        mass = np.exp(log_weight-log_weight.max())
        posterior = np.r_[0., mass/mass.sum()]
        np.testing.assert_allclose(posterior, [0.,3/8,5/8], atol=1e-15)
        np.testing.assert_allclose(r["independent_exact_active_posterior"]["underflow"], posterior)

    def test_projection_back_three_original_routes_and_conjugate(self):
        r = self.report["projection_back"]
        joint = np.array([[[1,0,1]], [[0,2,2]]])
        for name in ("ssspy_filter_route_output", "ssspy_joint_data_route_output"):
            np.testing.assert_allclose(complex_array(r[name]), joint, atol=1e-14)
        np.testing.assert_allclose(complex_array(r["ssspy_scaled_filter"]), [[[1,-2],[0,2]]])
        np.testing.assert_allclose(complex_array(r["pra_returned_conjugate_coefficients"]), [[2,2.5]])
        np.testing.assert_allclose(complex_array(r["pra_applied_output"]), [[[2,0,2]],[[0,2.5,2.5]]])
        np.testing.assert_allclose(complex_array(r["pra_complex_returned_coefficient"]), [[2-1j]])
        np.testing.assert_allclose(complex_array(r["pra_complex_applied_coefficient"]), [[2+1j]])
        # Joint fit explains x0 exactly; two separate fits each also absorb correlation.
        np.testing.assert_allclose(joint.sum(axis=0), [[1,2,3]])
        self.assertFalse(np.allclose(complex_array(r["pra_applied_output"]).sum(axis=0), [[1,2,3]]))


if __name__ == "__main__":
    unittest.main()
