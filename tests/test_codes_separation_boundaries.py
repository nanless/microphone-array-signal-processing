"""Independent scalar/exponent and GSS state-phase regression checks."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from codes.chapters.ch08.core.separation import masked_spatial_covariance, mask_mvdr_2x2
from codes.chapters.ch08.core.gss_teaching import guided_cacgmm_mvdr
from codes.chapters.ch00.cross_chapter.enhancement_step_exercises import ip_row
from codes.chapters.ch00.cross_chapter.enhancement_structure_exercises import cacg_relative_density, cacg_shape_step
from codes.chapters.ch08.examples.gss_teaching_demo import generate


class SeparationBoundaries(unittest.TestCase):
    def test_covariance_representable_extremes(self):
        for amplitude, weight, expected in [(1e100, 1e150, 1e200), (1., 1e308, 1.), (1e-100, 1e-150, 1e-200)]:
            r = masked_spatial_covariance(np.full((1, 1, 2), amplitude), np.full((1, 2), weight), epsilon=1e-310)
            self.assertAlmostEqual(float(r[0, 0, 0].real) / expected, 1, places=14)
        with self.assertRaises(ValueError):
            masked_spatial_covariance(np.full((1, 1, 2), 1e200), np.ones((1, 2)))

    def test_zero_weight_outlier_and_tiny_positive_mass(self):
        x = np.array([[[1e300, 1.]]])
        r = masked_spatial_covariance(x, [[0., 1.]])
        np.testing.assert_allclose(r, [[[1.]]])
        # Product 1e-300 * (1e150)^2 = 1, plus the ordinary second term.
        r = masked_spatial_covariance(np.array([[[1e150, 1.]]]), [[1e-300, 1.]])
        np.testing.assert_allclose(r, [[[2.]]])
        # Conjugation and enormous cross-channel scale ratio.
        x = np.array([[[1e150], [1e-150j]]])
        r = masked_spatial_covariance(x, [[1.]])
        np.testing.assert_allclose(r[0, 0, 1], -1j)
        self.assertAlmostEqual(r[0, 1, 1].real / 1e-300, 1)

    def test_covariance_floor_and_empty_contract(self):
        for weight, expected in [(5e-13, 10), (2.5e-13, 5)]:
            np.testing.assert_allclose(masked_spatial_covariance(np.array([[[2, 4]]]), [[weight, weight]]), [[[expected]]])
        for shape in [(0, 2, 2), (2, 0, 2), (2, 2, 0)]:
            with self.assertRaises(ValueError): masked_spatial_covariance(np.zeros(shape), np.zeros((shape[0], shape[2])))

    def test_integer_minimum_matches_float_input(self):
        value = np.iinfo(np.int64).min
        x = np.full((1, 2, 3), value, dtype=np.int64)
        mask = np.ones((1, 3))
        for weights in (mask, mask * 1e308):
            integer_r = masked_spatial_covariance(x, weights)
            float_r = masked_spatial_covariance(x.astype(float), weights)
            np.testing.assert_array_equal(integer_r, float_r)
            np.testing.assert_allclose(integer_r.real / float(value)**2, 1.)
        integer_y, integer_w, diagnostic = mask_mvdr_2x2(x, mask, mask, return_diagnostics=True)
        float_y, float_w = mask_mvdr_2x2(x.astype(float), mask, mask)
        np.testing.assert_array_equal(integer_y, float_y)
        np.testing.assert_array_equal(integer_w, float_w)
        np.testing.assert_allclose(integer_y / float(value), 1.)
        self.assertEqual(diagnostic['routes'], ['mvdr'])

    def test_covariance_cancellation_retains_tiny_cross_term(self):
        # First two cross products cancel exactly; the third remains 1e-300.
        x = np.array([[[1e150, 1e150, 1e-150],
                       [1e150, -1e150, 1e-150]]])
        r = masked_spatial_covariance(x, [[1., 1., 1.]])[0]
        self.assertAlmostEqual(r[0, 1].real / (1e-300 / 3), 1., places=14)
        self.assertAlmostEqual(r[1, 0].real / (1e-300 / 3), 1., places=14)
        np.testing.assert_allclose(r.diagonal().real / (2e300 / 3), 1.)

    def test_mvdr_huge_loading_and_diagnostics(self):
        x = np.array([[[1., 1.], [1., -1.]]])
        y, w, d = mask_mvdr_2x2(x, [[.9, .1]], [[.1, .9]], diagonal_loading=1e308, return_diagnostics=True)
        np.testing.assert_allclose(w, [[.5, .5]], atol=1e-15)
        np.testing.assert_allclose(y, [[1, 0]], atol=1e-15)
        self.assertEqual(d['routes'], ['mvdr'])
        _, _, d = mask_mvdr_2x2(x, [[0, 0]], [[1, 1]], return_diagnostics=True)
        self.assertEqual(d['routes'], ['empty_target_mask'])

    def test_ip_and_cacg_subnormal_scales(self):
        for scale in [1e-310, 1e300]:
            w = ip_row(np.eye(2), scale*np.eye(2), 0)
            np.testing.assert_allclose(w*np.sqrt(scale), [1, 0], atol=1e-15)
        np.testing.assert_allclose(cacg_relative_density(np.eye(2), 1e-310*np.eye(2)), [1, 1])
        raw, shape = cacg_shape_step(np.eye(2), [.75, .25], 1e-310*np.eye(2))
        np.testing.assert_allclose(raw.real/1e-310, np.diag([1.5, .5]), atol=1e-12)
        np.testing.assert_allclose(shape, np.diag([1.5, .5]))

    def test_gss_parameter_types_and_insufficient_shrinkage(self):
        x, a = np.ones((1, 2, 10)), np.ones((10, 1))
        for value in [True, 1+0j, np.array([.02]), np.nan]:
            with self.assertRaises(ValueError): guided_cacgmm_mvdr(x, a, shape_loading=value)
        with self.assertRaisesRegex(ValueError, 'positive definiteness'):
            guided_cacgmm_mvdr(x, a, shape_loading=1e-18)

    def test_gss_gate_and_finite_large_statistics(self):
        x = np.tile(np.eye(2), (1, 6))[None]
        a = np.ones((12, 2))
        tiny = guided_cacgmm_mvdr(x*1e-14, a)
        self.assertEqual(tiny['low_energy_bins'], 12)
        unthresholded = guided_cacgmm_mvdr(x*1e-14, a, absolute_energy_floor=0)
        np.testing.assert_allclose(unthresholded['posterior'], 1/3)
        huge = guided_cacgmm_mvdr(x*1e154, a)
        self.assertEqual(huge['low_energy_bins'], 0)
        self.assertTrue(np.isfinite(huge['target_scm']).all())

    def test_gss_e_snapshot_recomputes_posterior(self):
        x = np.array([[[1, 2, 1, 2, 1, 2], [0, 0, 1j, 2j, 1, 2]]], complex)
        activity = np.array([[1], [1], [0], [0], [1], [1]])
        r = guided_cacgmm_mvdr(x, activity, iterations=1)
        z = x[0].T / np.linalg.norm(x[0].T, axis=1)[:, None]
        scores = []
        for j, b in enumerate(r['e_step_shapes'][0]):
            # Independent 2x2 cofactor inverse, no general solver.
            det = (b[0, 0]*b[1, 1]-b[0, 1]*b[1, 0]).real
            inverse = np.array([[b[1, 1], -b[0, 1]], [-b[1, 0], b[0, 0]]])/det
            q = np.array([np.vdot(row, inverse@row).real for row in z])
            scores.append(r['e_step_priors'][0, j]/det/q**2)
        scores = np.array(scores).T
        scores[activity[:, 0] == 0, 0] = 0
        np.testing.assert_allclose(r['posterior'][0].T, scores/scores.sum(1, keepdims=True), atol=1e-14)
        self.assertGreater(np.max(abs(r['shape_matrices']-r['e_step_shapes'])), .001)

    def test_gss_assets_preserve_old_pcm_and_check_is_readonly(self):
        expected = {'source_1.wav': '6e3333c7dc244d2a09d9342dd9df0fd76ae35ef1cfbcd0d5f1637a0e4a366644',
                    'source_2.wav': '82d7c5b07a588fbd7a1dff6223453cf4cc7490e8dca4c22c1196b5692907f600',
                    'mixture.wav': '2e826e6f4b66f8e33d56c045da35cc937549cbf6d0124bfbb779079a273aec5b',
                    'enhanced_correct.wav': '13e818538c51c00296c8a04682d9903c09c3e0495b357e57d5994a70134da567',
                    'enhanced_missed.wav': 'ebfad9f73fdf061140f04917e1449b1d47187728bccba3c9fe332a60901ee247'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            generate(path)
            for name, digest in expected.items():
                self.assertEqual(hashlib.sha256((path/name).read_bytes()).hexdigest(), digest)
            generate(path, check=True)
            bad = path/'enhanced_correct.wav'
            bad.write_bytes(b'broken')
            with self.assertRaises(ValueError): generate(path, check=True)
            self.assertEqual(bad.read_bytes(), b'broken')
