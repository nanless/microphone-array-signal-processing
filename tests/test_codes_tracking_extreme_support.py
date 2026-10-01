"""Fraction, scalar geometry and filesystem oracles for Chapter 9 support limits."""
from fractions import Fraction
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from codes.chapters.ch09.core.tracking import (
    ConstantVelocityKalman, CircularParticleFilter, _checked_covariance,
    white_acceleration_covariance, wrap_angle,
)
from codes.chapters.ch09.core.moving_source import retarded_emission_times
from codes.chapters.ch09.core.tracking_audio import analyze_array
from codes.chapters.ch00.cross_chapter.tracking_time_exercises import predict_timestamped_direction
from codes.chapters.ch09.examples import chapter09_tracking_audio as tracking_export
from codes.chapters.ch09.examples import moving_source_audio as moving_export


class TrackingExtremeSupport(unittest.TestCase):
    def test_micro_angles_and_nearby_points_are_not_collapsed(self):
        values = np.array([-1e-100, -1e-14, 0., 1e-14, 1e-100])
        np.testing.assert_array_equal(wrap_angle(values), values)
        self.assertEqual(len(set(wrap_angle([0., 1e-100]).tolist())), 2)
        np.testing.assert_array_equal(wrap_angle([-540, -180, 180, 540]), [-180]*4)
        self.assertEqual(wrap_angle(1e308), (1e308 % 360+180) % 360-180)

    def test_joseph_two_subnormal_parts_must_be_merged(self):
        p = 1e-323
        model = ConstantVelocityKalman([0, 0], np.diag([p, p]), np.zeros((2, 2)))
        model.update(0, p)
        expected = float(Fraction(p)*Fraction(p)/(Fraction(p)+Fraction(p)))
        self.assertEqual(expected, math.ulp(0.))
        self.assertEqual(model.covariance[0, 0], expected)
        self.assertEqual(model.covariance[1, 1], p)

    def test_unrepresentable_positive_posterior_is_atomic_failure(self):
        p = math.ulp(0.)
        model = ConstantVelocityKalman([1, 2], np.diag([p, p]), np.zeros((2, 2)))
        before = model.state.copy(), model.covariance.copy()
        exact = Fraction(p)/2
        self.assertGreater(exact, 0)
        self.assertEqual(float(exact), 0)
        with self.assertRaisesRegex(ValueError, 'outside float64 support'):
            model.update(3, p)
        np.testing.assert_array_equal(model.state, before[0])
        np.testing.assert_array_equal(model.covariance, before[1])

    def test_compound_gain_recovers_representable_velocity(self):
        cross, r = 2e-18, 1e308
        model = ConstantVelocityKalman([0, 0], [[1, cross], [cross, 1]], np.zeros((2, 2)))
        model.update(179, r)
        total = Fraction(1)+Fraction(r)
        self.assertEqual(model.state[1], float(Fraction(cross)*179/total))
        self.assertEqual(model.state[1], math.ulp(0.))
        self.assertEqual(model.state[0], float(Fraction(179)/total))

    def test_prediction_compound_and_zero_are_distinct(self):
        dt = 1e-150
        model = ConstantVelocityKalman([0, 1e150], [[0, 0], [0, 1e150]], np.zeros((2, 2)))
        model.predict(dt)
        self.assertEqual(model.state[0], float(Fraction(dt)*Fraction(1e150)))
        self.assertEqual(model.covariance[0, 0], float(Fraction(dt)**2*Fraction(1e150)))
        self.assertEqual(model.covariance[0, 1], float(Fraction(dt)*Fraction(1e150)))
        zero = ConstantVelocityKalman([0, 0], np.zeros((2, 2)), np.zeros((2, 2)))
        zero.predict(1)
        np.testing.assert_array_equal(zero.covariance, np.zeros((2, 2)))

    def test_white_acceleration_full_monomials_fraction_oracle(self):
        for dt, density in ((1e-150, 1e300), (1e-200, 1e300), (1e100, 1e-100)):
            t, q = Fraction(dt), Fraction(density)
            expected = np.array([[float(q*t**3/3), float(q*t*t/2)], [float(q*t*t/2), float(q*t)]])
            np.testing.assert_array_equal(white_acceleration_covariance(dt, density), expected)
        np.testing.assert_array_equal(white_acceleration_covariance(.01, 100),
                                     100*np.array([[.01**3/3, .01**2/2], [.01**2/2, .01]]))
        for dt, q in ((0, 1e308), (1e308, 0)):
            np.testing.assert_array_equal(white_acceleration_covariance(dt, q), np.zeros((2, 2)))

    def test_q_boundary_and_real_type_contract(self):
        for dt, q in ((1e-200, 1), (1e100, 1e100)):
            with self.assertRaises(ValueError):
                white_acceleration_covariance(dt, q)
        for bad in (True, '1', 1+0j, [1]):
            for args in ((bad, 1), (1, bad)):
                with self.assertRaises(ValueError):
                    white_acceleration_covariance(*args)
        for args in ((-1, 1), (1, -1)):
            with self.assertRaises(ValueError):
                white_acceleration_covariance(*args)

    def test_covariance_zero_variance_and_normalization_support(self):
        with self.assertRaisesRegex(ValueError, 'zero variance'):
            _checked_covariance([[0, 1e-100], [1e-100, 1e300]])
        with self.assertRaisesRegex(ValueError, 'normalization loses'):
            _checked_covariance([[1e-200, 0], [0, 1e300]])
        np.testing.assert_array_equal(_checked_covariance([[0, 0], [0, 1e300]]), [[0, 0], [0, 1e300]])

    def test_persistent_particle_log_support_and_readonly_copy(self):
        pf = CircularParticleFilter([0, 10])
        pf.update(10, .01, clutter_probability=0)
        self.assertEqual(pf.weight_underflow_indices, [0])
        np.testing.assert_array_equal(pf.weights, [0, 1])
        # Equal initial priors, two equal-sigma observations at both locations:
        # each has equal total squared distance 100, so posterior exactly 1/2.
        logs = pf.log_weights
        self.assertAlmostEqual(logs[0], -500000, places=7)
        logs[:] = -np.inf
        self.assertTrue(np.isfinite(pf.log_weights[0]))
        pf.update(0, .01, clutter_probability=0)
        np.testing.assert_allclose(pf.weights, [.5, .5], atol=1e-10)
        self.assertEqual(pf.weight_underflow_indices, [])

    def test_explicit_same_display_prior_reset_removes_log_support(self):
        pf = CircularParticleFilter([0, 10])
        pf.update(10, .01, clutter_probability=0)
        np.testing.assert_array_equal(pf.weights, [0, 1])
        self.assertTrue(np.isfinite(pf.log_weights[0]))
        pf.set_prior_weights([0, 1])
        self.assertTrue(np.isneginf(pf.log_weights[0]))
        self.assertEqual(pf.weight_underflow_indices, [])
        pf.update(0, .01, clutter_probability=0)
        np.testing.assert_array_equal(pf.weights, [0, 1])

    def test_prior_setter_keeps_positive_log_probability_after_linear_underflow(self):
        pf = CircularParticleFilter([0, 10])
        small, large = math.ulp(0.), 1e308
        pf.set_prior_weights([small, large])
        self.assertEqual(pf.weight_underflow_indices, [0])
        np.testing.assert_array_equal(pf.weights, [0, 1])
        # Ratio is about 5e-632, but its logarithm is finite and independently
        # computable from input logarithms, unlike the overflowing mass sum.
        self.assertAlmostEqual(pf.log_weights[0], math.log(small)-math.log(large), places=12)
        self.assertEqual(pf.log_weights[1], 0)
        with np.errstate(all='raise'):
            pf.set_prior_weights([1e308, 1e308])
        np.testing.assert_array_equal(pf.weights, [.5, .5])
        np.testing.assert_allclose(pf.log_weights, [-math.log(2)]*2, atol=1e-15)

    def test_invalid_prior_setter_preserves_all_particle_state(self):
        pf = CircularParticleFilter([0, 10])
        pf.update(10, .01, clutter_probability=0)
        for invalid in ([0, 0], [-1, 1], [1], [[1, 1]], [1, math.nan],
                        [1, math.inf], [True, False], ['0', '1'], [0j, 1j]):
            before = copy.deepcopy(pf.__dict__)
            with self.assertRaises(ValueError):
                pf.set_prior_weights(invalid)
            self.assertEqual(pf.__dict__.keys(), before.keys())
            for key, expected in before.items():
                if isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(pf.__dict__[key], expected)
                else:
                    self.assertEqual(pf.__dict__[key], expected)

    def test_manual_zero_and_resampling_remove_support_explicitly(self):
        pf = CircularParticleFilter([0, 10])
        pf.weights[:] = [0, 1]
        pf.update(0, .01, clutter_probability=0)
        np.testing.assert_array_equal(pf.weights, [0, 1])
        self.assertTrue(np.isneginf(pf.log_weights[0]))
        self.assertEqual(pf.weight_underflow_indices, [])
        other = CircularParticleFilter([0, 10])
        other.update(10, .01, clutter_probability=0)
        self.assertTrue(other.resample_if_needed(np.random.default_rng(0), 2))
        np.testing.assert_array_equal(other.last_resample_ancestor_indices, [1, 1])
        self.assertEqual(other.last_resample_removed_indices, [0])
        np.testing.assert_array_equal(other.particles, [10, 10])
        np.testing.assert_array_equal(other.weights, [.5, .5])

    def test_unsupported_log_density_failure_does_not_remove_positive_support(self):
        pf = CircularParticleFilter([0, 10])
        before = pf.particles.copy(), pf.weights.copy(), pf.log_weights
        with self.assertRaisesRegex(ValueError, 'log likelihood'):
            pf.update(0, 1e-300, clutter_probability=0)
        for actual, expected in zip((pf.particles, pf.weights, pf.log_weights), before):
            np.testing.assert_array_equal(actual, expected)

    def test_ttl_clock_resolution_cannot_enlarge_application_lifetime(self):
        origin = 1e20
        for consume in (origin, np.nextafter(origin, np.inf)):
            with self.assertRaisesRegex(ValueError, 'resolution'):
                predict_timestamped_direction([0, 0], np.eye(2), origin, consume, .1, 0)
        # Zero lifetime accepts an identical timestamp, rejects a real next tick.
        predict_timestamped_direction([0, 0], np.eye(2), origin, origin, 0, 0)
        with self.assertRaises(ValueError):
            predict_timestamped_direction([0, 0], np.eye(2), origin, np.nextafter(origin, np.inf), 0, 0)

    def test_static_hypot_extremes_have_closed_form_delay(self):
        for distance, speed in ((1e-200, 343.), (1e-200, 1e-200), (1e200, 343.)):
            expected = -distance/speed
            actual = retarded_emission_times([0], [[0, 0], [0, 0]],
                source_start_xy=[distance, 0], source_velocity_xy=[0, 0], sound_speed=speed)
            np.testing.assert_array_equal(actual, [[expected], [expected]])
        with self.assertRaisesRegex(ValueError, 'singular'):
            retarded_emission_times([0], [[0, 0], [0, 0]],
                source_start_xy=[0, 0], source_velocity_xy=[0, 0])

    def test_rms_large_finite_not_infinity_and_tiny_not_false_zero(self):
        # Fake only DOA to isolate pre-DOA RMS, not the RMS formula/expectation.
        with patch('codes.chapters.ch09.core.tracking_audio.gcc_phat', return_value=(0., None, None, None)):
            report = analyze_array(np.full((2, 32000), 1e200))
        self.assertEqual(report['frames']['rms_before_export'], [1e200]*197)
        report = analyze_array(np.full((2, 32000), 1e-200))
        self.assertEqual(report['frames']['rms_before_export'], [1e-200]*197)
        self.assertEqual(report['scores']['valid_observation_count'], 0)
        with self.assertRaisesRegex(ValueError, 'gain correction loses'):
            analyze_array(np.full((2, 32000), 1e-300), export_gain=1e300)


class AudioDirectorySafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cached = {}
        for module in (tracking_export, moving_export):
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/'audio'
                module.generate(directory)
                cls.cached[module] = {p.name: p.read_bytes() for p in directory.iterdir()}

    def _copy(self, module, directory):
        directory.mkdir()
        for name, blob in self.cached[module].items():
            (directory/name).write_bytes(blob)

    def test_real_check_is_byte_and_mtime_readonly(self):
        for module in self.cached:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/'audio'; self._copy(module, directory)
                before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
                module.generate(directory, check=True)
                self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()})

    def test_member_symlink_cannot_write_external_file(self):
        for module in self.cached:
            for check in (False, True):
                with tempfile.TemporaryDirectory() as tmp:
                    parent = Path(tmp); directory = parent/'audio'; self._copy(module, directory)
                    outside = parent/'outside'; outside.write_bytes(self.cached[module]['source.wav'])
                    (directory/'source.wav').unlink(); (directory/'source.wav').symlink_to(outside)
                    snapshot = outside.read_bytes(), outside.stat().st_mtime_ns
                    with self.assertRaisesRegex(ValueError, 'ordinary file'):
                        module.generate(directory, check=check)
                    self.assertEqual(snapshot, (outside.read_bytes(), outside.stat().st_mtime_ns))

    def test_directory_and_parent_symlinks_are_rejected_before_writes(self):
        for module in self.cached:
            with tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp); real = parent/'real'; real.mkdir(); link = parent/'alias'; link.symlink_to(real, target_is_directory=True)
                for output in (link, link/'child'):
                    for check in (False, True):
                        with self.assertRaisesRegex(ValueError, 'symbolic link'):
                            module.generate(output, check=check)
                        self.assertEqual(list(real.iterdir()), [])

    def test_extras_missing_wrong_directory_and_check_type(self):
        for module in self.cached:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/'audio'; self._copy(module, directory)
                for check in ('yes', 1, None, np.bool_(True)):
                    with self.assertRaisesRegex(ValueError, 'bool'):
                        module.generate(directory, check=check)
                extra = directory/'extra'; extra.write_bytes(b'keep')
                for check in (False, True):
                    with self.assertRaises(ValueError):
                        module.generate(directory, check=check)
                    self.assertEqual(extra.read_bytes(), b'keep')
                extra.unlink(); (directory/'source.wav').unlink()
                for check in (False, True):
                    with self.assertRaises(ValueError):
                        module.generate(directory, check=check)
                file = Path(tmp)/'not-directory'; file.write_bytes(b'keep')
                with self.assertRaises(ValueError):
                    module.generate(file)
                self.assertEqual(file.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
