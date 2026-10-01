"""Independent support-boundary checks for Appendix B's finite room metrics."""
import copy
from fractions import Fraction
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from codes.chapters.appendix_b.core.room_metrics import drr_db, measured_t60_from_t20, t20_from_edc_points
from codes.chapters.appendix_b.examples import room_srp_exercise as room


class RoomMetricBoundaries(unittest.TestCase):
    def test_drr_common_scale_and_overflowing_ratio_have_finite_log_answer(self):
        for scale in (1., 1e-200, 1e200):
            full = [scale, scale/2]
            direct = [scale]
            # Independent exact binary-float energy ratio, not normalized helper.
            ed = Fraction.from_float(scale)**2
            er = Fraction.from_float(scale/2)**2
            self.assertAlmostEqual(drr_db(full, direct), 10*math.log10(float(ed/er)), places=11)
        reflected = 1e-160
        expected = -20*math.log10(reflected)
        self.assertAlmostEqual(drr_db([1., reflected], [1.]), expected, places=10)
        self.assertAlmostEqual(drr_db([reflected, 1.], [reflected]), -expected, places=10)

    def test_true_zero_and_unrepresentable_difference_are_errors(self):
        for full, direct in (([0, 1], [0]), ([1], [1]), ([1e308], [-1e308]),
                             ([1e308, 1e-308, 1], [1e308, 1e-308])):
            with self.assertRaises(ValueError):
                drr_db(full, direct)

    def test_exponential_t20_is_common_amplitude_invariant(self):
        h = np.array([math.exp(-3*math.log(10)*n/(16000*.6)) for n in range(32000)])
        for scale in (1., 1e-200, 1e200):
            self.assertAlmostEqual(measured_t60_from_t20(h*scale), .6, delta=2e-7)
        with self.assertRaisesRegex(ValueError, 'no energy'):
            measured_t60_from_t20([0., 0., 0.])

    def test_t20_conditioned_time_axis_matches_fraction_endpoint_slope(self):
        for times in ([1e16, 1e16+2, 1e16+4], [0., 1e-200, 2e-200], [0., 1e200, 2e200]):
            span = Fraction.from_float(times[-1])-Fraction.from_float(times[0])
            expected_slope = float(Fraction(-20)/span)
            expected_t60 = float(3*span)
            actual = t20_from_edc_points(times, [-5, -15, -25])
            self.assertLess(abs(actual['slope_db_per_s']/expected_slope-1), 1e-14)
            self.assertLess(abs(actual['t60_extrapolated_s']/expected_t60-1), 1e-14)
        with self.assertRaises(ValueError):
            t20_from_edc_points([0., 5e-324, 1e-323], [-5, -15, -25])

    def test_ordinary_fit_arithmetic_is_unchanged(self):
        times, decay = np.array([.05, .15, .25]), np.array([-5., -15., -25.])
        slope = float(np.polyfit(times, decay, 1)[0])
        self.assertEqual(t20_from_edc_points(times, decay)['slope_db_per_s'], slope)

    def test_real_domain_is_checked_before_cast(self):
        for bad in ([True, 1], ['1', '2'], np.array([1+1j, 2]), np.array([1, 2], dtype=object), [10**400, 1], [math.inf, 1]):
            with self.subTest(bad=type(bad)):
                with self.assertRaises(ValueError):
                    drr_db(bad, [1])
                with self.assertRaises(ValueError):
                    measured_t60_from_t20(bad)
        for bad in ([True, .1, .2], [0., .1+1j, .2], ['0', '.1', '.2'], [0, 10**400, 10**401]):
            with self.assertRaises(ValueError):
                t20_from_edc_points(bad, [-5, -15, -25])

    def test_configuration_rejects_nan_wrong_shapes_and_implicit_types(self):
        for key, bad in (('room_dimensions_m', [math.nan, 10, 6]),
                         ('room_dimensions_m', [12, 10]), ('microphones_m', [['6', '5', '3']]*4),
                         ('sound_speed_m_s', True), ('array_center_m', [6+1j, 5, 3])):
            config = room.configuration()
            config[key] = bad
            with self.assertRaises(ValueError):
                room.validate_configuration(config)

    def test_all_output_paths_preflight_before_simulation_or_partial_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root/'outside'; outside.mkdir()
            (root/'link').symlink_to(outside, target_is_directory=True)
            (root/'dangling.png').symlink_to(root/'missing.png')
            for kwargs in ({'plot': root/'same', 'audio': root/'same'},
                           {'plot': root/'audio'/'x.png', 'audio': root/'audio'},
                           {'plot': root/'dangling.png'}, {'plot': root/'link'/'out.png'},
                           {'plot': root/'link'/'..'/'out.png'}):
                with self.assertRaises(ValueError):
                    room.validate_output_paths(**kwargs)
            with patch('sys.argv', ['room', '--run', '--plot', str(root/'same'), '--audio-dir', str(root/'same')]), patch.object(room, 'run_experiment') as run:
                with self.assertRaises(SystemExit):
                    room.main()
                run.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])
            self.assertFalse((root/'same').exists())

    def test_nonfinite_duplicate_json_is_rejected(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                room.strict_json(data)


if __name__ == '__main__':
    unittest.main()
