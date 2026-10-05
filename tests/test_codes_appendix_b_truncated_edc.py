"""Independent finite geometric-series and scalar-fit checks for E15-15."""

from decimal import Decimal, localcontext
import math
import unittest

import numpy as np

from codes.chapters.appendix_b.appendix_b_experiments import truncated_edc_fixture
from codes.chapters.appendix_b.core.room_metrics import measured_t60_from_t20


def geometric_db(index, count):
    # Algebraically q = 10**(-1/1600). Work in log coordinates rather than
    # powering a previously rounded q, independently of reverse summation.
    rate = -math.log(10) / 1600
    return 10 / math.log(10) * (
        rate * index + math.log(-math.expm1(rate * (count - index)))
        - math.log(-math.expm1(rate * count)))


def scalar_fit(count, first, last):
    times = [n / 16000 for n in range(first, last + 1)]
    values = [geometric_db(n, count) for n in range(first, last + 1)]
    mean_t = math.fsum(times) / len(times)
    mean_d = math.fsum(values) / len(values)
    slope = math.fsum((t - mean_t) * (d - mean_d)
                      for t, d in zip(times, values)) / math.fsum(
                          (t - mean_t) ** 2 for t in times)
    return slope, -60 / slope


class TruncatedEDCTest(unittest.TestCase):
    def test_finite_half_tail_point_and_infinite_limit_are_different(self):
        result = truncated_edc_fixture()
        point = result['point']
        self.assertEqual((point['samples'], point['index']), (3200, 1600))
        self.assertAlmostEqual(point['finite_relative_energy'], 1 / 11, delta=3e-17)
        self.assertAlmostEqual(point['finite_decay_db'], -10 * math.log10(11), delta=1e-14)
        self.assertAlmostEqual(point['infinite_relative_energy'], .1, delta=3e-17)
        self.assertAlmostEqual(point['infinite_decay_db'], -10, delta=1e-14)

    def test_fixed_sample_intervals_and_scalar_regression(self):
        result = truncated_edc_fixture()
        self.assertEqual(result['main_samples'], [3200, 9600, 32000])
        expected = [(1920, 712, 1889), (3200, 786, 3011),
                    (4800, 799, 3810), (9600, 800, 4000), (32000, 800, 4000)]
        for row, (count, first, last) in zip(result['cases'], expected, strict=True):
            self.assertEqual((row['samples'], row['fit_start_index'],
                              row['fit_stop_index_inclusive']), (count, first, last))
            self.assertEqual(row['fit_samples'], last - first + 1)
            self.assertEqual(row['nominal_duration_s'], count / 16000)
            self.assertEqual(row['last_sample_time_s'], (count - 1) / 16000)
            slope, t60 = scalar_fit(count, first, last)
            self.assertAlmostEqual(row['slope_db_per_s'], slope, delta=6e-12)
            self.assertAlmostEqual(row['t60_extrapolated_s'], t60, delta=3e-14)
            self.assertAlmostEqual(row['t20_s'], t60 / 3, delta=1e-14)
            self.assertLess(row['geometric_vs_reverse_sum_max_absolute'], 4e-15)
            self.assertLessEqual(row['fit_start_db'], -5)
            self.assertLessEqual(row['fit_stop_db'], -25)

    def test_threshold_equality_uses_exact_model_not_a_wide_tolerance(self):
        # Finite N=32000 is infinitesimally below the infinite straight line
        # at n=800/4000. Decimal can resolve the finite-tail correction that
        # float64 cannot; no epsilon changes the crossing rule.
        with localcontext() as ctx:
            ctx.prec = 70
            for count, first, last in ((3200, 786, 3011),
                                       (9600, 800, 4000), (32000, 800, 4000)):
                def ratio(n):
                    return ((Decimal(10) ** (-Decimal(n) / 1600)
                             - Decimal(10) ** (-Decimal(count) / 1600))
                            / (1 - Decimal(10) ** (-Decimal(count) / 1600)))
                for index, threshold_db in ((first, -5), (last, -25)):
                    threshold = Decimal(10) ** (Decimal(threshold_db) / 10)
                    self.assertGreater(ratio(index - 1), threshold)
                    self.assertLessEqual(ratio(index), threshold)

    def test_equal_prefix_amplitudes_do_not_make_equal_normalized_tails(self):
        # Same h[0:3200], different support; no parameter or amplitude changes.
        short = np.exp(-math.log(10) * np.arange(3200) / 3200)
        long = np.exp(-math.log(10) * np.arange(32000) / 3200)
        np.testing.assert_array_equal(short, long[:3200])
        short_t60 = measured_t60_from_t20(short)
        long_t60 = measured_t60_from_t20(long)
        self.assertAlmostEqual(short_t60, .461163965881604, delta=3e-14)
        self.assertAlmostEqual(long_t60, .6, delta=3e-14)
        self.assertGreater(long_t60 - short_t60, .13)
        for scale in (1e-200, 1e200):
            self.assertAlmostEqual(measured_t60_from_t20(short * scale), short_t60, delta=3e-14)

    def test_short_unusable_decay_is_rejected_and_no_interval_is_invented(self):
        with self.assertRaisesRegex(ValueError, 'usable'):
            measured_t60_from_t20(np.exp(-math.log(10) * np.arange(4) / 3200))

    def test_returned_common_scales_keep_the_same_short_record_bias(self):
        controls = truncated_edc_fixture()['common_scale_controls']
        self.assertEqual([row['amplitude'] for row in controls], [1., 1e-200, 1e200])
        for row in controls:
            self.assertEqual(row['samples'], 3200)
            self.assertAlmostEqual(row['t60_extrapolated_s'], .461163965881604, delta=3e-14)


if __name__ == '__main__':
    unittest.main()
