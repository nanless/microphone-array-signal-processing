"""Component and fixed-delay arithmetic for the shared-input AEC experiment."""

import math
import unittest

import numpy as np

from codes.examples.aec_same_input_truth import (
    FAR_SCORE,
    NEAR_SCORE,
    PATH,
    RATE,
    SAMPLES,
    _impulse_lag,
    _score,
    echo_to_output_ratio_db,
    known_components,
)


class TestAECSameInputTruth(unittest.TestCase):
    def test_fixture_is_exactly_additive_and_fir_is_independently_recomputable(self):
        signals = known_components()
        reference = signals["render"].astype(np.float64) / 32768.0
        echo = signals["echo"].astype(np.int32)
        near = signals["near"].astype(np.int32)
        mixed = signals["mixed"].astype(np.int32)
        self.assertEqual(len(reference), 6 * RATE)
        np.testing.assert_array_equal(mixed, echo + near)
        self.assertEqual(int(np.count_nonzero(near[:NEAR_SCORE[0]])), 0)
        self.assertEqual(int(np.count_nonzero(near[NEAR_SCORE[1]:])), 0)
        self.assertGreater(int(np.count_nonzero(near[NEAR_SCORE[0]:NEAR_SCORE[1]])), 0)
        for n in (0, 1, 2, 4, 5, 16000, 32000, 48000):
            expected = sum(PATH[k] * reference[n - k] for k in range(min(n + 1, len(PATH))))
            self.assertEqual(int(echo[n]), int(np.rint(expected * 32768.0)))
        self.assertLess(int(np.max(np.abs(mixed))), 32767)

    def test_power_ratio_has_correct_db_convention(self):
        echo = np.array([100.0, -100.0, 50.0, -50.0])
        self.assertAlmostEqual(echo_to_output_ratio_db(echo, echo), 0.0)
        self.assertAlmostEqual(echo_to_output_ratio_db(echo, echo / 2),
                               20.0 * math.log10(2.0))
        self.assertIsNone(echo_to_output_ratio_db(echo, np.zeros_like(echo)))
        with self.assertRaises(ValueError):
            echo_to_output_ratio_db(np.zeros(4), np.ones(4))

    def test_fixed_lag_is_not_fitted_on_the_near_end_score(self):
        lag = 64
        echo = np.zeros(SAMPLES)
        near = np.zeros(SAMPLES)
        a, b = FAR_SCORE
        c, d = NEAR_SCORE
        echo[a:b] = 100.0
        near[c:d] = 100.0 * np.random.default_rng(9).standard_normal(d - c)
        base = np.zeros(SAMPLES)
        injected = np.zeros(SAMPLES)
        base[lag:] = echo[:-lag]
        injected[lag:] = (echo + near)[:-lag]
        result = _score(base, injected, near, echo, output_kind="test", lag_samples=lag)
        aligned = result["fixed_lag_compensated"]
        raw = result["same_sample_without_delay_compensation"]
        self.assertAlmostEqual(aligned["far_only_echo_to_total_output_power_ratio_db"], 0.0)
        self.assertAlmostEqual(aligned["known_near_injection_increment"]["increment_projection_gain_no_delay_fit"], 1.0)
        self.assertAlmostEqual(aligned["known_near_injection_increment"]["fixed_sample_relative_squared_error"], 0.0)
        self.assertGreater(raw["known_near_injection_increment"]["fixed_sample_relative_squared_error"], 1.0)
        with self.assertRaises(ValueError):
            _score(base, injected, near, echo, output_kind="test", lag_samples=-1)

    def test_independent_impulse_onset(self):
        probe = np.zeros(SAMPLES, dtype=np.int16)
        probe[48037 + 128] = 9000
        self.assertEqual(_impulse_lag(probe, 48037), 128)
        probe[48037 + 130] = 10000
        with self.assertRaises(ValueError):
            _impulse_lag(probe, 48037)


if __name__ == "__main__":
    unittest.main()
