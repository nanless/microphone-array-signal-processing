"""Independent scalar FIR, analytic phasor and numerical-boundary checks."""
import json
import math
import unittest

import numpy as np

from codes.chapters.ch01.core.spectral_cues import build_cases, measure_response


def independent_records(amplitudes=(.1, .1)):
    """Scalar source and two explicit delayed copies; no tested generator."""
    samples = []
    for n in range(32000):
        t = n / 16000
        samples.append(min(1., t / .02, (2. - t) / .02)
                       * (amplitudes[0] * math.sin(2 * math.pi * 1000 * t)
                          + amplitudes[1] * math.sin(2 * math.pi * 7000 * t)))
    source = np.array(samples)[None, :]
    stereo = np.zeros((2, 32001))
    for n in range(32001):
        now = samples[n] if n < 32000 else 0.
        previous = samples[n - 1] if n > 0 else 0.
        stereo[:, n] = now + .5 * previous, now - .5 * previous
    return source, stereo


class SpectralCueTest(unittest.TestCase):
    def test_build_has_shared_gain_causal_fir_and_complete_tail(self):
        records = build_cases()
        self.assertEqual(set(records), {'flat_source', 'flat_stereo', 'tilted_source', 'tilted_stereo'})
        for name, amplitudes in [('flat', (.1, .1)), ('tilted', (.12, .04))]:
            source, stereo = independent_records(amplitudes)
            self.assertEqual(records[name + '_source'].shape, (1, 32000))
            self.assertEqual(records[name + '_stereo'].shape, (2, 32001))
            np.testing.assert_allclose(records[name + '_source'], source, atol=1e-16, rtol=1e-14)
            np.testing.assert_allclose(records[name + '_stereo'], stereo, atol=1e-16, rtol=1e-14)
            self.assertNotEqual(stereo[0, -1], 0.)
            self.assertEqual(stereo[1, -1], -stereo[0, -1])

    def test_two_source_spectra_match_independent_power_and_complex_responses(self):
        cosine = math.sqrt(2 + math.sqrt(2)) / 2  # cos(pi/8)
        for amplitudes in [(.1, .1), (.12, .04)]:
            source, stereo = independent_records(amplitudes)
            before = source.copy(), stereo.copy()
            result = measure_response(source, stereo)
            source_power = sum(a*a/2 for a in amplitudes)
            left_power = sum(a*a/2 * (1.25+c) for a, c in zip(amplitudes, [cosine, -cosine]))
            right_power = sum(a*a/2 * (1.25-c) for a, c in zip(amplitudes, [cosine, -cosine]))
            self.assertAlmostEqual(result['source_mean_square'], source_power, places=14)
            self.assertAlmostEqual(result['left_mean_square'], left_power, places=14)
            self.assertAlmostEqual(result['right_mean_square'], right_power, places=14)
            self.assertAlmostEqual(result['ild_right_minus_left_db'],
                                   10 * math.log10(right_power/left_power), places=11)
            for row, amplitude, omega in zip(result['spectral'], amplitudes, [math.pi/8, 7*math.pi/8]):
                hl = 1 + .5 * complex(math.cos(omega), -math.sin(omega))
                hr = 1 - .5 * complex(math.cos(omega), -math.sin(omega))
                for key, expected in [('source_complex', -1j*amplitude),
                                      ('left_complex', -1j*amplitude*hl),
                                      ('right_complex', -1j*amplitude*hr)]:
                    np.testing.assert_allclose(row[key], [expected.real, expected.imag], atol=2e-13, rtol=0)
                self.assertAlmostEqual(row['left_power_gain'], 1.25 + math.cos(omega), places=12)
                self.assertAlmostEqual(row['right_power_gain'], 1.25 - math.cos(omega), places=12)
                self.assertAlmostEqual(row['ild_right_minus_left_db'],
                                       10*math.log10((1.25-math.cos(omega))/(1.25+math.cos(omega))), places=11)
                ratio = hr / hl
                expected_phase = math.atan2(ratio.imag, ratio.real)
                self.assertAlmostEqual(row['ipd_right_minus_left_rad'], expected_phase, places=12)
                self.assertAlmostEqual(row['ipd_right_minus_left_deg'], math.degrees(expected_phase), places=10)
            np.testing.assert_array_equal(source, before[0])
            np.testing.assert_array_equal(stereo, before[1])
            json.dumps(result, allow_nan=False)

    def test_common_large_and_small_scale_preserves_ratios_and_physical_powers(self):
        source, stereo = independent_records()
        for scale in (1e150, 1e-150):
            result = measure_response(source*scale, stereo*scale)
            self.assertAlmostEqual(result['source_mean_square'] / (scale*scale), .01, places=14)
            self.assertAlmostEqual(result['left_mean_square'] / (scale*scale), .0125, places=14)
            self.assertAlmostEqual(result['ild_right_minus_left_db'], 0, places=10)
            self.assertAlmostEqual(result['spectral'][0]['left_power_gain'],
                                   1.25 + math.sqrt(2 + math.sqrt(2))/2, places=12)

    def test_phase_branch_and_weight_sign_do_not_change_power(self):
        source, _ = independent_records()
        stereo = np.zeros((2, 32001))
        stereo[0, :32000] = source[0]
        stereo[1, :32000] = -source[0]
        result = measure_response(source, stereo)
        self.assertAlmostEqual(result['ild_right_minus_left_db'], 0.)
        for row in result['spectral']:
            self.assertAlmostEqual(row['right_power_gain'], 1.)
            self.assertAlmostEqual(row['ipd_right_minus_left_rad'], -math.pi)

    def test_shapes_types_and_nonfinite_samples_are_rejected(self):
        source, stereo = independent_records()
        for bad in [source[0], source[:, :-1], source.astype(complex), source.astype(bool),
                    np.full(source.shape, np.nan), np.full(source.shape, np.inf)]:
            with self.subTest(source_shape=bad.shape, dtype=bad.dtype):
                with self.assertRaises(ValueError):
                    measure_response(bad, stereo)
        for bad in [stereo.T, stereo[:, :-1], stereo.astype(complex), np.full(stereo.shape, np.inf)]:
            with self.assertRaises(ValueError):
                measure_response(source, bad)

    def test_silence_absent_lines_and_unrepresentable_powers_are_rejected(self):
        source, stereo = independent_records()
        for bad_source, bad_stereo in [(source*0, stereo), (source, stereo*0),
                                       (np.ones_like(source), stereo),
                                       (source*1e200, stereo*1e200),
                                       (source*1e-200, stereo*1e-200)]:
            with self.assertRaises(ValueError):
                measure_response(bad_source, bad_stereo)
        zero_ear = stereo.copy()
        zero_ear[1] = 0
        with self.assertRaises(ValueError):
            measure_response(source, zero_ear)
        # Valid total powers do not guarantee representable response gains.
        with self.assertRaises(ValueError):
            measure_response(source*1e-150, stereo*1e150)


if __name__ == '__main__':
    unittest.main()
