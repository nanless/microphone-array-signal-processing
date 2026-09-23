"""Fraction-oracle and boundary tests for the scalar Eq. (6-9) teaching case."""

from fractions import Fraction
import json
import unittest

from codes.examples.aec_kalman_scalar_demo import run_demo, scalar_kalman_step


class TestScalarKalmanStep(unittest.TestCase):
    def test_chapter_real_case_matches_independent_fraction_values(self):
        result = scalar_kalman_step(
            previous_weight=.2, previous_variance=.5, transition=1.,
            process_variance=.1, reference=2., observation=1.4,
            observation_variance=.4)
        # Hand fractions: P^-=3/5, E^-=1, S=14/5, K=3/7;
        # W+=1/5+3/7=22/35, P+=(1-6/7)*3/5=3/35,
        # E+=7/5-2*(22/35)=1/7.
        expected = {
            "prior_weight": Fraction(1, 5),
            "prior_variance": Fraction(3, 5),
            "prior_error": Fraction(1),
            "innovation_variance": Fraction(14, 5),
            "gain": Fraction(3, 7),
            "posterior_weight": Fraction(22, 35),
            "posterior_variance": Fraction(3, 35),
            "posterior_error": Fraction(1, 7),
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(result[key].real, float(value), places=12)

    def test_larger_assumed_observation_variance_reduces_gain(self):
        common = dict(previous_weight=.2, previous_variance=.5, transition=1.,
                      process_variance=.1, reference=2., observation=1.4)
        baseline = scalar_kalman_step(**common, observation_variance=.4)
        cautious = scalar_kalman_step(**common, observation_variance=4.)
        self.assertAlmostEqual(cautious["gain"].real, float(Fraction(3, 16)), places=12)
        self.assertAlmostEqual(cautious["posterior_weight"].real,
                               float(Fraction(31, 80)), places=12)
        self.assertAlmostEqual(cautious["posterior_variance"],
                               float(Fraction(3, 8)), places=12)
        self.assertAlmostEqual(cautious["posterior_error"].real,
                               float(Fraction(5, 8)), places=12)
        self.assertEqual(cautious["prior_error"], baseline["prior_error"])
        self.assertLess(abs(cautious["gain"]), abs(baseline["gain"]))

    def test_complex_reference_uses_conjugate_in_gain(self):
        result = scalar_kalman_step(
            previous_weight=0., previous_variance=1., transition=1.,
            process_variance=0., reference=1j, observation=1.,
            observation_variance=1.)
        # For X=i, S=2; the numerator P^- conj(X)=-i. A missing conjugate
        # would give +i/2 and the wrong posterior prediction.
        self.assertAlmostEqual(result["gain"].real, 0.)
        self.assertAlmostEqual(result["gain"].imag, -.5)
        self.assertAlmostEqual(result["posterior_weight"].imag, -.5)
        self.assertAlmostEqual(result["posterior_error"].real, .5)
        self.assertAlmostEqual(result["posterior_variance"], .5)

    def test_no_reference_leaves_prior_unchanged(self):
        result = scalar_kalman_step(
            previous_weight=.2, previous_variance=.5, transition=1.,
            process_variance=.1, reference=0., observation=1.4,
            observation_variance=.4)
        self.assertEqual(result["gain"], 0.)
        self.assertAlmostEqual(result["posterior_weight"].real, .2)
        self.assertAlmostEqual(result["posterior_variance"], .6)
        self.assertAlmostEqual(result["posterior_error"].real, 1.4)

    def test_invalid_variances_and_non_numeric_or_overflowing_input(self):
        base = dict(previous_weight=.2, previous_variance=.5, transition=1.,
                    process_variance=.1, reference=2., observation=1.4,
                    observation_variance=.4)
        bad_cases = (
            ("previous_variance", -.1), ("process_variance", -1.),
            ("observation_variance", 0.), ("observation_variance", 1 + 0j),
            ("reference", "2"), ("reference", True),
            ("observation", float("nan")), ("transition", float("inf")),
            ("reference", 1e308),
        )
        for key, value in bad_cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                scalar_kalman_step(**{**base, key: value})

    def test_demo_is_finite_json_and_states_scope(self):
        result = run_demo()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["chapter_inputs"]["observation"], 1.4)
        self.assertAlmostEqual(result["chapter_step"]["posterior_weight"], 22 / 35)
        self.assertEqual(result["complex_reference_check"]["gain"],
                         {"real": 0., "imag": -.5})
        self.assertIn("not complete FDKF/PBFDKF", result["scope"])
        self.assertIn("not measured double talk", result["scope"])


if __name__ == "__main__":
    unittest.main()
