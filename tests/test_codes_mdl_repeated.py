"""Independent model/score/interval checks for the resampled MDL exercise."""

import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from codes.array_tutorial.doa import mdl_source_count
from codes.examples.mdl_repeated_trials import run_experiment, wilson_interval


class RepeatedMDLTest(unittest.TestCase):
    def test_models_and_reproducibility(self):
        result = run_experiment(trials=9, seed=28)
        self.assertEqual(result, run_experiment(trials=9, seed=28))
        json.dumps(result, allow_nan=False)
        expected = [[5, 5, 1, 1], [5, 5, 1, 1], [1.4, 1.4, 1, 1],
                    [1.4, 1.4, 1, 1], [9, 1, 1, 1], [1, 1, 1, 1], [9, 4, 1, 1]]
        for index, (row, eigenvalues) in enumerate(zip(result["cases"], expected)):
            np.testing.assert_allclose(row["population_eigenvalues_descending"], eigenvalues)
            self.assertEqual(sum(row["selected_count_histogram_k0_to_k3"]), 9)
            self.assertEqual(row["seed_sequence"], [28, index])
            self.assertEqual(row["mdl_physical_source_model_valid"], index not in (4, 6))
            if index < 5:
                self.assertAlmostEqual(row["array_mean_snr_db"], 10*math.log10(2*(.1 if index in (2, 3) else 1)))
            else:
                self.assertIsNone(row["array_mean_snr_db"])

    def test_each_trial_agrees_with_literal_scalar_am_gm_score(self):
        observed = []

        def independent(values, snapshots):
            spectrum = sorted(values, reverse=True)
            scores = []
            for k in range(4):
                tail = spectrum[k:]
                geometric = math.prod(tail)**(1/len(tail))
                arithmetic = sum(tail)/len(tail)
                scores.append(snapshots*len(tail)*math.log(arithmetic/geometric)
                              + .5*k*(8-k)*math.log(snapshots))
            count, tested_scores = mdl_source_count(values, snapshots)
            self.assertEqual(count, min(range(4), key=scores.__getitem__))
            np.testing.assert_allclose(tested_scores, scores, atol=1e-11)
            observed.append(values.copy())
            return count, tested_scores

        with patch("codes.examples.mdl_repeated_trials.mdl_source_count", side_effect=independent):
            run_experiment(trials=3, seed=7)
        self.assertEqual(len(observed), 21)
        self.assertFalse(np.array_equal(observed[0], observed[1]))

    def test_wilson_interval_matches_score_test_roots(self):
        for successes, trials in ((0, 200), (200, 200), (163, 200), (4, 20)):
            lower, upper = wilson_interval(successes, trials)
            z2 = 1.959963984540054**2
            p = successes/trials
            roots = np.sort(np.roots([trials+z2, -2*trials*p-z2, trials*p*p]))
            np.testing.assert_allclose([lower, upper], roots, atol=1e-14)
            self.assertLessEqual(lower, p)
            self.assertGreaterEqual(upper, p)
        self.assertEqual(wilson_interval(0, 200)[0], 0.)
        self.assertEqual(wilson_interval(200, 200)[1], 1.)

    def test_invalid_configuration_is_not_coerced(self):
        for trials in (0, -1, True, 2., "2"):
            with self.subTest(trials=trials), self.assertRaises(ValueError):
                run_experiment(trials=trials)
        for seed in (-1, True, 2., "2"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                run_experiment(seed=seed)
        for pair in ((-1, 2), (3, 2), (0, 0), (True, 2), (1, 2.)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                wilson_interval(*pair)


if __name__ == "__main__":
    unittest.main()
