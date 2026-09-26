"""Explicit cross-module, chapter and research inventory for 117 exercises."""

import json
import re
import unittest
from pathlib import Path

from codes.examples import (aec_advanced_exercises, aec_algorithm_minicases, exercises_engineering,
                            exercises_enhancement, exercises_spatial,
                            tracking_crossing_dropout_demo)
from codes.examples import spatial_precision_exercises, enhancement_step_exercises, tracking_time_exercises
from codes.examples.spectral_subtraction_demo import run_demo as spectral_subtraction_demo
from codes.examples.coarray_covariance_exercise import run_exercise as coarray_exercise
from codes.examples.doa_resolution_trials import run_experiment as doa_resolution_experiment


from codes.examples import spatial_model_exercises, enhancement_structure_exercises, engineering_boundary_exercises, interpolation_exercise

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "spatial_model": {"E02-08", "E04-11", "E05-07", "E12-05"},
    "enhancement_structure": {"E07-07", "E08-11", "E09-09"},
    "engineering_boundary": {"E10-16", "E10-17", "E11-09"},
    "interpolation": {"E13-02"},
    "spatial_precision": {"E02-07", "E04-10", "E05-06"},
    "enhancement_steps": {"E06-21", "E07-06", "E08-08", "E08-09", "E08-10"},
    "time_state": {"E09-07", "E09-08", "E10-15", "E11-08"},
    "spatial": {
        "E01-01", "E01-02", "E02-01", "E02-02", "E02-03", "E03-01",
        "E03-02", "E04-01", "E04-02", "E04-03", "E05-01", "E05-02",
        "E01-03", "E02-04", "E02-05", "E03-03", "E03-04", "E04-04", "E05-03", "E05-04", "E04-05",
        "E02-06", "E03-05", "E03-06", "E04-06", "E04-07", "E04-09", "E05-05",
    },
    "enhancement": {
        "E06-01", "E06-02", "E06-03", "E07-01", "E07-02", "E07-03",
        "E08-01", "E08-02", "E08-03", "E09-01", "E09-02", "E09-03",
        "E06-04", "E06-05", "E07-04", "E07-05", "E08-04", "E08-05", "E09-04", "E09-05",
        "E06-06", "E08-06", "E08-07",
    },
    "aec_minicases": {"E06-07", "E06-08", "E06-09", "E06-10"},
    "aec_advanced": {f"E06-{number:02d}" for number in range(11, 21)},
    "tracking_crossing": {"E09-06"},
    "coarray": {"E03-07"},
    "doa_resolution": {"E04-08"},
    "engineering": {
        "E10-01", "E10-02", "E10-03", "E10-04", "E10-05", "E10-06",
        "E11-01", "E11-02", "E12-01", "E12-02", "E12-03", "E13-01",
        "E10-07", "E10-08", "E10-09", "E10-10", "E10-11", "E10-12", "E10-14",
        "E11-03", "E11-04", "E11-05", "E11-06", "E11-07", "E12-04",
    },
    "spectral_subtraction": {"E10-13"},
}
AEC_CASE_KEYS = {"E06-07": "overlap_save", "E06-08": "ipnlms",
                 "E06-09": "geigel", "E06-10": "delay_polarity"}
RUNNERS = {"spatial_model": spatial_model_exercises.run_exercises,
           "enhancement_structure": enhancement_structure_exercises.run_exercises,
           "engineering_boundary": engineering_boundary_exercises.run_exercises,
           "interpolation": interpolation_exercise.run_exercises,
           "spatial_precision": spatial_precision_exercises.run_exercises,
           "enhancement_steps": enhancement_step_exercises.run_exercises,
           "time_state": tracking_time_exercises.run_exercises,
           "spatial": exercises_spatial.run_exercises,
           "enhancement": exercises_enhancement.run_exercises,
           "aec_advanced": aec_advanced_exercises.run_exercises,
           "tracking_crossing": lambda: {"E09-06": tracking_crossing_dropout_demo.run_experiment()},
           "coarray": lambda: {"E03-07": coarray_exercise()},
           "doa_resolution": lambda: {"E04-08": doa_resolution_experiment(trials=4)},
           "engineering": exercises_engineering.run_exercises,
           "spectral_subtraction": lambda: {"E10-13": spectral_subtraction_demo()},
           "aec_minicases": lambda: {
               exercise_id: aec_algorithm_minicases.run_demo()[case_key]
               for exercise_id, case_key in AEC_CASE_KEYS.items()
           }}
ALL_IDS = set().union(*EXPECTED.values())


def documented_ids(text):
    """Expand only explicit same-chapter ranges such as E01-01～02."""
    found = set(re.findall(r"\bE\d{2}-\d{2}\b", text))
    for chapter, first, last in re.findall(r"\bE(\d{2})-(\d{2})～(\d{2})\b", text):
        if int(last) < int(first):
            raise ValueError("Exercise range must be ascending")
        found.update(f"E{chapter}-{number:02d}" for number in range(int(first), int(last) + 1))
    return found


class ExerciseCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: run() for name, run in RUNNERS.items()}

    def test_independent_inventory_has_117_unique_ids(self):
        self.assertEqual(len(ALL_IDS), 117)
        self.assertEqual(sum(map(len, EXPECTED.values())), 117)

    def test_each_module_returns_exact_assigned_ids(self):
        for name, results in self.results.items():
            with self.subTest(module=name):
                self.assertIsInstance(results, dict)
                self.assertEqual(set(results), EXPECTED[name])
                self.assertTrue(all(isinstance(value, dict) and value for value in results.values()))

    def test_results_are_strict_json_without_numpy_or_nonfinite_values(self):
        for name, results in self.results.items():
            with self.subTest(module=name):
                encoded = json.dumps(results, allow_nan=False, ensure_ascii=False)
                self.assertEqual(set(json.loads(encoded)), EXPECTED[name])

    def test_every_id_appears_in_its_own_chapter(self):
        for exercise_id in sorted(ALL_IDS):
            with self.subTest(exercise_id=exercise_id):
                chapters = list((ROOT / "chapters").glob(exercise_id[1:3] + "_*.md"))
                self.assertEqual(len(chapters), 1)
                text = chapters[0].read_text(encoding="utf-8")
                self.assertRegex(text, rf"\b{re.escape(exercise_id)}\b")

    def test_chapter_and_research_inventories_cover_exactly_117_ids(self):
        chapters = "\n".join(path.read_text(encoding="utf-8")
                             for path in (ROOT / "chapters").glob("*.md"))
        research = (ROOT / "codes/research/05_exercises_and_audio.md").read_text(encoding="utf-8")
        self.assertEqual(documented_ids(chapters), ALL_IDS)
        self.assertEqual(documented_ids(research), ALL_IDS)

    def test_range_parser_does_not_invent_ids_from_partial_labels(self):
        self.assertEqual(documented_ids("E01-01～02、E13-01；E09 章"),
                         {"E01-01", "E01-02", "E13-01"})
        self.assertEqual(documented_ids("E01-010、xE01-01、E01-01x"), set())
        with self.assertRaises(ValueError):
            documented_ids("E01-03～01")


if __name__ == "__main__":
    unittest.main()
