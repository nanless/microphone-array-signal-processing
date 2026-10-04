"""Explicit cross-module, chapter and research inventory for 343 exercises."""

import json
import re
import unittest
from pathlib import Path

from codes.chapters.ch14 import chapter14_exercises
from codes.chapters.ch15 import chapter15_exercises
from codes.chapters.ch06 import aec_advanced_exercises
from codes.chapters.ch06 import aec_algorithm_minicases
from codes.chapters.ch00.cross_chapter import exercises_engineering
from codes.chapters.ch00.cross_chapter import exercises_enhancement
from codes.chapters.ch00.cross_chapter import exercises_spatial
from codes.chapters.ch09 import tracking_crossing_dropout_demo
from codes.chapters.ch00.cross_chapter import spatial_precision_exercises
from codes.chapters.ch00.cross_chapter import enhancement_step_exercises
from codes.chapters.ch00.cross_chapter import tracking_time_exercises
from codes.chapters.ch10.spectral_subtraction_demo import run_demo as spectral_subtraction_demo
from codes.chapters.ch03.coarray_covariance_exercise import run_exercise as coarray_exercise
from codes.chapters.ch04.doa_resolution_trials import run_experiment as doa_resolution_experiment


from codes.chapters.ch01 import chapter01_experiments
from codes.chapters.ch02 import chapter02_experiments
from codes.chapters.ch03 import chapter03_experiments
from codes.chapters.ch04 import chapter04_experiments
from codes.chapters.ch05 import chapter05_experiments
from codes.chapters.ch06 import chapter06_experiments
from codes.chapters.ch06.aec_affine_projection_demo import run_demo as apa_demo
from codes.chapters.ch06.core.apa_audio import run_experiment as apa_audio_experiment
from codes.chapters.ch06.examples.generate_reference_audio import prepare_assets as reference_audio_assets
from codes.chapters.ch07 import chapter07_experiments
from codes.chapters.ch08 import chapter08_experiments
from codes.chapters.ch09 import chapter09_experiments
from codes.chapters.ch10 import chapter10_experiments
from codes.chapters.ch11 import chapter11_experiments
from codes.chapters.appendix_a import appendix_a_experiments
from codes.chapters.appendix_b import appendix_b_experiments
from codes.chapters.ch00.cross_chapter import spatial_model_exercises
from codes.chapters.ch00.cross_chapter import enhancement_structure_exercises
from codes.chapters.ch00.cross_chapter import engineering_boundary_exercises
from codes.chapters.appendix_b import interpolation_exercise

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "distributed": {f"E15-{n:02d}" for n in range(1, 25)},
    "imaging": {f"E14-{n:02d}" for n in range(1, 17)},
    "apa": {f"E06-{n:02d}" for n in range(34, 39)},
    "apa_audio": {"E06-39"},
    "reference_audio": {"E06-42"},
    "appendix_b": {f"E13-{n:02d}" for n in range(3, 15)},
    "appendix_a": {f"E12-{n:02d}" for n in range(6, 20)},
    "chapter11": {f"E11-{n:02d}" for n in range(10, 26)},
    "chapter10": {f"E10-{n:02d}" for n in range(18, 34)},
    "chapter09": {f"E09-{n:02d}" for n in range(10, 24)},
    "chapter08": {f"E08-{n:02d}" for n in range(12, 30)},
    "chapter07": {f"E07-{n:02d}" for n in range(8, 22)},
    "chapter06": {f"E06-{n:02d}" for n in range(22, 34)} | {"E06-40", "E06-41"},
    "chapter05": {f"E05-{n:02d}" for n in range(8, 25)},
    "chapter04": {f"E04-{n:02d}" for n in range(12, 26)},
    "chapter03": {f"E03-{n:02d}" for n in range(8, 19)},
    "chapter02": {f"E02-{n:02d}" for n in range(9, 21)},
    "chapter01": {"E01-04", "E01-05", "E01-06", "E01-07", "E01-08", "E01-09", "E01-10"},
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
RUNNERS = {"distributed": lambda: chapter15_exercises.run_experiments()["exercises"], "imaging": lambda: chapter14_exercises.run_experiments()["exercises"], "apa": lambda: {k: v for k, v in apa_demo().items() if k != "scope"}, "apa_audio": lambda: {"E06-39": apa_audio_experiment()[1]}, "reference_audio": lambda: {"E06-42": reference_audio_assets()[1]}, "appendix_b": appendix_b_experiments.run_exercises,
           "appendix_a": appendix_a_experiments.run_experiments,
           "chapter11": chapter11_experiments.run_experiments,
           "chapter09": chapter09_experiments.run_experiments,
           "chapter10": chapter10_experiments.run_experiments,
           "chapter08": chapter08_experiments.run_experiments,
           "chapter07": chapter07_experiments.run_experiments,
           "chapter06": chapter06_experiments.run_experiments,
           "chapter05": chapter05_experiments.run_exercises,
           "chapter04": chapter04_experiments.run_exercises,
           "chapter03": chapter03_experiments.run_exercises,
           "chapter02": chapter02_experiments.run_exercises,
           "chapter01": chapter01_experiments.run_exercises,
           "spatial_model": spatial_model_exercises.run_exercises,
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

    def test_independent_inventory_has_343_unique_ids(self):
        self.assertEqual(len(ALL_IDS), 343)
        self.assertEqual(sum(map(len, EXPECTED.values())), 343)

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

    def test_chapter_and_research_inventories_cover_exactly_343_ids(self):
        chapters = "\n".join(path.read_text(encoding="utf-8")
                             for path in (ROOT / "chapters").glob("*.md"))
        research = (ROOT / "codes/chapters/ch00/research/05_exercises_and_audio.md").read_text(encoding="utf-8")
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
