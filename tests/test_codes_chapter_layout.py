"""Guard the chapter-only layout and representative canonical entry points."""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
CHAPTERS = CODES / "chapters"
EXPECTED_DIRS = {*(f"ch{number:02d}" for number in range(12)),
                 "appendix_a", "appendix_b"}


class ChapterLayoutTest(unittest.TestCase):
    def test_all_repository_code_content_is_under_chapters(self):
        # Downloaded upstream repositories and Python bytecode are ignored.
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "codes"],
            cwd=ROOT, capture_output=True, check=True,
        )
        present = [ROOT / name.decode() for name in result.stdout.split(b"\0") if name]
        present = [path for path in present if path.is_file()]
        self.assertTrue(present)
        self.assertEqual([path for path in present if not path.is_relative_to(CHAPTERS)], [])

    def test_chapter_packages_and_module_sources(self):
        directories = {path.name for path in CHAPTERS.iterdir()
                       if path.is_dir() and not path.name.startswith("__")}
        self.assertEqual(directories, EXPECTED_DIRS)
        implementations = [path for path in sorted(CHAPTERS.glob("*/*.py"))
                           if path.name != "__init__.py"]
        self.assertEqual(len(implementations), 36)
        self.assertIn(CHAPTERS / "ch06/aec_affine_projection_demo.py", implementations)
        for path in implementations:
            with self.subTest(path=path):
                chapter = path.parent.name
                module = importlib.import_module(f"codes.chapters.{chapter}.{path.stem}")
                self.assertEqual(Path(module.__file__).resolve(), path.resolve())
        for module_name in (
            "codes.chapters.ch02.core.spectral",
            "codes.chapters.ch06.core.aec",
            "codes.chapters.ch08.examples.gss_teaching_demo",
            "codes.chapters.ch00.cross_chapter.exercises_spatial",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(CHAPTERS))

    def test_module_commands_produce_stable_results(self):
        for chapter, module in (("ch01", "chapter01_experiments"),
                                ("ch04", "chapter04_experiments"),
                                ("appendix_b", "appendix_b_experiments")):
            with self.subTest(module=module):
                command = [sys.executable, "-m", f"codes.chapters.{chapter}.{module}"]
                first = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       timeout=30, check=True).stdout
                second = subprocess.run(command, cwd=ROOT, capture_output=True,
                                        timeout=30, check=True).stdout
                self.assertTrue(first)
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
