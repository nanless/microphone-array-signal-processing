"""Guard the chapter-owned implementations and legacy entry-point contract."""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "codes" / "chapters"
EXPECTED_DIRS = {*(f"ch{number:02d}" for number in range(1, 12)),
                 "appendix_a", "appendix_b"}


class ChapterLayoutTest(unittest.TestCase):
    def test_canonical_modules_share_one_object_with_legacy_imports(self):
        directories = {path.name for path in CHAPTERS.iterdir()
                       if path.is_dir() and not path.name.startswith("__")}
        self.assertEqual(directories, EXPECTED_DIRS)
        implementations = sorted(CHAPTERS.glob("*/*.py"))
        implementations = [path for path in implementations if path.name != "__init__.py"]
        self.assertEqual(len(implementations), 35)
        for path in implementations:
            with self.subTest(path=path):
                chapter = path.parent.name
                canonical = importlib.import_module(f"codes.chapters.{chapter}.{path.stem}")
                legacy = importlib.import_module(f"codes.examples.{path.stem}")
                self.assertIs(legacy, canonical)
                self.assertEqual(Path(canonical.__file__).resolve(), path.resolve())

    def test_direct_script_and_module_commands_keep_same_results(self):
        for chapter, module in (("ch01", "chapter01_experiments"),
                                ("appendix_b", "appendix_b_experiments")):
            with self.subTest(module=module):
                commands = (
                    [sys.executable, "-m", f"codes.chapters.{chapter}.{module}"],
                    [sys.executable, "-m", f"codes.examples.{module}"],
                    [sys.executable, str(ROOT / "codes/examples" / f"{module}.py")],
                )
                outputs = [subprocess.run(command, cwd=ROOT, capture_output=True,
                                          timeout=25, check=True).stdout for command in commands]
                self.assertEqual(outputs[0], outputs[1])
                self.assertEqual(outputs[0], outputs[2])


if __name__ == "__main__":
    unittest.main()
