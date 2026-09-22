import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

from codes.upstream.fetch_upstreams import validate_project


ROOT = Path(__file__).resolve().parents[1]


class CodesIntegrationTests(unittest.TestCase):
    def test_documented_examples_run_from_repository_root(self):
        scripts = (
            "codes/examples/ch02_05_baselines.py",
            "codes/examples/ch06_09_baselines.py",
            "codes/examples/ch10_engineering_baselines.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                completed = subprocess.run(
                    [sys.executable, script],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(completed.stdout.strip())

    def test_upstream_lock_uses_full_immutable_git_revisions(self):
        lock_path = ROOT / "codes" / "SOURCES.lock.json"
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        ids = [project["id"] for project in data["projects"]]
        self.assertEqual(len(ids), len(set(ids)))
        for project in data["projects"]:
            self.assertRegex(project["revision"], re.compile(r"^[0-9a-f]{40}$"))
            self.assertTrue(project["url"].startswith("https://"))
            self.assertIn("license", project)

    def test_upstream_list_is_offline_and_matches_lock(self):
        completed = subprocess.run(
            [sys.executable, "codes/upstream/fetch_upstreams.py", "--list"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("pyroomacoustics", completed.stdout)
        self.assertIn("index-only", completed.stdout)

    def test_upstream_manifest_rejects_unsafe_identifiers_and_moving_refs(self):
        valid = {
            "id": "safe_project",
            "url": "https://example.invalid/project.git",
            "revision": "a" * 40,
        }
        validate_project(valid)
        for changed in (
            {**valid, "id": "../escape"},
            {**valid, "url": "file:///tmp/project"},
            {**valid, "url": "https://token@example.invalid/project.git"},
            {**valid, "revision": "main"},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_project(changed)


if __name__ == "__main__":
    unittest.main()
