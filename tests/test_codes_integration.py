import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch00.upstream.fetch_upstreams import inspect_project, run_git, validate_project


ROOT = Path(__file__).resolve().parents[1]


class CodesIntegrationTests(unittest.TestCase):
    def test_documented_examples_run_from_repository_root(self):
        modules = (
            "codes.chapters.ch00.cross_chapter.ch02_05_baselines",
            "codes.chapters.ch00.cross_chapter.ch06_09_baselines",
            "codes.chapters.ch10.examples.ch10_engineering_baselines",
        )
        for module in modules:
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-m", module],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(completed.stdout.strip())

    def test_upstream_lock_uses_full_immutable_git_revisions(self):
        lock_path = ROOT / "codes" / "chapters" / "ch00" / "SOURCES.lock.json"
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        ids = [project["id"] for project in data["projects"]]
        self.assertEqual(len(ids), len(set(ids)))
        for project in data["projects"]:
            self.assertRegex(project["revision"], re.compile(r"^[0-9a-f]{40}$"))
            self.assertTrue(project["url"].startswith("https://"))
            self.assertIn("license", project)

    def test_upstream_list_is_offline_and_matches_lock(self):
        completed = subprocess.run(
            [sys.executable, "-m", "codes.chapters.ch00.upstream.fetch_upstreams", "--list"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("pyroomacoustics", completed.stdout)
        self.assertIn("fetch", completed.stdout)

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
            {**valid, "entrypoints": ["../outside.py"]},
            {**valid, "entrypoints": ["/absolute.py"]},
            {**valid, "source_paths": ["../outside"]},
            {**valid, "source_paths": ["*.py"]},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_project(changed)

    def test_source_selection_accepts_literal_spaces_without_pattern_syntax(self):
        project = {"id": "fixture", "url": "https://example.invalid/official.git",
                   "revision": "a" * 40, "entrypoints": [],
                   "source_paths": ["source labview files/", "LICENSE"]}
        validate_project(project)
        for path in ("source\nfiles/", "source\tfiles/", "source*/", "source[1]/",
                     "/source files/", "source files/../outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_project({**project, "source_paths": [path]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "fixture" / ".git" / "info"
            info.mkdir(parents=True)
            (info / "sparse-checkout").write_text(
                "/source labview files/\n/LICENSE\n!**/*.exe\n")
            with patch("codes.chapters.ch00.upstream.fetch_upstreams.run_git",
                       side_effect=(project["revision"], project["url"], "")):
                result = inspect_project(project, root)
        self.assertEqual(result["status"], "source_verified")
        self.assertTrue(result["source_selection_verified"])
        self.assertEqual(result["requested_source_paths"], project["source_paths"])

    def test_git_environment_cannot_redirect_repository_or_inject_config(self):
        injected = {"GIT_DIR": "/outside/.git", "GIT_WORK_TREE": "/outside",
                    "GIT_INDEX_FILE": "/outside/index", "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.worktree", "GIT_CONFIG_VALUE_0": "/outside"}
        with patch.dict(os.environ, injected):
            with patch("codes.chapters.ch00.upstream.fetch_upstreams.subprocess.run") as execute:
                execute.return_value.stdout = "ok"
                run_git(["status"], cwd=ROOT)
                environment = execute.call_args.kwargs["env"]
                for key in injected:
                    self.assertNotIn(key, environment)
                self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)

    def test_source_selection_reports_actual_rules_and_rejects_stale_subset(self):
        project = {"id": "fixture", "url": "https://example.invalid/official.git",
                   "revision": "a" * 40, "entrypoints": [], "source_paths": ["README.md"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "fixture" / ".git" / "info"
            info.mkdir(parents=True)
            (info / "sparse-checkout").write_text("/metaaf/\n/README.md\n!**/*.bin\n")
            with patch("codes.chapters.ch00.upstream.fetch_upstreams.run_git",
                       side_effect=(project["revision"], project["url"], "")):
                result = inspect_project(project, root)
            self.assertEqual(result["status"], "source_selection_mismatch")
            self.assertFalse(result["source_selection_verified"])
            self.assertIn("/metaaf/", result["observed_sparse_patterns"])

    def test_checkout_verification_never_accepts_wrong_remote_or_dirty_files(self):
        project = {"id": "fixture", "url": "https://example.invalid/official.git",
                   "revision": "a" * 40, "entrypoints": ["missing.py"], "fetch_enabled": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(inspect_project(project, root)["status"], "missing")
            (root / "fixture" / ".git").mkdir(parents=True)
            for outputs in (("b" * 40, project["url"]),
                            (project["revision"], "https://example.invalid/other.git"),
                            (project["revision"], project["url"], " M algorithm.py")):
                with patch("codes.chapters.ch00.upstream.fetch_upstreams.run_git", side_effect=outputs):
                    with self.assertRaises(ValueError):
                        inspect_project(project, root)
            with patch("codes.chapters.ch00.upstream.fetch_upstreams.run_git",
                       side_effect=(project["revision"], project["url"], "")):
                result = inspect_project(project, root)
                self.assertEqual(result["status"], "entrypoints_missing")
                self.assertEqual(result["missing_entrypoints"], ["missing.py"])
                self.assertEqual(result["execution"], "not_run")

    def test_checkout_verification_rejects_symlink_before_invoking_git(self):
        project = {"id": "fixture", "url": "https://example.invalid/official.git", "revision": "a" * 40}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture").symlink_to(root / "elsewhere", target_is_directory=True)
            with patch("codes.chapters.ch00.upstream.fetch_upstreams.run_git") as git:
                with self.assertRaises(ValueError):
                    inspect_project(project, root)
                git.assert_not_called()


if __name__ == "__main__":
    unittest.main()
