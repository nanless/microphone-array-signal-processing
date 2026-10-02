"""Offline source-identity and report-boundary contracts, with isolated fixtures."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch00 import io_contracts
from codes.chapters.ch00.upstream import fetch_archives as archive
from codes.chapters.ch00.upstream import fetch_upstreams as git


# An independent, explicit fixture for the published source-selection policy.
EXCLUSIONS = (
    "wav", "flac", "mp3", "mp4", "ogg", "pt", "pth", "ckpt", "onnx", "tflite",
    "h5", "hdf5", "npz", "npy", "so", "dll", "dylib", "zip", "tar", "gz",
    "bin", "pb", "pkl", "pickle", "mat", "safetensors", "whl", "a", "o",
    "exe", "wasm", "weights",
)


class SourceAcquisitionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / "downloads"
        self.cache = self.root / "archive-cache"
        self.project = {
            "id": "fixture", "url": "https://example.invalid/official.git",
            "revision": "a" * 40, "entrypoints": [],
            "source_paths": ["source labview files/", "LICENSE"], "fetch_enabled": True,
        }
        self.archive_project = {
            "id": "fixture-1.0", "release": "1.0", "archive_root": "source",
            "archive": {"url": "https://example.invalid/source.zip", "sha256": "b" * 64,
                        "size_bytes": 1},
            "descriptor": {"url": "https://example.invalid/description", "sha256": "c" * 64},
            "archive_format": "zip", "source_paths": ["README", "src/"],
        }

    def invoke(self, module, report, *, verify=True):
        if module is git:
            args = argparse.Namespace(list=False, project=None, all=not verify, verify=verify,
                                      destination=self.destination, report=report)
            with patch.object(git, "parse_args", return_value=args):
                return git.main()
        argv = ["--verify" if verify else "--project", *([] if verify else ["fixture-1.0"]),
                "--destination", str(self.destination), "--cache", str(self.cache)]
        if report is not None:
            argv += ["--report", str(report)]
        return archive.main(argv)

    @contextlib.contextmanager
    def isolated_sources(self, module):
        project = self.project if module is git else self.archive_project
        with patch.object(module, "load_projects", return_value={project["id"]: project}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def sparse_fixture(self, patterns):
        info = self.destination / "fixture" / ".git" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "sparse-checkout").write_text("\n".join(patterns) + "\n")
        with patch.object(git, "run_git", side_effect=(self.project["revision"], self.project["url"], "")):
            return git.inspect_project(self.project, self.destination)

    def test_complete_policy_and_space_path_are_verified(self):
        rules = ["/source labview files/", "/LICENSE"] + [f"!**/*.{x}" for x in EXCLUSIONS]
        record = self.sparse_fixture(rules)
        self.assertEqual(record["status"], "source_verified")
        self.assertTrue(record["source_selection_verified"])
        self.assertEqual(record["observed_sparse_patterns"], rules)
        self.assertEqual(record["expected_sparse_patterns"], rules)

    def test_old_twenty_and_extra_negative_policy_are_not_verified(self):
        includes = ["/source labview files/", "/LICENSE"]
        normal = [f"!**/*.{x}" for x in EXCLUSIONS]
        for rules in (includes + normal[:20], includes + normal + ["!/source labview files/core.py"],
                      includes + normal[:-1], includes + list(reversed(normal))):
            with self.subTest(rules=rules):
                record = self.sparse_fixture(rules)
                self.assertEqual(record["status"], "source_selection_mismatch")
                self.assertFalse(record["source_selection_verified"])
                self.assertEqual(record["observed_sparse_patterns"], rules)

    def test_extra_positive_source_and_comments_cannot_hide_rule_changes(self):
        normal = ["/source labview files/", "/LICENSE"] + [f"!**/*.{x}" for x in EXCLUSIONS]
        for rules in (["/extra/", *normal], [*normal, "# local policy"], [*normal, ""]):
            with self.subTest(rules=rules):
                self.assertEqual(self.sparse_fixture(rules)["status"], "source_selection_mismatch")

    def test_preserved_nonsparse_checkout_requires_no_requested_subset(self):
        (self.destination / "fixture" / ".git").mkdir(parents=True)
        for project, expected in ((self.project, "source_selection_mismatch"),
                                  ({k: v for k, v in self.project.items() if k != "source_paths"},
                                   "source_verified")):
            with patch.object(git, "run_git", side_effect=(project["revision"], project["url"], "")):
                record = git.inspect_project(project, self.destination)
            self.assertEqual(record["status"], expected)
            self.assertEqual(record["asset_policy"], "existing_checkout_preserved")

    def test_both_lock_readers_reject_duplicate_and_nonfinite_json(self):
        for module, project in ((git, self.project), (archive, self.archive_project)):
            legal = json.dumps({"schema_version": 1, "projects": [project]})
            bad = (legal.replace('"id":', '"id": "discarded", "id":', 1),
                   legal[:-1] + ', "diagnostic": NaN}',
                   legal[:-1] + ', "diagnostic": Infinity}',
                   legal[:-1] + ', "diagnostic": 1e999}')
            for index, text in enumerate(bad):
                lock = self.root / f"{module.__name__}-{index}.json"
                lock.write_text(text)
                with self.subTest(module=module.__name__, index=index), \
                     patch.object(module, "LOCK_FILE", lock), self.assertRaises(ValueError):
                    module.load_projects()

    def test_nearby_json_names_and_independent_nested_ids_are_legal(self):
        for module, project in ((git, self.project), (archive, self.archive_project)):
            lock = self.root / f"{module.__name__}-legal.json"
            lock.write_text(json.dumps({"schema_version": 1, "projects": [project],
                                       "metadata": {"id": "separate", "id_suffix": "legal", "值": 1.2}}))
            with patch.object(module, "LOCK_FILE", lock):
                self.assertEqual(set(module.load_projects()), {project["id"]})

    def test_report_link_parent_traversal_and_special_leaf_refused_before_acquisition(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "keep.json"
        sentinel.write_bytes(b"KEEP")
        leaf = self.root / "leaf.json"
        leaf.symlink_to(sentinel)
        alias = self.root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        hard = self.root / "hard.json"
        os.link(sentinel, hard)
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        invalid = (leaf, alias / "keep.json", alias / ".." / "new.json", outside, hard, fifo)
        for module in (git, archive):
            operation = "fetch_project" if module is git else "acquire"
            for report in invalid:
                with self.subTest(module=module.__name__, report=str(report)), \
                     self.isolated_sources(module), patch.object(module, operation) as run, \
                     patch.object(module, "write_json_report") as write:
                    with self.assertRaises(ValueError):
                        self.invoke(module, report, verify=False)
                    run.assert_not_called()
                    write.assert_not_called()
                self.assertEqual(sentinel.read_bytes(), b"KEEP")
                self.assertFalse(self.destination.exists())
                self.assertFalse(self.cache.exists())

    def test_report_protected_source_cache_locks_and_parent_paths_refused(self):
        for module in (git, archive):
            other_lock = git.ARCHIVE_LOCK_FILE if module is git else archive.GIT_LOCK_FILE
            targets = (self.destination / "result.json", self.destination,
                       module.LOCK_FILE, other_lock, module.SHARED_IO_FILE,
                       Path(module.__file__), git.DEFAULT_DESTINATION / "elsewhere.json",
                       self.root)
            if module is archive:
                targets += (self.cache / "result.json",)
            for report in targets:
                operation = "fetch_project" if module is git else "acquire"
                with self.subTest(module=module.__name__, report=str(report)), \
                     self.isolated_sources(module), patch.object(module, operation) as run, \
                     patch.object(module, "write_json_report") as write:
                    with self.assertRaises(ValueError):
                        self.invoke(module, report, verify=False)
                    run.assert_not_called()
                    write.assert_not_called()

    def test_valid_sibling_report_replaces_ordinary_file_and_binds_shared_io(self):
        report = self.root / "downloads-report.json"
        for module in (git, archive):
            report.write_bytes(b"old ordinary report")
            with self.isolated_sources(module):
                self.assertEqual(self.invoke(module, report), 1)
            data = io_contracts.strict_json_loads(report.read_bytes())
            self.assertEqual(data["projects"][0]["status"], "missing")
            self.assertEqual(data["lock_sha256"], hashlib.sha256(module.LOCK_FILE.read_bytes()).hexdigest())
            self.assertEqual(set(data["report_sources"]), {
                Path(module.__file__).relative_to(module.REPOSITORY_ROOT).as_posix(),
                "codes/chapters/ch00/io_contracts.py"})
            for name, digest in data["report_sources"].items():
                self.assertEqual(digest, hashlib.sha256((module.REPOSITORY_ROOT / name).read_bytes()).hexdigest())
            self.assertFalse(self.destination.exists())
            self.assertFalse(self.cache.exists())

    def test_replace_failure_preserves_previous_report_and_cleans_temporary_file(self):
        report = self.root / "report.json"
        report.write_bytes(b"KEEP")
        for module in (git, archive):
            with self.isolated_sources(module), patch.object(io_contracts.os, "replace", side_effect=OSError("fixture failure")):
                with self.assertRaisesRegex(OSError, "fixture failure"):
                    self.invoke(module, report)
            self.assertEqual(report.read_bytes(), b"KEEP")
            self.assertEqual(list(self.root.glob(".report-*")), [])

    def test_no_report_is_readonly_and_uses_no_network(self):
        for module in (git, archive):
            with self.isolated_sources(module), patch.object(module, "write_json_report") as write:
                self.assertEqual(self.invoke(module, None), 1)
                write.assert_not_called()
            self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
