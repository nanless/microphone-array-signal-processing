"""Offline adversarial fixtures for the pinned archive acquisition boundary."""
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile
import tempfile
import stat
import zipfile
import warnings
import unittest
from unittest.mock import patch

from codes.chapters.ch00.upstream import fetch_archives as fetch


class ArchiveSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / "downloads"
        self.cache = self.root / "cache"
        self.descriptor = b"fixture descriptor, not a package signature\n"
        self.project = {
            "id": "example-1.0", "release": "1.0", "archive_root": "source",
            "archive": {"url": "https://example.invalid/source.tar.xz", "sha256": "0" * 64,
                        "size_bytes": 1},
            "descriptor": {"url": "https://example.invalid/source.dsc",
                           "sha256": hashlib.sha256(self.descriptor).hexdigest()},
            "source_paths": ["README", "src/"]}
        self.default_members = [("source/README", b"license and readme"),
                                ("source/src/main.py", b"raise RuntimeError('must not run')"),
                                ("source/omitted.wav", b"not selected")]

    def archive(self, members=None):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, content in self.default_members if members is None else members:
                member = tarfile.TarInfo(name)
                if isinstance(content, tuple):
                    member.type, member.linkname = content
                    tar.addfile(member)
                else:
                    member.size = len(content)
                    tar.addfile(member, io.BytesIO(content))
        data = lzma.compress(raw.getvalue())
        self.project["archive"].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        return data

    def prepare_cache(self, data):
        self.cache.mkdir(exist_ok=True)
        (self.cache / self.project["archive"]["sha256"]).write_bytes(data)
        (self.cache / self.project["descriptor"]["sha256"]).write_bytes(self.descriptor)

    def create(self):
        self.prepare_cache(self.archive())
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network forbidden")):
            return fetch.acquire(self.project, self.destination, self.cache, download=True)

    def test_fixed_lock_is_valid_and_restricted(self):
        projects = fetch.load_projects()
        item = projects["harktool5-3.5.0"]
        self.assertEqual(item["archive"]["size_bytes"], 1672544)
        self.assertEqual(item["archive"]["sha256"],
                         "26f7f75d8dbe9f8c3e3a83aef160472adf683b130ee12100d4c4e2af52b65d6d")
        self.assertIn("purpose restricted", item["license"])

    def test_report_binds_exact_lock_and_project_set(self):
        report = json.loads((fetch.LOCK_FILE.parent / "ARCHIVE_SOURCE_STATUS.json").read_text())
        projects = fetch.load_projects()
        self.assertEqual(report["lock_sha256"], hashlib.sha256(fetch.LOCK_FILE.read_bytes()).hexdigest())
        self.assertEqual(len(report["projects"]), len(projects))
        self.assertEqual({p["id"]: (p["archive_sha256"], p["descriptor_sha256"])
                          for p in report["projects"]},
                         {key: (p["archive"]["sha256"], p["descriptor"]["sha256"])
                          for key, p in projects.items()})
        for record in report["projects"]:
            self.assertEqual(record["requested_source_paths"], projects[record["id"]]["source_paths"])
            self.assertEqual(record["file_count"], len(record["files"]))
            self.assertEqual(record["execution"], "not_run")

    def test_hash_is_checked_before_any_decompression(self):
        self.archive()
        with patch.object(fetch.lzma, "LZMAFile", side_effect=AssertionError("parsed first")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                fetch.selected_files(b"x" * self.project["archive"]["size_bytes"], self.project)

    def test_reject_all_unsafe_paths_even_when_unselected(self):
        for name in ("../outside", "/absolute", "source/../outside", "source//alias",
                     "source/./alias", "source\\escape", "C:/evil", "other/evil",
                     "source/new\nline"):
            with self.subTest(name=name):
                data = self.archive(self.default_members + [(name, b"evil")])
                with self.assertRaises(ValueError):
                    fetch.selected_files(data, self.project)
                self.assertFalse(self.destination.exists())

    def test_reject_links_and_special_members_even_when_unselected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                     tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind):
                data = self.archive(self.default_members + [("source/omitted", (kind, "../../evil"))])
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    fetch.selected_files(data, self.project)

    def test_reject_duplicate_and_file_directory_collision(self):
        for extra in ([self.default_members[0]], [("source/src", b"collision")]):
            data = self.archive(self.default_members + extra)
            with self.assertRaises(ValueError):
                fetch.selected_files(data, self.project)

    def test_decompression_member_count_and_size_limits(self):
        data = self.archive()
        with patch.object(fetch, "MAX_TAR_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "decompressed"):
                fetch.selected_files(data, self.project)
        with patch.object(fetch, "MAX_FILE_BYTES", 2):
            with self.assertRaisesRegex(ValueError, "member size"):
                fetch.selected_files(data, self.project)
        with patch.object(fetch, "MAX_MEMBERS", 1):
            with self.assertRaisesRegex(ValueError, "member count"):
                fetch.selected_files(data, self.project)

    def test_required_selection_must_exist(self):
        data = self.archive()
        self.project["source_paths"].append("not-in-archive")
        with self.assertRaisesRegex(ValueError, "matched no files"):
            fetch.selected_files(data, self.project)

    def test_fetch_and_readonly_verify_exact_subset_without_execution(self):
        record = self.create()
        self.assertEqual(record["status"], "source_verified")
        self.assertEqual(record["file_count"], 2)
        self.assertEqual(record["execution"], "not_run")
        target = self.destination / self.project["id"]
        self.assertEqual((target / "README").read_bytes(), b"license and readme")
        self.assertEqual((target / "src/main.py").read_bytes(), b"raise RuntimeError('must not run')")
        self.assertFalse((target / "omitted.wav").exists())
        before = {p: p.stat().st_mtime_ns for p in target.rglob("*")}
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network forbidden")):
            self.assertEqual(fetch.acquire(self.project, self.destination, self.cache, download=False), record)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in target.rglob("*")})

    def test_missing_verify_does_not_create_or_fetch(self):
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network forbidden")):
            record = fetch.acquire(self.project, self.destination, self.cache, download=False)
        self.assertEqual(record["status"], "missing")
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.cache.exists())

    def test_modified_file_is_not_overwritten(self):
        self.create()
        readme = self.destination / self.project["id"] / "README"
        readme.write_bytes(b"user changes")
        for download in (True, False):
            with self.assertRaisesRegex(ValueError, "modified"):
                fetch.acquire(self.project, self.destination, self.cache, download=download)
            self.assertEqual(readme.read_bytes(), b"user changes")

    def test_extra_file_and_extra_empty_directory_rejected(self):
        self.create()
        target = self.destination / self.project["id"]
        for extra in (target / "extra", target / "empty"):
            if extra.name == "empty":
                extra.mkdir()
            else:
                extra.write_bytes(b"extra")
            with self.assertRaises(ValueError):
                fetch.acquire(self.project, self.destination, self.cache, download=False)
            extra.rmdir() if extra.is_dir() else extra.unlink()

    def test_missing_file_is_not_repaired(self):
        self.create()
        path = self.destination / self.project["id"] / "README"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            fetch.acquire(self.project, self.destination, self.cache, download=True)
        self.assertFalse(path.exists())

    def test_local_symlink_file_directory_target_and_cache_rejected(self):
        self.create()
        target = self.destination / self.project["id"]
        external = self.root / "external"
        external.mkdir()
        for place in (target / "linked-file", target / "linked-dir"):
            place.symlink_to(external if place.name == "linked-dir" else target / "README")
            with self.assertRaises(ValueError):
                fetch.acquire(self.project, self.destination, self.cache, download=False)
            place.unlink()
        alias = self.root / "alias"
        alias.symlink_to(self.cache)
        with self.assertRaises(ValueError):
            fetch.acquire(self.project, self.destination, alias, download=False)
        other = self.root / "other"
        other.mkdir()
        (other / self.project["id"]).symlink_to(target)
        with self.assertRaises(ValueError):
            fetch.acquire(self.project, other, self.cache, download=False)

    def test_corrupt_cache_refused_not_redownloaded(self):
        self.create()
        cache = self.cache / self.project["archive"]["sha256"]
        data = cache.read_bytes()
        cache.write_bytes(b"x" + data[1:])
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                fetch.acquire(self.project, self.destination, self.cache, download=True)

    def test_download_verifies_before_cache_publication(self):
        self.archive()
        with patch.object(fetch, "urlopen", return_value=io.BytesIO(b"bad download")):
            with self.assertRaises(ValueError):
                fetch.artifact(self.project["archive"], self.cache, fetch.MAX_ARCHIVE_BYTES, True)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_invalid_xz_is_recorded_as_failure_after_matching_hash(self):
        data = b"not an xz stream but deliberately pinned for this test"
        self.project["archive"].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        self.prepare_cache(data)
        report = self.root / "format-failure.json"
        with patch.object(fetch, "load_projects", return_value={self.project["id"]: self.project}):
            result = fetch.main(["--project", self.project["id"], "--destination", str(self.destination),
                                 "--cache", str(self.cache), "--report", str(report)])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(report.read_text())["projects"][0]["status"], "failed")
        self.assertFalse((self.destination / self.project["id"]).exists())

    def test_report_records_actual_failure(self):
        self.create()
        (self.destination / self.project["id"] / "README").write_bytes(b"changed")
        report = self.root / "report.json"
        with patch.object(fetch, "load_projects", return_value={self.project["id"]: self.project}):
            result = fetch.main(["--verify", "--destination", str(self.destination),
                                 "--cache", str(self.cache), "--report", str(report)])
        self.assertEqual(result, 1)
        records = json.loads(report.read_text())["projects"]
        self.assertEqual(records[0]["status"], "failed")
        self.assertIn("modified", records[0]["error"])


class ZipSourceTests(ArchiveSourceTests):
    """Run the shared acquisition checks against ZIP, plus ZIP-specific hazards."""
    def archive(self, members=None):
        raw = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in self.default_members if members is None else members:
                    member = zipfile.ZipInfo(name)
                    member.create_system = 3
                    if isinstance(content, tuple):
                        types = {tarfile.SYMTYPE: stat.S_IFLNK, tarfile.LNKTYPE: stat.S_IFLNK,
                                 tarfile.FIFOTYPE: stat.S_IFIFO, tarfile.CHRTYPE: stat.S_IFCHR,
                                 tarfile.BLKTYPE: stat.S_IFBLK}
                        member.external_attr = (types[content[0]] | 0o644) << 16
                        content = content[1].encode()
                    else:
                        member.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(member, content)
        data = raw.getvalue()
        self.project["archive_format"] = "zip"
        self.project["archive"]["url"] = "https://example.invalid/source.zip"
        self.project["archive"].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        return data

    def test_decompression_member_count_and_size_limits(self):
        data = self.archive()
        for limit, value, message in (("MAX_TAR_BYTES", 10, "total member"),
                                       ("MAX_FILE_BYTES", 2, "member size"),
                                       ("MAX_MEMBERS", 1, "member count")):
            with self.subTest(limit=limit), patch.object(fetch, limit, value):
                with self.assertRaisesRegex(ValueError, message):
                    fetch.selected_files(data, self.project)

    def test_invalid_xz_is_recorded_as_failure_after_matching_hash(self):
        self.project["archive_format"] = "zip"
        super().test_invalid_xz_is_recorded_as_failure_after_matching_hash()

    def test_encrypted_unselected_entry_rejected(self):
        data = self.archive()
        original = zipfile.ZipFile.infolist
        def entries(archive):
            members = original(archive)
            members[-1].flag_bits |= 1
            return members
        with patch.object(zipfile.ZipFile, "infolist", entries):
            with self.assertRaisesRegex(ValueError, "encrypted"):
                fetch.selected_files(data, self.project)

    def test_crc_corruption_rejected_before_publication(self):
        data = self.archive()
        # ZIP_STORED fixture: alter selected payload, retain its CRC, then pin
        # the corrupt archive to isolate the member-integrity boundary.
        corrupt = data.replace(b"license and readme", b"LICENSE and readme", 1)
        self.assertNotEqual(corrupt, data)
        self.project["archive"]["sha256"] = hashlib.sha256(corrupt).hexdigest()
        with self.assertRaises(zipfile.BadZipFile):
            fetch.selected_files(corrupt, self.project)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
