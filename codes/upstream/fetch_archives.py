#!/usr/bin/env python3
"""Fetch pinned .tar.xz source subsets; never build, import or execute upstream code.

Only standard-library code is used. Verify compressed bytes before parsing, bound
both compressed and decompressed sizes, reject unsafe members even when omitted,
and write ordinary files with exclusive creation. Existing targets are verified,
never repaired or replaced. --verify is offline and read-only except --report.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

LOCK_FILE = Path(__file__).resolve().parents[1] / "ARCHIVE_SOURCES.lock.json"
DEFAULT_DESTINATION = Path(__file__).resolve().parent / "_downloads"
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_DESCRIPTOR_BYTES = 128 * 1024
MAX_TAR_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 10000


def safe_name(name: str) -> str:
    """Literal POSIX paths only; reject aliases, Windows paths and traversal."""
    if not isinstance(name, str) or not name or len(name) > 1024:
        raise ValueError("invalid path")
    value = name[:-1] if name.endswith("/") else name
    if (not value or value.startswith("/") or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError(f"unsafe path: {name!r}")
    return value


def validate_project(project: dict) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}(?:\.[a-z0-9_-]+)*", project["id"]):
        raise ValueError("unsafe project id")
    if "/" in safe_name(project["archive_root"]):
        raise ValueError("archive_root must have one component")
    for key in ("archive", "descriptor"):
        item = project[key]
        url = urlparse(item["url"])
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.fragment):
            raise ValueError("invalid unauthenticated HTTP(S) URL")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("full lowercase SHA-256 required")
    size = project["archive"]["size_bytes"]
    if not isinstance(size, int) or not 0 < size <= MAX_ARCHIVE_BYTES:
        raise ValueError("archive size outside limit")
    paths = project["source_paths"]
    if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
        raise ValueError("nonempty unique source_paths required")
    for name in paths:
        safe_name(name)


def load_projects() -> dict[str, dict]:
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported lock schema")
    result = {}
    for project in data["projects"]:
        validate_project(project)
        if project["id"] in result:
            raise ValueError("duplicate project id")
        result[project["id"]] = project
    return result


def checked_bytes(data: bytes, item: dict, limit: int) -> bytes:
    if len(data) > limit or ("size_bytes" in item and len(data) != item["size_bytes"]):
        raise ValueError("artifact size mismatch or limit exceeded")
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("artifact SHA-256 mismatch")
    return data


def read_regular(path: Path, limit: int) -> bytes:
    # O_NOFOLLOW closes the last-component symlink race; callers also reject
    # symlink directories. Concurrent hostile mutation of parent dirs is outside
    # this local, single-writer acquisition tool's threat model.
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError(f"not a bounded, single-link regular file: {path}")
        return stream.read(limit + 1)


def local_root(path: Path, create: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"symlink directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def cache_path(cache: Path, item: dict) -> Path:
    return cache / item["sha256"]


def artifact(item: dict, cache: Path, limit: int, download: bool) -> bytes:
    path = cache_path(cache, item)
    if path.exists() or path.is_symlink():
        return checked_bytes(read_regular(path, limit), item, limit)
    if not download:
        raise FileNotFoundError(f"missing verified artifact cache: {path}")
    cache = local_root(cache, create=True)
    with urlopen(item["url"], timeout=60) as response:
        data = checked_bytes(response.read(limit + 1), item, limit)
    # Hard-link publication is atomic and refuses to replace any existing name.
    with tempfile.NamedTemporaryFile(dir=cache, prefix=".download-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
    try:
        os.link(temp, path)
    finally:
        temp.unlink()
    return data


def selected_files(data: bytes, project: dict) -> dict[str, bytes]:
    """Return audited subset; do not extract using tar member names."""
    validate_project(project)
    checked_bytes(data, project["archive"], MAX_ARCHIVE_BYTES)
    with lzma.LZMAFile(io.BytesIO(data)) as stream:
        raw = stream.read(MAX_TAR_BYTES + 1)
    if len(raw) > MAX_TAR_BYTES:
        raise ValueError("decompressed tar limit exceeded")
    selected, seen, regular, matched = {}, set(), set(), set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for index, member in enumerate(archive):
            if index >= MAX_MEMBERS:
                raise ValueError("archive member count limit exceeded")
            name = safe_name(member.name)
            if name in seen:
                raise ValueError(f"duplicate archive path: {name}")
            seen.add(name)
            if member.issparse() or not (member.isfile() or member.isdir()):
                raise ValueError(f"links/special/sparse members forbidden: {name}")
            parts = PurePosixPath(name).parts
            if parts[0] != project["archive_root"]:
                raise ValueError(f"member outside archive root: {name}")
            if member.size < 0 or member.size > MAX_FILE_BYTES:
                raise ValueError("archive member size limit exceeded")
            total += member.size
            if total > MAX_TAR_BYTES:
                raise ValueError("total member size limit exceeded")
            if member.isdir():
                continue
            regular.add(name)
            if len(parts) < 2:
                raise ValueError("archive root must be a directory")
            relative = "/".join(parts[1:])
            for rule in project["source_paths"]:
                if relative == rule or (rule.endswith("/") and relative.startswith(rule)):
                    matched.add(rule)
                    with archive.extractfile(member) as stream:
                        content = stream.read(MAX_FILE_BYTES + 1)
                    if len(content) != member.size:
                        raise ValueError("truncated member")
                    selected[relative] = content
        for name in seen:
            if any(str(p) in regular for p in PurePosixPath(name).parents):
                raise ValueError("file/directory path collision")
    if matched != set(project["source_paths"]):
        raise ValueError("some source selection rules matched no files")
    return selected


def expected_directories(files: dict[str, bytes]) -> set[str]:
    return {str(p) for name in files for p in PurePosixPath(name).parents if str(p) != "."}


def verify_tree(target: Path, files: dict[str, bytes]) -> None:
    if target.is_symlink() or not target.is_dir():
        raise ValueError("source target is not an ordinary directory")
    observed_files, observed_dirs = set(), set()
    for base, dirs, names in os.walk(target, followlinks=False):
        for name in dirs + names:
            path = Path(base) / name
            relative = path.relative_to(target).as_posix()
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                observed_dirs.add(relative)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                observed_files.add(relative)
                if relative not in files or read_regular(path, MAX_FILE_BYTES) != files[relative]:
                    raise ValueError(f"unexpected or modified source file: {relative}")
            else:
                raise ValueError(f"local link or special file: {relative}")
    if observed_files != set(files) or observed_dirs != expected_directories(files):
        raise ValueError("missing or extra source paths")


def acquire(project: dict, destination: Path, cache: Path, *, download: bool) -> dict:
    validate_project(project)
    destination, cache = local_root(destination), local_root(cache)
    target = destination / project["id"]
    if target.is_symlink():
        raise ValueError("symlink source target")
    record = {"id": project["id"], "release": project["release"],
              "archive_sha256": project["archive"]["sha256"],
              "descriptor_sha256": project["descriptor"]["sha256"],
              "execution": "not_run", "dependency_validation": "not_run"}
    if not download and not target.exists():
        return dict(record, status="missing")
    data = artifact(project["archive"], cache, MAX_ARCHIVE_BYTES, download)
    artifact(project["descriptor"], cache, MAX_DESCRIPTOR_BYTES, download)
    files = selected_files(data, project)
    if not target.exists():
        if not download:
            raise FileNotFoundError("source target disappeared during offline verification")
        destination.mkdir(parents=True, exist_ok=True)
        target.mkdir()  # Exclusive: never overwrite another process's target.
        for name, content in sorted(files.items()):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
    verify_tree(target, files)
    return dict(record, status="source_verified", source_selection_verified=True,
                requested_source_paths=project["source_paths"], file_count=len(files),
                files={name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--project")
    mode.add_argument("--verify", action="store_true", help="offline local verification")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--cache", type=Path, help="defaults to DESTINATION/.archive-cache")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    projects = load_projects()
    if args.list:
        print(json.dumps(projects, ensure_ascii=False, indent=2))
        return 0
    if args.project and args.project not in projects:
        parser.error("unknown locked project")
    selected = [projects[args.project]] if args.project else list(projects.values())
    records = []
    for project in selected:
        try:
            record = acquire(project, args.destination, args.cache or args.destination / ".archive-cache",
                             download=not args.verify)
        except (ValueError, OSError, tarfile.TarError, lzma.LZMAError, EOFError) as error:
            record = {"id": project["id"], "status": "failed", "error": str(error),
                      "execution": "not_run", "dependency_validation": "not_run"}
        records.append(record)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"schema_version": 1,
            "lock_sha256": hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest(),
            "projects": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in r.items() if k != "files"} for r in records], ensure_ascii=False))
    return 0 if all(r["status"] == "source_verified" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
