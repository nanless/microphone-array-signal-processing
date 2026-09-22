#!/usr/bin/env python3
"""Fetch allow-listed upstream references at exact Git revisions.

The script downloads source only. It never installs, builds, imports, or executes
upstream code, models, or datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse


LOCK_FILE = Path(__file__).resolve().parents[1] / "SOURCES.lock.json"
DEFAULT_DESTINATION = Path(__file__).resolve().parent / "_downloads"
PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
OMITTED_EXTENSIONS = ("wav", "flac", "mp3", "mp4", "ogg", "pt", "pth", "ckpt", "onnx", "tflite", "h5", "hdf5", "npz", "npy", "so", "dll", "dylib", "zip", "tar", "gz", "bin", "pb", "pkl", "pickle", "mat", "safetensors", "whl", "a", "o", "exe", "wasm", "weights")


def validate_project(project: dict[str, object]) -> None:
    project_id = project.get("id")
    url = project.get("url")
    revision = project.get("revision")
    if not isinstance(project_id, str) or not PROJECT_ID.fullmatch(project_id):
        raise ValueError("project id must be a safe lowercase slug")
    if not isinstance(revision, str) or not FULL_GIT_SHA.fullmatch(revision):
        raise ValueError(f"{project_id}: revision must be a full lowercase Git SHA")
    if not isinstance(url, str):
        raise ValueError(f"{project_id}: url must be a string")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{project_id}: url must be an unauthenticated https URL")
    for path in project.get("entrypoints", []):
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"{project_id}: entrypoint must be a relative repository path")
    for path in project.get("source_paths", []):
        if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", path) or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"{project_id}: source selection must use literal relative paths")


def load_projects() -> dict[str, dict[str, object]]:
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    projects = data.get("projects")
    if not isinstance(projects, list):
        raise ValueError("SOURCES.lock.json: projects must be a list")
    result: dict[str, dict[str, object]] = {}
    for project in projects:
        if not isinstance(project, dict):
            raise ValueError("SOURCES.lock.json: every project must be an object")
        validate_project(project)
        project_id = str(project["id"])
        if project_id in result:
            raise ValueError(f"SOURCES.lock.json: duplicate project id {project_id}")
        result[project_id] = project
    return result


def run_git(arguments: list[str], cwd: Path | None = None) -> str:
    # Repository-selection and injected configuration variables can override cwd.
    # Do not let a caller's outer Git operation redirect this independent checkout.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update(GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0",
                       GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    completed = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        env=environment,
    )
    return completed.stdout.strip()


def inspect_project(project: dict[str, object], destination: Path) -> dict[str, object]:
    """Inspect a checkout without fetching or changing its files."""
    validate_project(project)
    root = destination.resolve()
    target = root / str(project["id"])
    if target.is_symlink():
        raise ValueError(f"refusing symlink checkout: {target}")
    target.resolve().relative_to(root)
    record = {"id": project["id"], "revision": project["revision"], "status": "missing"}
    if not target.exists():
        if not project.get("fetch_enabled", False):
            record["status"] = "index_only"
        return record
    if not (target / ".git").is_dir():
        raise ValueError(f"not an independent Git checkout: {target}")
    actual = run_git(["rev-parse", "HEAD"], cwd=target)
    origin = run_git(["remote", "get-url", "origin"], cwd=target)
    if actual != project["revision"] or origin != project["url"]:
        raise ValueError(f"{project['id']}: checkout revision or origin does not match lock")
    dirty = run_git(["status", "--porcelain"], cwd=target)
    if dirty:
        raise ValueError(f"{project['id']}: local changes exist; preserve and inspect them")
    missing = [p for p in project.get("entrypoints", []) if not (target / p).exists()]
    sparse_file = target / ".git" / "info" / "sparse-checkout"
    sparse = sparse_file.exists()
    patterns = sparse_file.read_text(encoding="utf-8").splitlines() if sparse else []
    requested = ["/" + p for p in project["source_paths"]] if project.get("source_paths") else ["/*"]
    selected = [p for p in patterns if p and not p.startswith(("!", "#"))]
    scope_matches = selected == requested if sparse else not project.get("source_paths")
    record.update(status="entrypoints_missing" if missing else "source_verified", missing_entrypoints=missing,
                  execution="not_run", dependency_validation="not_run",
                  asset_policy="sparse_source_checkout" if sparse else "existing_checkout_preserved",
                  requested_source_paths=project.get("source_paths", ["repository"]),
                  observed_sparse_patterns=patterns, source_selection_verified=bool(scope_matches))
    if not scope_matches:
        record["status"] = "source_selection_mismatch"
    return record


def fetch_project(project: dict[str, object], destination: Path) -> dict[str, object]:
    validate_project(project)
    project_id = str(project["id"])
    if not project.get("fetch_enabled", False):
        raise ValueError(f"{project_id} is index-only; follow its official build instructions")
    url = str(project["url"])
    revision = str(project["revision"])
    destination = destination.resolve()
    target = destination / project_id
    if target.is_symlink():
        raise ValueError(f"refusing symlink checkout: {target}")
    target = target.resolve()
    try:
        target.relative_to(destination)
    except ValueError as error:
        raise ValueError(f"project target escapes destination: {target}") from error
    if target.exists():
        record = inspect_project(project, destination)
        print(f"{project_id}: existing {record['status']}", flush=True)
        return record

    destination.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    try:
        run_git(["init"], cwd=target)
        run_git(["remote", "add", "origin", url], cwd=target)
        print(f"{project_id}: fetching {revision}", flush=True)
        run_git(["fetch", "--depth", "1", "--filter=blob:none", "origin", revision], cwd=target)
        # Sparse working trees retain source and notices while omitting common
        # model/audio/archive assets. Git objects are still upstream-owned data.
        includes = ["/" + p for p in project["source_paths"]] if project.get("source_paths") else ["/*"]
        run_git(["sparse-checkout", "set", "--no-cone", *includes,
                 *[f"!**/*.{ext}" for ext in OMITTED_EXTENSIONS]], cwd=target)
        run_git(["checkout", "--detach", revision], cwd=target)
        actual = run_git(["rev-parse", "HEAD"], cwd=target)
        if actual != revision:
            raise RuntimeError(f"revision mismatch for {project_id}: {actual} != {revision}")
    except Exception:
        print(f"incomplete checkout retained for inspection: {target}", file=sys.stderr)
        raise
    record = inspect_project(project, destination)
    print(f"{project_id}: {actual} ({record['status']}) -> {target}", flush=True)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list locked projects without network access")
    group.add_argument("--project", help="fetch one allow-listed project id")
    group.add_argument("--all", action="store_true", help="fetch every project with fetch_enabled=true")
    group.add_argument("--verify", action="store_true", help="inspect local checkouts without network")
    parser.add_argument("--report", type=Path, help="write a JSON acquisition report (not runtime validation)")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="download root (default: codes/upstream/_downloads)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projects = load_projects()
    if args.list:
        for project_id, project in projects.items():
            mode = "fetch" if project.get("fetch_enabled", False) else "index-only"
            print(f"{project_id:16} {mode:10} {project['revision']} {project['url']}")
        return 0

    selected: list[dict[str, object]]
    if args.project:
        if args.project not in projects:
            print(f"unknown project: {args.project}", file=sys.stderr)
            return 2
        selected = [projects[args.project]]
    elif args.verify:
        selected = list(projects.values())
    else:
        selected = [project for project in projects.values() if project.get("fetch_enabled", False)]

    records = []
    for project in selected:
        try:
            record = (inspect_project(project, args.destination) if args.verify
                      else fetch_project(project, args.destination.resolve()))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            record = {"id": project["id"], "revision": project["revision"],
                      "status": "failed", "error": str(error)}
            print(f"{project['id']}: {error}", file=sys.stderr, flush=True)
        records.append(record)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"schema_version": 1,
                                         "lock_sha256": hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest(),
                                         "projects": records},
                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})}
    print(json.dumps(counts, ensure_ascii=False))
    return 0 if all(r["status"] in ("source_verified", "index_only") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
