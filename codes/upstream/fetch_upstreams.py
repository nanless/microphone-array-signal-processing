#!/usr/bin/env python3
"""Fetch allow-listed upstream references at exact Git revisions.

The script downloads source only. It never installs, builds, imports, or executes
upstream code, models, or datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse


LOCK_FILE = Path(__file__).resolve().parents[1] / "SOURCES.lock.json"
DEFAULT_DESTINATION = Path(__file__).resolve().parent / "_downloads"
PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


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
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def fetch_project(project: dict[str, object], destination: Path) -> None:
    validate_project(project)
    project_id = str(project["id"])
    if not project.get("fetch_enabled", False):
        raise ValueError(f"{project_id} is index-only; follow its official build instructions")
    url = str(project["url"])
    revision = str(project["revision"])
    destination = destination.resolve()
    target = (destination / project_id).resolve()
    try:
        target.relative_to(destination)
    except ValueError as error:
        raise ValueError(f"project target escapes destination: {target}") from error
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {target}")

    destination.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    try:
        run_git(["init"], cwd=target)
        run_git(["remote", "add", "origin", url], cwd=target)
        run_git(["fetch", "--depth", "1", "origin", revision], cwd=target)
        run_git(["checkout", "--detach", revision], cwd=target)
        actual = run_git(["rev-parse", "HEAD"], cwd=target)
        if actual != revision:
            raise RuntimeError(f"revision mismatch for {project_id}: {actual} != {revision}")
    except Exception:
        print(f"incomplete checkout retained for inspection: {target}", file=sys.stderr)
        raise
    print(f"{project_id}: {actual} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list locked projects without network access")
    group.add_argument("--project", help="fetch one allow-listed project id")
    group.add_argument("--all", action="store_true", help="fetch every project with fetch_enabled=true")
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
    else:
        selected = [project for project in projects.values() if project.get("fetch_enabled", False)]

    for project in selected:
        fetch_project(project, args.destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
