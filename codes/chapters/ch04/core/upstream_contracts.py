"""Shared fixed-source and finite write-preflight contracts, first used in chapter 4.

Used original blobs, complete acquisition selection, and actual method execution
are separate records. Checks do not eliminate concurrent filesystem races.
No fetching, installation, upstream edits, or implicit report writes occur here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from codes.chapters.ch00.io_contracts import (
    strict_json_loads, same_metadata, validate_parent_chain, validate_report_destination,
    write_json_report,
)
from codes.chapters.ch00.upstream.fetch_upstreams import run_git, sparse_patterns, validate_project

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
STATUS = LOCK.with_name("SOURCE_STATUS.json")
CACHE = LOCK.parent / "upstream/_downloads"
HISTORICAL_REVISION = "e4d54a158d892b224e37cc49b93066e667babf08"
PROJECTS = {
    "pyaec": ("https://github.com/ewan-xu/pyaec.git",
        "5b9c02c57075d790b7df8652884618189d49bbc4", "Apache-2.0", "LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "echocatzh-pfdkf": ("https://github.com/echocatzh/PFDKF.git",
        "7c8c86b5691966c330015d8e0960db0733b4844f", "MIT", "LICENSE",
        "b6c53ea4c5cf363eeef563f592f0b7308ad0943627e7803e9b0df05bf592b5d5"),
    "dtln_aec": ("https://github.com/breizhn/DTLN-aec.git",
        "9d24e128b4f409db18227b8babb343016625921f", "MIT", "LICENSE",
        "aa95acd8c8a7341bfcdb2823694dcbb27a2a4143e860bb52db1ff0f29c85e5a1"),
    "pb_bss": ("https://github.com/fgnt/pb_bss.git",
        "10acc347fc9ea21e3d312806a0bd751d0d0af183", "MIT", "LICENSE",
        "48241e1eae6ab4c15c5718992ca60d3d15961212a4e324e09e5dbfbc79b78214"),
    "sof": ("https://github.com/thesofproject/sof.git",
        "b6c6a05d52536313fe8e8752b1c4e069b1cc4002",
        "BSD-3-Clause and per-file licenses", "LICENCE",
        "cb6a9aa5baac3ea573feebcdc298526f701e2b7cd005a22208799778e26f0068"),
    "pyroomacoustics": ("https://github.com/LCAV/pyroomacoustics.git",
        "0dd39f2614b7fc44b2cc63dbe7d60f4641068890", "MIT", "LICENSE",
        "0922c9a0c1f5bb35a1e0df7b56954864e27cfc625fc3b041acde0707ce797e46"),
    "doatools": ("https://github.com/morriswmz/doatools.py.git",
        "9469db201e0418aef6b97583ef54b6fec2769502", "MIT", "LICENSE.md",
        "39ddd5dcfeb6eb77f18c99e8955341dc7b4d4823a3a121e0124e20de336b507a"),
    "sbl": ("https://github.com/gerstoft/SBL.git",
        "d4bba35e9b60907d3024473ba5a41046450baae0", "GPL-3.0", "LICENSE",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
    "smpphat": ("https://github.com/FrancoisGrondin/smpphat.git",
        "6fd33e6eb3251078a4cd9793dde909e2500265cc", "GPL-3.0", "LICENSE",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
    "said-spatial-imaging": ("https://github.com/IN03X/SAID.git",
        "cf52ede4f38361cdbb03aa93c8254109583dfe2b",
        "MIT AND Apache-2.0 (source components); separate non-commercial model-weight terms", "LICENSE",
        "d3757b5359fd0d6d1ed7dbe9b37da90b11c4a6cabaa11081b26b5f6aa95794af"),
}


def ordinary_file(path):
    path = validate_parent_chain(path)
    if not path.is_file():
        raise ValueError(f"Existing ordinary source file required: {path}")
    return path


def sha(path):
    return hashlib.sha256(ordinary_file(path).read_bytes()).hexdigest()


def git(checkout, *arguments):
    return run_git(list(arguments), cwd=checkout)


def source_documents(lock_path=None, status_path=None):
    lock_path = LOCK if lock_path is None else lock_path
    status_path = STATUS if status_path is None else status_path
    lock_bytes = ordinary_file(lock_path).read_bytes()
    status_bytes = ordinary_file(status_path).read_bytes()
    lock, status = strict_json_loads(lock_bytes), strict_json_loads(status_bytes)
    for document in (lock, status):
        if (type(document) is not dict or type(document.get("schema_version")) is not int
                or document["schema_version"] != 1 or type(document.get("projects")) is not list
                or any(type(row) is not dict for row in document["projects"])):
            raise ValueError("Source documents require schema 1 and object project records")
        ids = [row.get("id") for row in document["projects"]]
        if any(type(i) is not str for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("Source documents require unique string project IDs")
    digest = hashlib.sha256(lock_bytes).hexdigest()
    if status.get("lock_sha256") != digest:
        raise ValueError("Current source status must bind current lock bytes")
    return lock, status, digest, hashlib.sha256(status_bytes).hexdigest()


def record_files(identity, relatives):
    checkout = Path(identity["checkout"])
    for relative in relatives:
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise ValueError("Original source path must be relative to its checkout")
        path = ordinary_file(checkout / relative)
        payload = path.read_bytes()
        expected = git(checkout, "rev-parse", f"{identity['head']}:{relative}")
        actual = git(checkout, "hash-object", "--no-filters", "--", str(path))
        if actual != expected:
            raise ValueError(f"Original source differs from fixed Git blob: {relative}")
        record = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                  "git_blob": expected, "actual_blob": actual}
        previous = identity["used_files"].get(relative)
        if previous is not None and previous != record:
            raise ValueError(f"Original source changed during execution: {relative}")
        identity["used_files"][relative] = record


def verify_project(project, checkout=None, relatives=(), *, all_python=False,
                   lock_path=None, status_path=None):
    """Check fixed identity before use; never upgrade selection mismatch."""
    lock, status, lock_digest, status_digest = source_documents(lock_path, status_path)
    url, revision, license_name, license_path, license_digest = PROJECTS[project]
    entries = [p for p in lock["projects"] if p["id"] == project]
    records = [p for p in status["projects"] if p["id"] == project]
    if len(entries) != 1 or len(records) != 1:
        raise ValueError("Expected exactly one lock and status project record")
    entry, recorded = entries[0], records[0]
    validate_project(entry)
    if (entry["url"], entry["revision"], entry["license"]) != (url, revision, license_name):
        raise ValueError(f"Review changed fixed source lock: {project}")
    checkout = validate_parent_chain(CACHE / project if checkout is None else checkout)
    validate_parent_chain(checkout / ".git")
    if not checkout.is_dir() or not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"Existing independent checkout required: {checkout}")
    if git(checkout, "rev-parse", "--show-toplevel") != str(checkout.resolve()):
        raise ValueError("Source directory must be its own Git top level")
    origin, head = git(checkout, "remote", "get-url", "origin"), git(checkout, "rev-parse", "HEAD")
    if (origin, head) != (url, revision):
        raise ValueError("Original source origin or HEAD differs from fixed identity")
    if git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Original source worktree must be entirely clean")
    ignored = git(checkout, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
    sparse_file = validate_parent_chain(checkout / ".git/info/sparse-checkout")
    sparse = sparse_file.exists()
    patterns = ordinary_file(sparse_file).read_text().splitlines() if sparse else []
    expected = sparse_patterns(entry) if sparse else []
    matches = patterns == expected if sparse else not entry.get("source_paths")
    missing = [p for p in entry.get("entrypoints", []) if not (checkout / p).exists()]
    live = {"id": project, "revision": revision,
            "status": "source_selection_mismatch" if not matches else
                      "entrypoints_missing" if missing else "source_verified",
            "source_selection_verified": bool(matches), "missing_entrypoints": missing,
            "observed_sparse_patterns": patterns, "expected_sparse_patterns": expected,
            "requested_source_paths": entry.get("source_paths", ["repository"]),
            "asset_policy": "sparse_source_checkout" if sparse else "existing_checkout_preserved"}
    if checkout.resolve() == (CACHE / project).resolve():
        if any(not same_metadata(recorded.get(k), v) for k, v in live.items()):
            raise ValueError("Recorded complete selection differs from actual checkout")
    identity = {"project": project, "checkout": str(checkout.resolve()), "origin": origin,
                "head": head, "lock_entry": entry, "lock_sha256": lock_digest,
                "source_document_paths": {"lock": str(LOCK if lock_path is None else lock_path),
                                          "status": str(STATUS if status_path is None else status_path)},
                "status_sha256": status_digest, "recorded_complete_selection": recorded,
                "live_complete_selection": live, "used_source_identity": "verified",
                "file_identity_scope": "declared original files plus tracked Python preflight when requested; does not claim every verified file executed",
                "git_clean_scope": "tracked and nonignored untracked files; ignored members listed separately",
                "ignored_members_before": ignored,
                "used_files": {}, "clean_before": True, "clean_after": False}
    names = list(relatives)
    if all_python:
        names.extend(git(checkout, "ls-files", "--", "*.py").splitlines())
    record_files(identity, [license_path, *dict.fromkeys(names)])
    if identity["used_files"][license_path]["sha256"] != license_digest:
        raise ValueError("Original license differs from fixed verified SHA-256")
    return identity


def check_unchanged(identity):
    checkout = Path(identity["checkout"])
    if (git(checkout, "rev-parse", "HEAD") != identity["head"]
            or git(checkout, "remote", "get-url", "origin") != identity["origin"]
            or git(checkout, "status", "--porcelain", "--untracked-files=all")):
        raise ValueError("Original source changed during execution")
    if (sha(identity['source_document_paths']['lock']) != identity['lock_sha256']
            or sha(identity['source_document_paths']['status']) != identity['status_sha256']):
        raise ValueError("Source lock or recorded selection changed during execution")
    ignored_after = git(checkout, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
    if ignored_after != identity['ignored_members_before']:
        raise ValueError("Ignored source membership changed during execution")
    identity['ignored_members_after'] = ignored_after
    record_files(identity, tuple(identity["used_files"]))
    identity["clean_after"] = True
    return identity


def dependencies(script, extras=()):
    paths = (script, Path(__file__), ROOT / "codes/chapters/ch00/io_contracts.py",
             ROOT / "codes/chapters/ch00/upstream/fetch_upstreams.py", *extras)
    return {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in paths}


def report_target(path, current=None, protected=()):
    target = validate_report_destination(path, forbidden_roots=protected)
    if target.resolve().is_relative_to(ROOT.resolve()):
        if current is None or target.resolve() != Path(current).resolve():
            raise ValueError("Only the designated new current report may be written inside the repository")
    return target


def write_report(path, data, current=None, protected=()):
    # Repeat the finite preflight immediately before shared atomic replacement.
    target = report_target(path, current, protected)
    write_json_report(target, data, forbidden_roots=protected)


def work_target(path, protected=()):
    target = validate_parent_chain(path)
    if target.resolve().is_relative_to(ROOT.resolve()) or ROOT.resolve().is_relative_to(target.resolve()):
        raise ValueError("Temporary work directory must be outside the repository")
    if target.exists() and not target.is_dir():
        raise ValueError("Temporary work destination must be an ordinary directory")
    for root in protected:
        root = Path(root).resolve()
        if target.resolve().is_relative_to(root) or root.is_relative_to(target.resolve()):
            raise ValueError("Temporary work directory overlaps original source")
    return target


def historical_bytes(path):
    """Read the real pre-edit Git object; strip handling preserves exact bytes."""
    relative = Path(path).resolve().relative_to(ROOT)
    revision_path = HISTORICAL_REVISION + ":" + str(relative)
    payload = (git(ROOT, "show", revision_path) + "\n").encode()
    if len(payload) != int(git(ROOT, "cat-file", "-s", revision_path)):
        raise ValueError("Historical Git object cannot be represented byte-exactly")
    return payload
