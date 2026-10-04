"""Read-only fixed MeetEval native word-edit contracts; no Python scoring pipeline.

Only --report writes a JSON report. The complete original MIT header is included
without modification; the C++ caller is a self-authored token/interval adapter.
Compilation takes place in a temporary directory, never in the upstream tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch10.examples.run_industrial_interfaces import clean_environment, compiled_dependencies, verify_failure_sources

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
CACHE = ROOT / "codes/chapters/ch00/upstream/_downloads"
CURRENT = ROOT / "codes/chapters/ch11/reports/meeting_kernel_contracts_current.json"
REVISION = "6e3dc81284f2d6928f7ef9e620fd3b6906daa429"
ORIGIN = "https://github.com/fgnt/meeteval.git"
FILES = {
    "LICENSE": "5515c2bdda551fa20d771e99ab31deebbb4f9626bef6f6c25f26125c964fd34c",
    "README.md": "44f11b4e45e41e41e35a4e8c09b66b45a816c2c258463664b7853068daf201ef",
    "meeteval/wer/matching/levenshtein.h": "04e013ba7265a6df8f261f793624ba425b57b7beee9ec9b572f4dd4f1e79ce06",
    "meeteval/wer/matching/cy_levenshtein.pyx": "7f73fedd29315cc3dff9017833baf7f646ca0fc60b7b4e074e3659a834f265e2",
    "meeteval/wer/wer/cp.py": "2ead616c81b2119e6381532fb337e550c1b12e4ae0ecdfcc96557df32404c1f2",
    "meeteval/wer/wer/orc.py": "7aac2cfdaeb8fbe975aa5087baaa64bff99a08a6093aaffac014cfdb8b2ce9b7",
    "meeteval/wer/wer/time_constrained.py": "b2d6c0c18f4add17484edf53dbea8f9926f37216c9ccd7612871c92ce537435c",
}
# Each entry contains a hand-derived error count, not an upstream-generated oracle.
# Token strings are indivisible words: 春天 has one token, not two characters.
CASES = (
    ("same_word_distant", ["春天"], ["春天"], [[0, 1]], [[10, 11]], 0, 2),
    ("same_word_touching", ["春天"], ["春天"], [[0, 1]], [[1, 2]], 0, 2),
    ("same_word_positive_overlap", ["春天"], ["春天"], [[0, 1]], [[.5, 1.5]], 0, 0),
    ("different_word_overlap", ["春天"], ["夏天"], [[0, 1]], [[0, 1]], 1, 1),
    ("different_word_distant", ["春天"], ["夏天"], [[0, 1]], [[10, 11]], 1, 2),
    ("empty_reference", [], ["春天", "夏天"], [], [[0, 1], [2, 3]], 2, 2),
    ("empty_hypothesis", ["春天", "夏天"], [], [[0, 1], [2, 3]], [], 2, 2),
    ("both_empty", [], [], [], [], 0, 0),
    ("two_words_temporal_shift", ["春天", "夏天"], ["春天", "夏天"],
     [[0, 1], [2, 3]], [[10, 11], [12, 13]], 0, 4),
)
# Added current controls; the historical report's original nine cases stay fixed.
EXTRA_POINT_CASES = (
    ("interior_hypothesis_point", ["春天"], ["春天"], [[0, 3]], [[1.5, 1.5]], 0, 0),
    ("left_endpoint_hypothesis_point", ["春天"], ["春天"], [[0, 3]], [[0, 0]], 0, 2),
    ("right_endpoint_hypothesis_point", ["春天"], ["春天"], [[0, 3]], [[3, 3]], 0, 2),
    ("character_based_two_hypothesis_points", ["abc", "b"], ["abc", "b"],
     [[0, 3], [3, 4]], [[1.5, 1.5], [3.5, 3.5]], 0, 0),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


strict_loads = upstream.strict_json_loads


def ordinary_path(path, *, directory=False, allow_missing=False):
    path = upstream.validate_parent_chain(path)
    if path.exists():
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError("unexpected file type: " + str(path))
    elif not allow_missing:
        raise ValueError("required path is absent: " + str(path))
    return path


def verify_sources(cache=CACHE):
    identity = upstream.verify_project("meeteval", Path(cache) / "meeteval", FILES)
    for relative, expected in FILES.items():
        if identity["used_files"][relative]["sha256"] != expected:
            raise ValueError("fixed source digest differs: " + relative)
    sources = [{"path": path, **row,
                "role": "native include" if path.endswith(".h") else "static contract/identity evidence"}
               for path, row in identity["used_files"].items()]
    return {"checkout": identity["checkout"], "revision": identity["head"],
            "origin": identity["origin"], "source_lock_sha256": identity["lock_sha256"],
            "source_lock_project_count": len(upstream.source_documents()[0]["projects"]),
            "sources": sources, "clean": True, "source_identity": identity}


def driver_text(header):
    # Paths are encoded as C++ string literals; no shell or macro/source rewriting.
    include = json.dumps(str(header))
    lines = ["#include <iostream>", "#include <stdexcept>", "#include " + include,
             "using V=std::vector<unsigned int>; using T=std::vector<std::pair<double,double>>;",
             'int main(){std::cout<<"[";']
    cases = CASES + EXTRA_POINT_CASES
    words = {word: i + 1 for i, word in enumerate(sorted({w for c in cases for seq in c[1:3] for w in seq}))}
    vector = lambda sequence: "{" + ",".join(str(words[w]) for w in sequence) + "}"
    times = lambda sequence: "{" + ",".join("{" + str(a) + "," + str(b) + "}" for a, b in sequence) + "}"
    for i, (name, ref, hyp, rt, ht, _, _) in enumerate(cases):
        if i:
            lines.append('std::cout<<",";')
        lines.append("{V r=" + vector(ref) + ";V h=" + vector(hyp) + ";T rt=" + times(rt) + ";T ht=" + times(ht) + ";")
        prefix = json.dumps('{"name":"' + name + '","ordinary":')
        lines.append("std::cout<<" + prefix + "<<levenshtein_distance_(r,h)"
                     '<<",\\\"timed\\\":"<<time_constrained_levenshtein_distance_v2_(r,h,rt,ht,1,1,1,0)<<"}";}')
    lines.append('std::cout<<"]\\n";}')
    return "\n".join(lines) + "\n", words


def run_audit(cache=CACHE, compiler="c++"):
    before = verify_sources(cache)
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError("C++ compiler is unavailable: " + compiler)
    text, words = driver_text(Path(before["checkout"]) / "meeteval/wer/matching/levenshtein.h")
    with tempfile.TemporaryDirectory(prefix="masp-meeting-kernel-", dir="/private/tmp") as folder:
        source, binary = Path(folder) / "driver.cc", Path(folder) / "driver"
        source.write_text(text, encoding="utf-8")
        command = [executable, "-std=c++17", "-O2", "-MD", "-MF", str(Path(folder) / "driver.d"), str(source), "-o", str(binary)]
        environment = clean_environment()
        try:
            compiled = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60, env=environment)
            dependencies = compiled_dependencies(Path(folder), {"meeteval": before["source_identity"]}, [source])
            execution = subprocess.run([str(binary)], capture_output=True, text=True, check=True, timeout=10, env=environment)
            rows = strict_loads(execution.stdout)
        finally:
            error = sys.exception()
            if error is not None:
                verify_failure_sources({"meeteval": before["source_identity"]}, error)
            else:
                upstream.check_unchanged(before["source_identity"])
    cases = CASES + EXTRA_POINT_CASES
    if len(rows) != len(cases):
        raise ValueError("native result count changed")
    for index, (row, (name, ref, hyp, rt, ht, ordinary, timed)) in enumerate(zip(rows, cases)):
        if (row.get("name") != name or type(row.get("ordinary")) is not int or
                type(row.get("timed")) is not int or (row["ordinary"], row["timed"]) != (ordinary, timed)):
            raise ValueError("native result disagrees with hand oracle: " + name)
        row.update(reference_words=ref, hypothesis_words=hyp, reference_intervals_s=rt,
                   hypothesis_intervals_s=ht, expected_ordinary=ordinary, expected_timed=timed,
                   status="matched_independent_hand_expected",
                   case_group="original_nine" if index < len(CASES) else "current_point_controls")
    upstream.check_unchanged(before["source_identity"])
    return {"schema_version": 2, "actual_dependencies": upstream.dependencies(__file__, [Path(clean_environment.__code__.co_filename)]), "compiled_dependencies": dependencies,
            "created_utc": datetime.now(timezone.utc).isoformat(), "tool_sha256": digest(Path(__file__).read_bytes()),
            **before, "before_clean": True, "after_clean": True,
            "execution": {"original_functions": ["levenshtein_distance_", "time_constrained_levenshtein_distance_v2_"],
                          "scope": "complete original header, two native kernels only",
                          "not_executed": ["production Python cpWER/ORC-WER/tcpWER wrappers", "ASR", "diarization", "models", "hardware"],
                          "source_patch": False, "algorithm_substitutes": [],
                          "scaffold": "self-authored caller; unsigned whole-word tokens and nonnegative double intervals",
                          "driver_license": "self-authored repository code; original included header remains MIT",
                          "driver_sha256": digest(text.encode()), "token_encoding": words,
                          "costs": {"insertion": 1, "deletion": 1, "substitution": 1, "correct": 0},
                          "timing_rule": "a_begin < b_end and b_begin < a_end; zero collar; endpoint contact forbidden, interior hypothesis points accepted",
                          "historical_case_count": 9, "current_case_count": 13,
                          "compile_command": command, "compiler_binary": {"path": str(Path(executable).resolve()), "sha256": digest(Path(executable).resolve().read_bytes())}, "compiler": subprocess.check_output([executable, "--version"], text=True, env=environment),
                          "compile_stderr": compiled.stderr, "execution_stderr": execution.stderr},
            "license": {"name": "MIT", "path": "LICENSE", "sha256": FILES["LICENSE"]},
            "environment": {"python": sys.version, "platform": platform.platform(), "scientific_packages_required": False, "compile_environment_policy": "Git, Python, CMake, compiler, include and linker overrides removed"},
            "static_scope": "other six files verify identity, binding and wrapper contracts; not executed",
            "historical_failure": "2026-09-28 separate report retained: Python production cp call lacked cy_levenshtein; native success does not erase it",
            "cases": rows}


def write_report(path, report, cache=CACHE):
    upstream.write_report(path, report, CURRENT, (CACHE, Path(cache)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report is not None:
        upstream.report_target(args.report, CURRENT, (CACHE, args.cache))
    report = run_audit(args.cache, args.compiler)
    if args.report is not None:
        write_report(args.report, report, args.cache)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
