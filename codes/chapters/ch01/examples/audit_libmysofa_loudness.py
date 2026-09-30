"""Run the pinned libmysofa loudness function on two artificial structures.

Only the original loudness.c and tools.c are compiled, in a temporary directory.
This is not a SOFA parser, interpolation, resampling or binaural renderer test.
The upstream BSD notices remain in the unchanged checkout; no upstream source
or compiled binary is redistributed by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
UPSTREAM = ROOT / "codes/chapters/ch00/upstream/_downloads/libmysofa"
REVISION = "6cc5b15a73e9bd97810d03767082edda7f315881"
SOURCE_FILES = (
    "src/hrtf/loudness.c", "src/hrtf/tools.c", "src/hrtf/tools.h",
    "src/hrtf/mysofa.h", "src/resampler/speex_resampler.h", "LICENSE",
)
ABS_TOLERANCE = 2e-6
EXPORT_HEADER = "#ifndef MYSOFA_EXPORT\n#define MYSOFA_EXPORT\n#endif\n"

# This harness is written for the tutorial; it does not copy upstream functions.
HARNESS = r'''#include "mysofa.h"
#include <stdio.h>

static void probe(const char *name, float left, float right) {
  float position[3] = {0.f, 0.f, 1.f};
  float ir[2] = {left, right};
  struct MYSOFA_ATTRIBUTE type = {NULL, "Type", "spherical"};
  struct MYSOFA_HRTF h = {0};
  h.I = h.E = h.M = h.N = 1;
  h.C = 3;
  h.R = 2;
  h.SourcePosition.values = position;
  h.SourcePosition.elements = 3;
  h.SourcePosition.attributes = &type;
  h.DataIR.values = ir;
  h.DataIR.elements = 2;
  float factor = mysofa_loudness(&h);
  printf("%s %.9g %.9g %.9g\n", name, (double)factor,
         (double)ir[0], (double)ir[1]);
}

int main(void) {
  probe("nonzero", 1.f, .5f);
  probe("zero", 0.f, 0.f);
  return 0;
}
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path = ROOT) -> str:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                          check=True, timeout=60).stdout


def verify_sources(upstream: Path = UPSTREAM) -> dict:
    """Reject wrong revisions, dirty trees and bytes differing from Git blobs."""
    if not (upstream / ".git").exists():
        raise FileNotFoundError(f"Existing locked checkout is required: {upstream}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    entry = next(item for item in lock["projects"] if item["id"] == "libmysofa")
    if entry["revision"] != REVISION:
        raise ValueError("The source lock changed; review this audit before running")
    head = _run(["git", "rev-parse", "HEAD"], cwd=upstream).strip()
    if head != REVISION:
        raise ValueError(f"Unexpected upstream HEAD: {head}")
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"],
                  cwd=upstream)
    if status:
        raise ValueError("Upstream worktree is not clean; audit refuses to alter it")
    hashes = {}
    for relative in SOURCE_FILES:
        blob = subprocess.run(["git", "show", f"{REVISION}:{relative}"],
                              cwd=upstream, capture_output=True, check=True,
                              timeout=60).stdout
        actual = (upstream / relative).read_bytes()
        if actual != blob:
            raise ValueError(f"Source bytes differ from the locked revision: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return {"url": entry["url"], "revision": head, "release": "v1.3.5",
            "license": entry["license"], "lock_sha256": sha256(LOCK),
            "worktree_clean": True, "source_sha256": hashes}


def finite_record(value: float) -> dict:
    """Encode non-finite C results without JSON NaN/Infinity extensions."""
    if math.isnan(value):
        return {"value": None, "classification": "nan"}
    if math.isinf(value):
        kind = "positive_infinity" if value > 0 else "negative_infinity"
        return {"value": None, "classification": kind}
    return {"value": value, "classification": "finite"}


def evaluate(stdout: str) -> list[dict]:
    lines = [line.split() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 2 or [line[0] for line in lines] != ["nonzero", "zero"]:
        raise ValueError("Unexpected native probe output")
    if any(len(line) != 4 for line in lines):
        raise ValueError("Unexpected native probe field count")
    expected_factor = math.sqrt(2.0 / (1.0 + 0.5 ** 2))
    cases = []
    for name, factor_text, left_text, right_text in lines:
        factor, left, right = map(float, (factor_text, left_text, right_text))
        nonzero = name == "nonzero"
        inputs = [1.0, 0.5] if nonzero else [0.0, 0.0]
        record = {"name": name, "input_ir_left_right": inputs,
                  "factor": finite_record(factor),
                  "output_ir_left_right": [finite_record(left), finite_record(right)]}
        if nonzero:
            expected = [expected_factor, expected_factor * 0.5]
            ratio = left / right if right and math.isfinite(left / right) else None
            energy = left * left + right * right
            ild = (20.0 * math.log10(abs(right / left))
                   if all(math.isfinite(v) and v != 0 for v in (left, right)) else None)
            record.update({"expected_factor": expected_factor,
                           "expected_output_ir_left_right": expected,
                           "expected_energy_sum": 2.0, "energy_sum": finite_record(energy),
                           "expected_left_over_right": 2.0, "left_over_right": ratio,
                           "ild_definition": "20*log10(abs(right/left))",
                           "input_ild_db": 20 * math.log10(0.5), "output_ild_db": ild,
                           "normalization_success": all(
                               math.isclose(a, b, rel_tol=0, abs_tol=ABS_TOLERANCE)
                               for a, b in zip((factor, left, right, energy, ratio or 0),
                                               (expected_factor, *expected, 2.0, 2.0))),
                           "expected_behavior_observed": None})
            record["expected_behavior_observed"] = record["normalization_success"]
        else:
            observed = math.isinf(factor) and factor > 0 and math.isnan(left) and math.isnan(right)
            record.update({"normalization_success": False,
                           "expected_behavior": "positive infinity factor and two NaN outputs",
                           "expected_behavior_observed": observed,
                           "reason": "zero selected IR energy; division by zero and infinity times zero"})
        cases.append(record)
    return cases


def run_audit(*, compiler: str = "cc", upstream: Path = UPSTREAM) -> dict:
    source = verify_sources(upstream)
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(f"C compiler not available: {compiler}")
    compiler_version = _run([executable, "--version"])
    with tempfile.TemporaryDirectory(prefix="ch01-libmysofa-") as temp:
        build = Path(temp)
        harness = build / "probe.c"
        harness.write_text(HARNESS, encoding="utf-8")
        (build / "mysofa_export.h").write_text(EXPORT_HEADER, encoding="utf-8")
        binary = build / "probe"
        command = [executable, "-std=c99", "-O0", "-I", str(build), "-I",
                   str(upstream / "src/hrtf"), str(harness),
                   str(upstream / "src/hrtf/loudness.c"),
                   str(upstream / "src/hrtf/tools.c"), "-lm", "-o", str(binary)]
        compiled = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                  check=True, timeout=60)
        stdout = _run([str(binary)])
    if verify_sources(upstream) != source:
        raise ValueError("Upstream or source lock changed while the audit ran")
    cases = evaluate(stdout)
    return {"schema_version": 1,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_source": str(Path(__file__).resolve().relative_to(ROOT)),
            "audit_source_sha256": sha256(Path(__file__)),
            "scope": "original-method extracted call: mysofa_loudness only",
            "source": source, "platform": platform.platform(),
            "compiler": {"executable": executable, "version": compiler_version,
                         "command": command, "stdout": compiled.stdout,
                         "stderr": compiled.stderr, "flags": ["-std=c99", "-O0"]},
            "scaffold": {"export_header": EXPORT_HEADER,
                         "export_header_sha256": hashlib.sha256(EXPORT_HEADER.encode()).hexdigest(),
                         "harness_sha256": hashlib.sha256(HARNESS.encode()).hexdigest(),
                         "upstream_patched": False, "temporary_binary_retained": False},
            "config": {"dimensions": {"I": 1, "E": 1, "M": 1, "N": 1, "C": 3, "R": 2},
                       "source_position": [0.0, 0.0, 1.0],
                       "source_position_type": "spherical", "absolute_tolerance": ABS_TOLERANCE,
                       "randomness": "none; deterministic artificial coefficients"},
            "native_stdout": stdout, "cases": cases,
            "expected_behavior_observed": all(c["expected_behavior_observed"] for c in cases),
            "not_executed": ["SOFA parser", "SOFA measurement data", "mysofa_open",
                             "Data.Delay", "resampling", "interpolation",
                             "audio convolution/rendering", "device acceptance"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Save the actual run as strict JSON")
    parser.add_argument("--compiler", default="cc", help="C compiler executable (default: cc)")
    args = parser.parse_args()
    report = run_audit(compiler=args.compiler)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["expected_behavior_observed"]:
        raise SystemExit("Observed behavior differs from the two independently specified expectations")


if __name__ == "__main__":
    main()
