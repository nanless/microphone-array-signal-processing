"""Build three locked native libraries outside their trees and run original probes.

No downloads, package installations or source edits are performed.  Invoke from
the repository root with ``.venv/bin/python codes/examples/run_industrial_interfaces.py``.
Build products, callback traces and test WAV stay in a fresh ignored tmp directory.
Only the explicit JSON report is a publication artifact; this module is inert on import.
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
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codes.upstream.fetch_upstreams import inspect_project, load_projects  # noqa: E402

HARNESS = Path(__file__).with_name("industrial_interfaces.c")
DEFAULT_REPORT = ROOT / "codes/reports/industrial_interfaces.json"
PCM_EXPECTED = (-32768, 0, 0, 32767, 16384, -16384, 1, -1,
                12345, -23456, 0, 1000, 32767, -32768)
OPTIONS = {
    "libsoxr": ["-DBUILD_TESTS=OFF", "-DBUILD_EXAMPLES=OFF", "-DWITH_OPENMP=OFF",
                "-DWITH_LSR_BINDINGS=OFF", "-DWITH_CR32S=OFF", "-DWITH_CR64S=OFF",
                "-DWITH_PFFFT=OFF"],
    "libebur128": ["-DENABLE_TESTS=OFF", "-DENABLE_FUZZER=OFF"],
    "libsndfile": ["-DBUILD_TESTING=OFF", "-DBUILD_PROGRAMS=OFF", "-DBUILD_EXAMPLES=OFF",
                  "-DENABLE_EXTERNAL_LIBS=OFF", "-DENABLE_MPEG=OFF", "-DENABLE_CPACK=OFF"],
}
LIBRARY_FILES = {"libsoxr": "src/libsoxr.a", "libebur128": "libebur128.a",
                 "libsndfile": "libsndfile.a"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_measurements(data: dict) -> None:
    """Check independent arithmetic and protocol invariants, not saved outputs."""
    def reject_nonfinite(value):
        if isinstance(value, dict):
            for item in value.values():
                reject_nonfinite(item)
        elif isinstance(value, list):
            for item in value:
                reject_nonfinite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite numbers must be represented by null and a reason")

    reject_nonfinite(data)
    src = data["src"]
    for name in ("whole", "chunk127", "chunk509"):
        result = src[name]
        if result["output_frames"] != 16000:
            raise ValueError("SRC duration must be 48000 * 16000 / 48000 frames")
        if result["flush_frames"] <= 0 or result["before_flush_frames"] + result["flush_frames"] != 16000:
            raise ValueError("SRC flush must restore the pending tail")
        if abs(result["delay_after_flush_output_samples"]) > 1e-9:
            raise ValueError("SRC delay remains after draining")
    for name in ("chunk127_max_abs_error", "chunk509_max_abs_error"):
        if not math.isfinite(src[name]) or not 0 <= src[name] < 1e-12:
            raise ValueError("same-state SRC chunking mismatch")
    if src["whole"]["partial_consumption_calls"] < 1:
        raise ValueError("small output buffer did not exercise partial input consumption")
    if src["impulse_output_peak_frame"] != 4000:
        raise ValueError("impulse time must scale from input frame 12000 to output frame 4000")
    if not 0 <= src["sine_max_abs_error_frames_320_to_15680"] < 1e-5:
        raise ValueError("steady 1 kHz sine differs from analytic output")
    reset = src["clear_at_input_frame_24000_without_flush"]
    if reset["output_frames"] == 16000 and src["reset_equal_index_overlap_max_abs_difference"] <= 1e-6:
        raise ValueError("reset negative control unexpectedly preserved output")
    loudness = data["loudness"]
    delta = loudness["half_lufs"] - loudness["full_lufs"]
    if not math.isfinite(delta) or abs(delta - 20 * math.log10(0.5)) > 1e-10:
        raise ValueError("half-amplitude LUFS change must be -6.020599913... LU")
    if abs(loudness["half_minus_full_lu"] - delta) > 1e-12:
        raise ValueError("inconsistent reported loudness delta")
    if abs(loudness["chunk127_minus_whole_lu"]) > 1e-10:
        raise ValueError("loudness chunking mismatch")
    for kind in ("sample_peak", "true_peak"):
        if abs(loudness[f"half_{kind}"] / loudness[f"full_{kind}"] - 0.5) > 1e-12:
            raise ValueError("peak amplitude must halve")
    if loudness["silence_lufs"] is not None or loudness["silence_reason"] != "negative_infinity_no_gated_energy":
        raise ValueError("silence must remain null with its explicit non-finite reason")
    if loudness["silence_sample_peak"] != 0 or loudness["silence_true_peak"] != 0:
        raise ValueError("silence peaks must be zero")
    pcm = data["pcm16"]
    if pcm["frame_read_counts"] != [3, 3, 1, 0] or pcm["item_read_counts"] != [6, 6, 2, 0]:
        raise ValueError("incorrect short-read counts or frame/item units")
    if tuple(pcm["pcm_interleaved"]) != PCM_EXPECTED or pcm["float_max_abs_error"] != 0:
        raise ValueError("PCM16 interleaving or exact /32768 normalization mismatch")


def verify_wave(path: Path) -> dict:
    """Use Python's independent standard-library decoder for the native WAV."""
    with wave.open(str(path), "rb") as source:
        shape = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes())
        values = struct.unpack("<14h", source.readframes(7))
        trailing = source.readframes(1)
    if shape != (2, 2, 16000, 7) or values != PCM_EXPECTED or trailing:
        raise ValueError("standard-library WAV decoder disagrees with explicit fixture")
    return {"decoder": "Python stdlib wave + struct <14h", "status": "passed",
            "sha256": digest(path)}


def record_final_source_status(report: dict, projects: dict, source_root: Path) -> list[str]:
    """Record post-build failures without replacing an earlier experiment error."""
    errors = []
    for name, source in report["sources"].items():
        try:
            status = inspect_project(projects[name], source_root)["status"]
            source["final_status"] = status
            if status != "source_verified":
                errors.append(f"{name}: final source status {status}")
        except Exception as error:
            source["final_status"] = "verification_failed"
            errors.append(f"{name}: {error}")
    if errors:
        report["status"] = "failed"
    return errors


def run(report_path: Path = DEFAULT_REPORT) -> dict:
    cmake, compiler = shutil.which("cmake"), shutil.which("clang")
    if not cmake or not compiler:
        raise RuntimeError("Existing cmake and clang are required; nothing is installed automatically")
    projects = load_projects()
    source_root = ROOT / "codes/upstream/_downloads"
    sources = {}
    for name in OPTIONS:
        record = inspect_project(projects[name], source_root)
        if record["status"] != "source_verified":
            raise RuntimeError(f"{name}: expected existing verified source, found {record['status']}")
        sources[name] = {"revision": projects[name]["revision"], "origin": projects[name]["url"],
                         "initial_status": record["status"]}
    (ROOT / "tmp").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="industrial-interfaces-", dir=ROOT / "tmp"))
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GIT_", "CMAKE_", "DYLD_")) and key not in
                   {"CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH",
                    "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "PKG_CONFIG_PATH"}}
    commands = []

    def portable(text: str) -> str:
        return text.replace(str(ROOT), "${REPO_ROOT}").replace(cmake, "cmake").replace(compiler, "clang")

    def execute(arguments: list[str], label: str) -> subprocess.CompletedProcess:
        completed = subprocess.run(arguments, cwd=work, env=environment, capture_output=True,
                                   text=True, timeout=240, check=False)
        log = work / f"{label}.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        commands.append({"argv": [portable(arg) for arg in arguments], "returncode": completed.returncode,
                         "log": str(log.relative_to(ROOT)), "log_sha256": digest(log)})
        if completed.returncode:
            raise RuntimeError(f"{label} failed ({completed.returncode}); see {log}")
        return completed

    report = {"schema_version": 1, "status": "started", "sources": sources,
              "scope": "Synthetic native interface invariants; not standards certification, device audio or real-time benchmarking",
              "artifacts": {"harness": {"path": str(HARNESS.relative_to(ROOT)), "sha256": digest(HARNESS)},
                            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(Path(__file__))}},
              "environment": {"python": platform.python_version(), "system": platform.system(),
                              "release": platform.release(), "machine": platform.machine()},
              "work_directory": str(work.relative_to(ROOT)), "commands": commands}
    primary_error = None
    try:
        report["environment"]["cmake"] = execute([cmake, "--version"], "cmake-version").stdout.splitlines()[0]
        report["environment"]["compiler"] = execute([compiler, "--version"], "compiler-version").stdout.splitlines()[0]
        for name, options in OPTIONS.items():
            source, build = source_root / name, work / name
            execute([cmake, "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
                     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", f"-DCMAKE_C_COMPILER={compiler}",
                     *options], name + "-configure")
            execute([cmake, "--build", str(build), "--parallel", "2"], name + "-build")
        executable = work / "industrial_interfaces"
        execute([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", str(HARNESS),
                 "-I", str(source_root / "libsoxr/src"),
                 "-I", str(source_root / "libebur128/ebur128"),
                 "-I", str(source_root / "libsndfile/include"),
                 *(str(work / name / filename) for name, filename in LIBRARY_FILES.items()),
                 "-lm", "-o", str(executable)], "harness-build")
        wav_path = work / "fixture_pcm16.wav"
        result = execute([str(executable), str(wav_path)], "harness-run")
        data = json.loads(result.stdout)
        validate_measurements(data)
        report["measurements"] = data
        report["independent_wave_check"] = verify_wave(wav_path)
        report["binary_sha256"] = digest(executable)
        report["static_library_sha256"] = {name: digest(work / name / filename)
                                           for name, filename in LIBRARY_FILES.items()}
        report["callback_trace_rows"] = len(result.stderr.splitlines()) - 1
        report["status"] = "passed"
    except Exception as error:
        primary_error = error
        report["status"] = "failed"
        report["failure"] = portable(str(error))
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        verification_errors = record_final_source_status(report, projects, source_root)
        if verification_errors:
            report["post_build_source_errors"] = [portable(error) for error in verification_errors]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if verification_errors and primary_error is None:
            raise RuntimeError("Post-build source verification failed; see the report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    result = run(arguments.report)
    print(json.dumps({"status": result["status"], "report": str(arguments.report),
                      "measurements": result["measurements"]}, indent=2, allow_nan=False))
