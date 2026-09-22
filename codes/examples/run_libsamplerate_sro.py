"""Offline 100/150 ppm clock-drift probe using the locked libsamplerate C API.

The upstream source is linked from its ignored, independently verified checkout.
This program downloads nothing and writes build products only to a fresh system
temporary directory.  Its pulse-time expectations come from two clock equations,
not from libsamplerate or an earlier run.  No clock estimator or feedback loop is
implemented here.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codes.upstream.fetch_upstreams import inspect_project, load_projects  # noqa: E402

REFERENCE_RATE = 16_000
FIRST_PPM = 100
SECOND_PPM = 150
SWITCH_SECONDS = 5
DURATION_SECONDS = 10
MARKER_SECONDS = (1, 3, 4, 6, 8, 9)
CHUNK_FRAMES = 257
OUTPUT_CAPACITY = 64

# Original teaching harness.  It uses the public C API but copies no upstream
# implementation.  The final data-bearing call sets end_of_input=1; zero-input
# calls then collect the remaining output until the API generates none.
HARNESS = r'''
#include <samplerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REF 16000
#define F1 (REF * (1.0 + 100.0 / 1000000.0))
#define F2 (REF * (1.0 + 150.0 / 1000000.0))
#define SPLIT ((long) llround(5.0 * F1))
#define TOTAL ((long) llround(5.0 * F1 + 5.0 * F2))
#define CHUNK 257
#define OUT_CAP 64

static void fail(const char *where, int error) {
    if (error) fprintf(stderr, "%s: %s\n", where, src_strerror(error));
    else fprintf(stderr, "%s\n", where);
    exit(2);
}

static void write_call(FILE *trace, FILE *output, float *samples,
                       const char *phase, long start, long offered,
                       SRC_DATA *data, double ratio) {
    if (data->input_frames_used < 0 || data->input_frames_used > offered ||
        data->output_frames_gen < 0 || data->output_frames_gen > OUT_CAP)
        fail("invalid API frame counts", 0);
    if (fwrite(samples, sizeof(float), (size_t)data->output_frames_gen, output)
        != (size_t)data->output_frames_gen)
        fail("output write", 0);
    fprintf(trace, "%s,%ld,%ld,%ld,%ld,%d,%.17g\n", phase, start,
            offered, data->input_frames_used, data->output_frames_gen,
            data->end_of_input, ratio);
}

int main(int argc, char **argv) {
    if (argc != 4 || (strcmp(argv[3], "corrected") != 0 &&
                      strcmp(argv[3], "uncorrected") != 0)) {
        fprintf(stderr, "usage: probe output.f32 trace.csv corrected|uncorrected\n");
        return 2;
    }
    const int corrected = strcmp(argv[3], "corrected") == 0;
    float *input = calloc((size_t)TOTAL, sizeof(float));
    FILE *output = fopen(argv[1], "wb");
    FILE *trace = fopen(argv[2], "w");
    int error = 0;
    SRC_STATE *state = src_new(SRC_SINC_MEDIUM_QUALITY, 1, &error);
    if (!input || !output || !trace || !state)
        fail("setup", error);
    const double seconds[] = {1, 3, 4, 6, 8, 9};
    for (size_t i = 0; i < sizeof(seconds) / sizeof(seconds[0]); ++i) {
        double t = seconds[i];
        long n = (long) llround(t <= 5 ? t * F1 : 5.0 * F1 + (t - 5.0) * F2);
        if (n < 0 || n >= TOTAL) fail("marker index", 0);
        input[n] = 0.8f;
    }
    fprintf(trace, "phase,input_start,offered,used,generated,eof,ratio\n");
    long cursor = 0;
    for (int segment = 0; segment < 2; ++segment) {
        long end = segment == 0 ? SPLIT : TOTAL;
        double ratio = corrected ? REF / (segment == 0 ? F1 : F2) : 1.0;
        while (cursor < end) {
            long block_end = cursor + CHUNK;
            if (block_end > end) block_end = end;
            while (cursor < block_end) {
                float out[OUT_CAP];
                SRC_DATA data = {0};
                data.data_in = input + cursor;
                data.input_frames = block_end - cursor;
                data.data_out = out;
                data.output_frames = OUT_CAP;
                data.src_ratio = ratio;
                data.end_of_input = (segment == 1 && block_end == TOTAL);
                error = src_process(state, &data);
                if (error) fail("src_process", error);
                if (data.input_frames_used == 0 && data.output_frames_gen == 0)
                    fail("no forward progress", 0);
                write_call(trace, output, out, "input", cursor,
                           block_end - cursor, &data, ratio);
                cursor += data.input_frames_used;
            }
        }
    }
    for (int calls = 0; calls < 4096; ++calls) {
        float out[OUT_CAP];
        SRC_DATA data = {0};
        data.data_in = input + TOTAL;
        data.input_frames = 0;
        data.data_out = out;
        data.output_frames = OUT_CAP;
        data.src_ratio = corrected ? REF / F2 : 1.0;
        data.end_of_input = 1;
        error = src_process(state, &data);
        if (error) fail("drain", error);
        write_call(trace, output, out, "drain", TOTAL, 0, &data,
                   data.src_ratio);
        if (data.output_frames_gen == 0) break;
        if (calls == 4095) fail("drain did not end", 0);
    }
    if (fclose(trace) || fclose(output)) fail("close", 0);
    printf("%s\n", src_get_version());
    src_delete(state);
    free(input);
    return 0;
}
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock_frame(seconds: int) -> int:
    """Independent piecewise clock: device samples, not reference samples."""
    first = REFERENCE_RATE * (1 + FIRST_PPM / 1_000_000)
    second = REFERENCE_RATE * (1 + SECOND_PPM / 1_000_000)
    source_time = (min(seconds, SWITCH_SECONDS) * first
                   + max(0, seconds - SWITCH_SECONDS) * second)
    return math.floor(source_time + 0.5)


def inspect_trace(path: Path, output_frames: int, corrected: bool) -> dict:
    """Check every public-API count, including unconsumed input and the EOF drain."""
    consumed = generated = calls = partial = drain_generated = eof_calls = 0
    ratios = set()
    last_phase = None
    final_made = None
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            phase = row["phase"]
            start, offered, used, made, eof = (int(row[key]) for key in
                                               ("input_start", "offered", "used", "generated", "eof"))
            ratio = float(row["ratio"])
            if phase not in ("input", "drain") or (phase == "input" and last_phase == "drain"):
                raise ValueError("invalid trace phase order")
            if start != consumed or used < 0 or offered < used or made < 0 or made > OUTPUT_CAPACITY:
                raise ValueError("lost or duplicated source frame, or invalid API count")
            if phase == "drain" and (start != clock_frame(DURATION_SECONDS) or offered or used or not eof):
                raise ValueError("invalid EOF drain call")
            if phase == "input" and (not offered or not (used or made)):
                raise ValueError("input call made no forward progress")
            if corrected and phase == "input" and start < clock_frame(SWITCH_SECONDS) < start + offered:
                raise ValueError("input block crosses the clock-rate change")
            expected = (1.0 if not corrected else REFERENCE_RATE /
                        (REFERENCE_RATE * (1 + (FIRST_PPM if start < clock_frame(SWITCH_SECONDS)
                                                else SECOND_PPM) / 1_000_000)))
            if not math.isfinite(ratio) or abs(ratio - expected) > 1e-14:
                raise ValueError("wrong per-block resampling ratio")
            if eof and phase == "input" and start + offered < clock_frame(DURATION_SECONDS):
                raise ValueError("end_of_input set before final source block")
            if used < offered:
                partial += 1
            if phase == "drain":
                drain_generated += made
            consumed += used
            generated += made
            calls += 1
            eof_calls += eof
            ratios.add(ratio)
            last_phase = phase
            final_made = made
    if (consumed != clock_frame(DURATION_SECONDS) or generated != output_frames):
        raise ValueError("source/output frame totals disagree with trace")
    if last_phase != "drain" or final_made != 0 or not eof_calls or not partial or not drain_generated:
        raise ValueError("partial consumption or final drain was not exercised")
    if len(ratios) != (2 if corrected else 1):
        raise ValueError("ratio did not change at the clock boundary")
    return {"api_calls": calls, "partial_input_calls": partial,
            "input_frames_used_total": consumed, "output_frames_gen_total": generated,
            "drain_output_frames": drain_generated, "eof_calls": eof_calls,
            "ratio_values": sorted(ratios)}


def marker_offsets(path: Path) -> tuple[int, dict[int, int]]:
    if path.stat().st_size % 4:
        raise ValueError("float32 output has a partial trailing item")
    samples = array("f")
    with path.open("rb") as source:
        samples.fromfile(source, path.stat().st_size // 4)
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("non-finite or empty audio output")
    offsets = {}
    for seconds in MARKER_SECONDS:
        ideal = seconds * REFERENCE_RATE
        low, high = max(0, ideal - 80), min(len(samples), ideal + 81)
        if high <= low:
            raise ValueError("marker is outside output interval")
        frame = max(range(low, high), key=lambda index: abs(samples[index]))
        if abs(samples[frame]) < 0.3:
            raise ValueError("pulse marker lost or too weak")
        offsets[seconds] = frame - ideal
    return len(samples), offsets


def validate_clock_result(uncorrected: dict, corrected: dict) -> dict:
    """Use clock arithmetic for the expected slopes; do not self-oracle the SRC."""
    raw = {int(k): int(v) for k, v in uncorrected["marker_offsets_frames"].items()}
    fixed = {int(k): int(v) for k, v in corrected["marker_offsets_frames"].items()}
    if set(raw) != set(MARKER_SECONDS) or set(fixed) != set(MARKER_SECONDS):
        raise ValueError("missing clock markers")
    raw_span = raw[9] - raw[1]
    expected_span = clock_frame(9) - clock_frame(1) - 8 * REFERENCE_RATE
    fixed_span = fixed[9] - fixed[1]
    if abs(raw_span - expected_span) > 2 or abs(fixed_span) > 3:
        raise ValueError("observed residual drift disagrees with independent clock model")
    for first, last in ((1, 4), (6, 9)):
        expected = clock_frame(last) - clock_frame(first) - (last - first) * REFERENCE_RATE
        if abs((raw[last] - raw[first]) - expected) > 2 or abs(fixed[last] - fixed[first]) > 3:
            raise ValueError("segment drift disagrees with the two clock rates")
    if abs(corrected["output_frames"] - DURATION_SECONDS * REFERENCE_RATE) > 3:
        raise ValueError("corrected output length disagrees with reference time base")
    return {"uncorrected_span_frames": raw_span, "independent_expected_span_frames": expected_span,
            "corrected_residual_span_frames": fixed_span,
            "residual_drift_reduction_frames": abs(raw_span) - abs(fixed_span)}


def _run_command(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True, timeout=240, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(arguments[0]).name} exited {result.returncode}:\n{result.stderr[-3000:]}")
    return result


def run() -> dict:
    projects = load_projects()
    project = projects["libsamplerate"]
    downloaded = ROOT / "codes/upstream/_downloads"
    source = downloaded / "libsamplerate"
    status = inspect_project(project, downloaded)
    if status["status"] != "source_verified":
        raise RuntimeError(f"locked libsamplerate checkout not verified: {status['status']}")
    cmake, compiler = shutil.which("cmake"), shutil.which("clang") or shutil.which("cc")
    if not cmake or not compiler:
        raise RuntimeError("local CMake and C compiler are required; no tool is installed automatically")
    with tempfile.TemporaryDirectory(prefix="libsamplerate-sro-") as temporary:
        work = Path(temporary)
        harness, build = work / "sro_probe.c", work / "build"
        harness.write_text(HARNESS, encoding="utf-8")
        _run_command([cmake, "-S", str(source), "-B", str(build),
                      "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
                      "-DBUILD_TESTING=OFF", "-DLIBSAMPLERATE_EXAMPLES=OFF",
                      "-DLIBSAMPLERATE_INSTALL=OFF", f"-DCMAKE_C_COMPILER={compiler}"], work)
        _run_command([cmake, "--build", str(build), "--target", "samplerate", "--parallel", "2"], work)
        library = build / "src/libsamplerate.a"
        executable = work / "sro_probe"
        _run_command([compiler, "-std=c99", "-O2", "-Wall", "-Wextra",
                      "-I", str(source / "include"), str(harness), str(library),
                      "-lm", "-o", str(executable)], work)
        cases = {}
        for name in ("uncorrected", "corrected"):
            audio, trace = work / f"{name}.f32", work / f"{name}.csv"
            version = _run_command([str(executable), str(audio), str(trace), name], work).stdout.strip()
            frames, offsets = marker_offsets(audio)
            case = {"output_frames": frames, "marker_offsets_frames": offsets,
                    "trace": inspect_trace(trace, frames, name == "corrected"),
                    "audio_sha256": sha256(audio), "trace_sha256": sha256(trace)}
            cases[name] = case
        result = validate_clock_result(cases["uncorrected"], cases["corrected"])
        final_status = inspect_project(project, downloaded)["status"]
        if final_status != "source_verified":
            raise RuntimeError(f"source changed during experiment: {final_status}")
        return {"status": "passed", "scope": "synthetic fixed-known two-clock correction, not estimation or hardware",
                "source": {"id": "libsamplerate", "revision": project["revision"],
                           "license": project["license"], "final_status": final_status},
                "build": {"cmake": _run_command([cmake, "--version"], work).stdout.splitlines()[0],
                          "compiler": _run_command([compiler, "--version"], work).stdout.splitlines()[0],
                          "library_sha256": sha256(library), "harness_sha256": sha256(harness),
                          "host": f"{platform.system()} {platform.machine()}"},
                "configuration": {"reference_rate_hz": REFERENCE_RATE,
                                  "first_device_ppm": FIRST_PPM, "second_device_ppm": SECOND_PPM,
                                  "switch_seconds": SWITCH_SECONDS, "duration_seconds": DURATION_SECONDS,
                                  "marker_seconds": MARKER_SECONDS, "input_frames": clock_frame(DURATION_SECONDS),
                                  "input_block_frames": CHUNK_FRAMES, "output_capacity_frames": OUTPUT_CAPACITY,
                                  "converter": "SRC_SINC_MEDIUM_QUALITY", "libsamplerate_version": version,
                                  "ratio_rule": "output sample rate / input sample rate",
                                  "alignment": "No fitted shift or gain; compare pulse peaks against reference clock"},
                "cases": cases, "independent_clock_check": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
