"""Reproduce locked smpphat SRP/SMP-PHAT with an independent NumPy baseline.

The default path is offline and never installs a dependency.  Pass either an
existing single-precision FFTW prefix or ``--download-fftw``.  The latter
downloads one checksum-pinned official archive into an isolated work directory.
No upstream source or compiled executable is copied into this repository.
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
import subprocess
import tarfile
import tempfile
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "6fd33e6eb3251078a4cd9793dde909e2500265cc"
UPSTREAM_URL = "https://github.com/FrancoisGrondin/smpphat"
FFTW_VERSION = "3.3.10"
FFTW_URL = f"https://fftw.org/pub/fftw/fftw-{FFTW_VERSION}.tar.gz"
FFTW_SHA256 = "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
FRAME_SIZE = 64
INTERPOLATION_RATE = 4
SAMPLE_RATE = 16_000
SOUND_SPEED = 343.0
CHANNELS = 4
POINTS = 24
TRUE_INDEX = 3  # 45 degrees on the 15-degree grid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_directions() -> np.ndarray:
    """Horizontal grid: zero is +y and positive azimuth rotates toward +x."""
    angles = np.deg2rad(np.arange(POINTS, dtype=np.float64) * 15.0)
    return np.column_stack((np.sin(angles), np.cos(angles), np.zeros(POINTS))).astype("<f4")


def fixed_geometries() -> list[tuple[str, np.ndarray, str]]:
    """A synthetic diamond and a copy with one microphone moved by 0.5 mm."""
    regular = np.array([
        [-0.0320, 0.0000, 0.0000],
        [0.0000, -0.0320, 0.0000],
        [0.0320, 0.0000, 0.0000],
        [0.0000, 0.0320, 0.0000],
    ], dtype="<f4")
    perturbed = regular.copy()
    perturbed[3, 1] += np.float32(0.0005)
    return [
        ("regular_diamond", regular, "four microphones on a 32 mm-radius diamond"),
        ("mic4_y_plus_0p5mm", perturbed, "same geometry; microphone 4 y increased by 0.5 mm"),
    ]


def pair_vectors(geometry: np.ndarray) -> np.ndarray:
    return np.array([geometry[first] - geometry[second]
                     for first in range(CHANNELS)
                     for second in range(first + 1, CHANNELS)], dtype=np.float64)


def make_phat(geometry: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Unit-magnitude Xi Xj* for the upstream TDOA sign convention."""
    delays = -(SAMPLE_RATE / SOUND_SPEED) * (pair_vectors(geometry) @ direction)
    bins = np.arange(FRAME_SIZE // 2 + 1, dtype=np.float64)
    phase = -2.0 * np.pi * delays[:, None] * bins[None, :] / FRAME_SIZE
    spectrum = np.exp(1j * phase)
    interleaved = np.empty((*spectrum.shape, 2), dtype="<f4")
    interleaved[..., 0] = spectrum.real
    interleaved[..., 1] = spectrum.imag
    # A real frame's original Nyquist endpoint must be real.  It is omitted
    # rather than inventing a fractional-delay value at that single endpoint.
    interleaved[:, -1, :] = 0.0
    return interleaved


def merge_plan(geometry: np.ndarray, epsilon: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Independent translation of the documented equal, parallel baseline rule."""
    vectors = pair_vectors(geometry)
    groups = np.zeros(len(vectors), dtype=np.uint32)
    polarities = np.zeros(len(vectors), dtype=np.float64)
    group = 0
    for first in range(len(vectors)):
        if polarities[first] != 0:
            continue
        groups[first] = group
        polarities[first] = 1.0
        for second in range(len(vectors)):
            if polarities[second] != 0:
                continue
            dot = float(vectors[first] @ vectors[second])
            norm_first = float(np.linalg.norm(vectors[first]))
            norm_second = float(np.linalg.norm(vectors[second]))
            if (abs(abs(dot) - norm_first * norm_second) < epsilon
                    and abs(norm_first - norm_second) < epsilon):
                groups[second] = group
                polarities[second] = 1.0 if dot > 0 else -1.0
        group += 1
    return groups, polarities


def _round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _unscaled_c2r(spectrum: np.ndarray) -> np.ndarray:
    """Direct FFTW c2r definition after zero-padding, without calling an FFT."""
    length = FRAME_SIZE * INTERPOLATION_RATE
    bins = np.arange(spectrum.size, dtype=np.float64)
    times = np.arange(length, dtype=np.float64)[:, None]
    # All supplied positive bins except DC lie below the padded transform's
    # Nyquist bin and therefore have distinct negative-frequency conjugates.
    return (float(spectrum[0].real)
            + 2.0 * np.real(np.exp(2j * np.pi * times * bins[None, :] / length)
                            * spectrum[None, :])[:, 1:].sum(axis=1))


def _tdoas(geometry: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return -(SAMPLE_RATE / SOUND_SPEED) * (directions.astype(np.float64) @ pair_vectors(geometry).T)


def independent_scores(
    geometry: np.ndarray, directions: np.ndarray, phat: np.ndarray
) -> dict[str, object]:
    """Calculate SRP and SMP scores directly from the DFT definition."""
    complex_phat = phat[..., 0].astype(np.float64) + 1j * phat[..., 1].astype(np.float64)
    delays = _tdoas(geometry, directions)
    half_ranges = np.ceil(np.max(np.abs(delays), axis=0) * INTERPOLATION_RATE).astype(int)
    srp_ranges = 2 * half_ranges + 1
    srp_offsets = np.concatenate(([0], np.cumsum(srp_ranges[:-1])))
    correlations = np.array([_unscaled_c2r(row) for row in complex_phat])
    srp_scores = []
    srp_lookups = np.empty((len(directions), len(complex_phat)), dtype=int)
    for point in range(len(directions)):
        score = 0.0
        for pair in range(len(complex_phat)):
            lag = _round_away_from_zero(float(delays[point, pair]) * INTERPOLATION_RATE)
            if abs(lag) > half_ranges[pair]:
                raise ArithmeticError("lookup lies outside independently calculated range")
            srp_lookups[point, pair] = srp_offsets[pair] + half_ranges[pair] + lag
            score += correlations[pair, lag % correlations.shape[1]]
        srp_scores.append(score)

    groups, polarities = merge_plan(geometry)
    group_count = int(groups.max()) + 1
    group_spectra = np.zeros((group_count, complex_phat.shape[1]), dtype=np.complex128)
    reference_pairs = np.empty(group_count, dtype=int)
    for group in range(group_count):
        reference_pairs[group] = int(np.flatnonzero(groups == group)[0])
    for pair, group in enumerate(groups):
        group_spectra[group] += (complex_phat[pair] if polarities[pair] > 0
                                 else complex_phat[pair].conj())
    group_correlations = np.array([_unscaled_c2r(row) for row in group_spectra])
    group_half_ranges = half_ranges[reference_pairs]
    smp_ranges = 2 * group_half_ranges + 1
    smp_offsets = np.concatenate(([0], np.cumsum(smp_ranges[:-1])))
    smp_scores = []
    smp_lookups = np.empty((len(directions), group_count), dtype=int)
    for point in range(len(directions)):
        score = 0.0
        for group, reference_pair in enumerate(reference_pairs):
            lag = _round_away_from_zero(
                float(delays[point, reference_pair]) * INTERPOLATION_RATE)
            smp_lookups[point, group] = (
                smp_offsets[group] + group_half_ranges[group] + lag)
            score += group_correlations[group, lag % group_correlations.shape[1]]
        smp_scores.append(score)
    return {
        "groups": groups.tolist(),
        "polarities": polarities.tolist(),
        "srp_ranges": srp_ranges.tolist(),
        "smp_ranges": smp_ranges.tolist(),
        "srp_lookups": srp_lookups.reshape(-1).tolist(),
        "smp_lookups": smp_lookups.reshape(-1).tolist(),
        "srp_scores": srp_scores,
        "smp_scores": smp_scores,
        "srp_peak_index": int(np.argmax(srp_scores)),
        "smp_peak_index": int(np.argmax(smp_scores)),
    }


def _flatten_lag_windows(correlations: np.ndarray, ranges: list[int]) -> np.ndarray:
    windows = []
    for correlation, size in zip(correlations, ranges):
        half = (int(size) - 1) // 2
        windows.append(np.concatenate((correlation[-half:] if half else np.empty(0),
                                       correlation[:half + 1])))
    return np.concatenate(windows)


def direct_scores_at_upstream_lookups(phat: np.ndarray, upstream: dict) -> dict[str, list[float]]:
    """Isolate FFT values from the upstream lookup-table behavior."""
    complex_phat = phat[..., 0].astype(np.float64) + 1j * phat[..., 1].astype(np.float64)
    pair_correlations = np.array([_unscaled_c2r(row) for row in complex_phat])
    srp_flat = _flatten_lag_windows(pair_correlations, upstream["srp_ranges"])
    srp_lookups = np.asarray(upstream["srp_lookups"], dtype=int).reshape(POINTS, -1)
    srp_scores = srp_flat[srp_lookups].sum(axis=1)

    groups = np.asarray(upstream["groups"], dtype=int)
    polarities = np.asarray(upstream["polarities"], dtype=float)
    group_count = int(upstream["groups_count"])
    group_spectra = np.zeros((group_count, complex_phat.shape[1]), dtype=np.complex128)
    for pair, group in enumerate(groups):
        group_spectra[group] += (complex_phat[pair] if polarities[pair] > 0
                                 else complex_phat[pair].conj())
    group_correlations = np.array([_unscaled_c2r(row) for row in group_spectra])
    smp_flat = _flatten_lag_windows(group_correlations, upstream["smp_ranges"])
    smp_lookups = np.asarray(upstream["smp_lookups"], dtype=int).reshape(POINTS, -1)
    smp_scores = smp_flat[smp_lookups].sum(axis=1)
    return {"srp_scores": srp_scores.tolist(), "smp_scores": smp_scores.tolist()}


def verify_upstream(source_dir: Path) -> dict[str, str]:
    """Require the exact clean checkout before compiling its unchanged files."""
    source_dir = source_dir.resolve()
    if not (source_dir / ".git").exists():
        raise FileNotFoundError("locked smpphat checkout is absent")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source_dir), *arguments], env=env, text=True,
            stderr=subprocess.PIPE).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source_dir:
        raise ValueError("smpphat directory is not its own checkout")
    if git("rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("smpphat revision differs from the experiment lock")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("smpphat worktree contains modified or untracked files")
    files = ["CMakeLists.txt", "LICENSE", "src/signal.c", "src/system.c",
             "include/smpphat/const.h", "include/smpphat/signal.h",
             "include/smpphat/system.h"]
    return {name: sha256(source_dir / name) for name in files}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            resolved = (destination / member.name).resolve()
            if destination != resolved and destination not in resolved.parents:
                raise ValueError("FFTW archive contains a path outside the work directory")
        bundle.extractall(destination, filter="data")


def _download_fftw_archive(archive: Path, work_dir: Path) -> None:
    temporary = tempfile.NamedTemporaryFile(
        prefix="fftw-download-", suffix=".tar.gz", dir=work_dir, delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        urllib.request.urlretrieve(FFTW_URL, temporary_path)
        if sha256(temporary_path) != FFTW_SHA256:
            raise ValueError("downloaded FFTW archive SHA-256 does not match the pinned value")
        os.replace(temporary_path, archive)
    finally:
        temporary_path.unlink(missing_ok=True)


def ensure_fftw(work_dir: Path, *, download: bool) -> tuple[Path, dict[str, object]]:
    """Build checksum-pinned FFTW single precision under work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "fftw-prefix-verified"
    library = prefix / "lib/libfftw3f.a"
    header = prefix / "include/fftw3.h"
    manifest_path = work_dir / "fftw-prefix-verified-manifest.json"
    archive = work_dir / f"fftw-{FFTW_VERSION}.tar.gz"
    if library.is_file() and header.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "source": "checksum_verified_official_archive",
            "version": FFTW_VERSION,
            "archive_url": FFTW_URL,
            "archive_sha256": FFTW_SHA256,
            "header_sha256": sha256(header),
            "library_sha256": sha256(library),
        }
        if manifest != expected:
            raise ValueError("cached FFTW prefix manifest does not match its files")
        if not archive.is_file() or sha256(archive) != FFTW_SHA256:
            if not download:
                raise ValueError("cached FFTW archive does not match its manifest")
            _download_fftw_archive(archive, work_dir)
        return prefix, manifest
    if library.is_file() and header.is_file() and not download:
        raise ValueError("cached FFTW prefix lacks a verified build manifest")
    archive_valid = archive.is_file() and sha256(archive) == FFTW_SHA256
    if not archive_valid:
        if not download:
            if archive.exists():
                raise ValueError("FFTW archive SHA-256 does not match the pinned value")
            raise FileNotFoundError(
                "FFTW3f is absent; pass --fftw-prefix or explicitly opt in with --download-fftw")
        _download_fftw_archive(archive, work_dir)
    build_root = Path(tempfile.mkdtemp(prefix="fftw-verified-build-", dir=work_dir))
    _safe_extract(archive, build_root)
    source = build_root / f"fftw-{FFTW_VERSION}"
    configure = [
        str(source / "configure"), f"--prefix={prefix}", "--enable-float",
        "--disable-shared", "--enable-static", "--disable-fortran",
    ]
    subprocess.run(configure, cwd=source, check=True)
    subprocess.run(["make", "-j2"], cwd=source, check=True)
    subprocess.run(["make", "install"], cwd=source, check=True)
    if not library.is_file() or not header.is_file():
        raise FileNotFoundError("isolated FFTW build did not produce fftw3f")
    manifest = {
        "source": "checksum_verified_official_archive",
        "version": FFTW_VERSION,
        "archive_url": FFTW_URL,
        "archive_sha256": FFTW_SHA256,
        "header_sha256": sha256(header),
        "library_sha256": sha256(library),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return prefix, manifest


def validate_fftw_prefix(prefix: Path) -> tuple[Path, dict[str, object]]:
    prefix = prefix.resolve()
    header = prefix / "include/fftw3.h"
    library = prefix / "lib/libfftw3f.a"
    if not header.is_file() or not library.is_file():
        raise FileNotFoundError("FFTW prefix must contain include/fftw3.h and lib/libfftw3f.a")
    return prefix, {
        "source": "provided_prefix",
        "version": "not verified from an archive",
        "archive_url": None,
        "archive_sha256": None,
        "header_sha256": sha256(header),
        "library_sha256": sha256(library),
    }


def compile_harness(source_dir: Path, fftw_prefix: Path, build_dir: Path) -> tuple[Path, list[str], str]:
    build_dir.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(os.environ.get("CC", "clang"))
    if compiler is None:
        raise FileNotFoundError("a C compiler is required")
    executable = build_dir / "reproduce_smpphat_harness"
    command = [
        compiler, "-std=c11", "-O2", "-Wall", "-Wextra",
        "-I", str(fftw_prefix / "include"), "-I", str(source_dir / "include"),
        str(Path(__file__).with_name("reproduce_smpphat_harness.c")),
        str(source_dir / "src/signal.c"), str(source_dir / "src/system.c"),
        str(fftw_prefix / "lib/libfftw3f.a"), "-lm", "-o", str(executable),
    ]
    subprocess.run(command, cwd=build_dir, check=True)
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    return executable, command, version


def _write_float32(path: Path, values: np.ndarray) -> None:
    payload = np.asarray(values, dtype="<f4")
    path.write_bytes(payload.tobytes(order="C"))


def _run_case(executable: Path, work_dir: Path, name: str, geometry: np.ndarray,
              directions: np.ndarray, description: str) -> dict[str, object]:
    case_dir = work_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = case_dir / "geometry.f32"
    directions_path = case_dir / "directions.f32"
    phat_path = case_dir / "phat.f32"
    phat = make_phat(geometry, directions[TRUE_INDEX])
    _write_float32(geometry_path, geometry)
    _write_float32(directions_path, directions)
    _write_float32(phat_path, phat)
    output = subprocess.check_output(
        [str(executable), str(geometry_path), str(directions_path), str(phat_path)],
        text=True)
    upstream = json.loads(output)
    expected = independent_scores(geometry, directions, phat)
    diagnosed = direct_scores_at_upstream_lookups(phat, upstream)
    srp_intended_error = float(np.max(np.abs(
        np.asarray(upstream["srp_scores"]) - np.asarray(expected["srp_scores"]))))
    smp_intended_error = float(np.max(np.abs(
        np.asarray(upstream["smp_scores"]) - np.asarray(expected["smp_scores"]))))
    srp_diagnosed_error = float(np.max(np.abs(
        np.asarray(upstream["srp_scores"]) - np.asarray(diagnosed["srp_scores"]))))
    smp_diagnosed_error = float(np.max(np.abs(
        np.asarray(upstream["smp_scores"]) - np.asarray(diagnosed["smp_scores"]))))
    srp_smp_error = float(np.max(np.abs(
        np.asarray(upstream["srp_scores"]) - np.asarray(upstream["smp_scores"]))))
    intended_equivalence_error = float(np.max(np.abs(
        np.asarray(expected["srp_scores"]) - np.asarray(expected["smp_scores"]))))
    actual_lookups = np.asarray(upstream["srp_lookups"], dtype=int).reshape(POINTS, -1)
    intended_lookups = np.asarray(expected["srp_lookups"], dtype=int).reshape(POINTS, -1)
    mismatch_positions = np.argwhere(actual_lookups != intended_lookups)
    interpolated_delays = _tdoas(geometry, directions) * INTERPOLATION_RATE
    lookup_examples = [{
        "point_index": int(point), "pair_index": int(pair),
        "interpolated_tdoa": float(interpolated_delays[point, pair]),
        "intended_signed_lookup": int(intended_lookups[point, pair]),
        "upstream_lookup": int(actual_lookups[point, pair]),
    } for point, pair in mismatch_positions[:8]]

    expected_group_count = 4 if name == "regular_diamond" else 6
    if upstream["srp_status"] != 0 or upstream["smp_status"] != 0:
        raise ArithmeticError("upstream SRP or SMP-PHAT returned an error")
    if upstream["pairs_count"] != 6 or upstream["groups_count"] != expected_group_count:
        raise ArithmeticError("upstream pair/group count differs from the analytic geometry")
    if upstream["groups"] != expected["groups"] or upstream["polarities"] != expected["polarities"]:
        raise ArithmeticError("upstream grouping differs from the independent baseline")
    if upstream["srp_peak_index"] != TRUE_INDEX or upstream["smp_peak_index"] != TRUE_INDEX:
        raise ArithmeticError("the fixed true direction is not the upstream peak")
    if max(srp_diagnosed_error, smp_diagnosed_error) >= 2e-4:
        raise ArithmeticError("direct DFT at upstream lookups does not explain the C scores")
    if intended_equivalence_error >= 1e-9:
        raise ArithmeticError("the independent signed-lookup SRP/SMP equivalence check failed")
    if name != "regular_diamond" and srp_smp_error >= 2e-4:
        raise ArithmeticError("unmerged SMP must equal SRP for the perturbed array")
    expected_equivalence_failed = srp_smp_error >= 2e-4
    intended_signed_lookup_failed = max(srp_intended_error, smp_intended_error) >= 2e-4
    return {
        "case": name,
        "description": description,
        "geometry_m": geometry.tolist(),
        "phat_float32_sha256": sha256(phat_path),
        "upstream_c": upstream,
        "independent_signed_lookup_direct_dft": expected,
        "direct_dft_at_upstream_lookups": diagnosed,
        "lookup_diagnosis": {
            "srp_lookup_mismatch_count": int(len(mismatch_positions)),
            "examples": lookup_examples,
            "cause": "negative rounded TDOA is converted to unsigned before addition",
            "language_status": "negative out-of-range floating-to-unsigned conversion is undefined in C",
            "upstream_source_locations": ["src/system.c:588", "src/system.c:824"],
        },
        "comparison": {
            "c_srp_vs_intended_signed_lookup_max_abs": srp_intended_error,
            "c_smp_vs_intended_signed_lookup_max_abs": smp_intended_error,
            "c_srp_vs_direct_dft_at_upstream_lookup_max_abs": srp_diagnosed_error,
            "c_smp_vs_direct_dft_at_upstream_lookup_max_abs": smp_diagnosed_error,
            "c_srp_vs_c_smp_max_abs": srp_smp_error,
            "independent_srp_vs_smp_max_abs": intended_equivalence_error,
            "same_peak": upstream["srp_peak_index"] == upstream["smp_peak_index"]
                         == expected["srp_peak_index"] == expected["smp_peak_index"],
            "expected_equivalence_failed": expected_equivalence_failed,
            "intended_signed_lookup_failed": intended_signed_lookup_failed,
            "diagnosis_verified": max(srp_diagnosed_error, smp_diagnosed_error) < 2e-4,
        },
    }


def run_experiment(*, source_dir: Path | None = None, fftw_prefix: Path | None = None,
                   work_dir: Path | None = None, download_fftw: bool = False) -> dict[str, object]:
    source_dir = (source_dir or ROOT / "upstream/_downloads/smpphat").resolve()
    source_hashes = verify_upstream(source_dir)
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="smpphat-reference-"))
    else:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    prefix, fftw_provenance = (
        validate_fftw_prefix(fftw_prefix) if fftw_prefix is not None
        else ensure_fftw(work_dir, download=download_fftw))
    executable, compile_command, compiler_version = compile_harness(
        source_dir, prefix, work_dir / "build")
    directions = fixed_directions()
    cases = [_run_case(executable, work_dir / "inputs", name, geometry, directions, description)
             for name, geometry, description in fixed_geometries()]
    final_source_hashes = verify_upstream(source_dir)
    if final_source_hashes != source_hashes:
        raise ValueError("locked smpphat source changed during the experiment")
    runtime_versions = {case["upstream_c"]["fftw_runtime_version"] for case in cases}
    if len(runtime_versions) != 1:
        raise ValueError("FFTW runtime version changed between cases")
    runtime_version = runtime_versions.pop()
    fftw_provenance = {**fftw_provenance, "runtime_version": runtime_version}
    if (fftw_provenance["source"] == "checksum_verified_official_archive"
            and runtime_version != f"fftw-{FFTW_VERSION}"):
        raise ValueError("verified FFTW archive produced an unexpected runtime version")
    numerical_failure = any(
        case["comparison"]["expected_equivalence_failed"]
        or case["comparison"]["intended_signed_lookup_failed"]
        for case in cases)
    return {
        "schema_version": 1,
        "experiment": "locked_smpphat_srp_vs_merged_pairs",
        "numerical_status": (
            "failed_portability" if numerical_failure
            else "passed_numerically_with_undefined_lookup_conversion"),
        "provenance": {
            "upstream_url": UPSTREAM_URL,
            "upstream_revision": UPSTREAM_REVISION,
            "upstream_license": "GPL-3.0; separately obtained and compiled without modification",
            "upstream_file_sha256": source_hashes,
            "upstream_verified_on": "2026-09-22",
            "fftw": {**fftw_provenance,
                     "license": "GPL-2.0-or-later; separately obtained, not distributed here"},
            "runner_sha256": sha256(Path(__file__)),
            "harness_sha256": sha256(Path(__file__).with_name("reproduce_smpphat_harness.c")),
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "compiler": compiler_version,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "build": {
            "upstream_cmake_used": False,
            "reason": "upstream CMake writes into its source tree and forces unsupported arm64 -msse3",
            "algorithm_source_modified": False,
            "fftw_backend_replaced": False,
            "compile_command": [
                "<compiler>" if item == compile_command[0]
                else "<upstream>" + item[len(str(source_dir)):] if item.startswith(str(source_dir))
                else "<fftw-prefix>" + item[len(str(prefix)):] if item.startswith(str(prefix))
                else "<work-dir>" + item[len(str(work_dir)):] if item.startswith(str(work_dir))
                else "<repo>/codes/examples/reproduce_smpphat_harness.c"
                if item == str(Path(__file__).with_name("reproduce_smpphat_harness.c")) else item
                for item in compile_command
            ],
        },
        "parameters": {
            "channels": CHANNELS,
            "pairs": CHANNELS * (CHANNELS - 1) // 2,
            "frame_size": FRAME_SIZE,
            "interpolation_rate": INTERPOLATION_RATE,
            "sample_rate_hz": SAMPLE_RATE,
            "sound_speed_m_s": SOUND_SPEED,
            "candidate_azimuth_deg": (np.arange(POINTS) * 15).tolist(),
            "candidate_vectors": directions.tolist(),
            "true_direction_index": TRUE_INDEX,
            "true_azimuth_deg": 45,
            "phat_definition": "Xi*conj(Xj)=exp(-j*2*pi*k*tdoa/frame_size)",
            "phat_bins": "DC has magnitude 1; bins 1..31 have magnitude 1; original bin 32 is zero",
            "tdoa_definition_samples": "-(sample_rate/sound_speed)*dot(direction, ri-rj)",
            "azimuth_convention": "0 degrees is +y; positive angles rotate toward +x",
            "score_scaling": "FFTW unnormalized c2r; absolute scores are not probabilities",
        },
        "cases": cases,
        "limitations": [
            "deterministic far-field unit-magnitude PHAT spectra, not microphone recordings",
            "horizontal candidate grid only; no elevation, noise, reverberation, or near-field propagation",
            "one fixed direction; this verifies algebraic equivalence, grouping, and peak, not localization accuracy",
            "no timing claim: compilation and one-frame execution were not used as a benchmark",
            "the 0.5 mm perturbation intentionally breaks the upstream 1e-5 equal-baseline grouping test",
            "the locked source converts negative rounded TDOAs to unsigned before addition; behavior is compiler dependent",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--fftw-prefix", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--download-fftw", action="store_true",
                        help="explicitly allow the pinned official FFTW archive download")
    parser.add_argument("--allow-known-portability-failure", action="store_true",
                        help="write the diagnosed failure report but return success")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_experiment(source_dir=arguments.source_dir, fftw_prefix=arguments.fftw_prefix,
                            work_dir=arguments.work_dir, download_fftw=arguments.download_fftw)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if report["numerical_status"] == "failed_portability" \
            and not arguments.allow_known_portability_failure:
        parser.exit(2, "locked smpphat failed the all-grid SRP/SMP equivalence check; "
                       "report saved, pass --allow-known-portability-failure only to archive this diagnosis\n")


if __name__ == "__main__":
    main()
