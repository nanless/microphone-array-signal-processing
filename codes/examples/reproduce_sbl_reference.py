"""Run the locked, unmodified GPL SBL in its separate local checkout.

No download, installation, or upstream code copy is performed. Importing this
module does not execute the experiment. The data are synthetic narrowband
complex snapshots, not recordings or a reproduction of a paper's benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVISION = "d4bba35e9b60907d3024473ba5a41046450baae0"
MODULE_SHA256 = "6c4a80a53766a04f68850bab5d285b72b04ae5cf231b3cea444ead55d8fe43c6"
SEED = 20260922
GRID = np.arange(-80.0, 81.0)


def load_locked_source(source_dir: Path | None = None):
    """Verify the checkout and module before executing its unchanged source."""
    source_dir = (source_dir or ROOT / "upstream/_downloads/sbl").resolve()
    source = source_dir / "SBL_MF_Python/sbl.py"
    if not source.is_file() or not (source_dir / ".git").exists():
        raise FileNotFoundError("Fetch the locked sbl source separately before running")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_")}

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source_dir), *args], env=env, text=True,
            stderr=subprocess.PIPE).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source_dir:
        raise ValueError("SBL directory is not its own checkout")
    if git("rev-parse", "HEAD") != REVISION:
        raise ValueError("SBL revision differs from the experiment lock")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("SBL tracked worktree is modified")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODULE_SHA256:
        raise ValueError("SBL core SHA-256 differs from the experiment lock")
    module = types.ModuleType("locked_sbl_reference")
    module.__file__ = str(source)
    # Execute the verified bytes without creating a pycache in the checkout.
    exec(compile(payload, str(source), "exec"), module.__dict__)
    return module


def steering(angles_deg: np.ndarray) -> np.ndarray:
    """Four half-wavelength microphones; zero broadside, positive toward +x.

    Relative delays are -m*d*sin(theta)/c, so exp(-j*2*pi*f*tau)
    is exp(+j*pi*m*sin(theta)). Columns have unit channel amplitude.
    """
    angles = np.asarray(angles_deg)
    if angles.dtype.kind not in "fi" or angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be a finite real one-dimensional array")
    return np.exp(1j * np.pi * np.arange(4)[:, None] * np.sin(np.deg2rad(angles)))


def independent_covariance(y: np.ndarray) -> np.ndarray:
    """Scalar sums, independently of the upstream einsum and matrix product."""
    channels, snapshots = y.shape
    covariance = np.empty((channels, channels), dtype=complex)
    for row in range(channels):
        for column in range(channels):
            products = [complex(y[row, t]) * complex(y[column, t]).conjugate()
                        for t in range(snapshots)]
            covariance[row, column] = complex(
                math.fsum(value.real for value in products),
                math.fsum(value.imag for value in products)) / snapshots
    return covariance


def independent_peaks(power: np.ndarray, count: int) -> np.ndarray:
    """Strict positive local maxima; never fill missing peaks with index zero."""
    values = np.asarray(power)
    if (values.ndim != 1 or values.dtype.kind not in "fi"
            or not np.all(np.isfinite(values)) or np.any(values < 0)):
        raise ValueError("spectrum must be finite, real, nonnegative and one-dimensional")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("peak count must be a positive integer")
    candidates = [i for i, value in enumerate(values)
                  if value > 0 and (i == 0 or value > values[i - 1])
                  and (i == len(values) - 1 or value > values[i + 1])]
    return np.array(sorted(candidates, key=lambda i: (-values[i], i))[:count], dtype=int)


def validate_inputs(a: np.ndarray, y: np.ndarray, source_count: int) -> None:
    """Protect the known-source-count NumPy interface, not a general SBL API."""
    if (a.ndim != 3 or y.ndim != 3 or a.shape[0] != y.shape[0]
            or a.shape[2] != y.shape[2] or min(a.shape) < 1 or min(y.shape) < 1):
        raise ValueError("expected A=M x G x F and Y=M x L x F with matching M,F")
    if (isinstance(source_count, bool) or not isinstance(source_count, (int, np.integer))
            or not 0 < source_count < a.shape[0]):
        raise ValueError("source count must be an integer strictly between zero and M")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(y)):
        raise ValueError("dictionary and observations must be finite")
    if np.any(np.max(np.abs(a), axis=0) == 0) or np.max(np.abs(y)) == 0:
        raise ValueError("zero dictionary columns or zero observations are unsupported")


def synthetic_cases() -> list[dict]:
    """Share latent draws across cases; only the named condition is changed."""
    rng = np.random.default_rng(SEED)
    sources = (rng.standard_normal((2, 200)) + 1j * rng.standard_normal((2, 200))) / np.sqrt(2)
    white = (rng.standard_normal((4, 200)) + 1j * rng.standard_normal((4, 200))) / np.sqrt(2)
    definitions = [
        ("on_grid", [-30.0, 30.0], 0.02, 2, False),
        ("off_grid", [-29.5, 30.5], 0.02, 2, False),
        ("low_snr", [-30.0, 30.0], 20.0, 2, False),
        ("colored_noise", [-30.0, 30.0], 0.02, 2, True),
        ("wrong_source_count", [-30.0, 30.0], 0.02, 1, False),
    ]
    cases = []
    for name, angles, variance, count, colored in definitions:
        correlation = (0.9 ** np.abs(np.arange(4)[:, None] - np.arange(4))
                       if colored else np.eye(4))
        noise = np.sqrt(variance) * np.linalg.cholesky(correlation) @ white
        signal = steering(np.array(angles)) @ sources
        expected = steering(np.array(angles)) @ steering(np.array(angles)).conj().T + variance * correlation
        cases.append(dict(name=name, angles=angles, variance=variance, count=count,
                          correlation=correlation, y=(signal + noise)[:, :, None],
                          population=expected))
    return cases


def _complex_parts(values: np.ndarray) -> dict:
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def run_experiment(source_dir: Path | None = None, *, max_iterations: int = 1000) -> dict:
    """Run five paired single-realization cases; never infer aggregate accuracy."""
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    upstream = load_locked_source(source_dir)
    options_dict = dict(convergence_error=1e-5, gamma_range=1e-4,
                        convergence_maxiter=max_iterations, convergence_min_iteration=10,
                        status_report=100, fixedpoint=1, flag=0)
    a = steering(GRID)[:, :, None]
    results = []
    for case in synthetic_cases():
        y, count = case["y"], case["count"]
        validate_inputs(a, y, count)
        options = upstream.Options(Nsource=count, **options_dict)
        with contextlib.redirect_stdout(io.StringIO()):
            gamma, report = upstream.SBL(a.copy(), y.copy(), options)
        if not np.all(np.isfinite(gamma)) or np.any(gamma < 0):
            raise ArithmeticError("upstream returned invalid candidate powers")
        iteration = int(report.results_iteration)
        errors = report.results_error[:iteration + 1]
        if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(report.results_noisepower)):
            raise ArithmeticError("upstream returned nonfinite diagnostics")
        converged = iteration > options.convergence_min_iteration and errors[-1] < options.convergence_error
        peak_indices = independent_peaks(gamma, count)
        peak_angles = np.sort(GRID[peak_indices])
        covariance = independent_covariance(y[:, :, 0])
        _, vectors = np.linalg.eigh(covariance)
        noise_subspace = vectors[:, :4 - count]
        music = 1.0 / np.maximum(np.sum(np.abs(noise_subspace.conj().T @ a[:, :, 0]) ** 2, axis=0), 1e-12)
        music_angles = np.sort(GRID[independent_peaks(music, count)])
        # Two sorted equal-size sets minimize sum squared error on this sector.
        matched_errors = (peak_angles - case["angles"]).tolist() if len(peak_angles) == 2 else None
        near_truth = np.min(np.abs(GRID[:, None] - case["angles"]), axis=1) <= 1.0
        results.append({
            "case": case["name"], "true_angles_deg": case["angles"],
            "assumed_source_count": count, "noise_variance_per_microphone": case["variance"],
            "noise_correlation": case["correlation"].tolist(),
            "expected_total_source_to_noise_db": 10 * math.log10(2 / case["variance"]),
            "sbl_peak_angles_deg": peak_angles.tolist(), "matched_signed_errors_deg": matched_errors,
            "missing_peak_count_relative_to_truth": max(0, 2 - len(peak_angles)),
            "music_peak_angles_deg": music_angles.tolist(),
            "gamma": gamma.tolist(),
            "gamma_fraction_beyond_one_degree_of_truth": float(np.sum(gamma[~near_truth]) / np.sum(gamma)),
            "upstream_noise_variance": np.asarray(report.results_noisepower).tolist(),
            "iterations_executed": iteration + 1, "upstream_zero_based_iteration": iteration,
            "stop_reason": "convergence_threshold" if converged else "iteration_limit",
            "final_relative_update": float(errors[-1]), "relative_update_trace": errors.tolist(),
            "sample_covariance": _complex_parts(covariance),
            "population_covariance": _complex_parts(case["population"]),
            "population_eigenvalues": np.linalg.eigvalsh(case["population"]).tolist(),
            "scalar_vs_matrix_covariance_max_abs": float(np.max(np.abs(covariance - y[:, :, 0] @ y[:, :, 0].conj().T / 200))),
            "sample_vs_population_relative_frobenius": float(np.linalg.norm(covariance - case["population"]) / np.linalg.norm(case["population"])),
        })
    return {
        "schema_version": 1, "experiment": "locked_sbl_four_microphone_single_realization",
        "provenance": {"upstream_url": "https://github.com/gerstoft/SBL",
                       "revision": REVISION, "module_sha256": MODULE_SHA256,
                       "license": "GPL-3.0; separately obtained upstream, not copied into this harness",
                       "source_verified_on": "2026-09-22",
                       "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "source_check_scope": "HEAD and all tracked worktree files; untracked files not audited",
                       "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "python": platform.python_version(), "numpy": np.__version__,
                       "platform": platform.platform()},
        "parameters": {"seed": SEED, "snapshots": 200, "channels": 4, "frequency_hz": 1000,
                       "sound_speed_m_s": 343, "spacing_m": 0.1715,
                       "steering": "exp(+j*pi*m*sin(theta)); reference=m0; theta=0 broadside",
                       "source_power_each": 1, "dictionary_column_squared_norm": 4,
                       "grid_deg": GRID.tolist(), "options": options_dict,
                       "input_axes": {"A": "M,G,F", "Y": "M,L,F"},
                       "sampling": "paired CN(0,1) source/noise draws; no sample normalization; F=1",
                       "sample_rate_and_stft": "not applicable: synthetic narrowband snapshots"},
        "limitations": ["one paired realization per condition, not success-rate estimation",
                        "no recording, reverberation, motion, latency or device benchmark",
                        "MUSIC uses the same assumed source count; not ground truth",
                        "convergence measures gamma updates, not localization correctness",
                        "missing source count is not converted into a deceptively small two-source error"],
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON report explicitly")
    args = parser.parse_args()
    result = run_experiment(args.source_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
