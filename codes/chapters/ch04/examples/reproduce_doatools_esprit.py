"""Offline, fixed-source ESPRIT diagnostic; no install or upstream modification.

The optional execution environment needs SciPy. Ordinary tests only check the
saved report and NumPy reference, and never import or download doatools. The
reference is an independent expression of LS/TLS, not an upstream patch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REVISION = "9469db201e0418aef6b97583ef54b6fec2769502"
SOURCE_HASHES = {
    "doatools/estimation/esprit.py": "06f24db5d0f04928dca8c7821fe53c821534c9473c6afef399ffd0746d474711",
    "doatools/estimation/core.py": "63a0e2d9a3561c0270aaa2d6f207777257ccff607bd11d72dab5c3f5419649d8",
    "doatools/model/sources.py": "0fe676c55c4b6f2e73ad99323398bd71bc8d80bbf8c56b24f9313af85e980e25",
    "doatools/utils/conversion.py": "41e3089e456c617046302a7cbc1c5a417872db0513dffb29ac52e31ae6ba80e3",
    "LICENSE.md": "39ddd5dcfeb6eb77f18c99e8955341dc7b4d4823a3a121e0124e20de336b507a",
}
CONFIG = {
    "channels": 8, "source_angles_deg": [-5.0, 10.0], "source_count": 2,
    "wavelength_m": 1.0, "spacing_m": 0.5, "source_powers": [1.0, 1.0],
    "white_noise_power_per_channel": 0.1, "displacement": 1,
    "angle_convention": "broadside zero, positive toward +x; exp(+j*pi*m*sin(theta))",
    "input": "exact population covariance A A^H + 0.1 I; no sampled snapshots",
    "randomness": "none", "direction_error_tolerance_deg": 1e-8,
}


def configuration_sha256() -> str:
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def population_input() -> tuple[np.ndarray, np.ndarray]:
    a = np.exp(1j * np.pi * np.arange(CONFIG["channels"])[:, None]
               * np.sin(np.deg2rad(CONFIG["source_angles_deg"])))
    return a, a @ a.conj().T + CONFIG["white_noise_power_per_channel"] * np.eye(a.shape[0])


def safe_rotation(subspace: np.ndarray, formulation: str, weighted: bool = True) -> dict:
    """Independent LS (lstsq) or TLS (SVD), with separate weighted arrays.

    This fixed diagnostic accepts any full-rank M x K signal basis, including
    analytic steering columns; it is not a general production DOA interface.
    """
    e = np.asarray(subspace, dtype=complex)
    if (e.ndim != 2 or not 0 < e.shape[1] < e.shape[0]
            or not np.all(np.isfinite(e))):
        raise ValueError("expected a finite M x K basis with 0 < K < M")
    if formulation not in ("ls", "tls"):
        raise ValueError("formulation must be ls or tls")
    first, second = e[:-1].copy(), e[1:].copy()
    n = first.shape[0]
    weights = np.sqrt(np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)))
    if weighted:
        first *= weights[:, None]
        second *= weights[:, None]
    k = e.shape[1]
    if np.linalg.matrix_rank(first) < k:
        raise ValueError("first subarray has insufficient rank")
    if formulation == "ls":
        rotation = np.linalg.lstsq(first, second, rcond=None)[0]
    else:
        # The K smallest right singular vectors of [E1 E2] partition as
        # [V12; V22]; E1*(-V12/V22) approximates E2.
        _, _, vh = np.linalg.svd(np.column_stack((first, second)), full_matrices=True)
        null = vh.conj().T[:, -k:]
        top, bottom = null[:k], null[k:]
        rotation = -np.linalg.solve(bottom.T, top.T).T
    modes = np.linalg.eigvals(rotation)
    angles = np.sort(np.rad2deg(np.arcsin(np.angle(modes) / np.pi)))
    return {
        "angles_deg": angles.tolist(),
        "rotation_real": rotation.real.tolist(), "rotation_imag": rotation.imag.tolist(),
        "rotation_moduli": np.sort(np.abs(modes)).tolist(),
        "subarray_residual_fro": float(np.linalg.norm(second - first @ rotation)),
        "subarrays_share_memory": bool(np.shares_memory(first, second)),
    }


def verify_source(source_dir: Path) -> None:
    if not (source_dir / ".git").exists():
        raise FileNotFoundError("Fetch locked doatools separately; this script never downloads it")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source_dir), *args],
                                       env=env, text=True, stderr=subprocess.PIPE).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source_dir:
        raise ValueError("source directory must be its own checkout")
    if git("rev-parse", "HEAD") != REVISION:
        raise ValueError("doatools revision does not match the lock")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked upstream files are modified")
    # Avoid executing untracked Python shadows in this otherwise clean checkout.
    if git("ls-files", "--others", "--exclude-standard", "--", "*.py"):
        raise ValueError("untracked Python source in the upstream checkout")
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("source hash mismatch: " + name)


def run_experiment(source_dir: Path | None = None) -> dict:
    source_dir = (source_dir or ROOT / "chapters/ch00/upstream/_downloads/doatools").resolve()
    verify_source(source_dir)
    if any(name == "doatools" or name.startswith("doatools.") for name in sys.modules):
        raise RuntimeError("run in a fresh process, without previously imported doatools")
    original_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_dir))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module("doatools.estimation.esprit")
        if Path(module.__file__).resolve() != source_dir / "doatools/estimation/esprit.py":
            raise RuntimeError("unexpected upstream import location")
        import scipy
        a, covariance = population_input()
        _, eigenvectors = np.linalg.eigh(covariance)
        subspace = eigenvectors[:, -CONFIG["source_count"]:]
        truth = np.array(CONFIG["source_angles_deg"])
        results = []
        for formulation in ("ls", "tls"):
            for row_weights in ("default", "none"):
                resolved, output = module.Esprit1D(CONFIG["wavelength_m"]).estimate(
                    covariance.copy(), CONFIG["source_count"], d0=CONFIG["spacing_m"],
                    displacement=CONFIG["displacement"],
                    formulation=formulation, row_weights=row_weights, unit="deg")
                angles = np.sort(output.locations)
                error = float(np.max(np.abs(angles - truth)))
                results.append({"formulation": formulation, "row_weights": row_weights,
                                "resolved": bool(resolved), "angles_deg": angles.tolist(),
                                "max_abs_direction_error_deg": error,
                                "direction_check": "passed" if error <= CONFIG["direction_error_tolerance_deg"] else "failed"})
        references = {form: safe_rotation(subspace, form) for form in ("ls", "tls")}
        analytic = {form: safe_rotation(a, form) for form in ("ls", "tls")}
        expected_failure = all(
            row["direction_check"] == ("failed" if row["row_weights"] == "default" else "passed")
            for row in results)
        reference_passed = all(
            np.max(np.abs(np.array(result["angles_deg"]) - truth)) <= CONFIG["direction_error_tolerance_deg"]
            for group in (references, analytic) for result in group.values())
        return {
            "schema_version": 1,
            "status": "failed_default_row_weighting" if expected_failure and reference_passed else "unexpected_diagnostic_result",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {"revision": REVISION, "source_sha256": SOURCE_HASHES,
                           "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "configuration_sha256": configuration_sha256(),
                           "source_dir": str(source_dir), "upstream_modified": False,
                           "upstream_signature": str(inspect.signature(module.Esprit1D.estimate))},
            "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                            "platform": platform.platform(), "machine": platform.machine(),
                            "executable": sys.executable, "import_warnings": [str(w.message) for w in caught]},
            "configuration": CONFIG, "covariance_real": covariance.real.tolist(),
            "covariance_imag": covariance.imag.tolist(), "upstream_results": results,
            "safe_copy_eigenspace_references": references,
            "analytic_steering_basis_references": analytic,
            "limits": ["One exact covariance diagnostic, not a success-rate or speed benchmark",
                       "No microphone audio, room, finite snapshots or device validation",
                       "Only the locked ESPRIT interface was executed; optional sparse solvers were not",
                       "Safe-copy references are independent NumPy calculations, not a modified upstream release"],
        }
    finally:
        sys.path.remove(str(source_dir))
        sys.dont_write_bytecode = original_bytecode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_experiment(args.source_dir), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
