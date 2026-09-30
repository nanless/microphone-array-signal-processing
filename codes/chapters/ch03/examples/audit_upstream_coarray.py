"""Audit selected original doatools methods without importing the full package.

Read-only by default; --report explicitly writes a strict JSON result. The
upstream checkout is never edited or fetched. AST extraction preserves selected
definitions, including their bodies and decorators. A local NumPy facade supplies
the removed float_/complex_ names; it is scaffolding, not an upstream patch.
Original MIT source and its license stay in the separate pinned checkout.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
CACHE = ROOT / "codes/chapters/ch00/upstream/_downloads"
REVISION = "9469db201e0418aef6b97583ef54b6fec2769502"
URL = "https://github.com/morriswmz/doatools.py.git"
DEFINITIONS = {
    "doatools/model/array_elements.py": ("ArrayElement", "IsotropicScalarSensor"),
    "doatools/model/perturbations.py": (
        "ArrayPerturbation", "LocationErrors", "GainErrors", "PhaseErrors", "MutualCoupling"),
    "doatools/model/arrays.py": (
        "ArrayDesign", "GridBasedArrayDesign", "UniformLinearArray", "NestedArray", "CoPrimeArray"),
    "doatools/model/coarray.py": ("compute_location_differences", "WeightFunction1D"),
    "doatools/utils/math.py": ("vec",),
    "doatools/estimation/core.py": ("ensure_covariance_size",),
    "doatools/estimation/coarray.py": ("CoarrayACMBuilder1D",),
}
ABS_TOLERANCE = 1e-14
SCALE_ABS_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(checkout: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=checkout, check=True,
                          capture_output=True, timeout=60).stdout


def verify_sources(cache: Path = CACHE, lock_path: Path = LOCK) -> dict:
    """Bind the selected lock entry, clean HEAD and actual Git blob bytes."""
    checkout = cache / "doatools"
    if not (checkout / ".git").exists():
        raise FileNotFoundError(f"Existing fixed checkout required: {checkout}")
    lock_bytes = lock_path.read_bytes()
    entries = [p for p in json.loads(lock_bytes)["projects"] if p["id"] == "doatools"]
    if len(entries) != 1:
        raise ValueError("Expected exactly one doatools lock entry")
    entry = entries[0]
    if (entry["revision"], entry["license"], entry["url"].rstrip("/")) != (REVISION, "MIT", URL):
        raise ValueError("Review the changed doatools lock before running")
    head = _git(checkout, "rev-parse", "HEAD").decode().strip()
    if head != REVISION:
        raise ValueError(f"Unexpected doatools HEAD: {head}")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("The doatools worktree is not clean")
    files = []
    for relative in (*DEFINITIONS, "LICENSE.md"):
        actual = (checkout / relative).read_bytes()
        if actual != _git(checkout, "show", f"{head}:{relative}"):
            raise ValueError(f"Source bytes differ from fixed Git blob: {relative}")
        files.append({"path": relative, "sha256": hashlib.sha256(actual).hexdigest(),
                      "git_blob_oid": _git(checkout, "rev-parse", f"{head}:{relative}").decode().strip(),
                      "definitions": list(DEFINITIONS.get(relative, ()))})
    canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "lock_entry_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "url": entry["url"], "revision": head, "license": "MIT",
            "implementation_identity": "maintainer's independent implementation, not Pal author code",
            "worktree_clean": True, "files": files}


class CompatibleNumpy:
    """Provide old names only inside the extracted definitions' namespace."""
    def __getattr__(self, name):
        if name == "float_":
            return np.float64
        if name == "complex_":
            return np.complex128
        return getattr(np, name)


def extract(path: Path, names: tuple[str, ...], namespace: dict) -> list[dict]:
    """Compile named top-level definitions; do not run imports or module code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes, evidence = [], []
    for name in names:
        matches = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one original definition: {path}:{name}")
        node = matches[0]
        nodes.append(node)
        evidence.append({"path": str(path.relative_to(ROOT)), "name": name,
                         "start_line": node.lineno, "end_line": node.end_lineno,
                         "definition_ast_sha256": hashlib.sha256(ast.dump(
                             node, include_attributes=False).encode()).hexdigest(),
                         "body_or_decorators_changed": False})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return evidence


def encode_complex(values) -> dict:
    a = np.asarray(values)
    return {"real": a.real.tolist(), "imag": a.imag.tolist()}


def assert_close(actual, expected, *, atol=ABS_TOLERANCE) -> float:
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(e)):
        raise AssertionError("Nonfinite or differently shaped numerical comparison")
    error = float(np.max(np.abs(a-e)))
    if error > atol:
        raise AssertionError(f"Independent expectation mismatch: {error} > {atol}")
    return error


def validate_physical_covariance(values, sensors: int) -> np.ndarray:
    """Own guard for the two small physical fixtures; no repairs or loading."""
    a = np.asarray(values)
    if a.shape != (sensors, sensors) or not np.all(np.isfinite(a)):
        raise ValueError("Expected a finite M by M physical covariance")
    if not np.allclose(a, a.conj().T, atol=ABS_TOLERANCE, rtol=0):
        raise ValueError("Physical covariance must be Hermitian")
    if float(np.linalg.eigvalsh(a)[0]) < -ABS_TOLERANCE:
        raise ValueError("Physical covariance must be positive semidefinite")
    return a


def trial(name, function) -> dict:
    try:
        value = function()
    except Exception as error:
        return {"case": name, "outcome": "exception", "exception_type": type(error).__name__,
                "message": str(error)}
    return {"case": name, "outcome": "returned", "value": value}


def load_methods(checkout: Path) -> tuple[dict, list]:
    namespace = {"np": CompatibleNumpy(), "ABC": ABC, "abstractmethod": abstractmethod,
                 "gcd": math.gcd, "warnings": warnings, "copy": copy}
    extractions = []
    for relative, names in DEFINITIONS.items():
        extractions.extend(extract(checkout / relative, names, namespace))
        if relative == "doatools/model/array_elements.py":
            namespace["ISOTROPIC_SCALAR_SENSOR"] = namespace["IsotropicScalarSensor"]()
    return namespace, extractions


def audit_layouts(ns: dict) -> list:
    cases = (("nested", ns["NestedArray"](3, 3, .02), [0, 1, 2, 3, 7, 11], 23, 12),
             ("coprime_m", ns["CoPrimeArray"](3, 4, .02, mode="m"), [0, 3, 6, 9, 4, 8], 13, 7),
             ("coprime_default_2m", ns["CoPrimeArray"](3, 4, .02),
              [0, 3, 6, 9, 4, 8, 12, 16, 20], 29, 15))
    results = []
    for name, array, indices, central, virtual in cases:
        assert_close(array.element_indices.ravel(), indices)
        assert_close(array.element_locations.ravel(), [.02*p for p in indices])
        w = ns["WeightFunction1D"](array)
        # Enumerate ordered physical pairs independently of WeightFunction1D.
        counts = {}
        for i in indices:
            for j in indices:
                counts[i-j] = counts.get(i-j, 0) + 1
        assert_close(w.differences(), sorted(counts))
        assert_close(w.weights(), [counts[k] for k in sorted(counts)])
        if (w.get_central_ula_size(), w.get_central_ula_size(True)) != (central, virtual):
            raise AssertionError("Unexpected central continuous lag sizes")
        results.append({"case": name, "d0_m": .02, "physical_count": len(indices),
                        "indices_in_original_channel_order": array.element_indices.ravel().tolist(),
                        "positions_m": array.element_locations.ravel().tolist(),
                        "signed_lags": w.differences().tolist(), "multiplicity": w.weights().tolist(),
                        "ordered_pair_count": sum(counts.values()),
                        "central_signed_size": central, "virtual_size": virtual})
    return results


def audit_covariances(ns: dict) -> tuple[list, object, np.ndarray]:
    array = ns["GridBasedArrayDesign"](np.array([0, 1, 3]).reshape(-1, 1), d0=.02)
    builder = ns["CoarrayACMBuilder1D"](array)
    a = np.array([1, 1j, -1j])
    ideal = np.ones((3, 3)) + np.outer(a, a.conj()) + .1*np.eye(3)
    snapshot = np.outer([1, 0, 1], [1, 0, 1]).astype(complex)
    results = []
    for name, R, golden_da, golden_ss in (
        ("ideal_E03_07", ideal, [.1, .1, 4.1, 4.1], [.0025, .0025, 4.2025, 4.2025]),
        ("single_snapshot_E03_07", snapshot, [-1/3, 2/3, 2/3, 5/3], [1/36, 1/9, 1/9, 25/36]),
    ):
        validate_physical_covariance(R, 3)
        # Own pair selection and lag averaging, without calling original helpers.
        positions, lag_values = [0, 1, 3], {}
        for lag in range(-3, 4):
            selected = [R[i, j] for i, p in enumerate(positions)
                        for j, q in enumerate(positions) if p-q == lag]
            lag_values[lag] = sum(selected)/len(selected)
        expected_da = np.array([[lag_values[i-j] for j in range(4)] for i in range(4)])
        expected_ss = expected_da @ expected_da.conj().T / 4
        da, ss = builder(R, method="da"), builder(R)  # Default is original 'ss'.
        da_error, ss_error = assert_close(da, expected_da), assert_close(ss, expected_ss)
        assert_close(np.linalg.eigvalsh(da), golden_da)
        assert_close(np.linalg.eigvalsh(ss), golden_ss)
        da_scale = assert_close(builder(4*R, method="da"), 4*expected_da, atol=SCALE_ABS_TOLERANCE)
        ss_scale = assert_close(builder(4*R), 16*expected_ss, atol=SCALE_ABS_TOLERANCE)
        results.append({"case": name, "physical_indices": positions, "physical_count": 3,
                        "virtual_count": 4, "input": encode_complex(R),
                        "physical_eigenvalues": np.linalg.eigvalsh(R).tolist(),
                        "lags_in_order": list(range(-3, 4)),
                        "independent_lag_values": encode_complex([lag_values[k] for k in range(-3, 4)]),
                        "da": encode_complex(da), "ss_default": encode_complex(ss),
                        "da_eigenvalues": np.linalg.eigvalsh(da).tolist(),
                        "ss_eigenvalues": np.linalg.eigvalsh(ss).tolist(),
                        "hand_expected_da_eigenvalues": golden_da, "hand_expected_ss_eigenvalues": golden_ss,
                        "da_max_expected_error": da_error, "ss_max_expected_error": ss_error,
                        "amplitude_scale": 2, "physical_covariance_scale": 4,
                        "da_scale": 4, "ss_scale": 16,
                        "scaled_da_max_error": da_scale, "scaled_ss_max_error": ss_scale,
                        "input_psd": True, "da_psd": name == "ideal_E03_07", "ss_psd": True})
    return results, builder, ideal


def audit_perturbations(ns: dict) -> list:
    A = np.array([[1, 1j], [1, -1]], dtype=complex)
    cases = (("gain", ns["GainErrors"](np.array([.1, -.2])), [.1, -.2], np.diag([1.1, .8]) @ A),
             ("phase", ns["PhaseErrors"](np.array([.2, -.3])), [.2, -.3],
              np.diag(np.exp(1j*np.array([.2, -.3]))) @ A),
             ("coupling", ns["MutualCoupling"](np.array([[1., .1], [.2, 1.]])),
              [[1., .1], [.2, 1.]], np.array([[1., .1], [.2, 1.]]) @ A))
    results = []
    for name, obj, parameters, expected in cases:
        output, derivatives = obj.perturb_steering_matrix(A, [2*A])
        error = assert_close(output, expected)
        derivative_error = assert_close(derivatives[0], 2*expected)
        results.append({"case": name, "parameters": parameters,
                        "parameter_units": "radians" if name == "phase" else "dimensionless",
                        "input": encode_complex(A), "output": encode_complex(output),
                        "independent_expected": encode_complex(expected), "max_expected_error": error,
                        "derivative_input": encode_complex(2*A),
                        "derivative_output": encode_complex(derivatives[0]),
                        "derivative_max_expected_error": derivative_error,
                        "unknown_parameter_estimation_executed": False})
    locations = np.array([[0.], [.04]])
    errors = np.array([[.001, .002], [-.001, .003]])
    output = ns["LocationErrors"](errors).perturb_sensor_locations(locations)
    error = assert_close(output, [[.001, .002], [.039, .003]])
    results.append({"case": "locations_1d_to_2d", "input_m": locations.tolist(),
                    "errors_m": errors.tolist(), "output_m": output.tolist(),
                    "max_expected_error_m": error, "unknown_parameter_estimation_executed": False})
    return results


def audit_invalid_inputs(ns: dict, builder, ideal) -> list:
    # Deliberately bypass the own guard: observe the unchanged original checks.
    results = [
        trial("GainErrors_list", lambda: ns["GainErrors"]([.1, .2]).params.tolist()),
        trial("PhaseErrors_list", lambda: ns["PhaseErrors"]([.1, .2]).params.tolist()),
        trial("MutualCoupling_2x3_constructor", lambda: list(ns["MutualCoupling"](np.ones((2, 3))).params.shape)),
        trial("MutualCoupling_1d_constructor", lambda: list(ns["MutualCoupling"](np.ones(2)).params.shape)),
        trial("covariance_3x4_da", lambda: encode_complex(builder(np.column_stack([ideal, np.ones(3)*99]), "da"))),
        trial("covariance_3x2_da", lambda: encode_complex(builder(ideal[:, :2], "da"))),
    ]
    golden = ["AttributeError", "AttributeError", "returned", "IndexError", "returned", "IndexError"]
    inputs = ([.1, .2], [.1, .2], [[1, 1, 1], [1, 1, 1]], [1, 1],
              encode_complex(np.column_stack([ideal, np.ones(3)*99])), encode_complex(ideal[:, :2]))
    for case, expected, actual_input in zip(results, golden, inputs):
        observed = case["exception_type"] if case["outcome"] == "exception" else "returned"
        if observed != expected:
            raise AssertionError(f"Fixed-source negative path changed: {case}")
        case["input"] = actual_input
        case["stage"] = "coarray_da_transform" if case["case"].startswith("covariance") else "constructor"
        case["classification"] = "source_input_validation_defect"
        case["valid_model_input"] = False
    extra = np.asarray(results[4]["value"]["real"]) + 1j*np.asarray(results[4]["value"]["imag"])
    assert_close(extra, builder(ideal, "da"))
    results[4]["extra_fourth_column"] = [99, 99, 99]
    results[4]["extra_column_silently_ignored"] = True
    return results


def run_audit(cache: Path = CACHE) -> dict:
    tool_digest = sha256(Path(__file__))
    sources_before = verify_sources(cache)
    ns, extractions = load_methods(cache / "doatools")
    layouts = audit_layouts(ns)
    covariances, builder, ideal = audit_covariances(ns)
    perturbations = audit_perturbations(ns)
    negatives = audit_invalid_inputs(ns, builder, ideal)
    if verify_sources(cache) != sources_before or sha256(Path(__file__)) != tool_digest:
        raise ValueError("Audit source, lock or upstream changed during execution")
    executed_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 1, "executed_at_utc": executed_at.isoformat(),
        "executed_at_local": executed_at.astimezone().isoformat(),
        "audit_source": str(Path(__file__).resolve().relative_to(ROOT)),
        "audit_source_sha256": tool_digest, "sources": sources_before,
        "upstream_clean_before_and_after": True,
        "scope": "unchanged original-method AST extraction with a local NumPy compatibility facade",
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "executable": sys.executable, "platform": platform.platform()},
        "tolerances": {"absolute": ABS_TOLERANCE, "relative": 0., "amplitude_scale_absolute": SCALE_ABS_TOLERANCE,
                       "physical_covariance_guard_absolute": ABS_TOLERANCE,
                       "guard_scope": "only the fixed small physical fixtures; not a general scale-invariant validator"},
        "scaffold": {"upstream_files_modified": False, "full_package_import_executed": False,
                     "definitions_changed": False, "extractions": extractions,
                     "local_numpy_aliases": {"float_": "float64", "complex_": "complex128"},
                     "global_numpy_modified": False,
                     "own_guard_used_for_positive_physical_inputs_only": True,
                     "independent_expectations": ["ordered-pair enumeration", "lag averaging and direct Toeplitz fill",
                                                  "TT^H/4", "hand-derived eigenvalues", "diagonal or matrix multiplication"]},
        "results": {"layouts": layouts, "covariances": covariances,
                    "forward_given_perturbations": perturbations, "negative_inputs": negatives},
        "all_expected_behaviors_observed": True,
        "not_executed": ["full doatools import and dependency validation", "DOA or source-count estimator",
                         "unknown-error calibration", "StructureFromSound MATLAB system",
                         "HARK measurement or Infineon firmware", "hardware or equipment acceptance"]}
    json.dumps(report, allow_nan=False)
    return report


def write_report(path: Path, report: dict) -> None:
    """Publish only after the audit succeeds; never permit an upstream write."""
    if path.resolve().is_relative_to(CACHE.resolve()):
        raise ValueError("Report output must be outside upstream checkouts")
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".upstream-coarray-", suffix=".tmp", delete=False) as out:
            temporary = Path(out.name)
            out.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Explicitly generate a current strict JSON report")
    args = parser.parse_args()
    report = run_audit()
    report["command"] = [sys.executable, "-m", "codes.chapters.ch03.examples.audit_upstream_coarray", *sys.argv[1:]]
    if args.report is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        write_report(args.report, report)
        print(f"Original-method audit completed; report: {args.report}")


if __name__ == "__main__":
    main()
