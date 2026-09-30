"""Execute selected unchanged methods from three existing pinned checkouts.

AST extraction avoids importing optional SciPy/Traits/Numba packages. Only
Acoular's JIT decorator is removed in memory; its numerical body is unchanged.
The small grid/array adapters below are tutorial scaffolding, not upstream
implementations. No files in the upstream checkouts are written or downloaded.
This is method-level evidence, not a full-package or equipment measurement.
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
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
CACHE = ROOT / "codes/chapters/ch00/upstream/_downloads"
PROJECTS = {
    "pyroomacoustics": {
        "revision": "0dd39f2614b7fc44b2cc63dbe7d60f4641068890",
        "license": "MIT", "files": (
            "pyroomacoustics/parameters.py", "pyroomacoustics/doa/doa.py",
            "pyroomacoustics/acoustics.py", "pyroomacoustics/experimental/rt60.py", "LICENSE")},
    "acoular": {
        "revision": "13d3d7df74ac1a8135c7ec71da098cbbc03d8652",
        "license": "BSD-3-Clause", "files": (
            "acoular/fastFuncs.py", "acoular/fbeamform.py", "LICENSE")},
    "doatools": {
        "revision": "9469db201e0418aef6b97583ef54b6fec2769502",
        "license": "MIT", "files": (
            "doatools/model/sources.py", "doatools/utils/math.py",
            "doatools/performance/utils.py", "doatools/performance/crb.py", "LICENSE.md")},
}
ABS_TOLERANCE = 1e-12
ACOULAR_ABS_TOLERANCE = 2e-6  # Original kernel rounds its phase to float32.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(checkout: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=checkout, capture_output=True,
                          check=True, timeout=60).stdout


def verify_sources(cache: Path = CACHE, lock_path: Path = LOCK) -> dict:
    """Reject changed locks, wrong HEADs, dirty trees and changed source bytes."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = {}
    for project, spec in PROJECTS.items():
        checkout = cache / project
        if not (checkout / ".git").exists():
            raise FileNotFoundError(f"Existing locked checkout required: {checkout}")
        entries = [p for p in lock["projects"] if p["id"] == project]
        if len(entries) != 1:
            raise ValueError(f"Expected one lock entry for {project}")
        entry = entries[0]
        if (entry["revision"], entry["license"]) != (spec["revision"], spec["license"]):
            raise ValueError(f"Review changed source lock before running: {project}")
        head = _git(checkout, "rev-parse", "HEAD").decode().strip()
        if head != spec["revision"]:
            raise ValueError(f"Unexpected upstream HEAD: {project}: {head}")
        if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"Upstream worktree is not clean: {project}")
        hashes = {}
        for relative in spec["files"]:
            actual = (checkout / relative).read_bytes()
            if actual != _git(checkout, "show", f"{head}:{relative}"):
                raise ValueError(f"Source bytes differ from Git blob: {project}/{relative}")
            hashes[relative] = hashlib.sha256(actual).hexdigest()
        result[project] = {"url": entry["url"], "revision": head,
                           "release": entry.get("release"), "license": entry["license"],
                           "worktree_clean": True, "source_sha256": hashes,
                           "lock_entry_sha256": hashlib.sha256(json.dumps(
                               entry, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode()).hexdigest()}
    return {"lock_sha256": sha256(lock_path), "projects": result}


def extract(path: Path, names: tuple[str, ...], namespace: dict, *,
            remove_decorators: tuple[str, ...] = ()) -> list[dict]:
    """Compile selected top-level definitions only; preserve their bodies."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes, evidence = [], []
    for name in names:
        matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one definition: {path}:{name}")
        node = matches[0]
        removed = []
        if name in remove_decorators:
            removed = [ast.unparse(n) for n in node.decorator_list]
            node.decorator_list = []
        nodes.append(node)
        evidence.append({"path": str(path.relative_to(ROOT)), "name": name,
                         "start_line": node.lineno, "end_line": node.end_lineno,
                         "removed_decorators": removed})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return evidence


def complex_values(values) -> list[dict]:
    return [{"real": float(z.real), "imag": float(z.imag)} for z in np.ravel(values)]


def assert_close(actual, expected, *, atol=ABS_TOLERANCE, rtol=0.0):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False):
        raise AssertionError(f"Unexpected extracted-method result: {actual!r} != {expected!r}")


def audit_pra(cache: Path, extractions: list) -> dict:
    base = cache / "pyroomacoustics/pyroomacoustics"
    ns = {"np": np}
    for relative, names in (
        ("parameters.py", ("calculate_speed_of_sound",)),
        ("doa/doa.py", ("ModeVector",)),
        ("acoustics.py", ("rt60_eyring",)),
        ("experimental/rt60.py", ("_fit_exp_and_extrapolate", "measure_rt60")),
    ):
        extractions.extend(extract(base / relative, names, ns))
    speeds = []
    for t, h, p, expected in ((20, 0, 101.325, 343.4),
                               (20, 50, 101.325, 344.02), (20, 50, 80, 344.02)):
        value = float(ns["calculate_speed_of_sound"](t, h, p))
        assert_close(value, expected)
        speeds.append({"temperature_c": t, "relative_humidity_percent": h,
                       "pressure_kpa": p, "output_m_s": value, "expected_m_s": expected})

    locations = np.array([[-.1, 0, .1], [0, 0, 0]])
    modes = []
    for mode, grid_x in (("near", .2), ("far", 1.0)):
        grid = SimpleNamespace(x=np.array([grid_x]), y=np.array([0.]), z=np.array([0.]))
        # Independent scalar geometry and trigonometry, not a second ModeVector.
        distances = [.3, .2, .1] if mode == "near" else [-.1, 0., .1]
        expected = [complex(math.cos(2*math.pi*1000*(r-distances[1])/343),
                            math.sin(2*math.pi*1000*(r-distances[1])/343)) for r in distances]
        for precompute in (False, True):
            obj = ns["ModeVector"](locations, 8000, 8, 343, grid,
                                   mode=mode, precompute=precompute)
            value = obj[1, slice(None), 0]
            relative = value / value[1]
            assert_close(relative, expected)
            modes.append({"mode": mode, "precompute": precompute,
                          "grid_xyz": [grid_x, 0., 0.],
                          "grid_identity": "absolute source position in metres" if mode == "near"
                                           else "unit direction; not a source range",
                          "relative_response": complex_values(relative),
                          "expected_response": complex_values(expected)})
    physical = [(0.2/r) * complex(math.cos(-2*math.pi*1000*(r-.2)/343),
                                 math.sin(-2*math.pi*1000*(r-.2)/343)) for r in [.3, .2, .1]]
    try:
        ns["ModeVector"](locations, 8000, 7, 343, grid)
    except ValueError as error:
        odd_fft = {"classification": "rejected", "exception": str(error)}
    else:
        raise AssertionError("Expected odd FFT length rejection")

    eyring = []
    for m in (0., .001, .01, .1):
        value = float(ns["rt60_eyring"](100., 56., .2, m, 343.))
        expected = -(24*math.log(10)/343)*56 / (100*math.log(.8)+4*m*56)
        reference = (24*math.log(10)/343)*56 / (-100*math.log(.8)+4*m*56)
        assert_close(value, expected)
        eyring.append({"air_absorption_parameter_per_m": m, "output_s": value,
                       "expected_fixed_source_s": expected,
                       "independent_positive_absorption_reference_s": reference,
                       "classification": "zero_air_reference" if m == 0 else
                       ("positive_but_increases_with_absorption" if value > 0 else "negative_time_invalid")})

    fs, target, length = 16000, .6, 32000
    impulse = np.exp(-3*math.log(10)*np.arange(length)/(fs*target))
    fixtures = (("analytic_exponential", impulse, target, True),
                ("zero_energy", np.zeros(8), 0., False),
                ("below_5db_decay", np.array([0., 0., 1.]), 0., False))
    decay = []
    for name, h, expected, valid in fixtures:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = float(ns["measure_rt60"](h, fs=fs, decay_db=20,
                                            energy_thres=1., plot=False, linear_domain_fit=False))
        assert_close(value, expected, atol=1e-10)
        decay.append({"name": name, "sample_count": len(h),
                      "input": "exp(-3*ln(10)*n/(fs*target_t60))" if valid else h.tolist(),
                      "input_float64_sha256": hashlib.sha256(h.astype('<f8').tobytes()).hexdigest(),
                      "output_s": value, "expected_s": expected,
                      "measurement_valid_for_fixture": valid,
                      "classification": "valid_analytic_decay" if valid else name,
                      "warnings": [{"category": type(w.message).__name__, "message": str(w.message)}
                                   for w in caught],
                      "zero_energy_edc_classification": "undefined_log_and_normalization" if name == "zero_energy" else None})
    return {"speed": speeds,
            "mode_vector": {"sensor_xy_m": locations.T.tolist(), "fs_hz": 8000,
                            "nfft": 8, "frequency_index": 1, "frequency_hz": 1000,
                            "c_m_s": 343, "reference_channel_zero_based": 1, "cases": modes,
                            "independent_physical_near_response": complex_values(physical),
                            "physical_near_amplitudes": [2/3, 1., 2.], "odd_fft": odd_fft,
                            "upper_doa_constructor_executed": False},
            "eyring": {"surface_m2": 100., "volume_m3": 56., "energy_absorption": .2,
                       "c_m_s": 343., "cases": eyring, "room_simulation_executed": False},
            "measure_rt60": {"fs_hz": fs, "target_t60_s": target, "decay_db": 20,
                             "energy_thres": 1., "linear_domain_fit": False, "plot": False,
                             "expected_derivation": "h squared has -60/target dB per second; geometric tail sum preserves slope away from truncation",
                             "cases": decay}}


def audit_acoular(cache: Path, extractions: list) -> dict:
    base = cache / "acoular/acoular"
    ns = {"np": np}
    extractions.extend(extract(base / "fastFuncs.py", ("_transferCoreFunc",), ns,
                               remove_decorators=("_transferCoreFunc",)))
    tree = ast.parse((base / "fbeamform.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SteeringVector")
    node = next(n for n in cls.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_steer_funcs_freq" for t in n.targets))
    literal = node.value.args[0]
    if not isinstance(literal, ast.Dict):
        raise ValueError("Expected original steering lambda dictionary")
    functions = eval(compile(ast.Expression(body=literal), str(base / "fbeamform.py"), "eval"), ns)
    extractions.append({"path": str((base / "fbeamform.py").relative_to(ROOT)),
                        "name": "SteeringVector._steer_funcs_freq dictionary only",
                        "start_line": node.lineno, "end_line": node.end_lineno,
                        "removed_decorators": [], "traits_Dict_constructor_executed": False})
    distances = [.3, .2, .1]
    value = np.empty(3, dtype=np.complex128)
    ns["_transferCoreFunc"](np.array([.2]), np.array(distances),
                            np.array([2*math.pi*1000/343]), value)
    expected = [(0.2/r)*complex(math.cos(-2*math.pi*1000*(r-.2)/343),
                                math.sin(-2*math.pi*1000*(r-.2)/343)) for r in distances]
    assert_close(value, expected, atol=ACOULAR_ABS_TOLERANCE)
    fractions = {"classic": (11/9, 1/3), "inverse": (1., 7/18),
                 "true level": (1., 9/49), "true location": (7/(3*math.sqrt(3)), 1/3)}
    cases = []
    for kind, (response, norm) in fractions.items():
        weights = functions[kind](value[None, :])[0]
        observed_response = np.vdot(weights, value)
        observed_norm = float(np.vdot(weights, weights).real)
        assert_close(observed_response, response, atol=ACOULAR_ABS_TOLERANCE)
        assert_close(observed_norm, norm, atol=ACOULAR_ABS_TOLERANCE)
        cases.append({"steer_type": kind, "weights": complex_values(weights),
                      "target_response": complex_values([observed_response])[0],
                      "squared_weight_norm": observed_norm,
                      "expected_target_response": response, "expected_squared_weight_norm": norm})
    return {"reference_distance_m": .2, "mic_distances_m": distances,
            "frequency_hz": 1000, "c_m_s": 343, "transfer": complex_values(value),
            "independent_transfer": complex_values(expected), "cases": cases,
            "numba_compilation_executed": False, "traits_pipeline_executed": False}


class ArrayAdapter:
    """Own ideal array adapter; original source class supplies phase derivatives."""
    size = 6
    locations = np.arange(size, dtype=float)[:, None] * .014

    def steering_matrix(self, sources, wavelength, derivatives, perturbations):
        if not derivatives or perturbations != 'all':
            raise ValueError("This adapter supports only the audited derivative call")
        phase, derivative = sources.phase_delay_matrix(self.locations, wavelength, True)
        response = np.exp(1j*phase)
        return response, 1j*response*derivative


def audit_crb(cache: Path, extractions: list) -> dict:
    base = cache / "doatools/doatools"
    ns = {"np": np, "ABC": ABC, "abstractmethod": abstractmethod, "copy": copy}
    for relative, names in (
        ("model/sources.py", ("_validate_sensor_location_ndim", "SourcePlacement", "FarField1DSourcePlacement")),
        ("utils/math.py", ("projm",)),
        ("performance/utils.py", ("reduce_output_matrix",)),
        ("performance/crb.py", ("crb_det_farfield_1d",)),
    ):
        extractions.extend(extract(base / relative, names, ns))
    array, cases = ArrayAdapter(), []
    for unit in ("rad", "deg"):
        for degrees in (0., 60.):
            theta = math.radians(degrees)
            source = ns["FarField1DSourcePlacement"]([theta if unit == "rad" else degrees], unit)
            variance = float(ns["crb_det_farfield_1d"](
                array, source, .343, np.array([[10.]]), 1., n_snapshots=100)[0, 0])
            expected_rad = 6 / (100*10*6*(6**2-1)*(2*math.pi*.014*math.cos(theta)/.343)**2)
            expected = expected_rad if unit == "rad" else expected_rad*(180/math.pi)**2
            assert_close(variance, expected, atol=1e-14, rtol=1e-12)
            cases.append({"angle_deg": degrees, "source_unit": unit,
                          "variance_unit": f"{unit}^2", "variance": variance,
                          "expected_variance": expected,
                          "standard_deviation_deg": math.sqrt(variance)*(180/math.pi if unit == "rad" else 1.)})
    try:
        ns["crb_det_farfield_1d"](array, source, .343, np.array([10.]), 1., n_snapshots=100)
    except ValueError as error:
        rejection = {"classification": "rejected_1d_P", "exception": str(error)}
    else:
        raise AssertionError("Expected source covariance dimension rejection")
    return {"array": "own six-sensor ideal ULA adapter", "spacing_m": .014,
            "wavelength_m": .343, "source_second_moment": [[10.]],
            "noise_variance": 1., "independent_snapshots": 100, "cases": cases,
            "independent_formula": "6/(T*SNR*M*(M^2-1)*(2*pi*d*cos(theta)/lambda)^2)",
            "invalid_P": rejection, "full_ArrayDesign_or_DOA_estimator_executed": False}


def run_audit(cache: Path = CACHE) -> dict:
    sources = verify_sources(cache)
    extractions = []
    results = {"pyroomacoustics": audit_pra(cache, extractions),
               "acoular": audit_acoular(cache, extractions),
               "doatools": audit_crb(cache, extractions)}
    if verify_sources(cache) != sources:
        raise ValueError("Source or lock changed while the audit ran")
    report = {"schema_version": 1, "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source": str(Path(__file__).resolve().relative_to(ROOT)),
              "audit_source_sha256": sha256(Path(__file__)), "sources": sources,
              "scope": "original-method AST extraction and execution; not full-package validation",
              "environment": {"python": sys.version, "executable": sys.executable,
                              "numpy": np.__version__, "platform": platform.platform()},
              "tolerances": {"default_absolute": ABS_TOLERANCE, "default_relative": 0.,
                             "acoular_absolute": ACOULAR_ABS_TOLERANCE,
                             "rt60_absolute_s": 1e-10, "crb_absolute": 1e-14, "crb_relative": 1e-12},
              "scaffold": {"upstream_modified": False, "upstream_imports_executed": False,
                           "extractions": extractions,
                           "own_adapters": ["SimpleNamespace grid xyz", "ArrayAdapter.steering_matrix"],
                           "runtime_not_required": ["SciPy nonlinear fit", "Traits", "Numba JIT"]},
              "all_expected_behaviors_observed": True, "results": results,
              "not_executed": ["full-package import", "room generation", "upper DOA pipeline",
                               "pyfar deconvolution", "HARK complex spectrum loading",
                               "LabVIEW humid-air model", "equipment or ISO acceptance"]}
    json.dumps(report, allow_nan=False)  # Refuse non-standard NaN/Infinity JSON.
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write strict JSON report atomically")
    args = parser.parse_args()
    report = run_audit()
    report["command"] = [sys.executable, "-m", "codes.chapters.ch02.examples.audit_upstream_models", *sys.argv[1:]]
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report is None:
        print(text, end="")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.report.parent,
                                             prefix=".upstream-models-", suffix=".tmp", delete=False) as out:
                temporary = Path(out.name)
                out.write(text)
            os.replace(temporary, args.report)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Verified extracted methods; report: {args.report}")


if __name__ == "__main__":
    main()
