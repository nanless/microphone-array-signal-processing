"""Execute selected unchanged methods from four existing pinned checkouts.

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
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import warnings

import numpy as np

from codes.chapters.ch00.io_contracts import (
    strict_json_loads, validate_parent_chain, validate_report_destination,
    write_json_report,
)
from codes.chapters.ch00.upstream.fetch_upstreams import (
    inspect_project, run_git, validate_project,
)

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
CACHE = ROOT / "codes/chapters/ch00/upstream/_downloads"
STATUS = LOCK.with_name("SOURCE_STATUS.json")
CURRENT_REPORT = ROOT / "codes/chapters/ch02/reports/upstream_model_contracts.json"
HISTORICAL_REPORT = CURRENT_REPORT.with_name("upstream_models.json")
HISTORICAL_PROJECTS = ("pyroomacoustics", "acoular", "doatools")
HISTORICAL_SCRIPT_REVISION = "82911654fae648f85dea5870b61fc6ad5422be78"
HISTORICAL_SCRIPT_SHA256 = "86f824d086eb19782eb25e1ff93a1deb841d7bc623162cc1badd619d0625fcf0"
SOURCE_DEPENDENCIES = (
    "codes/chapters/ch02/examples/audit_upstream_models.py",
    "codes/chapters/ch00/io_contracts.py",
    "codes/chapters/ch00/upstream/fetch_upstreams.py",
)
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
    "pyfar": {
        "revision": "0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a",
        "license": "MIT", "files": (
            "pyfar/signals/deterministic.py", "pyfar/dsp/dsp.py", "LICENSE", "pyproject.toml")},
    "speed-of-sound-in-air": {
        "revision": "5c7e6652cf2fd7b4c52e83d8fa3782b3c56e61de",
        "license": "GPL-3.0", "files": ("ReadMe.txt", "LICENSE")},
}
ORIGINS = {
    "pyroomacoustics": "https://github.com/LCAV/pyroomacoustics.git",
    "acoular": "https://github.com/acoular/acoular.git",
    "doatools": "https://github.com/morriswmz/doatools.py.git",
    "pyfar": "https://github.com/pyfar/pyfar.git",
    "speed-of-sound-in-air": "https://github.com/RobertoGavioso/Speed-of-sound-in-air.git",
}
ABS_TOLERANCE = 1e-12
ACOULAR_ABS_TOLERANCE = 2e-6  # Original kernel rounds its phase to float32.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordinary_file(path: Path) -> Path:
    path = validate_parent_chain(path)
    if not path.is_file():
        raise ValueError(f"An existing ordinary file is required: {path}")
    return path


def report_target(path: Path) -> Path:
    target = validate_report_destination(path, forbidden_roots=(CACHE, LOCK, STATUS,
        HISTORICAL_REPORT, ROOT / "codes/chapters/ch00/source_snapshots"))
    if target.resolve().is_relative_to(ROOT.resolve()) and target.resolve() != CURRENT_REPORT.resolve():
        raise ValueError("Only the new current model-contract report may be written inside the repository")
    return target


def _git(checkout: Path, *arguments: str) -> str:
    return run_git(list(arguments), cwd=checkout)


def verify_sources(cache: Path = CACHE, lock_path: Path = LOCK,
                   status_path: Path = STATUS) -> dict:
    """Verify used files; report the full selection separately, without repairing it."""
    lock_raw = ordinary_file(lock_path).read_bytes()
    status_raw = ordinary_file(status_path).read_bytes()
    lock, status = strict_json_loads(lock_raw), strict_json_loads(status_raw)
    lock_sha = hashlib.sha256(lock_raw).hexdigest()
    for document in (lock, status):
        if (type(document) is not dict or type(document.get("schema_version")) is not int
                or document["schema_version"] != 1 or type(document.get("projects")) is not list
                or any(type(row) is not dict for row in document["projects"])):
            raise ValueError("Source documents require schema 1 with object project records")
        ids = [row.get("id") for row in document["projects"]]
        if any(type(name) is not str for name in ids) or len(set(ids)) != len(ids):
            raise ValueError("Source project records require unique string IDs")
    if status.get("lock_sha256") != lock_sha:
        raise ValueError("Current source status must bind the actual current lock bytes")
    result = {}
    for project, spec in PROJECTS.items():
        checkout = validate_parent_chain(cache / project)
        validate_parent_chain(checkout / ".git")
        if not checkout.is_dir() or not (checkout / ".git").is_dir():
            raise ValueError(f"Existing independent ordinary checkout required: {checkout}")
        entries = [p for p in lock["projects"] if p["id"] == project]
        if len(entries) != 1:
            raise ValueError(f"Expected one lock entry for {project}")
        entry = entries[0]
        validate_project(entry)
        if (entry["revision"], entry["license"]) != (spec["revision"], spec["license"]):
            raise ValueError(f"Review changed source lock before running: {project}")
        if entry["url"] != ORIGINS[project]:
            raise ValueError(f"Unexpected official URL in source lock: {project}")
        if _git(checkout, "rev-parse", "--show-toplevel") != str(checkout.resolve()):
            raise ValueError(f"Checkout must be its own Git top level: {project}")
        origin = _git(checkout, "remote", "get-url", "origin")
        if origin != ORIGINS[project]:
            raise ValueError(f"Unexpected actual upstream origin: {project}")
        head = _git(checkout, "rev-parse", "HEAD")
        if head != spec["revision"]:
            raise ValueError(f"Unexpected upstream HEAD: {project}: {head}")
        if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"Upstream worktree is not clean: {project}")
        hashes, files = {}, {}
        for relative in spec["files"]:
            path = ordinary_file(checkout / relative)
            actual = path.read_bytes()
            blob = _git(checkout, "rev-parse", f"{head}:{relative}")
            actual_blob = _git(checkout, "hash-object", "--", str(path))
            if actual_blob != blob:
                raise ValueError(f"Source bytes differ from Git blob: {project}/{relative}")
            hashes[relative] = hashlib.sha256(actual).hexdigest()
            files[relative] = {"sha256": hashes[relative], "bytes": len(actual),
                               "head_blob": blob, "actual_blob": actual_blob}
        states = [row for row in status["projects"] if row["id"] == project]
        if len(states) != 1 or states[0].get("revision") != head:
            raise ValueError(f"Current acquisition record must bind fixed revision: {project}")
        if (type(states[0].get("status")) is not str
                or type(states[0].get("source_selection_verified")) is not bool):
            raise ValueError(f"Acquisition state must retain status and boolean selection: {project}")
        live = inspect_project(entry, cache)
        result[project] = {"url": entry["url"], "revision": head,
                           "actual_origin": origin, "used_source_identity_verified": True,
                           "used_file_identity": files, "lock_entry": entry,
                           "recorded_acquisition": states[0], "live_acquisition_scope": live,
                           "release": entry.get("release"), "license": entry["license"],
                           "worktree_clean": True, "source_sha256": hashes,
                           "lock_entry_sha256": hashlib.sha256(json.dumps(
                               entry, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode()).hexdigest()}
    return {"lock_sha256": lock_sha, "status_sha256": hashlib.sha256(status_raw).hexdigest(),
            "projects": result}


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


class SignalAdapter:
    """Own real Signal/FFT scaffold: only normalization 'none' is supported.

    This is deliberately not pyfar's Signal class or its arithmetic machinery.
    The audited controls use one real channel and explicit inverse-FFT length.
    """
    def __init__(self, time, sampling_rate, fft_norm="none", comment=""):
        if fft_norm != "none" or np.iscomplexobj(time):
            raise ValueError("The adapter supports only real normalization-none controls")
        self.time = np.atleast_2d(np.array(time, dtype=float))
        self.sampling_rate, self.fft_norm, self.comment = sampling_rate, fft_norm, comment

    @property
    def n_samples(self):
        return self.time.shape[-1]

    @property
    def n_bins(self):
        return self.n_samples // 2 + 1

    @property
    def freq(self):
        return np.fft.rfft(self.time)

    @freq.setter
    def freq(self, values):
        self.time = np.fft.irfft(values, n=self.n_samples)

    @property
    def freq_raw(self):
        return self.freq

    def copy(self):
        return copy.deepcopy(self)

    def find_nearest_frequency(self, values):
        frequencies = np.fft.rfftfreq(self.n_samples, 1 / self.sampling_rate)
        return np.argmin(abs(frequencies[:, None] - np.asarray(values)), axis=0)

    def __mul__(self, other):
        if self.n_samples != other.n_samples or self.sampling_rate != other.sampling_rate:
            raise ValueError("Adapter multiplication requires identical lengths and rates")
        result = self.copy()
        result.freq = self.freq * other.freq
        return result


def adapter_pad_zeros(signal, samples):
    return SignalAdapter(np.pad(signal.time, ((0, 0), (0, int(samples)))),
                         signal.sampling_rate, signal.fft_norm, signal.comment)


def adapter_match_norm(first, second, division=False):
    if first != "none" or second != "none" or not division:
        raise ValueError("Only the audited normalization-none division is supported")
    return "none"


class AdapterDeprecationWarning(UserWarning):
    """Own warning class supplied to original functions; not pyfar's class."""


def audit_pyfar(cache: Path, extractions: list) -> dict:
    base = cache / "pyfar/pyfar"
    namespace = {"np": np, "warnings": warnings,
        "PyfarDeprecationWarning": AdapterDeprecationWarning,
        "pyfar": SimpleNamespace(Signal=SignalAdapter,
            dsp=SimpleNamespace(pad_zeros=adapter_pad_zeros),
            classes=SimpleNamespace(audio=SimpleNamespace(_match_fft_norm=adapter_match_norm)))}
    for relative, names in (
        ("signals/deterministic.py", ("exponential_sweep_time", "_time_domain_sweep", "_exponential_sweep")),
        ("dsp/dsp.py", ("regularized_spectrum_inversion", "_cross_fade", "deconvolve")),
    ):
        extractions.extend(extract(base / relative, names, namespace))
    # Independent four-point DFT coefficients, including DC and Nyquist.
    x, h, y = [1., .5], [1., 0., .5], [1., .5, .5, .25]
    X, H, noise = np.array([1.5, 1-.5j, .5]), np.array([1.5, .5, 1.5]), np.array([.1, -.1j, -.1])
    epsilon = .25
    cases = (
        ("noiseless_regularized", x, y, H * abs(X)**2 / (abs(X)**2 + epsilon)),
        ("noise_regularized", x, [1., .6, .5, .25], (H*X + noise)*X.conj() / (abs(X)**2 + epsilon)),
        ("truncated_output", x, y[:2], abs(X)**2 / (abs(X)**2 + epsilon)),
        ("unexcited_DC", [1., -1.], [1., -1., .5, -.5],
         np.array([0., .5*2/(2+epsilon), 1.5*4/(4+epsilon)])),
    )
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, xx, yy, expected in cases:
            result = namespace["deconvolve"](SignalAdapter(yy, 8000), SignalAdapter(xx, 8000),
                                             fft_length=4, regu_final=epsilon)
            dc, z, nyquist = expected
            manual_waveform = np.array([(dc.real+nyquist.real+2*z.real)/4,
                (dc.real-nyquist.real-2*z.imag)/4, (dc.real+nyquist.real-2*z.real)/4,
                (dc.real-nyquist.real+2*z.imag)/4])
            assert_close(result.freq[0], expected)
            assert_close(result.time[0], manual_waveform)
            rows.append({"name": name, "input": xx, "output": yy, "epsilon": epsilon,
                         "fft_length": 4, "response": complex_values(result.freq[0]),
                         "expected_response": complex_values(expected),
                         "waveform": result.time[0].tolist(), "manual_waveform": manual_waveform.tolist()})
        fs, samples, fft_length = 8000, 1024, 2048
        sweep = namespace["exponential_sweep_time"](samples, [100, 3000], sampling_rate=fs,
                                                   n_fade_out=32, amplitude=.2)
        t = np.arange(samples)/fs
        scale = (samples/fs)/math.log(30)
        reference_sweep = .2*np.sin(2*math.pi*100*scale*(np.exp(t/scale)-1))
        reference_sweep[-32:] *= np.cos(np.linspace(0, math.pi/2, 32))**2
        assert_close(sweep.time[0], reference_sweep)
        fir = np.array([.8, 0., 0., .25])
        output = np.convolve(sweep.time[0], fir)
        reference_fir = np.pad(fir, (0, fft_length-len(fir)))
        exact = namespace["deconvolve"](SignalAdapter(output, fs), sweep,
                                        fft_length=fft_length, regu_final=0.)
        default = namespace["deconvolve"](SignalAdapter(output, fs), sweep, fft_length=fft_length)
        spectrum = np.fft.rfft(reference_sweep, fft_length)
        default_epsilon = float(1e-10*np.max(abs(spectrum)**2))
        expected_default = np.fft.rfft(reference_fir)*abs(spectrum)**2/(abs(spectrum)**2+default_epsilon)
        assert_close(exact.time[0], reference_fir, atol=2e-12)
        assert_close(default.freq[0], expected_default, atol=2e-12)
        rows.append({"name": "digital_ESS_full_support", "fs_hz": fs, "sweep_samples": samples,
            "record_samples": len(output), "fft_length": fft_length, "frequency_range_hz": [100, 3000],
            "amplitude": .2, "n_fade_out": 32, "fir": fir.tolist(),
            "sweep_vs_independent_max_error": float(np.max(abs(sweep.time[0]-reference_sweep))),
            "unregularized_full_fir_max_error": float(np.max(abs(exact.time[0]-reference_fir))),
            "default_epsilon": default_epsilon,
            "default_fir_max_error": float(np.max(abs(default.time[0]-reference_fir))),
            "minimum_input_spectral_power": float(np.min(abs(spectrum)**2)),
            "input_float64_sha256": hashlib.sha256(sweep.time.astype('<f8').tobytes()).hexdigest(),
            "output_float64_sha256": hashlib.sha256(output.astype('<f8').tobytes()).hexdigest(),
            "relation_to_chapter_audio": "Independent 8-kHz/1024-point contract; not the 16-kHz chapter audio experiment"})
    return {"scope": "Six complete unchanged original function bodies with own minimal real Signal/FFT/pad_zeros/normalization-none adapters",
        "short_control_fir": h, "short_control_epsilon": epsilon, "cases": rows,
        "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
        "zero_bin_boundary": "A zero DFT bin blocks pointwise division at that bin; it does not prove nonidentifiability of known-length FIR from complete linear output",
        "not_executed": ["pyfar package import", "original Signal class, arithmetic and FFT normalization machinery",
            "original pad_zeros implementation", "original warning class", "original convolve wrapper",
            "Farina weighted reversed-sweep inverse", "nonlinear harmonic separation", "playback or capture",
            "equipment, calibration or ISO measurement"]}


def source_dependencies() -> dict:
    return {relative: sha256(ordinary_file(ROOT / relative)) for relative in SOURCE_DEPENDENCIES}


def run_audit(cache: Path = CACHE) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    dependencies = source_dependencies()
    sources = verify_sources(cache)
    extractions = []
    results = {"pyroomacoustics": audit_pra(cache, extractions),
               "acoular": audit_acoular(cache, extractions),
               "doatools": audit_crb(cache, extractions),
               "pyfar": audit_pyfar(cache, extractions)}
    sources_after = verify_sources(cache)
    if sources_after != sources:
        raise ValueError("Source or lock changed while the audit ran")
    if source_dependencies() != dependencies:
        raise ValueError("Audit or shared source changed while the audit ran")
    report = {"schema_version": 2, "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source": str(Path(__file__).resolve().relative_to(ROOT)),
              "audit_source_sha256": sha256(Path(__file__)), "sources": sources,
              "source_sha256": dependencies,
              "source_verification": {"started_at_utc": started,
                  "before_after_equal": True,
                  "canonical_sources_before_sha256": hashlib.sha256(json.dumps(
                      sources, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                  "canonical_sources_after_sha256": hashlib.sha256(json.dumps(
                      sources_after, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                  "shared_dependencies_before_after_equal": True,
                  "scope": "Actual origin/HEAD/used blobs/licences/clean tree, complete live selection, lock/status bytes and local helper SHA rechecked after execution"},
              "scope": "original-method AST extraction and execution; not full-package validation",
              "method_scopes": {
                  "pyroomacoustics": "Original speed/ModeVector/Eyring/RT60 functions with own grid; no room simulation or upper DOA constructor",
                  "acoular": "Original transfer kernel body with JIT decorator removed and four dictionary lambdas; no Traits/Numba pipeline",
                  "doatools": "Original source/derivative/projection/CRB definitions with own ideal ArrayAdapter; no full ArrayDesign or estimator",
                  "pyfar": results["pyfar"]["scope"],
                  "speed-of-sound-in-air": "ReadMe/LICENSE identity only; no VI, executable or LabVIEW runtime executed"},
              "selection_status_boundary": "Full acquisition selection is independent of used-file identity and numerical behavior; source_selection_mismatch is not upgraded",
              "environment": {"python": sys.version, "executable": sys.executable,
                              "numpy": np.__version__, "platform": platform.platform()},
              "tolerances": {"default_absolute": ABS_TOLERANCE, "default_relative": 0.,
                             "acoular_absolute": ACOULAR_ABS_TOLERANCE,
                             "rt60_absolute_s": 1e-10, "crb_absolute": 1e-14, "crb_relative": 1e-12},
              "scaffold": {"upstream_modified": False, "upstream_imports_executed": False,
                           "extractions": extractions,
                           "own_adapters": ["SimpleNamespace grid xyz", "ArrayAdapter.steering_matrix",
                                            "SignalAdapter with NumPy real FFT", "adapter_pad_zeros",
                                            "adapter_match_norm", "AdapterDeprecationWarning"],
                           "runtime_not_required": ["SciPy nonlinear fit", "Traits", "Numba JIT"]},
              "all_expected_behaviors_observed": True, "results": results,
              "not_executed": ["full-package import", "room generation", "upper DOA pipeline",
                               "original pyfar Signal and package pipeline", "HARK complex spectrum loading",
                               "LabVIEW humid-air model", "equipment or ISO acceptance"]}
    strict_json_loads(json.dumps(report, allow_nan=False))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write strict JSON report atomically")
    args = parser.parse_args(argv)
    target = report_target(args.report) if args.report is not None else None
    report = run_audit()
    report["command"] = [sys.executable, "-m", "codes.chapters.ch02.examples.audit_upstream_models", *(sys.argv[1:] if argv is None else argv)]
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report is None:
        print(text, end="")
    else:
        # Recheck the caller policy and shared ordinary-path contract before replace.
        write_json_report(report_target(target), report, forbidden_roots=(CACHE, LOCK, STATUS, HISTORICAL_REPORT))
        print(f"Verified extracted methods; report: {target}")


if __name__ == "__main__":
    main()
