"""Fixed tracking interface probes; no downloads or upstream modifications.

FilterPy is imported from its verified checkout. Stone Soup reducer methods
are extracted unchanged from the AST and called with explicit minimal state
containers: this is not execution of the Stone Soup package or its PHD filter.
"""
from __future__ import annotations

# Keep both documented direct-file and module entry points.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from operator import attrgetter

import numpy as np
from codes.chapters.ch04.core import upstream_contracts as contracts

ROOT = Path(__file__).resolve().parents[4]
CACHE = contracts.CACHE
CURRENT = ROOT / "codes/chapters/ch09/reports/tracking_upstream_interfaces_current.json"
PROTECTED = (CACHE,)

CODES = Path(__file__).resolve().parents[3]
SOURCES = {'filterpy': {'files': {'LICENSE': '8ffce1097f1b1c0fba42e4449ef49aa05cb7b45566dd0989407839dba07af99c',
                        'filterpy/__init__.py': '329536b1ff39c3fdc3621983a98ecd0ad4a2327ea0694750f46d6358a89fe67e',
                        'filterpy/common/helpers.py': '45ac0d338a5ae96caae37ccaced7ad1e45a4d848da17d4a8023e59213ef62f02',
                        'filterpy/kalman/EKF.py': '44421460b96bc911e2cf81ca97009d84594de5a448be1c8a3a5de752575a547a',
                        'filterpy/kalman/IMM.py': '22b86f0fef3989445e61243cfe78f1b141ab373622a4583c249ba34f2d9ef556',
                        'filterpy/kalman/kalman_filter.py': '602923a445aa167a43a721bef5b8e17bb31b0f43e1b08c880f414460eebcf689',
                        'filterpy/kalman/sigma_points.py': 'cfa457511eb6f1dbe95f452e1b2e5b489187b64707c5e7bc2b15f30dad17df9e',
                        'filterpy/kalman/unscented_transform.py': '3c499c584c7f6e3f0091b1817853926533a80efa689285d7a4fcfa73bd55be75',
                        'filterpy/stats/stats.py': '576b407938a36b7292f862cf414527e1adc8419d19241f529b6ec2942f31f374'},
              'revision': '3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33'},
 'stonesoup': {'files': {'LICENSE': '2462f3d8a857f601e266f048a4ff051366c1e05f098cf3a1524eaf922f879815',
                         'stonesoup/mixturereducer/gaussianmixture.py': 'ac8d5eacebb7e2e3458ab7125707915b77f57e23b9b19e87c2c8d52c4b803aa4'},
               'revision': '8d1edeb07ef8505ed065cbef435cfb5e517d9bdc'}}
CONFIG = {
    "imm": {"means": [0., 10.], "variances": [1., 4.], "mu": [.75, .25],
            "transition_rows_old_columns_new": [[.9, .1], [.2, .8]],
            "F": 1., "Q": 0., "H": 1., "R": 1., "predict_only_steps": 3,
            "stale_likelihood_sequence": ["predict", "update(0)", "predict", "update(None)"]},
    "ekf": {"x": 1., "P": 1., "F": 2., "Q": 0., "R": 1., "z": 4.,
            "h": "x**2", "HJacobian": "2*x"},
    "ekf_angle": {"x_degrees": 179., "z_degrees": -179., "P": 1., "R": 1.},
    "unscented_transform": {"mean": 0., "variance": 1., "alpha": 1., "beta": 2.,
                            "kappa": 0., "transform": "x**2"},
    "reducer": {"prune_threshold": .2, "max_number_components": 1,
                "prune_weights": [.1, .4, .5], "all_pruned_weights": [.1, .1],
                "truncate_weights": [.4, .6], "merge_weights": [.8, .7],
                "component_means": "zero-based component index", "component_variance": 1.},
    "scope": "dimensionless scalar arithmetic except explicitly labelled degree example; no acoustic data or hardware timing",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_sha256():
    return hashlib.sha256(json.dumps({"sources": SOURCES, "config": CONFIG},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sources(root=CACHE):
    identities = {}
    for name, spec in SOURCES.items():
        identity = contracts.verify_project(name, Path(root) / name,
            relatives=tuple(spec["files"]), all_python=name == "filterpy")
        acquisition = identity["recorded_complete_selection"]
        selection = acquisition.get("status") == "source_verified"
        if (acquisition.get("status") not in ("source_verified", "source_selection_mismatch")
                or acquisition.get("revision") != identity["head"]
                or acquisition.get("missing_entrypoints") != []
                or acquisition.get("source_selection_verified") is not selection):
            raise ValueError("source identity unavailable: " + name)
        for relative, expected in spec["files"].items():
            if identity["used_files"][relative]["sha256"] != expected:
                raise ValueError("source digest mismatch: " + name + "/" + relative)
        identities[name] = identity
    return identities


def loaded_filterpy_modules(identity):
    """All imported original modules must belong to the preflighted checkout.

    Tracked Python files are checked before import. This records which were
    actually imported; checking a file does not assert its every method ran.
    External NumPy/SciPy installations are environment dependencies, not the
    fixed author repository, and are identified separately in the report.
    """
    checkout = Path(identity["checkout"])
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if name == "filterpy" or name.startswith("filterpy."):
            filename = getattr(module, "__file__", None)
            if not filename:
                raise ValueError("FilterPy module has no ordinary source: " + name)
            path = contracts.ordinary_file(filename)
            try:
                relative = path.relative_to(checkout).as_posix()
            except ValueError as error:
                raise ValueError("Imported a foreign FilterPy module: " + name) from error
            if relative not in identity["used_files"]:
                raise ValueError("FilterPy module was not verified before import: " + relative)
            loaded[name] = relative
    return loaded


def scalar(x):
    return float(np.asarray(x).item())


def run_filterpy(root):
    sys.dont_write_bytecode = True
    identity = contracts.verify_project("filterpy", Path(root) / "filterpy",
        relatives=tuple(SOURCES["filterpy"]["files"]), all_python=True)
    loaded_filterpy_modules(identity)  # Reject preexisting foreign modules first.
    sys.path.insert(0, str((root / "filterpy").resolve()))
    import filterpy
    import scipy
    from filterpy.kalman import (KalmanFilter, IMMEstimator, ExtendedKalmanFilter,
                                MerweScaledSigmaPoints, unscented_transform)
    if Path(filterpy.__file__).resolve().parent != (root / "filterpy/filterpy").resolve():
        raise ValueError("imported an unexpected FilterPy")

    def make_imm():
        cfg = CONFIG["imm"]
        fs = []
        for mean, variance in zip(cfg["means"], cfg["variances"]):
            f = KalmanFilter(1, 1)
            f.x = np.array([[mean]])
            for field, value in {"P": variance, "F": 1., "Q": 0., "H": 1., "R": 1.}.items():
                setattr(f, field, np.array([[value]]))
            fs.append(f)
        return IMMEstimator(fs, np.array(cfg["mu"]),
                            np.array(cfg["transition_rows_old_columns_new"]))

    def snapshot(f):
        return {"mu": f.mu.tolist(), "cbar": f.cbar.tolist(), "omega": f.omega.tolist(),
                "mean": scalar(f.x), "variance": scalar(f.P),
                "second_raw_moment": scalar(f.P) + scalar(f.x)**2,
                "component_means": [scalar(g.x) for g in f.filters],
                "component_variances": [scalar(g.P) for g in f.filters]}

    f = make_imm()
    states = [snapshot(f)]
    for _ in range(CONFIG["imm"]["predict_only_steps"]):
        f.predict()
        states.append(snapshot(f))
    f = make_imm()
    f.predict()
    before_none = snapshot(f)
    f.update(None)
    fresh_none = {"before": before_none, "after": snapshot(f), "likelihoods": f.likelihood.tolist()}
    f = make_imm()
    f.predict()
    f.update(np.array([[0.]]))
    measured = snapshot(f)
    cached_likelihoods = [g.likelihood for g in f.filters]
    f.predict()
    before_none = snapshot(f)
    f.update(None)
    stale_none = {"after_real_measurement": measured, "before": before_none,
                  "after": snapshot(f), "previous_measurement_likelihoods": cached_likelihoods,
                  "retained_innovation_variances": [scalar(g.S) for g in f.filters],
                  "likelihoods_used_by_none": f.likelihood.tolist()}

    def make_ekf(angle=False):
        f = ExtendedKalmanFilter(1, 1)
        f.x = np.array([[179. if angle else 1.]])
        f.F = np.array([[1. if angle else 2.]])
        f.P = np.array([[1.]])
        f.Q = np.array([[0.]])
        f.R = np.array([[1.]])
        return f

    ekf = {}
    for name in ["predict_update", "predict_then_update"]:
        f = make_ekf()
        seen = []
        def jacobian(x):
            seen.append({"function": "HJacobian", "x": scalar(x)})
            return 2*x
        def hx(x):
            seen.append({"function": "Hx", "x": scalar(x)})
            return x*x
        if name == "predict_update":
            f.predict_update(np.array([[4.]]), jacobian, hx)
        else:
            f.predict()
            f.update(np.array([[4.]]), jacobian, hx)
        ekf[name] = {"calls": seen, "mean": scalar(f.x), "variance": scalar(f.P),
                     "gain": scalar(f.K), "innovation_variance": scalar(f.S),
                     "stored_x_prior": scalar(f.x_prior), "stored_P_prior": scalar(f.P_prior)}
    # The docstring promises prediction for None, but this fixed combined
    # implementation subtracts None after changing S/K. Keep the actual error.
    f = make_ekf()
    before_none = {"mean": scalar(f.x), "variance": scalar(f.P),
                   "gain": scalar(f.K), "innovation_variance": scalar(f.S)}
    try:
        f.predict_update(None, lambda x: 2*x, lambda x: x*x)
    except Exception as error:
        none_error = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("Fixed original EKF None failure changed")
    none_record = {"before": before_none, "exception": none_error,
        "after": {"mean": scalar(f.x), "variance": scalar(f.P),
                  "gain": scalar(f.K), "innovation_variance": scalar(f.S)},
        "scope": "original combined method failed after partial state mutation; not an atomic prediction"}
    angle = {}
    for name in ["merged_default", "separate_wrapped"]:
        f = make_ekf(True)
        jacobian = lambda x: np.ones((1,1))
        hx = lambda x: x
        if name == "merged_default":
            f.predict_update(np.array([[-179.]]), jacobian, hx)
        else:
            f.predict()
            f.update(np.array([[-179.]]), jacobian, hx,
                     residual=lambda z, h: (z-h+180.) % 360.-180.)
        angle[name] = {"innovation_degrees": scalar(f.y), "unwrapped_mean_degrees": scalar(f.x)}

    points = MerweScaledSigmaPoints(1, alpha=1., beta=2., kappa=0.)
    sigmas = points.sigma_points(np.array([0.]), np.array([[1.]]))
    mean, cov = unscented_transform(sigmas**2, points.Wm, points.Wc)
    loaded = loaded_filterpy_modules(identity)
    contracts.check_unchanged(identity)
    return {"execution": "original package classes/functions imported from verified checkout",
            "source_identity": identity, "imported_modules": loaded,
            "imported_module_count": len(loaded),
            "external_dependencies": {"numpy": {"version": np.__version__, "file": np.__file__},
                                      "scipy": {"version": scipy.__version__, "file": scipy.__file__}},
            "ekf_predict_update_none": none_record,
            "filterpy_version": filterpy.__version__, "scipy_version": scipy.__version__,
            "imm_predict_only": states, "imm_update_none_fresh": fresh_none,
            "imm_update_none_after_measurement": stale_none,
            "ekf": ekf, "angle": angle,
            "unscented_transform": {"points": sigmas.tolist(), "Wm": points.Wm.tolist(),
                                    "Wc": points.Wc.tolist(), "mean": scalar(mean), "variance": scalar(cov)}}


def run_reducer_extraction(root):
    # Do not import the package and leave a partly initialized Stone Soup in
    # sys.modules. These three original bodies use explicit minimal adapters.
    package_probe = "not_run: package import deliberately excluded from this AST-only current audit; historical failed import remains in its original report"
    source = root / "stonesoup/stonesoup/mixturereducer/gaussianmixture.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianMixtureReducer")
    class MinimalWeightedState:
        def __init__(self, state_vector, covar, weight, timestamp=None):
            self.state_vector = np.asarray(state_vector)
            self.mean = self.state_vector
            self.covar = np.asarray(covar)
            self.weight = weight
            self.timestamp = timestamp
    class MinimalTaggedState(MinimalWeightedState):
        pass
    namespace = {"WeightedGaussianState": MinimalWeightedState,
                 "TaggedWeightedGaussianState": MinimalTaggedState, "attrgetter": attrgetter}
    names = ["prune", "truncate", "merge_components"]
    for name in names:
        method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    reducer = SimpleNamespace(prune_threshold=.2, max_number_components=1)
    records = []
    for name, key in [("prune", "prune_weights"), ("prune", "all_pruned_weights"),
                      ("truncate", "truncate_weights"), ("merge_components", "merge_weights")]:
        weights = CONFIG["reducer"][key]
        components = [MinimalWeightedState([[float(i)]], [[1.]], w) for i,w in enumerate(weights)]
        output = ([namespace[name](reducer, *components)] if name == "merge_components"
                  else namespace[name](reducer, components))
        records.append({"method": name, "input_weights": weights,
                        "input_means": list(range(len(weights))), "input_variances": [1.]*len(weights),
                        "output_weights": [float(c.weight) for c in output],
                        "output_means": [scalar(c.mean) for c in output],
                        "output_variances": [scalar(c.covar) for c in output],
                        "mass_before": sum(weights), "mass_after": sum(float(c.weight) for c in output)})
    return {"execution": "unchanged AST-extracted original methods with minimal state containers",
            "stonesoup_package_executed": False,
            "package_import_probe": package_probe,
            "not_tested": ["PHDUpdater", "TaggedWeightedGaussianState branch", "merge orchestration", "MHT/OR-Tools"],
            "records": records}


def run(root=CACHE):
    sys.dont_write_bytecode = True
    deps = contracts.dependencies(Path(__file__))
    identities = verify_sources(root)
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
              "source_lock_sha256": contracts.sha(contracts.LOCK),
              "source_status_sha256": contracts.sha(contracts.STATUS),
              "actual_dependencies_sha256": deps,
              "sources": SOURCES, "config": CONFIG,
              "source_identities": identities,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "numpy": np.__version__},
              "filterpy": run_filterpy(root),
              "stonesoup_reducer": run_reducer_extraction(root),
              "static_only": ["ODAS", "SAF tracker3d", "icoDOA/Cross3D", "StoneSoup JPDA/GOSPA/MHT"],
              "not_run": ["audio corpus", "hardware", "neural inference/training", "MATLAB Vo toolbox", "StoneSoup package import/PHD"]}
    for identity in identities.values():
        contracts.check_unchanged(identity)
    if contracts.dependencies(Path(__file__)) != deps:
        raise ValueError("Actual audit dependencies changed during execution")
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.suffix != ".json":
        raise ValueError("Report destination must use .json")
    target = None if args.output is None else contracts.report_target(
        args.output, CURRENT, (*PROTECTED, args.source_root))
    report = run(args.source_root)
    if target is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        contracts.write_report(target, report, CURRENT, (*PROTECTED, args.source_root))
        print(target)


if __name__ == "__main__":
    main()
