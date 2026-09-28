"""Fixed tracking interface probes; no downloads or upstream modifications.

FilterPy is imported from its verified checkout. Stone Soup reducer methods
are extracted unchanged from the AST and called with explicit minimal state
containers: this is not execution of the Stone Soup package or its PHD filter.
"""
from __future__ import annotations

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

CODES = Path(__file__).resolve().parents[1]
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


def verify_sources(root):
    locks = {p["id"]: p for p in json.loads((CODES / "SOURCES.lock.json").read_text())["projects"]}
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for name, spec in SOURCES.items():
        checkout = root / name
        def git(*args):
            return subprocess.check_output(["git", "-C", str(checkout), *args],
                                           env=env, text=True, stderr=subprocess.PIPE).strip()
        if locks[name]["revision"] != spec["revision"] or git("rev-parse", "HEAD") != spec["revision"]:
            raise ValueError("revision mismatch: " + name)
        if Path(git("rev-parse", "--show-toplevel")).resolve() != checkout.resolve():
            raise ValueError("expected independent checkout: " + name)
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("upstream tracked files modified: " + name)
        for relative, digest in spec["files"].items():
            if sha256(checkout / relative) != digest:
                raise ValueError("source digest mismatch: " + name + "/" + relative)


def scalar(x):
    return float(np.asarray(x).item())


def run_filterpy(root):
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
    return {"execution": "original package classes/functions imported from verified checkout",
            "filterpy_version": filterpy.__version__, "scipy_version": scipy.__version__,
            "imm_predict_only": states, "imm_update_none_fresh": fresh_none,
            "imm_update_none_after_measurement": stale_none,
            "ekf": ekf, "angle": angle,
            "unscented_transform": {"points": sigmas.tolist(), "Wm": points.Wm.tolist(),
                                    "Wc": points.Wc.tolist(), "mean": scalar(mean), "variance": scalar(cov)}}


def run_reducer_extraction(root):
    sys.path.insert(0, str((root / "stonesoup").resolve()))
    try:
        from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
    except ImportError as exc:
        package_probe = type(exc).__name__ + ": " + str(exc)
    else:
        package_probe = "import succeeded; reducer calls below still use AST extraction"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=CODES / "upstream/_downloads")
    parser.add_argument("--output", type=Path, default=CODES / "reports/tracking_upstream_interfaces.json")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    verify_sources(args.source_root)
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
              "sources": SOURCES, "config": CONFIG,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "numpy": np.__version__},
              "filterpy": run_filterpy(args.source_root),
              "stonesoup_reducer": run_reducer_extraction(args.source_root),
              "static_only": ["ODAS", "SAF tracker3d", "icoDOA/Cross3D", "StoneSoup JPDA/GOSPA/MHT"],
              "not_run": ["audio corpus", "hardware", "neural inference/training", "MATLAB Vo toolbox"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
