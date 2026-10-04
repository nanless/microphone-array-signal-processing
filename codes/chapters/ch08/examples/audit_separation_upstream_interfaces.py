"""Bounded, unmodified separation interfaces; no downloads, audio or weights.

PRA runs the installed original methods only after their files match the locked
source hashes. GSS extracts one unchanged arithmetic function with NumPy in
place of CuPy; this is neither a GPU test nor execution of the GSS package.
"""
from __future__ import annotations

# Preserve the direct-file and module entries.
if __name__ == "__main__" and not __package__:
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[4]))

import argparse
import contextlib
import functools
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts

ROOT = Path(__file__).resolve().parents[4]
CURRENT_REPORT = ROOT / "codes/chapters/ch08/reports/separation_upstream_interfaces_current.json"
CODES = Path(__file__).resolve().parents[3]
SOURCES = {'asteroid': {'files': {'LICENSE': 'c12aebc7a4eeeec2e482414004fd5d68275d7608552031cd48f4088b403f902d',
                        'asteroid/masknn/norms.py': '80bc9d54d9bbae5b3a110516cb927b5fc9a8ff72e10a257acb6ed4569c1e605b',
                        'asteroid/models/conv_tasnet.py': '29e0decce888c967d22c83a69b1d75d784854f8d345b9f62581568beba37eb3f'},
              'revision': 'fce87469132760fbab41c20616ea0f0e079aad38'},
 'espnet': {'files': {'LICENSE': '4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5',
                      'espnet2/enh/separator/tfgridnet_separator.py': 'e08616b3e1964c7b0019320fff47110a4708fff44752173faff95aefed50e1f6'},
            'revision': 'be79590bb2ff26ffb01bc825c5f68cb9418b7f0d'},
 'gss': {'files': {'LICENSE': '5658d3e38fcd75f27608d9ba7ee83a2c1a04cc7a76334509907b41c87b92fc8b',
                   'gss/cacgmm/cacgmm.py': 'dcc46c325d71fe1861828d5330a919155e480f60a79273553e98deaed89b1e9b',
                   'gss/cacgmm/cacgmm_trainer.py': 'a3ecae7552203930392b6b5a75a26608aa238e3ae8b7ced2fcd30021f4c50c11',
                   'gss/cacgmm/utils.py': '065fea8ce2271a7ccb04f58faddb34ab6d94531ccaecd9b908a0f977a48f615e',
                   'gss/core/enhancer.py': '2e4ce14a9bd82e15471fa38e702aceb126db381d6642797f77b29bdaba208099',
                   'gss/core/gss.py': '22a56720db745c65be9cedf7c25980f4a3eac12a84ad1729e710071dd2f8a669'},
         'revision': '10fad18cae85e2e4342c77421abc70c9c5da23ed'},
 'pyroomacoustics': {'files': {'LICENSE': '0922c9a0c1f5bb35a1e0df7b56954864e27cfc625fc3b041acde0707ce797e46',
                               'pyroomacoustics/bss/auxiva.py': '61016f0b27b8b4f1cfecdddb99c3f11f038552695abb299d10b6855d3c24ad6a',
                               'pyroomacoustics/bss/common.py': 'd142a27bdbd80e8a4f0c31e78759c9884f6ed44c0609b89e38902d86e50ee11c',
                               'pyroomacoustics/bss/fastmnmf.py': 'a13c253bc971d4dab7ae8f3d87ab7c309eb6b560b44e2b4c09b7018b50540518',
                               'pyroomacoustics/bss/fastmnmf2.py': '378ff43327cb7968e1ae08e0acf7b7099ad0267e11ae0559cf4d82a18b44093c',
                               'pyroomacoustics/bss/ilrma.py': '1bed68575f0bbfc899aade4f2f17f5210004b30ea7f0737b8acba3ccb5bd15b3',
                               'pyroomacoustics/bss/trinicon.py': '4202bb2b0834f982e246db37131c902fb078b533989512f81a6e162742b663fe'},
                     'revision': '0dd39f2614b7fc44b2cc63dbe7d60f4641068890'},
 'ssspy': {'files': {'LICENSE': '809ba860a2750092fd00a8ca4cb4dc459ae4f9f6e538f574ea52b8b1daae91b6',
                     'ssspy/algorithm/projection_back.py': '5cf34568d63971e296caafad84dd5ccf24cc09d220cb1e20569772ad972b17b6'},
           'revision': '38b9389e8b1914422561f1936d9b28d042d62d2c'}}
CONFIG = {
    "spectrum": {"seed": 8, "generator": "numpy.default_rng/PCG64",
                 "shape": [9, 3, 2], "axes": "time,frequency,channel",
                 "recipe": "standard_normal(shape)+1j*standard_normal(shape); X[:,:,1]*=3",
                 "dtype": "complex128"},
    "auxiva": {"n_src": 1, "init_eig": True, "n_iter": 0,
               "proj_back": False, "return_filters": True},
    "ilrma": {"legacy_numpy_seed": 7, "n_iter": 1, "n_components": 2,
              "proj_back": False, "return_filters": True},
    "fastmnmf": {"legacy_numpy_seed": 7, "n_src": 3, "n_iter": 0,
                 "n_components": 2, "mic_index": "all", "W0": "2*identity per frequency"},
    "trinicon": {"lengths": [16, 17, 19, 23], "input": "arange(2*S).reshape(2,S)*0.001",
                 "filter_length": 4, "block_length": 8, "n_blocks": 2,
                 "j_max": 0, "hop": 4, "default_filter_delay_samples": 2},
    "gss": {"weights": [0.2, 0.3, 0.5], "eps": 1e-10,
            "clip_log_pdf": [0., 0., 0.], "clip_activity": [False, False, True],
            "underflow_log_pdf": [1000., 0., 0.],
            "underflow_activity": [False, True, True]},
    "projection_back": {"separated_source_time": [[1,0,1],[0,1,1]],
                        "mixture_channel_time": [[1,2,3],[0,1,1]],
                        "filter_source_channel": [[1,-2],[0,1]],
                        "complex_output": "[1,j]", "complex_reference_gain": "2+j"},
    "scope": "dimensionless controlled arrays; no separation-quality or timing estimate",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_sha256():
    return hashlib.sha256(json.dumps({"sources": SOURCES, "config": CONFIG},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def complex_record(x):
    return {"real": np.asarray(x).real.tolist(), "imag": np.asarray(x).imag.tolist()}


def verify_sources(root):
    root = contracts.validate_parent_chain(root)
    identities = {}
    for project, spec in SOURCES.items():
        names = list(spec['files'])
        if project == 'ssspy':
            names.extend(name for name in contracts.git(root/project, 'ls-files', '--', '*.py').splitlines()
                         if (root/project/name).is_file())
        identity = contracts.verify_project(project, root/project, names)
        for name, digest in spec['files'].items():
            if identity['used_files'][name]['sha256'] != digest:
                raise ValueError('source hash mismatch: '+project+'/'+name)
        identities[project] = identity
    return identities


@contextlib.contextmanager
def source_imports(package):
    """Read target-package Python sources directly; leave existing pyc untouched."""
    class OriginalSourceLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source = contracts.ordinary_file(self.path).read_bytes()
            return compile(source, self.path, 'exec', dont_inherit=True)

        def exec_module(self, module):
            super().exec_module(module)
            module.__micarray_original_source_sha256__ = sha256(self.path)

    class OriginalSourceFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != package and not fullname.startswith(package+'.'):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                spec.loader = OriginalSourceLoader(fullname, spec.origin)
            return spec

    finder = OriginalSourceFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def original_source_imports(package):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with source_imports(package):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def package_preflight(package, checkout, expected_root=None):
    """Read every discoverable installed Python file before importing the package.

    Python source identity is separate from compiled extensions, third-party
    dependencies and method execution. It is not a proof against concurrent edits.
    """
    spec = importlib.machinery.PathFinder.find_spec(package, sys.path)
    if spec is None or not spec.origin or not spec.submodule_search_locations:
        raise ModuleNotFoundError('No original package available: '+package)
    root = contracts.validate_parent_chain(Path(spec.origin).parent)
    if expected_root is not None and root != Path(expected_root).resolve():
        raise ValueError('Imported package root differs from fixed checkout')
    for name, module in tuple(sys.modules.items()):
        if name == package or name.startswith(package+'.'):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise ValueError('Preloaded package module differs from checked root')
            if filename and filename.endswith('.py') and getattr(module, '__micarray_original_source_sha256__', None) != sha256(contracts.ordinary_file(filename)):
                raise ValueError('Preloaded Python module has no verified direct-source import identity')
    files = {}
    revision = contracts.git(checkout, 'rev-parse', 'HEAD')
    for path in sorted(root.rglob('*.py')):
        path = contracts.ordinary_file(path)
        relative = package+'/'+str(path.relative_to(root))
        blob = contracts.git(checkout, 'rev-parse', revision+':'+relative)
        actual = contracts.git(checkout, 'hash-object', '--no-filters', '--', str(path))
        if actual != blob:
            raise ValueError('Installed original Python blob differs: '+relative)
        files[relative] = {'path': str(path), 'sha256': sha256(path), 'git_blob': blob}
    cache_files = {str(p.relative_to(root)): sha256(contracts.ordinary_file(p)) for p in sorted(root.rglob('*.pyc'))}
    return {'package_root': str(root), 'python_files': files,
            'existing_pyc_sha256': cache_files, 'existing_pyc_execution': 'bypassed by direct-source loader',
            'scope': 'all discovered package Python files, before import; compiled extensions and external dependencies are not fixed-source Python blobs'}


def package_unchanged(record):
    root = Path(record['package_root'])
    if sorted(str(p.relative_to(root)) for p in root.rglob('*.py')) != sorted(
            str(Path(r['path']).relative_to(root)) for r in record['python_files'].values()):
        raise ValueError('Package Python membership changed')
    for row in record['python_files'].values():
        if sha256(contracts.ordinary_file(row['path'])) != row['sha256']:
            raise ValueError('Package Python changed during execution')
    after_cache = {str(p.relative_to(root)): sha256(contracts.ordinary_file(p)) for p in sorted(root.rglob('*.pyc'))}
    if after_cache != record['existing_pyc_sha256']:
        raise ValueError('Existing package bytecode cache changed')
    record['unchanged_after'] = True
    return record


@original_source_imports("pyroomacoustics")
def run_pra(source_root):
    sys.dont_write_bytecode = True
    installed = package_preflight("pyroomacoustics", source_root/"pyroomacoustics")
    import pyroomacoustics as pra
    import scipy
    package_root = Path(pra.__file__).parent
    for name, digest in SOURCES["pyroomacoustics"]["files"].items():
        if name.startswith("pyroomacoustics/"):
            if sha256(package_root / name.removeprefix("pyroomacoustics/")) != digest:
                raise ValueError("installed PRA differs: " + name)
    rng = np.random.default_rng(CONFIG["spectrum"]["seed"])
    shape = CONFIG["spectrum"]["shape"]
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x[:, :, 1] *= 3
    try:
        pra.bss.auxiva(x.copy(), **CONFIG["auxiva"])
    except Exception as exc:
        auxiva = {"exception_type": type(exc).__name__, "message": str(exc)}
    else:
        auxiva = {"exception_type": None}

    # Observe the original frame immediately before its final normalization.
    # sys.settrace reads state; it changes neither the method nor its arrays.
    source = source_root / "pyroomacoustics/pyroomacoustics/bss/ilrma.py"
    tree = ast.parse(source.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and any(isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Subscript)
                        and isinstance(a.targets[0].value, ast.Name)
                        and a.targets[0].value.id == "lambda_aux" for a in n.body))
    captured = {}
    def observer(frame, event, arg):
        if frame.f_code.co_name == "ilrma":
            if event == "line" and frame.f_lineno == loop.lineno and not captured:
                captured["before"] = frame.f_locals["W"].copy()
            if event == "return":
                captured["scale"] = frame.f_locals["lambda_aux"].copy()
        return observer
    params = dict(CONFIG["ilrma"])
    np.random.seed(params.pop("legacy_numpy_seed"))
    old_trace = sys.gettrace()
    try:
        sys.settrace(observer)
        y, w = pra.bss.ilrma(x.copy(), **params)
    finally:
        sys.settrace(old_trace)
    before = captured["before"]
    scale = captured["scale"]
    wx = np.einsum("fsc,tfc->tfs", w, x)
    row_normalized = np.einsum("fsc,tfc->tfs", before * scale[None, :, None], x)
    ilrma = {
        "returned_output": complex_record(y), "returned_filters": complex_record(w),
        "filters_before_normalization": complex_record(before), "normalization_scale": scale.tolist(),
        "max_output_filter_difference": float(np.max(np.abs(y-wx))),
        "max_output_pre_normalization_difference": float(np.max(np.abs(y-np.einsum('fsc,tfc->tfs', before, x)))),
        "max_returned_filter_column_scaling_difference": float(np.max(np.abs(w-before*scale[None,None,:]))),
        "independent_row_normalization_mean_power": np.mean(abs(row_normalized)**2, axis=(0,1)).tolist(),
        "finite": bool(np.isfinite(y).all() and np.isfinite(w).all()),
    }
    fast = {}
    for name in ("fastmnmf", "fastmnmf2"):
        params = dict(CONFIG["fastmnmf"])
        np.random.seed(params.pop("legacy_numpy_seed"))
        params.pop("W0")
        initial = np.tile(2*np.eye(2), (3,1,1)).astype(complex)
        saved = initial.copy()
        z = getattr(pra.bss, name)(x.copy(), W0=initial, **params)
        fast[name] = {"output_shape": list(z.shape), "output": complex_record(z),
                      "caller_W0_after": complex_record(initial),
                      "max_caller_W0_change": float(np.max(np.abs(initial-saved))),
                      "output_axes": "microphone,time,frequency,source",
                      "max_source_image_sum_error": float(np.max(np.abs(z.sum(axis=-1)-x.transpose(2,0,1)))),
                      "finite": bool(np.isfinite(z).all())}
    tails = []
    for length in CONFIG["trinicon"]["lengths"]:
        wave = np.arange(2*length).reshape(2,length)*.001
        z = pra.bss.trinicon(wave, filter_length=4, block_length=8, n_blocks=2, j_max=0)
        expected = np.pad(wave, ((0,0),(2,0)))[:, :length]
        mismatch = np.flatnonzero(np.any(np.abs(z[:,:length]-expected)>1e-14, axis=0))
        tails.append({"input_length": length, "output": z.tolist(),
                      "output_length": z.shape[1], "mismatched_original_sample_indices": mismatch.tolist(),
                      "expected_same_length_delayed_input": expected.tolist()})
    return {"execution_kind": "original_installed_methods_matching_locked_file_hashes",
            "installed_python_identity": package_unchanged(installed),
            "python_loading": "direct checked source; existing package pyc bypassed and preserved",
            "package_version": pra.__version__, "package_directory": str(package_root),
            "scipy_version": scipy.__version__, "input": complex_record(x),
            "auxiva_eigen_initialization": auxiva, "ilrma": ilrma,
            "fastmnmf": fast, "trinicon": tails}


def extract_function(text, name, namespace):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<original-function-numpy-backend>", "exec"), namespace)
    return namespace[name]


def load_original_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(contracts.ordinary_file(path).read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def run_projection_back(root):
    sss = load_original_module(root / "ssspy/ssspy/algorithm/projection_back.py", "sss_projection_audit")
    pra = load_original_module(root / "pyroomacoustics/pyroomacoustics/bss/common.py", "pra_projection_audit")
    cfg = CONFIG["projection_back"]
    y = np.array(cfg["separated_source_time"], dtype=complex)[:,None,:]
    x = np.array(cfg["mixture_channel_time"], dtype=complex)[:,None,:]
    w = np.array(cfg["filter_source_channel"], dtype=complex)[None,:,:]
    scaled_w = sss.projection_back(w, reference_id=0)
    scaled_y_from_w = (scaled_w @ x.transpose(1,0,2)).transpose(1,0,2)
    joint = sss.projection_back(y, reference=x, reference_id=0)
    coefficient = pra.projection_back(y.transpose(2,1,0), x[0].T)
    independent = y * coefficient.conj().T[:,:,None]
    yc = np.array([1,1j])[:,None,None]
    zc = pra.projection_back(yc, (2+1j)*yc[:,:,0])
    return {"execution_kind": "hash_checked_original_modules_direct_function_calls",
            "ssspy_scaled_filter": complex_record(scaled_w),
            "ssspy_filter_route_output": complex_record(scaled_y_from_w),
            "ssspy_joint_data_route_output": complex_record(joint),
            "pra_returned_conjugate_coefficients": complex_record(coefficient),
            "pra_applied_output": complex_record(independent),
            "pra_complex_returned_coefficient": complex_record(zc),
            "pra_complex_applied_coefficient": complex_record(zc.conj()),
            "full_bss_estimation_executed_for_this_case": False}


def run_gss_arithmetic(root):
    fn = extract_function((root / "gss/gss/cacgmm/utils.py").read_text(),
                          "log_pdf_to_affiliation", {"cp": np})
    cfg = CONFIG["gss"]
    weight = np.asarray(cfg["weights"])[:,None]
    results = {}
    for case, eps in (("clip", cfg["eps"]), ("underflow", 0.)):
        result = fn(weight, np.asarray(cfg[case+"_log_pdf"])[:,None],
                    np.asarray(cfg[case+"_activity"], dtype=bool)[:,None], eps)
        results[case] = {"affiliation": result[:,0].tolist(), "sum": float(result.sum())}
    return {"execution_kind": "unchanged_function_ast_with_numpy_replacing_cupy",
            "cupy_or_gss_package_executed": False, "results": results,
            "independent_exact_active_posterior": {"clip": [0.,0.,1.], "underflow": [0.,.375,.625]}}


def static_facts(root):
    tree = ast.parse((root / "gss/gss/core/gss.py").read_text())
    calls = [{"line": n.lineno, "keywords": [k.arg for k in n.keywords]}
             for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "predict"]
    return {"execution_kind": "static_source_read_only",
            "gss_final_predict_calls": sorted(calls, key=lambda x:x["line"]),
            "gss_default_post_iterations": 1,
            "gss_log_likelihood": "logsumexp of component log_pdf; no weights or activity argument",
            "gss_training_clipping": "epsilon 1e-10, no renormalization; final predict defaults epsilon zero",
            "asteroid_norm_aliases": {"cLN": "ChanLN", "cgLN": "CumLN"},
            "tfgridnet_input_normalization": "global sample std division without epsilon",
            "neural_frameworks_executed": False}


def run_audit(source_root=CODES / 'chapters/ch00/upstream/_downloads'):
    sys.dont_write_bytecode = True
    identities = verify_sources(source_root)
    report = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
              "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
              "actual_dependency_sha256": contracts.dependencies(Path(__file__)),
              "source_identities": identities,
              "lock_sha256": identities['gss']['lock_sha256'],
              "source_status_sha256": identities['gss']['status_sha256'],
              "sources": SOURCES, "config": CONFIG,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "bytecode_writes": False},
              "pra": run_pra(source_root), "gss_arithmetic": run_gss_arithmetic(source_root),
              "projection_back": run_projection_back(source_root),
              "static": static_facts(source_root),
              "not_executed": ["GPU GSS pipeline", "neural forward or training", "weights",
                               "speech data", "ASR", "separation quality", "hardware timing"]}
    for identity in identities.values():
        contracts.check_unchanged(identity)
    if contracts.dependencies(Path(__file__)) != report['actual_dependency_sha256']:
        raise ValueError('Current tool dependencies changed during audit')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=CODES / "chapters/ch00/upstream/_downloads")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    target = contracts.report_target(args.report, CURRENT_REPORT, protected=(args.source_root,)) if args.report else None
    report = run_audit(args.source_root)
    if target:
        contracts.write_report(target, report, CURRENT_REPORT, protected=(args.source_root,))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
