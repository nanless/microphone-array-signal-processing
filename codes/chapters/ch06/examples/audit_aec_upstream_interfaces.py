"""Bounded upstream AEC interface diagnostics; no downloads or source patches.

Pyaec and echocatzh: import fixed original Python files and call their functions.
DTLN: execute only its unchanged process_file AST with in-memory I/O and fake
interpreters. This tests file orchestration, never neural inference or audio quality.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts

CODES = Path(__file__).resolve().parents[3]
CURRENT_REPORT = CODES / "chapters/ch06/reports/aec_upstream_interfaces_current.json"
SOURCES = {
    "pyaec": {
        "revision": "5b9c02c57075d790b7df8652884618189d49bbc4",
        "files": {
            "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            "time_domain_adaptive_filters/rls.py": "f154c7cbfdca33c86efcf4ed6a9fc846a4330769ebc43a659680c16288467a16",
            "time_domain_adaptive_filters/kalman.py": "b9b678fb120f570cac61c909403f47fba5107a43718dcb7db86cf4f3ad9df6f0",
            "frequency_domain_adaptive_filters/fdkf.py": "8f8520e126d8efded74cbec9003a4d5451eb21958840a8a2ab50f0ce1019f758",
            "frequency_domain_adaptive_filters/pfdkf.py": "970cb59273da20b611bbc1044572f01452a34ed2b53bdff61b9af3a2b6e41387",
        },
    },
    "echocatzh-pfdkf": {
        "revision": "7c8c86b5691966c330015d8e0960db0733b4844f",
        "files": {
            "LICENSE": "b6c53ea4c5cf363eeef563f592f0b7308ad0943627e7803e9b0df05bf592b5d5",
            "pfdkf.py": "43d7b53953f44b12310b0ec68f77a42c4dbad886ac57bdb5d7d1f7c280d64410",
        },
    },
    "dtln_aec": {
        "revision": "9d24e128b4f409db18227b8babb343016625921f",
        "files": {
            "LICENSE": "aa95acd8c8a7341bfcdb2823694dcbb27a2a4143e860bb52db1ff0f29c85e5a1",
            "run_aec.py": "99e4c179b8a2838bb0f257308e24a28a8565d1f93f40546bbdf585f4305dc317",
        },
    },
}
CONFIG = {
    "randomness": "none; dimensionless synthetic arrays, no recordings",
    "pyaec": {"x": [0.] * 6, "d": [1., 2., 3., 4., 5., 6.], "N": 2,
              "fdkf_M": 4, "fdkf_input_length": 8, "pfdkf_N": 2, "pfdkf_M": 4},
    "echocatzh": {"N": 1, "M": 2, "x": [1., 1.], "d": [1., 1.],
                  "update_called": False, "res_modes": [False, True]},
    "dtln": {"sample_rate_hz": 16000, "mic_samples": 768, "reference_samples": 640,
             "input_values": 0., "fake_stage1_mask": 1., "fake_stage2_constants": [-0.5, 0.5],
             "real_tensorflow_or_weights_loaded": False, "real_soundfile_write": False},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_sha256() -> str:
    return hashlib.sha256(json.dumps({"sources": SOURCES, "config": CONFIG},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source(path: Path, spec: dict) -> dict:
    """Verify fixed origin, original blobs and selection before execution."""
    names = [name for name, known in SOURCES.items() if known == spec]
    if len(names) != 1:
        raise ValueError("Expected a declared fixed AEC source specification")
    identity = contracts.verify_project(names[0], path, spec["files"])
    for name, expected in spec["files"].items():
        if identity["used_files"][name]["sha256"] != expected:
            raise ValueError("source hash mismatch: " + name)
    return identity


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("aec_audit_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pyaec(root: Path) -> dict:
    c = CONFIG["pyaec"]
    rows = {}
    for name in ("rls", "kalman"):
        module = load_original(root / "time_domain_adaptive_filters" / (name + ".py"))
        result = getattr(module, name)(np.array(c["x"]), np.array(c["d"]), N=c["N"])
        rows[name] = {"execution_kind": "original_source_function_call",
                      "actual_residual": result.tolist(), "input_samples": len(c["d"]),
                      "returned_samples": int(result.size),
                      "unprocessed_sample_indices": list(range(result.size, len(c["d"]))),
                      "analytic_zero_reference_residual": c["d"],
                      "status": "tail_not_processed" if result.size == 4 else "unexpected"}
    for name in ("fdkf", "pfdkf"):
        module = load_original(root / "frequency_domain_adaptive_filters" / (name + ".py"))
        try:
            if name == "fdkf":
                module.fdkf(np.zeros(c["fdkf_input_length"]), np.zeros(c["fdkf_input_length"]),
                            M=c["fdkf_M"])
            else:
                module.PFDKF(c["pfdkf_N"], c["pfdkf_M"])
        except Exception as exc:
            rows[name] = {"execution_kind": "original_source_function_call" if name == "fdkf"
                          else "original_source_constructor_call",
                          "exception_type": type(exc).__name__, "message": str(exc).splitlines()[0],
                          "status": "numpy_complex_removed" if isinstance(exc, AttributeError)
                          and "complex" in str(exc) else "unexpected_failure", "output_computed": False}
        else:
            rows[name] = {"status": "constructor_or_function_completed",
                          "scope": "No convergence or acoustic performance evaluated"}
    return rows


def run_echocatzh(root: Path) -> dict:
    module = load_original(root / "pfdkf.py")
    c = CONFIG["echocatzh"]
    rows = []
    for res in c["res_modes"]:
        state = module.PFDKF(c["N"], c["M"], res=res)
        error, echo = state.filt(np.array(c["x"]), np.array(c["d"]))
        rows.append({"res": res, "error": error.tolist(), "reported_echo": echo.tolist(),
                     "filter_coefficients_remain_zero": bool(np.all(state.H == 0)),
                     "update_called": False})
    return {"execution_kind": "original_source_method_call", "rows": rows,
            "analytic_reference": {"bare_linear_error": [1., 1.],
                "res_error": [0.5 + 1e-10 / 4 * (1 / (8 + 1e-10) + 1 / (4 + 1e-10))] * 2,
                "derivation": "Four-point X=E=[2,-1+i,0]; P=1, m=|E|^2/2. W=1-|X|^2/(2|X|^2+1e-10). Valid IFFT samples equal W0/2+W1/2."},
            "scope": "One initial block only; residual suppression precedes any learned path update"}


class MemoryAudio:
    """No filesystem I/O: collect the array which upstream would write."""
    def read(self, name):
        c = CONFIG["dtln"]
        size = c["mic_samples"] if name.endswith("mic.wav") else c["reference_samples"]
        return np.full(size, c["input_values"]), c["sample_rate_hz"]

    def write(self, name, values, sample_rate):
        self.result = np.array(values, copy=True)
        self.sample_rate = sample_rate


class FakeInterpreter:
    """An explicit fixture, with no trained model and no TensorFlow dependency."""
    def __init__(self, stage, constant):
        self.stage = stage
        self.constant = constant
        self.invocations = 0

    def get_input_details(self):
        return [{"index": 0, "shape": [1, 1, 257 if self.stage == 1 else 512]},
                {"index": 1, "shape": [1, 1, 2]}, {"index": 2, "shape": [1, 1, 512]}]

    def get_output_details(self):
        return [{"index": 0}, {"index": 1}]

    def set_tensor(self, index, values):
        pass

    def invoke(self):
        self.invocations += 1

    def get_tensor(self, index):
        if index == 1:
            return np.zeros((1, 1, 2), dtype=np.float32)
        if self.stage == 1:
            return np.ones((1, 1, 257), dtype=np.float32)
        return np.full((1, 1, 512), self.constant, dtype=np.float32)


def run_dtln_ast(root: Path) -> dict:
    path = root / "run_aec.py"
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_file")
    rows = []
    for value in CONFIG["dtln"]["fake_stage2_constants"]:
        audio = MemoryAudio()
        namespace = {"np": np, "sf": audio}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        first, second = FakeInterpreter(1, value), FakeInterpreter(2, value)
        namespace["process_file"](first, second, "fixture_mic.wav", "memory-only.wav")
        rows.append({"fake_stage2_constant": value, "output_samples": int(audio.result.size),
                     "minimum": float(audio.result.min()), "maximum": float(audio.result.max()),
                     "samples_outside_unit_range": int(np.count_nonzero(np.abs(audio.result) > 1)),
                     "stage_invocation_counts": [first.invocations, second.invocations]})
    return {"execution_kind": "unmodified_function_ast_with_fake_interpreters_and_memory_io",
            "function_first_line": node.lineno, "function_last_line": node.end_lineno,
            "function_sha256": hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(),
            "real_neural_inference": False, "real_audio_write": False, "rows": rows,
            "analytic_reference": "512/128=4 overlapping constant blocks: +/-0.5 yields +/-2. Positive maximum guard rescales +2 to .99, but leaves -2 unchanged.",
            "scope": "Orchestration and asymmetric output range guard only; not a model failure or measured PCM clipping"}


def run_audit(download_root: Path | None = None) -> dict:
    root = contracts.validate_parent_chain(
        download_root or CODES / "chapters/ch00/upstream/_downloads")
    identities = {name: verify_source(root / name, spec) for name, spec in SOURCES.items()}
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        results = {"pyaec": run_pyaec(root / "pyaec"),
                   "echocatzh": run_echocatzh(root / "echocatzh-pfdkf"),
                   "dtln": run_dtln_ast(root / "dtln_aec")}
    finally:
        sys.dont_write_bytecode = old
    for identity in identities.values():
        contracts.check_unchanged(identity)
    return {"schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "platform": platform.platform(), "executable": sys.executable},
            "script_sha256": sha256(Path(__file__)), "binding_sha256": binding_sha256(),
            "actual_dependency_sha256": contracts.dependencies(Path(__file__)),
            "sources": SOURCES, "config": CONFIG, "source_identities": identities,
            "source_verification": "fixed used original blobs and licenses verified before and after; complete selection recorded separately",
            "upstream_modified": False, "performance_evaluation": False,
            "write_boundary": "Only explicit safe current report or external ordinary report; finite preflight does not eliminate concurrent races",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    protected = (args.download_root or contracts.CACHE,)
    target = contracts.report_target(args.report, CURRENT_REPORT, protected) if args.report else None
    report_data = run_audit(args.download_root)
    report = json.dumps(report_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if target is not None:
        contracts.write_report(target, report_data, CURRENT_REPORT, protected)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
