"""Fixed upstream diagnostics, without downloads, patches or dependency installs.

pb_bss: execute one unmodified function AST, not the installed package.
pyroomacoustics: call the installed package after matching its source bytes.
Ordinary tests inspect the saved report and independent analytic fixtures only.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import traceback
import warnings

import numpy as np
from codes.chapters.ch04.core import upstream_contracts as contracts

ROOT = Path(__file__).resolve().parents[3]
CURRENT_REPORT = ROOT / 'chapters/ch05/reports/beamformer_reference_current.json'
SOURCES = {
    "pb_bss": {
        "revision": "10acc347fc9ea21e3d312806a0bd751d0d0af183",
        "files": {
            "pb_bss/extraction/beamformer.py": "8bc6700fbe07b50caeb640503bbf827798c883a3fc12b9790f93ab2ee9453e53",
            "LICENSE": "48241e1eae6ab4c15c5718992ca60d3d15961212a4e324e09e5dbfbc79b78214",
        },
    },
    "pyroomacoustics": {
        "revision": "0dd39f2614b7fc44b2cc63dbe7d60f4641068890",
        "files": {
            "pyroomacoustics/beamforming.py": "04142595a0fd0b76404d92dbe2aaa93613b4037a774e309595d3c7792a492387",
            "pyroomacoustics/soundsource.py": "1e367bc85867ad7f94dad47fdf283e3d5282d83533e48ddfab906c0195a35f4c",
            "LICENSE": "0922c9a0c1f5bb35a1e0df7b56954864e27cfc625fc3b041acde0707ce797e46",
        },
    },
}
CONFIG = {
    "pb_bss": {
        "function": "get_mvdr_vector_merl",
        "shape": "(frequency=1, channels=2, channels=2)",
        "target_diagonal_cases": [[1.0, 4.0], [4.0, 1.0]],
        "noise_covariance": [[1.0, 0.0], [0.0, 1.0]],
        "matrix_type": "exact diagonal PSDs; no audio, snapshots or randomness",
    },
    "pyroomacoustics": {
        "method": "Beamformer.rake_distortionless_filters",
        "microphone_positions_m": [[0.0, 0.05], [0.0, 0.0]],
        "sample_rate_hz": 8000, "fft_length": 64, "filter_length": 32,
        "source_position_m": [1.0, 1.0],
        "interferer_position_m": [-1.0, 1.0],
        "noise_covariance": "identity of dimension 64 (channels * filter_length)",
        "delay_s": 0.001, "epsilon": 0.005,
        "sound_speed_m_s": 343.0,
        "randomness": "none; analytic direct-path SoundSource objects, no room simulation",
    },
}


def digest(path: Path) -> str:
    return contracts.sha(path)


def configuration_sha256() -> str:
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source(path: Path, specification: dict) -> dict:
    project = next(name for name, spec in SOURCES.items() if spec == specification)
    identity = contracts.verify_project(project, path, specification['files'], all_python=True)
    for name, expected in specification["files"].items():
        if identity['used_files'][name]['sha256'] != expected:
            raise ValueError("upstream source hash mismatch: " + name)
    return identity


def independent_diagonal_reference(diagonal: list[float]) -> dict:
    """Scalar analytic fixture for Rn=I, Rs=diag(p0,p1), p_i>0.

    Candidate c has weight p_c/(p0+p1) at c. Its output SNR equals
    p_c; these expressions do not use the upstream tensor contractions.
    """
    if len(diagonal) != 2 or any(not np.isfinite(p) or p <= 0 for p in diagonal):
        raise ValueError("two finite positive powers required")
    selected = 0 if diagonal[0] >= diagonal[1] else 1
    weights = [0.0, 0.0]
    weights[selected] = diagonal[selected] / sum(diagonal)
    return {"candidate_output_snr_linear": diagonal.copy(),
            "selected_channel": selected, "weights": weights}


def run_pb_bss(source: Path) -> dict:
    """Probe normal import, then explicitly execute only the original AST node."""
    if any(n == "pb_bss" or n.startswith("pb_bss.") for n in sys.modules):
        raise RuntimeError("run in a fresh process without pb_bss imported")
    sys.path.insert(0, str(source))
    try:
        try:
            imported = importlib.import_module("pb_bss.extraction.beamformer")
            if Path(imported.__file__).resolve() != source / "pb_bss/extraction/beamformer.py":
                raise RuntimeError("unexpected pb_bss import path")
            package_probe = {"status": "imported_only", "algorithm_called": False}
        except ImportError as exc:
            package_probe = {"status": "failed", "exception_type": type(exc).__name__,
                             "message": str(exc), "algorithm_called": False}
    finally:
        sys.path.pop(0)
    filename = source / "pb_bss/extraction/beamformer.py"
    text = filename.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == CONFIG["pb_bss"]["function"])
    # Exact function node, without edits or execution of the module imports.
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), "exec"), namespace)
    rows = []
    for diagonal in CONFIG["pb_bss"]["target_diagonal_cases"]:
        actual = namespace[function.name](np.diag(diagonal)[None], np.eye(2)[None])
        expected = independent_diagonal_reference(diagonal)
        rows.append({"target_diagonal": diagonal, "actual_weights": actual.tolist(),
                     "analytic_reference": expected,
                     "reference_selection_correct": bool(np.allclose(actual[0], expected["weights"],
                                                                      atol=1e-14, rtol=0))})
    return {"execution_kind": "unmodified_function_ast_extraction",
            "package_import_probe": package_probe, "upstream_function_modified": False,
            "function_sha256": hashlib.sha256(ast.get_source_segment(text, function).encode()).hexdigest(),
            "function_first_line": function.lineno, "function_last_line": function.end_lineno,
            "results": rows,
            "status": "failed_reference_selection" if [r["reference_selection_correct"] for r in rows] == [False, True]
                      else "unexpected_diagnostic_result"}


def run_pra(source: Path) -> dict:
    if any(n == 'pyroomacoustics' or n.startswith('pyroomacoustics.') for n in sys.modules):
        raise RuntimeError('run in a fresh process without pyroomacoustics imported')
    specification = importlib.util.find_spec('pyroomacoustics')
    if specification is None or not specification.submodule_search_locations or len(specification.submodule_search_locations) != 1:
        raise ImportError('one ordinary installed pyroomacoustics package required')
    installed = contracts.validate_parent_chain(Path(next(iter(specification.submodule_search_locations))))
    installed_hashes = {}
    # Check every installed original Python source against its fixed Git blob,
    # including modules imported by the package initialization.
    installed_python = {}
    for path in sorted(installed.rglob('*.py')):
        relative = 'pyroomacoustics/' + str(path.relative_to(installed))
        original = contracts.ordinary_file(source / relative)
        if (digest(path) != contracts.sha(original)
                or contracts.git(source, 'hash-object', '--no-filters', '--', str(original))
                != contracts.git(source, 'rev-parse', 'HEAD:' + relative)):
            raise ValueError('installed original Python source differs: ' + relative)
        installed_python[relative] = digest(path)
    for name in ("beamforming.py", "soundsource.py"):
        value = digest(installed / name)
        if value != SOURCES["pyroomacoustics"]["files"]["pyroomacoustics/" + name]:
            raise ValueError("installed pyroomacoustics source differs: " + name)
        installed_hashes[name] = value
    binaries = {str(path.relative_to(installed)): {'sha256': digest(path), 'size_bytes': path.stat().st_size}
                for path in sorted(installed.rglob('*')) if path.suffix in ('.so', '.pyd', '.dylib')}
    # The top-level package has not executed before its Python-source preflight.
    # Binary hashes identify installed artifacts; they do not establish their build provenance.
    import pyroomacoustics as pra
    from pyroomacoustics.soundsource import SoundSource
    if pra.__version__ != "0.10.0" or Path(pra.__file__).parent != installed:
        raise ValueError("expected verified installed pyroomacoustics 0.10.0 path")
    c = CONFIG["pyroomacoustics"]
    if pra.constants.get("c") != c["sound_speed_m_s"]:
        raise ValueError("installed pyroomacoustics sound-speed default differs")
    beamformer = pra.Beamformer(np.array(c["microphone_positions_m"]), c["sample_rate_hz"],
                               N=c["fft_length"], Lg=c["filter_length"])
    try:
        beamformer.rake_distortionless_filters(
            SoundSource(np.array(c["source_position_m"])),
            SoundSource(np.array(c["interferer_position_m"])), np.eye(64),
            delay=c["delay_s"], epsilon=c["epsilon"])
    except Exception as exc:
        frames = [{"file": Path(t.filename).name, "line": t.lineno, "function": t.name,
                   "statement": t.line} for t in traceback.extract_tb(exc.__traceback__)]
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": frames}
    else:
        failure = None
    expected = bool(failure and failure["type"] == "TypeError"
                    and "slice indices" in failure["message"]
                    and failure["traceback"][-1]["line"] == 1375)
    for relative, expected_sha in installed_python.items():
        if digest(installed / Path(relative).relative_to('pyroomacoustics')) != expected_sha:
            raise ValueError('installed original source changed during method call')
    for relative, metadata in binaries.items():
        if digest(installed / relative) != metadata['sha256']:
            raise ValueError('installed binary artifact changed during method call')
    return {"execution_kind": "installed_original_package_method_call", "package_version": pra.__version__,
            "installed_path": str(installed), "installed_source_sha256": installed_hashes,
            "installed_python_preflight_sha256": installed_python,
            "installed_artifact_sha256": binaries,
            "installed_artifact_scope": "installed binary identities only; no fixed-Git binary or reproducible-build attestation",
            "installed_sources_unchanged_after": True,
            "exception": failure, "status": "failed_float_slice_index" if expected else "unexpected_diagnostic_result",
            "filters_computed": failure is None, "upstream_modified": False,
            "scope": "Only this method/input was called; no original Rake performance or other methods validated"}


def run_experiment(download_root: Path | None = None) -> dict:
    base = contracts.validate_parent_chain(download_root or ROOT / "chapters/ch00/upstream/_downloads")
    identities = {}
    for name, specification in SOURCES.items():
        identities[name] = verify_source(base / name, specification)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import scipy
        pb = run_pb_bss(base / "pb_bss")
        pra = run_pra(base / 'pyroomacoustics')
    finally:
        sys.dont_write_bytecode = bytecode
    expected = (pb["status"] == "failed_reference_selection"
                and pra["status"] == "failed_float_slice_index")
    for identity in identities.values():
        contracts.check_unchanged(identity)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "original_failures_preserved" if expected else "unexpected_diagnostic_result",
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "platform": platform.platform(),
                            "machine": platform.machine(), "executable": sys.executable},
            "configuration": CONFIG,
            "provenance": {"sources": SOURCES, "harness_sha256": digest(Path(__file__)),
                           "configuration_sha256": configuration_sha256(), "upstream_modified": False},
            "source_identities": identities,
            "report_source_sha256": contracts.dependencies(Path(__file__)),
            "pb_bss": pb, "pyroomacoustics": pra,
            "limitations": ["No upstream patch, package installation or network request",
                            "No enhanced waveform, perceptual score, benchmark timing or full upstream validation",
                            "A failing original method is not a source acquisition failure"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--report", dest='output', type=Path,
                        help='Explicit new current report or ordinary path outside the repository; default stdout')
    args = parser.parse_args()
    target = contracts.report_target(args.output, CURRENT_REPORT, (contracts.CACHE,)) if args.output else None
    report = run_experiment()
    if target is not None:
        contracts.write_report(target, report, CURRENT_REPORT, (contracts.CACHE,))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if "unexpected_diagnostic_result" in (report["pb_bss"]["status"], report["pyroomacoustics"]["status"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
