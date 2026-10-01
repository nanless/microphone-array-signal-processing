"""Read-only fixed pb_bss stable_solve contracts and separate NumPy comparisons.

Load the complete original solve.py, without AST extraction or patching. Only
--report writes; no full pb_bss import, beamformer, model or hardware is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
REVISION = '10acc347fc9ea21e3d312806a0bd751d0d0af183'
ORIGIN = 'https://github.com/fgnt/pb_bss.git'
FILES = {
    'LICENSE': '48241e1eae6ab4c15c5718992ca60d3d15961212a4e324e09e5dbfbc79b78214',
    'pb_bss/math/solve.py': 'be440ee6f252e42a6ae5d7b6f3cc7205da5d4c29bcb01cae7885ab9f8c6a53ed',
}
ATOL = 1e-14


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_loads(data):
    def reject(value):
        raise ValueError('non-finite JSON constant: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('non-finite JSON number: ' + value)
        return number
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(data, parse_constant=reject, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def ordinary_path(path, *, directory=False, allow_missing=False):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('lexical parent traversal is forbidden')
    path = path.absolute()
    for current in reversed((path, *path.parents)):
        if current.is_symlink():
            raise ValueError('symbolic link is forbidden: ' + str(current))
        if current != path and not current.is_dir():
            raise ValueError('parent is not a directory: ' + str(current))
    if path.exists():
        mode = path.stat().st_mode
        if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
            raise ValueError('unexpected file type: ' + str(path))
    elif not allow_missing:
        raise ValueError('required path is absent: ' + str(path))
    return path


def report_target(path, cache=CACHE):
    target = ordinary_path(path, allow_missing=True)
    cache_path = ordinary_path(cache, directory=True)
    if target.is_relative_to(CACHE.absolute()) or target.is_relative_to(cache_path):
        raise ValueError('report must not be inside upstream cache')
    return target


def verify_sources(cache=CACHE):
    checkout = ordinary_path(Path(cache) / 'pb_bss', directory=True)
    lock_bytes = ordinary_path(LOCK).read_bytes()
    projects = strict_loads(lock_bytes)['projects']
    matching = [p for p in projects if p['id'] == 'pb_bss']
    if len(matching) != 1:
        raise ValueError('pb_bss lock entry must be unique')
    entry = matching[0]
    if (entry['revision'] != REVISION or entry['url'] != ORIGIN or
            entry['license'] != 'MIT' or any(p not in entry['entrypoints'] for p in FILES)):
        raise ValueError('fixed pb_bss lock identity changed')
    # An outer GIT_DIR/GIT_WORK_TREE/config injection cannot redirect this audit.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    def git(*args, binary=False):
        result = subprocess.check_output(['git', '-C', str(checkout), *args], env=env,
                                         stderr=subprocess.PIPE, timeout=20)
        return result if binary else result.decode().strip()
    if git('rev-parse', 'HEAD') != REVISION or git('remote', 'get-url', 'origin') != ORIGIN:
        raise ValueError('upstream HEAD or origin changed')
    if Path(git('rev-parse', '--show-toplevel')) != checkout:
        raise ValueError('not an independent checkout')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('upstream worktree is not clean')
    sources = []
    for relative, expected in FILES.items():
        content = ordinary_path(checkout / relative).read_bytes()
        if content != git('show', REVISION + ':' + relative, binary=True) or digest(content) != expected:
            raise ValueError('source digest/blob mismatch: ' + relative)
        sources.append({'path': relative, 'sha256': digest(content),
                        'git_blob': git('rev-parse', REVISION + ':' + relative),
                        'role': 'complete original Python module' if relative.endswith('.py') else 'license'})
    return {'checkout': str(checkout), 'revision': REVISION, 'origin': ORIGIN,
            'source_lock_sha256': digest(lock_bytes), 'source_lock_project_count': len(projects),
            'sources': sources, 'clean': True}


def load_original(checkout):
    path = ordinary_path(Path(checkout) / 'pb_bss/math/solve.py')
    spec = importlib.util.spec_from_file_location('_masp_original_pb_bss_solve', path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def compare(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise ValueError('unexpected shape or non-finite solver result')
    return bool(np.allclose(actual, expected, rtol=0, atol=ATOL))


def solver_cases(module):
    tenth, fifth = float(Fraction(1, 10)), float(Fraction(1, 5))
    expected = [[tenth, fifth], [tenth, fifth]]
    definitions = [
        ('singular_float_rhs', [[1., 1.], [2., 2.]], np.eye(2), expected, True),
        ('singular_integer_rhs', [[1., 1.], [2., 2.]], np.eye(2, dtype=np.int64), expected, False),
        ('nonsingular_integer_rhs', [[1., 0.], [0., 1.]], np.eye(2, dtype=np.int64), [[1., 0.], [0., 1.]], True),
        ('psd_nullspace_ls', [[0., 0.], [0., 1.]], np.ones((2, 1)), [[0.], [1.]], True),
    ]
    rows = []
    for name, matrix, rhs, oracle, should_match in definitions:
        a = np.array(matrix)
        a_before, rhs_before = a.copy(), rhs.copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            output = module.stable_solve(a, rhs)
        matched = compare(output, oracle)
        if matched != should_match:
            raise ValueError('fixed original behavior changed: ' + name)
        if not np.array_equal(a, a_before) or not np.array_equal(rhs, rhs_before):
            raise ValueError('input changed: ' + name)
        row = {'name': name, 'execution': 'original stable_solve', 'matrix': a.tolist(),
               'rhs': rhs.tolist(), 'matrix_dtype': str(a.dtype), 'rhs_dtype': str(rhs.dtype),
               'output_dtype': str(output.dtype), 'output': output.tolist(), 'independent_expected': oracle,
               'matched_expected': matched,
               'classification': 'matched_independent_expected' if matched else 'observed_integer_fallback_truncation',
               'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in caught],
               'error': None, 'inputs_unchanged': True}
        if not should_match:
            if not np.array_equal(output, np.zeros((2, 2), dtype=np.int64)):
                raise ValueError('integer fallback no longer produces the recorded zero result')
            row['cause_static'] = 'original fallback allocates zeros_like(B), then assigns floating lstsq results'
        if name == 'psd_nullspace_ls':
            row['independent_mvdr_model'] = {'covariance': matrix, 'steering': [1, 1],
                'constraint': 'w^H a = 1', 'optimal_weight': [1, 0], 'optimal_noise_power': 0,
                'normalized_ls_weight': [0, 1], 'normalized_ls_noise_power': 1,
                'derivation': 'noise power = |w2|^2; w1+w2=1 admits w2=0',
                'execution': 'hand-derived optimization only; no upstream beamformer called'}
        rows.append(row)
    return rows


def numpy_cases():
    epsilon = 1e-8
    a, b = np.diag([1., epsilon]), np.array([1., epsilon])
    rows = []
    for rcond, expected, rank, residual in ((1e-6, [1., 0.], 1, epsilon),
                                            (1e-10, [1., 1.], 2, 0.)):
        solution, returned, actual_rank, singular = np.linalg.lstsq(a, b, rcond=rcond)
        actual_residual = float(np.linalg.norm(a @ solution - b))
        if (not compare(solution, expected) or actual_rank != rank or
                returned.size != 0 or abs(actual_residual - residual) > ATOL):
            raise ValueError('NumPy result disagrees with diagonal closed-form oracle')
        rows.append({'name': 'numpy_rcond_' + str(rcond), 'execution': 'independent NumPy comparison; not pb_bss parameter',
                     'matrix': a.tolist(), 'rhs': b.tolist(), 'dtype': str(a.dtype), 'rcond': rcond,
                     'cutoff': rcond, 'singular_values': singular.tolist(), 'rank': int(actual_rank),
                     'output': solution.tolist(), 'independent_expected': expected,
                     'returned_residual_array': returned.tolist(), 'explicit_residual_norm': actual_residual,
                     'expected_residual_norm': residual, 'empty_residual_reason': 'M=N, so M<=N in this API',
                     'classification': 'matched_independent_expected'})
    return rows


def run_audit(cache=CACHE):
    before = verify_sources(cache)
    original = load_original(before['checkout'])
    cases = solver_cases(original)
    comparisons = numpy_cases()
    after = verify_sources(cache)
    if before != after:
        raise ValueError('upstream or lock changed during audit')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'tool_sha256': digest(Path(__file__).read_bytes()), **before,
              'before_clean': True, 'after_clean': True,
              'license': {'name': 'MIT', 'path': 'LICENSE', 'sha256': FILES['LICENSE']},
              'execution': {'module': 'pb_bss/math/solve.py', 'original_functions_called': ['stable_solve'],
                            'scope': 'complete unmodified module loaded; one function called',
                            'source_patch': False, 'ast_extraction': False, 'substitutes': [],
                            'not_executed': ['full pb_bss package', '_lstsq helper', 'beamformers', 'models', 'audio', 'hardware']},
              'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                              'numpy_linalg_wrapper_sha256': digest(Path(inspect.getsourcefile(
                                  inspect.unwrap(np.linalg.lstsq))).read_bytes())},
              'original_lstsq_default': {'argument': 'rcond omitted by original stable_solve',
                                        'runtime_default': 'NumPy 2.x eps * max(M,N)',
                                        'float64_2_by_2_ratio': float(np.finfo(float).eps * 2)},
              'comparison_tolerance': {'absolute': ATOL, 'relative': 0},
              'oracle': 'independent Fraction 1/10,1/5 and diagonal/constraint closed forms',
              'cases': cases, 'numpy_comparisons': comparisons,
              'counts': {'original_function_cases': len(cases), 'independent_numpy_cases': len(comparisons),
                         'original_matches': 3, 'original_known_dtype_failures': 1}}
    strict_loads(json.dumps(report, allow_nan=False))
    return report


def write_report(path, report, cache=CACHE):
    target = report_target(path, cache)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    strict_loads(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.solver-report-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        report_target(target, cache)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.report is not None:
        report_target(args.report, args.cache)
    report = run_audit(args.cache)
    if args.report is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        write_report(args.report, report, args.cache)


if __name__ == '__main__':
    main()
