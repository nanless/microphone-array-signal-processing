"""Read-only fixed pb_bss stable_solve contracts and separate NumPy comparisons.

Compile the complete verified solve.py bytes without cached bytecode, AST
extraction or patching. Default output is stdout. Only an explicit ordinary
--report writes the new current report or an external report; history is kept.
No full pb_bss import, beamformer, model or hardware is run. Finite pre/post
checks do not eliminate concurrent races or guarantee crash persistence.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from codes.chapters.ch00.io_contracts import (
    same_metadata, strict_json_loads, validate_parent_chain, write_json_report,
)
from codes.chapters.ch04.core import upstream_contracts as upstream

LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
STATUS = LOCK.with_name('SOURCE_STATUS.json')
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
HISTORICAL_REPORT = ROOT / 'codes/chapters/appendix_a/reports/upstream_solver_contracts.json'
CURRENT_REPORT = HISTORICAL_REPORT.with_name('upstream_solver_contracts_current.json')
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
    return strict_json_loads(data)


def ordinary_path(path, *, directory=False, allow_missing=False):
    path = validate_parent_chain(path)
    if not path.parent.is_dir():
        raise ValueError('parent is not an existing ordinary directory')
    if path.exists():
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError('unexpected file type: ' + str(path))
    elif not allow_missing:
        raise ValueError('required path is absent: ' + str(path))
    return path


def report_target(path, cache=CACHE):
    target = ordinary_path(path, allow_missing=True)
    cache_path = ordinary_path(cache, directory=True)
    return upstream.report_target(target, CURRENT_REPORT, protected=(
        CACHE, cache_path, LOCK, STATUS, HISTORICAL_REPORT,
        ROOT / 'codes/chapters/ch00/source_snapshots', ROOT / 'reviews'))


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
    identity = upstream.verify_project('pb_bss', checkout, relatives=tuple(FILES),
                                       lock_path=LOCK, status_path=STATUS)
    if identity['lock_sha256'] != digest(lock_bytes):
        raise ValueError('source lock changed during preflight')
    sources = []
    for relative, expected in FILES.items():
        record = identity['used_files'][relative]
        if record['sha256'] != expected:
            raise ValueError('source digest/blob mismatch: ' + relative)
        sources.append({'path': relative, 'sha256': record['sha256'],
                        'git_blob': record['git_blob'],
                        'role': 'complete original Python module' if relative.endswith('.py') else 'license'})
    return {'checkout': str(checkout), 'revision': REVISION, 'origin': ORIGIN,
            'source_lock_sha256': digest(lock_bytes), 'source_lock_project_count': len(projects),
            'source_status_sha256': identity['status_sha256'],
            'sources': sources, 'clean': True, 'source_identity': identity}


def load_original(checkout):
    path = ordinary_path(Path(checkout) / 'pb_bss/math/solve.py')
    payload = path.read_bytes()
    if digest(payload) != FILES['pb_bss/math/solve.py']:
        raise ValueError('source bytes changed before complete original module execution')
    spec = importlib.util.spec_from_file_location('_masp_original_pb_bss_solve', path)
    module = importlib.util.module_from_spec(spec)
    # SourceFileLoader.exec_module may read an existing valid .pyc even under
    # -B. Execute these exact verified full bytes; preserve module metadata and
    # ordinary original imports, without reading, deleting or updating caches.
    module.__cached__ = None
    exec(compile(payload, str(path), 'exec'), module.__dict__)
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


def complex_array(value):
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise ValueError('non-finite additional solver control')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'real': value.real.tolist(), 'imag': value.imag.tolist()}


def additional_solver_cases(module):
    """Four separate original calls; never alter the historical four-case set."""
    eye = np.eye(2)
    definitions = (
        ('vector_rhs', eye, np.ones(2), 'IndexError', None),
        ('rectangular_matrix', np.ones((3, 2)), np.ones((3, 1)), 'AssertionError', None),
        ('complex_single_real_rhs', 1j * eye, eye, None, -1j * eye),
        ('complex_batch_real_rhs', np.stack((1j * eye, np.zeros((2, 2)))),
         np.stack((eye, np.zeros((2, 2)))), None,
         np.stack((-1j * eye, np.zeros((2, 2))))),
    )
    rows = []
    for name, a, b, expected_error, expected in definitions:
        saved_a, saved_b = a.copy(), b.copy()
        output, error = None, None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                output = module.stable_solve(a, b)
            except (IndexError, AssertionError) as exception:
                error = {'type': type(exception).__name__, 'message': str(exception)}
        if (error['type'] if error else None) != expected_error:
            raise ValueError('fixed additional original behavior changed: ' + name)
        matched = None
        if expected_error is None:
            if output.shape != expected.shape or not np.all(np.isfinite(output)):
                raise ValueError('unexpected additional solver output: ' + name)
            matched = bool(np.allclose(output, expected, rtol=0, atol=ATOL))
            if name == 'complex_single_real_rhs':
                if not matched or output.dtype != np.dtype('complex128') or caught:
                    raise ValueError('complex direct solve behavior changed')
            else:
                if (matched or output.dtype != np.dtype('float64') or
                        not np.array_equal(output, np.zeros((2, 2, 2))) or
                        [w.category.__name__ for w in caught] != ['ComplexWarning'] * 2):
                    raise ValueError('complex batch fallback behavior changed')
        if not np.array_equal(a, saved_a) or not np.array_equal(b, saved_b):
            raise ValueError('additional original call changed inputs: ' + name)
        rows.append({'name': name, 'execution': 'original stable_solve; separate additional control',
                     'matrix': complex_array(a), 'rhs': complex_array(b),
                     'output': None if output is None else complex_array(output),
                     'independent_expected': None if expected is None else complex_array(expected),
                     'expected_error_type': expected_error, 'error': error,
                     'matched_expected': matched, 'inputs_unchanged': True,
                     'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in caught],
                     'classification': ('observed_shape_exception' if expected_error else
                                        'observed_complex_batch_fallback_discard' if not matched else
                                        'matched_independent_expected'),
                     'scope': 'generic linear-system control; matrix is not claimed to be a covariance or beamformer'})
    return rows


def actual_dependencies():
    """Four real local source files; not the installed dependency closure."""
    return upstream.dependencies(__file__)


def numpy_identities():
    paths = {'entry': np.__file__,
             'linalg_lstsq_wrapper': inspect.getsourcefile(inspect.unwrap(np.linalg.lstsq))}
    return {name: {'path': str(ordinary_path(path)),
                   'sha256': digest(ordinary_path(path).read_bytes()), 'version': np.__version__,
                   'scope': 'actual imported entry or Python wrapper file only; not whole package or native LAPACK'}
            for name, path in paths.items()}


def run_audit(cache=CACHE):
    direct_before, numpy_before = actual_dependencies(), numpy_identities()
    before = verify_sources(cache)
    source_before = copy.deepcopy(before['source_identity'])
    original = load_original(before['checkout'])
    cases = solver_cases(original)
    comparisons = numpy_cases()
    additional = additional_solver_cases(original)
    after = upstream.check_unchanged(before['source_identity'])
    direct_after, numpy_after = actual_dependencies(), numpy_identities()
    if (not same_metadata(direct_before, direct_after)
            or not same_metadata(numpy_before, numpy_after)):
        raise ValueError('actual direct source or NumPy file identity changed during audit')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'tool_sha256': direct_before[str(Path(__file__).resolve().relative_to(ROOT))], **before,
              'before_clean': True, 'after_clean': True,
              'source_identity_before': source_before, 'source_identity_after': after,
              'direct_sources_before': direct_before, 'direct_sources_after': direct_after,
              'direct_source_scope': 'four local Python source files; no complete transitive or installed dependency closure claim',
              'external_numpy_files_before': numpy_before, 'external_numpy_files_after': numpy_after,
              'license': {'name': 'MIT', 'path': 'LICENSE', 'sha256': FILES['LICENSE']},
              'execution': {'module': 'pb_bss/math/solve.py', 'original_functions_called': ['stable_solve'],
                            'scope': 'complete unmodified source bytes compiled/executed; one function called',
                            'bytecode_cache_read': False, 'bytecode_cache_written': False,
                            'source_patch': False, 'ast_extraction': False, 'substitutes': [],
                            'not_executed': ['full pb_bss package', '_lstsq helper', 'beamformers', 'models', 'audio', 'hardware']},
              'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                              'numpy_linalg_wrapper_sha256': numpy_before['linalg_lstsq_wrapper']['sha256']},
              'original_lstsq_default': {'argument': 'rcond omitted by original stable_solve',
                                        'runtime_default': 'NumPy 2.x eps * max(M,N)',
                                        'float64_2_by_2_ratio': float(np.finfo(float).eps * 2)},
              'comparison_tolerance': {'absolute': ATOL, 'relative': 0},
              'oracle': 'independent Fraction 1/10,1/5 and diagonal/constraint closed forms',
              'cases': cases, 'numpy_comparisons': comparisons,
              'additional_original_controls': additional,
              'additional_counts': {'original_function_cases': 4, 'shape_exceptions': 2,
                                    'matches': 1, 'complex_batch_dtype_differences': 1},
              'counts': {'original_function_cases': len(cases), 'independent_numpy_cases': len(comparisons),
                         'original_matches': 3, 'original_known_dtype_failures': 1}}
    strict_loads(json.dumps(report, allow_nan=False))
    return report


def write_report(path, report, cache=CACHE):
    target = report_target(path, cache)
    # The shared writer validates strict finite JSON and checks ordinary
    # membership/protected roots again immediately before atomic replacement.
    write_json_report(target, report, forbidden_roots=(
        CACHE, Path(cache), LOCK, STATUS, HISTORICAL_REPORT,
        ROOT / 'codes/chapters/ch00/source_snapshots', ROOT / 'reviews'))


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
