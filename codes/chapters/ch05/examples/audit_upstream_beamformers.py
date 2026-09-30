"""Read-only, fixed pb_bss function diagnostics for Chapter 5.

Requires NumPy 2 and SciPy; uses the existing locked checkout, without downloads,
package imports, Cython, dependency installation, or upstream patches. Twelve
unchanged function ASTs are executed with their NumPy/SciPy dependencies. This
is not a complete package, enhanced-waveform, paper benchmark, or device test.
Only an explicit --report PATH writes an output file.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
PROJECT = 'pb_bss'
REVISION = '10acc347fc9ea21e3d312806a0bd751d0d0af183'
SOURCE_SHA = {
    'pb_bss/math/solve.py': 'be440ee6f252e42a6ae5d7b6f3cc7205da5d4c29bcb01cae7885ab9f8c6a53ed',
    'pb_bss/extraction/beamformer.py': '8bc6700fbe07b50caeb640503bbf827798c883a3fc12b9790f93ab2ee9453e53',
    'LICENSE': '48241e1eae6ab4c15c5718992ca60d3d15961212a4e324e09e5dbfbc79b78214',
}
FUNCTIONS = (
    'stable_solve', 'get_power_spectral_density_matrix', 'get_mvdr_vector',
    'get_mvdr_vector_merl', 'get_gev_vector', '_get_gev_vector',
    'get_lcmv_vector', 'blind_analytic_normalization',
    'get_optimal_reference_channel', 'get_mvdr_vector_souden',
    'get_wmwf_vector', 'get_lcmv_vector_souden',
)
ATOL = 2e-14


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(directory, *args, binary=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    value = subprocess.check_output(['git', '-C', str(directory), *args],
                                    env=env, text=not binary)
    return value if binary else value.strip()


def checkout_state(directory):
    if not (directory / '.git').exists():
        raise FileNotFoundError('Existing independent locked pb_bss checkout required')
    if Path(git(directory, 'rev-parse', '--show-toplevel')).resolve() != directory.resolve():
        raise RuntimeError('Expected an independent upstream checkout')
    state = {'head': git(directory, 'rev-parse', 'HEAD'),
             'status': git(directory, 'status', '--porcelain', '--untracked-files=no'),
             'untracked_python': [p for p in git(directory, 'ls-files', '--others',
                                                '--exclude-standard').splitlines()
                                  if p.endswith('.py')]}
    if state['head'] != REVISION or state['status'] or state['untracked_python']:
        raise RuntimeError('Upstream revision or source cleanliness differs')
    return state


def encode(value):
    """Strict JSON retains complex values and explicit nonfinite classifications."""
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'real': encode(value.real.tolist()), 'imag': encode(value.imag.tolist()),
                'finite': bool(np.isfinite(value).all())}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, complex):
        return {'real': encode(value.real), 'imag': encode(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return {'value': None, 'classification': 'nan' if math.isnan(value)
                else ('positive_infinity' if value > 0 else 'negative_infinity')}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def _matches(actual, expected, direction=False, atol=ATOL):
    if isinstance(expected, tuple):
        return (isinstance(actual, tuple) and len(actual) == len(expected)
                and all(_matches(a, e, atol=atol) for a, e in zip(actual, expected)))
    if isinstance(expected, np.ndarray):
        actual = np.asarray(actual)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            return False
        if direction:
            # Eigenvector sign/phase is arbitrary. This checks its span, not scale.
            unit = expected / np.linalg.norm(expected)
            residual = actual - np.sum(unit.conj()*actual, axis=-1, keepdims=True)*unit
            return bool(np.linalg.norm(actual) > 0 and np.linalg.norm(residual) <= atol)
        return bool(np.allclose(actual, expected, atol=atol, rtol=0))
    return bool(actual == expected)


def extract_functions(directory):
    import scipy
    from scipy.linalg import eig, eigh
    namespace = {'np': np, 'solve': np.linalg.solve, 'eig': eig, 'eigh': eigh,
                 'operator': operator, 'functools': functools,
                 'c_gev_available': False, 'c_eig_available': False}
    files, functions = {}, {}
    for relative, expected_sha in SOURCE_SHA.items():
        path = directory / relative
        data = path.read_bytes()
        original = git(directory, 'show', 'HEAD:'+relative, binary=True)
        if data != original or sha(path) != expected_sha:
            raise RuntimeError('Original file hash differs: '+relative)
        files[relative] = {'sha256': sha(path),
                           'git_blob': git(directory, 'rev-parse', 'HEAD:'+relative)}
        if not relative.endswith('.py'):
            continue
        text = data.decode('utf-8')
        with warnings.catch_warnings(record=True) as parse_warnings:
            warnings.simplefilter('always', SyntaxWarning)
            tree = ast.parse(text, filename=str(path))
        files[relative]['parse_warnings'] = [str(w.message) for w in parse_warnings]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                # Execute the original node, including its original body/defaults.
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
                     namespace)
                segment = ast.get_source_segment(text, node)
                functions[node.name] = {'file': relative, 'first_line': node.lineno,
                    'last_line': node.end_lineno,
                    'source_segment_sha256': hashlib.sha256(segment.encode()).hexdigest(),
                    'body_modified': False}
    if set(functions) != set(FUNCTIONS):
        raise RuntimeError('Expected twelve original functions')
    return namespace, files, functions, scipy.__version__


def run_cases(namespace):
    """Hand fixtures; known failures are matched, never relabeled as success."""
    rows = {}

    def case(id_, function, args, expected=None, *, kwargs=None,
             classification='valid_model', explanation='', exception=None,
             message_contains='', direction=False, reference=None, observable=None,
             atol=ATOL):
        row = {'function': function, 'inputs': encode(args), 'kwargs': encode(kwargs or {}),
               'classification': classification, 'explanation': explanation,
               'expected_output': encode(expected), 'independent_reference': encode(reference),
               'comparison': 'span_residual' if direction else 'absolute_tolerance',
               'absolute_tolerance': atol, 'warnings': []}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                output = namespace[function](*args, **(kwargs or {}))
            row['warnings'] = [str(w.message) for w in caught]
            row['output'] = encode(output)
            row['exception'] = None
            row['expected_behavior_verified'] = exception is None and _matches(output, expected, direction, atol)
            if observable is not None:
                row['observables'] = encode(observable(output))
        except Exception as error:
            row['output'] = None
            row['exception'] = {'type': type(error).__name__, 'message': str(error)}
            row['expected_behavior_verified'] = (
                exception is not None and isinstance(error, exception)
                and message_contains in str(error))
        finally:
            if 'caught' in locals():
                row['warnings'] = [str(w.message) for w in caught]
        rows[id_] = row

    identity = np.eye(2, dtype=complex)[None]
    b = np.array([2, 1+1j], dtype=complex)
    rank_one = (3*np.outer(b, b.conj()))[None]
    colored_noise = np.diag([2, 1]).astype(complex)[None]
    full_rank = np.diag([2, 1]).astype(complex)[None]
    dominant_second = np.diag([1, 4]).astype(complex)[None]

    case('stable_solve_regular', 'stable_solve',
         (np.diag([2., 4.]), np.array([[2.], [8.]])), np.array([[1.], [2.]]))
    case('stable_solve_singular', 'stable_solve',
         (np.diag([0., 1.]), np.ones((2, 1))), np.array([[0.], [1.]]),
         classification='least_squares_fallback', explanation='Inconsistent first equation is not satisfied.')
    case('stable_solve_vector_rhs', 'stable_solve',
         (np.diag([2., 4.]), np.array([2., 8.])),
         exception=IndexError, message_contains='tuple index out of range',
         classification='expected_exception', explanation='Original helper requires a matrix RHS; NumPy solve vector API is not inherited.')
    case('mvdr_regular', 'get_mvdr_vector', (np.ones((1, 2), complex), identity),
         np.array([[.5, .5]], complex))
    case('mvdr_singular', 'get_mvdr_vector',
         (np.ones((1, 2), complex), np.diag([0., 1.])[None]), np.array([[0., 1.]]),
         classification='model_boundary', explanation='Finite feasible output has noise 1; true PSD optimum [1,0] has noise 0.',
         reference={'true_optimum': [1, 0], 'true_minimum_noise': 0},
         observable=lambda w: {'target_response': np.vdot(w[0], np.ones(2)),
                                'output_noise_power': abs(w[0, 1])**2})

    # q = 4 from scalar 4/2 + |1+j|²/1, not an upstream trace.
    for ref, expected in ((0, [[.5, .5+.5j]]), (1, [[.25-.25j, .5]])):
        case('souden_rank_one_ref'+str(ref), 'get_mvdr_vector_souden',
             (rank_one, colored_noise), np.array(expected, complex), kwargs={'ref_channel': ref},
             reference={'q': 4., 'reference_response': b[ref],
                        'formula': 'conj(b_ref) * Rn^-1 b / q'},
             observable=lambda w: {'physical_target_response': np.vdot(w[0], b)})
    case('souden_full_rank', 'get_mvdr_vector_souden', (full_rank, identity),
         np.array([[2/3, 0]], complex), kwargs={'ref_channel': 0},
         classification='model_boundary', explanation='Full-rank trace expression is not distortionless for every target image.')
    case('souden_auto_reference', 'get_mvdr_vector_souden', (dominant_second, identity),
         (np.array([[0, .8]], complex), 1), kwargs={'return_ref_channel': True},
         reference={'candidate_output_snr': [1, 4]})
    case('optimal_reference_direct', 'get_optimal_reference_channel',
         (np.diag([.2, .8])[None].astype(complex), dominant_second, identity), 1,
         reference={'candidate_output_snr': [1, 4]})

    for id_, speech, expected in (
            ('wmwf_full_rank', full_rank, [[.5, 0]]),
            ('wmwf_rank_one', np.diag([2, 0])[None].astype(complex), [[2/3, 0]])):
        case(id_, 'get_wmwf_vector', (speech, identity), np.array(expected, complex),
             kwargs={'reference_channel': 0, 'distortion_weight': 1.},
             classification='model_boundary' if id_ == 'wmwf_full_rank' else 'valid_model',
             reference={'general_mwf_weights': [2/3, 0],
                        'general_mwf_scalar_calculation': '2/(2+1)',
                        'trace_scalar_calculation': '2/(1+3)' if id_ == 'wmwf_full_rank' else '2/(1+2)'})

    for name, scale in (('positive', 1.), ('positive_double', 2.), ('negative', -1.), ('phase_j', 1j)):
        expected = np.array([[1., 1.]])*scale/abs(scale)/math.sqrt(2)
        case('ban_'+name, 'blind_analytic_normalization',
             (scale*np.ones((1, 2), complex), identity), expected,
             reference={'norm': 1., 'target_response': math.sqrt(2)*(scale/abs(scale)).conjugate()},
             observable=lambda w: {'weight_norm': float(np.linalg.norm(w)),
                                    'target_response': np.vdot(w[0], np.ones(2))})
    case('ban_zero', 'blind_analytic_normalization', (np.zeros((1, 2), complex), identity),
         np.zeros((1, 2), complex), classification='degenerate_zero_output',
         explanation='Explicit zero-denominator branch; not a normalized target response.')

    for function in ('get_gev_vector', '_get_gev_vector'):
        case('gev_'+('public' if function == 'get_gev_vector' else 'fallback'), function,
             (np.diag([4, 1])[None].astype(complex), colored_noise[:, ::-1, ::-1]),
             np.array([[1., 0.]], complex), direction=True,
             reference={'generalized_eigenvalues': [4., .5]})
    case('gev_singular_noise', 'get_gev_vector',
         (np.diag([4, 1])[None].astype(complex), np.diag([1, 0])[None].astype(complex)),
         exception=ValueError, message_contains='frequency 0', classification='expected_exception')

    case('lcmv_complex_response', 'get_lcmv_vector',
         (np.eye(2, dtype=complex)[:, None, :], np.array([1, -1j]), identity),
         np.array([[1, -1j]], complex), reference={'desired_wH_C': [1, 1j]},
         observable=lambda w: {'wH_C': w[0].conj(), 'CH_w': w[0]})
    case('lcmv_incompatible_constraints', 'get_lcmv_vector',
         (np.ones((2, 1, 2), complex), np.array([1, 0]), identity), np.array([[.25, .25]], complex),
         classification='constraint_violation', explanation='Least-squares result violates incompatible hard constraints.',
         reference={'requested_CH_w': [1, 0], 'least_squares_CH_w': [.5, .5],
                    'constraint_residual_norm': 1/math.sqrt(2)},
         observable=lambda w: {'CH_w': np.array([sum(w[0]), sum(w[0])]),
                                'constraint_residual_norm': float(np.linalg.norm(
                                    np.array([sum(w[0]), sum(w[0])])-np.array([1, 0])))})
    case('lcmv_souden_unimplemented', 'get_lcmv_vector_souden', (rank_one, rank_one, identity),
         kwargs={'ref_channel': 0}, exception=NotImplementedError,
         message_contains='not yet thoroughly tested', classification='expected_exception')

    for diagonal, expected, correct in (([1, 4], [[.2, 0]], [0, .8]),
                                       ([4, 1], [[.8, 0]], [.8, 0])):
        case('merl_reference_'+str(diagonal[0]), 'get_mvdr_vector_merl',
             (np.diag(diagonal)[None], np.eye(2)[None]), np.array(expected),
             classification='reference_selection_failure' if diagonal[0] == 1 else 'diagnostic_control',
             reference={'correct_weights': correct, 'candidate_output_snr': diagonal})

    snapshots = np.eye(2, dtype=complex)[None]  # F=1, C=2, T=2; no audio.
    for name, mask, expected, exception, classification in (
        ('float', np.ones((1, 2)), np.eye(2)[None]/2, None, 'valid_model'),
        ('integer', np.ones((1, 2), dtype=int), None, TypeError, 'expected_exception'),
        ('boolean', np.ones((1, 2), dtype=bool), None, AttributeError, 'expected_exception'),
        ('negative', np.array([[2., -1.]]), np.diag([2., -1.])[None], None, 'invalid_psd_accepted'),
        ('zero', np.zeros((1, 2)), np.zeros((1, 2, 2)), None, 'degenerate_zero_support'),
        ('tiny', np.full((1, 2), 1e-300), np.eye(2)[None]*1e-290, None, 'denominator_floor'),
    ):
        case('scm_'+name, 'get_power_spectral_density_matrix', (snapshots, mask), expected,
             exception=exception, classification=classification,
             message_contains='Cannot cast' if name == 'integer' else ('asfarray' if name == 'boolean' else ''),
             reference={'unfloored_nonzero_normalized_diagonal': [.5, .5]} if name == 'tiny' else None,
             atol=1e-303 if name == 'tiny' else ATOL)
    return rows


def build_report(cache=CACHE):
    if int(np.__version__.split('.')[0]) < 2:
        raise RuntimeError('This fixed compatibility audit requires NumPy >=2')
    directory = Path(cache).resolve() / PROJECT
    lock_sha = sha(LOCK)
    entries = [p for p in json.loads(LOCK.read_text())['projects'] if p['id'] == PROJECT]
    if len(entries) != 1 or entries[0]['revision'] != REVISION or entries[0]['license'] != 'MIT':
        raise RuntimeError('Locked pb_bss identity or license differs')
    before = checkout_state(directory)
    namespace, files, functions, scipy_version = extract_functions(directory)
    results = run_cases(namespace)
    after = checkout_state(directory)
    if after != before:
        raise RuntimeError('Upstream checkout changed during audit')
    if sha(LOCK) != lock_sha:
        raise RuntimeError('Source lock changed during audit')
    # Re-read original files after calls; no silently changed source is accepted.
    for relative, metadata in files.items():
        if sha(directory / relative) != metadata['sha256']:
            raise RuntimeError('Upstream source changed during audit: '+relative)
    utc = datetime.now(timezone.utc)
    verified = all(row['expected_behavior_verified'] for row in results.values())
    return {'schema_version': 1, 'created_utc': utc.isoformat(),
        'verified_date_asia_shanghai': utc.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat(),
        'audit_source_sha256': sha(__file__), 'lock_sha256': lock_sha,
        'lock_entry': entries[0], 'before': before, 'after': after,
        'original_files': files, 'original_functions': functions,
        'environment': {'executable': sys.executable, 'python': platform.python_version(),
                        'numpy': np.__version__, 'scipy': scipy_version,
                        'platform': platform.platform()},
        'scope': {'execution': 'Twelve unchanged original function ASTs with original Python bodies',
                  'dependencies': 'NumPy/SciPy linalg and original stable_solve; no compatibility facade',
                  'backend': 'Explicit c_gev_available=False and c_eig_available=False; SciPy fallback only',
                  'excluded': 'No complete pb_bss/ESPnet/Torch/Cython import, waveform, paper benchmark, firmware or device validation',
                  'inputs': 'Deterministic exact two-channel covariance and complex snapshots; no random trials',
                  'upstream_modified': False},
        'results': results,
        'status': 'expected_behaviors_verified_failures_preserved' if verified else 'unexpected_behavior',
        'case_count': len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Explicit opt-in output path; default writes no file')
    args = parser.parse_args()
    report = build_report()
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.report is not None:
        target = args.report.resolve()
        if target == Path(__file__).resolve() or target == LOCK.resolve() or CACHE.resolve() in target.parents:
            raise ValueError('Refusing to overwrite source, lock, or upstream cache')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized)
    print(json.dumps({'status': report['status'], 'case_count': report['case_count'],
                      'report': str(args.report) if args.report else None,
                      'unexpected_cases': [k for k, v in report['results'].items()
                                           if not v['expected_behavior_verified']]}, ensure_ascii=False))
    if report['status'] == 'unexpected_behavior':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
