"""Bounded calls to Chardon's fixed original Python module, in a fresh worker.

No demo, data, MATLAB, MEX, fetching, training or package installation is run.
The complete original damas.py bytes (including its imports) execute unchanged;
known failures and conditional uniqueness claims remain visible. Default stdout
is read-only. Only an explicit safe current/external --report publishes JSON.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
from datetime import datetime, timezone
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch10.examples.run_industrial_interfaces import clean_environment

ROOT = upstream.ROOT
CURRENT_REPORT = ROOT / 'codes/chapters/ch12/reports/damas_author_contracts_current.json'
REVISION = '61987952e2237e6b088a169ee891dd96576f2565'
SOURCE_SHA = '461858876d630240388b7c7972ce9997cb492690edd9ef757371f498e6639336'
EXTRA_DEPENDENCY = Path(clean_environment.__code__.co_filename)
MODULE = 'codes.chapters.ch12.examples.audit_damas_author_contracts'
ATOL = 2e-13


def identity(cache):
    value = upstream.verify_project('damas-author', Path(cache) / 'damas-author', relatives=('damas.py',))
    if value['head'] != REVISION or value['used_files']['damas.py']['sha256'] != SOURCE_SHA:
        raise ValueError('fixed author Python source changed')
    return value


def report_target(path, cache=upstream.CACHE):
    return upstream.report_target(path, CURRENT_REPORT,
        protected=(upstream.CACHE, cache, upstream.LOCK, upstream.STATUS, ROOT / 'reviews'))


def worker(cache):
    before = identity(cache)
    path = Path(before['checkout']) / 'damas.py'
    raw = upstream.ordinary_file(path).read_bytes()
    tree = ast.parse(raw)
    imports = [(n.names[0].name, n.names[0].asname) for n in tree.body if isinstance(n, ast.Import)]
    if imports != [('numpy', 'np'), ('scipy.linalg', 'la'), ('time', None), ('matplotlib.pyplot', 'plt')]:
        raise ValueError('original module import closure changed')
    external = {}
    for name in ('numpy', 'scipy.linalg', 'matplotlib.pyplot'):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise ImportError('original dependency unavailable: ' + name)
        source = upstream.ordinary_file(Path(spec.origin))
        external[name] = {'path': str(source), 'sha256_before': upstream.sha(source),
                         'scope': 'direct imported module entry file, not its full transitive library/binary closure'}
    module = types.ModuleType('fixed_damas_original')
    module.__file__ = str(path)
    module.__package__ = ''
    module.__spec__ = importlib.util.spec_from_file_location(module.__name__, path)
    sys.modules[module.__name__] = module
    # Compile the verified raw source itself: do not consume an ignored pyc.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    np = module.np
    rows = []

    def call(name, function, expected=None, expected_error=None, **details):
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                output = function()
            value = output[0] if isinstance(output, tuple) else output
            value = np.asarray(value)
            if expected_error or not np.isfinite(value).all() or not np.allclose(value, expected, rtol=0, atol=ATOL):
                raise ValueError('bounded original numeric contract failed: ' + name)
            row = {'name': name, 'execution': 'original complete Python module function',
                   'actual': value.tolist(), 'independent_expected': np.asarray(expected).tolist(),
                   'matched': True, 'original_stdout': captured.getvalue(), **details}
            if isinstance(output, tuple):
                row['original_unique_flag'] = bool(output[1])
        except Exception as error:
            if type(error).__name__ != expected_error:
                raise
            row = {'name': name, 'execution': 'original function raised',
                   'error_type': type(error).__name__, 'error_message': str(error),
                   'expected_original_failure_retained': True,
                   'original_stdout': captured.getvalue(), **details}
        rows.append(row)

    d = np.array([[1., 1.], [1., np.exp(-2j * np.pi / 3)]])
    v = np.sqrt(4 / 3) * np.array([1., np.exp(1j * np.pi / 3)])
    r = np.outer(v, v.conj())
    q = np.array([1., .2])
    call('full_gram_product', lambda: module.proddamas(d, q), [4.2, 1.8])
    call('removed_diagonal_gram_product', lambda: module.proddamasdr(d, q), [1.8, -.6])
    for name, expected in (('cmf_nnls_lh', [1., 0.]), ('damas_nnls_lh', [16/17, 0.]),
                           ('cmf_nnls_dr_lh', [2/3, 0.]), ('damas_nnls_dr_lh', [16/15, 0.])):
        call(name, lambda name=name: getattr(module, name)(d, r), expected,
             target='original full/removed-diagonal CSM objective or unnormalized dirty-map objective; not interchangeable')
    call('zero_csm_failure', lambda: module.cmf_nnls_lh(d, np.zeros((2, 2))),
         expected_error='NameError', boundary='original except e masks uninitialized Gram; source not repaired')
    call('duplicate_columns_conditional_unique_flag',
         lambda: module.cmf_nnls_lh(np.ones((2, 2)), np.ones((2, 2))), [1., 0.],
         independent_alternative_solution=[0., 1.],
         boundary='positive-definite active-support Gram does not prove global uniqueness with duplicate inactive columns')
    call('single_source_full_cleansc_two_steps',
         lambda: module.CLEANSC(np.ones((2, 1)), 2 * np.ones((2, 2)), 2, .6), [1.68],
         boundary='known rank-one CSM; no data benchmark, diagonal removal or marker optimization')
    for name, record in external.items():
        loaded = sys.modules[name]
        if str(Path(loaded.__file__).resolve()) != record['path']:
            raise ValueError('loaded dependency origin differs from preflight: ' + name)
        record['sha256_after'] = upstream.sha(record['path'])
        if record['sha256_before'] != record['sha256_after']:
            raise ValueError('external dependency entry changed: ' + name)
    external['versions'] = {name: importlib.import_module(name).__version__
                            for name in ('numpy', 'scipy', 'matplotlib')}
    return {'source_identity_before': before,
            'source_identity_after': upstream.check_unchanged(copy.deepcopy(before)),
            'module_identity': {'name': module.__name__, 'file': module.__file__, 'sha256': SOURCE_SHA,
                                'complete_module_bytes_executed': True, 'cached_bytecode_consumed': False,
                                'original_imports': imports},
            'external_dependencies': external,
            'known_input': {'transfer': {'real': d.real.tolist(), 'imag': d.imag.tolist()},
                            'csm': {'real': r.real.tolist(), 'imag': r.imag.tolist()}},
            'cases': rows, 'counts': {'bounded_cases': len(rows),
                                     'numeric_matches': sum(row.get('matched') is True for row in rows),
                                     'original_failures_retained': sum(row.get('expected_original_failure_retained') is True for row in rows)},
            'unexecuted': ['MATLAB methods', 'MEX/C compilation', 'original demos and MAT datasets',
                           'paper large-scale benchmark', 'successor acosolo algorithms']}


def run_audit(cache=upstream.CACHE, timeout=30):
    deps = upstream.dependencies(__file__, (EXTRA_DEPENDENCY,))
    before = identity(cache)
    with tempfile.TemporaryDirectory(prefix='masp-damas-author-') as folder:
        env = clean_environment()
        env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', MPLCONFIGDIR=folder)
        try:
            result = subprocess.run([sys.executable, '-B', '-m', MODULE, '--worker', '--cache', str(cache)],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout, check=False)
            if result.returncode:
                execution = {'execution': 'original worker failed', 'returncode': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr}
            else:
                execution = upstream.strict_json_loads(result.stdout)
                execution['worker_returncode'] = result.returncode
                execution['worker_stderr'] = result.stderr
        except subprocess.TimeoutExpired as error:
            execution = {'execution': 'original worker timed out', 'timeout_seconds': timeout,
                         'stdout': (error.stdout or b'').decode() if isinstance(error.stdout, bytes) else error.stdout,
                         'stderr': (error.stderr or b'').decode() if isinstance(error.stderr, bytes) else error.stderr}
    after = upstream.check_unchanged(copy.deepcopy(before))
    if deps != upstream.dependencies(__file__, (EXTRA_DEPENDENCY,)):
        raise ValueError('actual author-audit dependency changed during execution')
    return {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'limited_execution_with_original_failure' if 'cases' in execution else 'original_worker_not_completed',
            'actual_dependencies_before': deps, 'actual_dependencies_after': deps,
            'source_identity_before': before, 'source_identity_after': after,
            'worker': execution,
            'scope': '21 acquired source texts versus LICENSE and one original Python module used; no complete benchmark claim'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=upstream.CACHE)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker and args.report is not None:
        raise ValueError('worker cannot publish a report')
    destination = report_target(args.report, args.cache) if args.report is not None else None
    report = worker(args.cache) if args.worker else run_audit(args.cache)
    if destination is not None:
        report_target(destination, args.cache)
        upstream.write_report(destination, report, CURRENT_REPORT,
                              protected=(upstream.CACHE, args.cache, upstream.LOCK, upstream.STATUS, ROOT/'reviews'))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
