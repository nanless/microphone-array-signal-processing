"""Read-only original pyaec APA calls; only --report writes a report.

No AST extraction, patches, compatibility facade, downloads, or audio/model
assets. The complete fixed single-file module is loaded without bytecode writes.
This verifies four deterministic interface cases, not a complete AEC system.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads/pyaec'
CURRENT_REPORT = ROOT / 'codes/chapters/ch06/reports/upstream_apa_current.json'
REVISION = '5b9c02c57075d790b7df8652884618189d49bbc4'
SOURCE = 'time_domain_adaptive_filters/apa.py'
SOURCE_SHA = {SOURCE: 'c6c703bd912c1477eea126fa64f1795366766ba608cdc5981c8fb1ee656ab821',
              'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}
ATOL = 2e-14


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(directory, *args):
    return contracts.git(directory, *args)


def checkout_state(directory):
    identity = contracts.verify_project('pyaec', directory, SOURCE_SHA)
    return {'head': identity['head'], 'status': '', 'untracked_python': []}


def encode(value):
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'real': encode(value.real.tolist()), 'imag': encode(value.imag.tolist()),
                'finite': bool(np.isfinite(value).all())}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'value': None, 'classification': 'nan' if math.isnan(value) else
                ('positive_infinity' if value > 0 else 'negative_infinity')}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def independent_expectations():
    # P=1,N=1: scalar residual r[n+1] = x[n+1]/x[n] *
    # r[n] * .01/(x[n]**2+.01), with x[n]=n+1 and d=2*x.
    f = Fraction
    scalar = [f(2)]
    for x in (1, 2, 3):
        scalar.append(scalar[-1] * f(x+1, x) / (100*x*x+1))
    # P=2: at n=0 w=(100/101,0). At n=1, use the explicit
    # two-by-two adjugate for Gram+.01I = [[2.01,1],[1,1.01]].
    e0, e1, det = f(203, 101), f(1, 101), f(10301, 10000)
    second_weight = (f(101, 100)*e0-e1)/det
    return {'zero_reference': [1., 2., 3., 4.],
            'order_one': [float(v) for v in scalar],
            'order_two': [1., float(e0), float(2-second_weight)],
            'complex_discard': [1., 1., 1., 1.]}


def run_audit(directory=CACHE):
    directory = contracts.validate_parent_chain(directory)
    identity = contracts.verify_project('pyaec', directory, SOURCE_SHA)
    entry = identity['lock_entry']
    if SOURCE not in entry['entrypoints']:
        raise RuntimeError('APA source must be registered in the shared lock')
    for name, expected in SOURCE_SHA.items():
        if identity['used_files'][name]['sha256'] != expected:
            raise RuntimeError('Original source hash differs: '+name)
    before = {'head': identity['head'], 'status': '', 'untracked_python': []}
    files = {name: dict(identity['used_files'][name]) for name in SOURCE_SHA}
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('fixed_original_pyaec_apa', directory/SOURCE)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
    finally:
        sys.dont_write_bytecode = old_bytecode
    fixtures = {
        'zero_reference': (np.zeros(6), np.arange(1., 7.), 2, 2),
        'order_one': (np.arange(1., 6.), np.arange(2., 11., 2.), 1, 1),
        'order_two': (np.array([1., 1., 0., 0., 0.]), np.array([1., 3., 2., 0., 0.]), 2, 2),
        'complex_discard': (np.ones(5, dtype=complex)*1j, np.ones(5, dtype=complex), 1, 1),
    }
    expected = independent_expectations()
    rows = {}
    for name, (x, d, n, p) in fixtures.items():
        row = {'inputs': {'x': encode(x), 'd': encode(d), 'N': n, 'P': p, 'mu': 1.},
               'expected_residual': expected[name], 'absolute_tolerance': ATOL,
               'unprocessed_tail_indices': list(range(len(x)-n, min(len(x), len(d)))),
               'classification': 'complex_input_discarded' if name == 'complex_discard'
                                 else ('zero_reference_tail_not_processed' if name == 'zero_reference'
                                       else 'regularized_real_apa'),
               'warnings': [], 'exception': None}
        row['input_sha256'] = hashlib.sha256(json.dumps(row['inputs'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = original.apa(x, d, N=n, P=p, mu=1.)
                row['residual'] = encode(result)
                row['expected_behavior_verified'] = bool(result.shape == (len(expected[name]),)
                    and np.isfinite(result).all()
                    and np.allclose(result, expected[name], atol=ATOL, rtol=0))
            except Exception as error:
                row['residual'] = None
                row['exception'] = {'type': type(error).__name__, 'message': str(error)}
                row['expected_behavior_verified'] = False
        row['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
                            'file': str(Path(w.filename).relative_to(directory)),
                            'line': w.lineno} for w in caught]
        if name == 'complex_discard':
            row['expected_behavior_verified'] &= bool(row['warnings'] and all(
                w['category'] == 'ComplexWarning' for w in row['warnings']))
        else:
            row['expected_behavior_verified'] &= not row['warnings']
        rows[name] = row
    contracts.check_unchanged(identity)
    after = {'head': identity['head'], 'status': '', 'untracked_python': []}
    if before != after:
        raise RuntimeError('Upstream state changed during the audit')
    now = datetime.now(timezone.utc)
    return {'schema_version': 2, 'created_utc': now.isoformat(),
            'verified_date_asia_shanghai': now.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat(),
            'audit_source_sha256': sha(__file__), 'lock_sha256': identity['lock_sha256'], 'lock_entry': entry,
            'actual_dependency_sha256': contracts.dependencies(Path(__file__)),
            'source_identity': identity,
            'reproduction_command': '.venv/bin/python -m codes.chapters.ch06.examples.audit_upstream_apa',
            'original_files': files, 'before': before, 'after': after,
            'environment': {'python': sys.version, 'numpy': np.__version__,
                            'executable': sys.executable, 'platform': platform.platform()},
            'scope': {'execution': 'original_complete_single_file_module_and_function_call',
                      'identity': 'maintainer independent educational implementation; not Ozeki/Umeda author code',
                      'compatibility': 'no AST extraction, patches or compatibility facade',
                      'excluded': 'No full pyaec package, real audio, double-talk controller, convergence benchmark or device run',
                      'fixed_regularization': '.01 I; explicit inverse; real buffers; state resets per call'},
            'case_count': len(rows), 'results': rows,
            'status': 'expected_behaviors_verified_warnings_preserved' if all(
                row['expected_behavior_verified'] for row in rows.values()) else 'verification_failed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    target = contracts.report_target(args.report, CURRENT_REPORT, (contracts.CACHE,)) if args.report else None
    report = run_audit()
    serialized = json.dumps(encode(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if target is not None:
        contracts.write_report(target, encode(report), CURRENT_REPORT, (contracts.CACHE,))
    else:
        print(serialized, end='')
    return 0 if report['status'] == 'expected_behaviors_verified_warnings_preserved' else 1


if __name__ == '__main__':
    raise SystemExit(main())
