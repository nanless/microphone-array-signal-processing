"""Audit the pinned original mysofa_interpolate on five artificial controls.

Compile complete, unchanged interpolate.c and tools.c; the caller supplies a
two-position structure and neighbor list. No SOFA parsing, lookup, resampling,
rendering, measurement data or device validation is performed. Mathematical
delay preservation and observation of the known upstream failure are separate.
Default output is stdout; only explicit --report writes a current JSON report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from codes.chapters.ch00.io_contracts import (
    strict_json_loads, validate_parent_chain, validate_report_destination,
    write_json_report,
)
from codes.chapters.ch00.upstream.fetch_upstreams import inspect_project, run_git

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
STATUS = ROOT / 'codes/chapters/ch00/SOURCE_STATUS.json'
CURRENT_REPORT = ROOT / 'codes/chapters/ch01/reports/libmysofa_interpolation.json'
REVISION = '6cc5b15a73e9bd97810d03767082edda7f315881'
ORIGIN = 'https://github.com/hoene/libmysofa.git'
FILES = {
    'src/hrtf/interpolate.c': 'a554ad7b3c25f9d2a685b60e1fe9887b6167f9e81e4e114c570c5a51fbce7e44',
    'src/hrtf/tools.c': '55238724d257d3221c6c16bfcf62d6990c176b248ad81786e0f100e172f4fa22',
    'src/hrtf/tools.h': '2fb4fa5f73609ab963fbd5c94f24fd2411d45ebec138b4af48d82ba75b3fb59e',
    'src/hrtf/mysofa.h': 'fe24a1a424dae943624b942723af48b0f345d8d13911be88b0202096e546f834',
    'src/resampler/speex_resampler.h': 'ced479f7a027e2601c2c0ec129ba280d56e0021b0aa65f2e6f51e767efb3c622',
    'LICENSE': '9c8713ccc5d6d93d20c7a1cfcb876e64baf551bfbe67faa4e60d39cda9ffd023',
}
ATOL = 2e-6
EXPORT_HEADER = '#ifndef MYSOFA_EXPORT\n#define MYSOFA_EXPORT\n#endif\n'

# This caller is our own scaffold. It neither copies nor replaces the algorithm.
HARNESS = r'''#include "mysofa.h"
#include <stdio.h>
static void probe(const char *name, int elements, float *delays, float x) {
  float positions[6] = {-1.f,0.f,0.f, 1.f,0.f,0.f};
  float ir[4] = {2.f,4.f, 4.f,6.f};
  float query[3] = {x,0.f,0.f};
  int neighbors[6] = {1,-1,-1,-1,-1,-1};
  float result[2] = {0.f,0.f}, output_delays[2] = {0.f,0.f};
  struct MYSOFA_ATTRIBUTE type = {NULL, "Type", "cartesian"};
  struct MYSOFA_HRTF h = {0};
  h.M=2; h.N=1; h.R=2; h.C=3; h.I=h.E=1;
  h.SourcePosition.values=positions; h.SourcePosition.elements=6;
  h.SourcePosition.attributes=&type;
  h.DataIR.values=ir; h.DataIR.elements=4;
  h.DataDelay.values=delays; h.DataDelay.elements=elements;
  mysofa_interpolate(&h, query, 0, neighbors, result, output_delays);
  printf("%s %.9g %.9g %.9g %.9g\n", name,
         (double)result[0], (double)result[1],
         (double)output_delays[0], (double)output_delays[1]);
}
int main(void) {
  float common[2]={8.f,16.f}, zero[2]={0.f,0.f};
  float repeated[4]={8.f,16.f,8.f,16.f};
  float varying[4]={8.f,16.f,12.f,20.f};
  probe("global_exact",2,common,-1.f);
  probe("global_midpoint",2,common,0.f);
  probe("global_zero_midpoint",2,zero,0.f);
  probe("per_direction_same_midpoint",4,repeated,0.f);
  probe("per_direction_vary_midpoint",4,varying,0.f);
  return 0;
}
'''

# Direct hand expectations: distance 1 from each position gives weights 1/2.
# The midpoint IR is [(2+4)/2,(4+6)/2]; a common delay must remain common.
CONTROLS = (
    ('global_exact', 2, [8., 16.], -1., [2., 4.], [8., 16.], [8., 16.]),
    ('global_midpoint', 2, [8., 16.], 0., [3., 5.], [8., 16.], [4., 8.]),
    ('global_zero_midpoint', 2, [0., 0.], 0., [3., 5.], [0., 0.], [0., 0.]),
    ('per_direction_same_midpoint', 4, [8., 16., 8., 16.], 0., [3., 5.], [8., 16.], [8., 16.]),
    ('per_direction_vary_midpoint', 4, [8., 16., 12., 20.], 0., [3., 5.], [10., 18.], [10., 18.]),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ordinary_file(path):
    path = validate_parent_chain(path)
    if not path.is_file():
        raise ValueError('required ordinary file is absent: ' + str(path))
    return path


def report_target(path):
    """Preflight before compilation; repository writes allow only this new report."""
    target = validate_report_destination(path, forbidden_roots=(CACHE, LOCK, STATUS,
        ROOT / 'reviews', ROOT / 'codes/chapters/ch00/source_snapshots',
        Path(__file__), ROOT / 'tests/test_codes_ch01_libmysofa_interpolation.py'))
    normalized = target.resolve()
    if normalized.is_relative_to(ROOT.resolve()) and normalized != CURRENT_REPORT.resolve():
        raise ValueError('only the current interpolation report may be written inside the repository')
    return target


def verify_checkout(checkout, *, revision, origin, files):
    """Check complete required file identities without altering the checkout.

    Parameters support offline temporary-Git tests; production uses fixed values.
    This identity check does not assert that the full sparse selection is valid.
    """
    checkout = validate_parent_chain(checkout)
    if not checkout.is_dir() or not (checkout / '.git').is_dir():
        raise ValueError('an independent ordinary Git checkout is required')
    validate_parent_chain(checkout / '.git')
    if (type(revision) is not str or len(revision) != 40
            or any(c not in '0123456789abcdef' for c in revision)):
        raise ValueError('a complete fixed Git revision is required')
    def git(*args):
        return run_git(list(args), cwd=checkout)
    if git('rev-parse', '--show-toplevel') != str(checkout.resolve()):
        raise ValueError('checkout must be its own Git top level')
    if git('remote', 'get-url', 'origin') != origin:
        raise ValueError('official origin mismatch')
    if git('rev-parse', 'HEAD') != revision:
        raise ValueError('fixed HEAD mismatch')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('upstream worktree must be completely clean')
    identity = {}
    for relative, expected_sha in files.items():
        if (not relative or Path(relative).is_absolute() or '..' in Path(relative).parts):
            raise ValueError('source path must be an ordinary relative member')
        path = ordinary_file(checkout / relative)
        raw = path.read_bytes()
        head_blob = git('rev-parse', revision + ':' + relative)
        actual_blob = git('hash-object', '--', str(path))
        if digest(raw) != expected_sha or head_blob != actual_blob:
            raise ValueError('source SHA/blob mismatch: ' + relative)
        identity[relative] = {'sha256': digest(raw), 'bytes': len(raw),
                             'head_blob': head_blob, 'actual_blob': actual_blob}
    return {'origin': origin, 'head': revision, 'worktree_clean': True,
            'required_source_identity_verified': True, 'files': identity}


def verify_sources():
    lock_raw, status_raw = ordinary_file(LOCK).read_bytes(), ordinary_file(STATUS).read_bytes()
    lock, status = strict_json_loads(lock_raw), strict_json_loads(status_raw)
    lock_sha = digest(lock_raw)
    if status.get('lock_sha256') != lock_sha:
        raise ValueError('current source status does not bind the current lock bytes')
    rows = [r for r in lock['projects'] if r['id'] == 'libmysofa']
    states = [r for r in status['projects'] if r['id'] == 'libmysofa']
    if len(rows) != 1 or len(states) != 1:
        raise ValueError('libmysofa lock and acquisition records must be unique')
    entry, recorded = rows[0], states[0]
    if (entry['revision'] != REVISION or entry['url'] != ORIGIN
            or recorded['revision'] != REVISION
            or entry['license'] != 'BSD-3-Clause; third-party code notices separate'):
        raise ValueError('fixed libmysofa source identity changed')
    identity = verify_checkout(CACHE / 'libmysofa', revision=REVISION, origin=ORIGIN, files=FILES)
    # Inspect the complete selection independently; retain mismatch if present.
    live = inspect_project(entry, CACHE)
    return {**identity, 'source_version': 'v1.3.5',
            'source_lock_sha256': lock_sha, 'source_lock_entry': entry,
            'source_status_sha256': digest(status_raw), 'recorded_acquisition': recorded,
            'live_acquisition_scope': live,
            'license': {'name': entry['license'], 'path': 'LICENSE', 'sha256': FILES['LICENSE'],
                        'retention': 'Unchanged upstream headers and root BSD notice retained; no upstream code or binary copied into this report',
                        'supporting_header': 'speex_resampler.h retains Jean-Marc Valin 2007 BSD notice; its resampler is not compiled or called'}}


def evaluate(stdout):
    lines = [line.split() for line in stdout.splitlines() if line.strip()]
    if (len(lines) != len(CONTROLS) or any(len(line) != 5 for line in lines)
            or [line[0] for line in lines] != [row[0] for row in CONTROLS]):
        raise ValueError('unexpected native interpolation output rows')
    result = []
    for line, (name, elements, delays, x, ir, mathematical_delay, upstream_delay) in zip(lines, CONTROLS):
        observed = [float(v) for v in line[1:]]
        if not all(math.isfinite(v) for v in observed):
            raise ValueError('native interpolation output must be finite')
        ir_ok = all(math.isclose(a, b, rel_tol=0, abs_tol=ATOL) for a, b in zip(observed[:2], ir))
        delay_ok = all(math.isclose(a, b, rel_tol=0, abs_tol=ATOL) for a, b in zip(observed[2:], mathematical_delay))
        upstream_ok = ir_ok and all(math.isclose(a, b, rel_tol=0, abs_tol=ATOL) for a, b in zip(observed[2:], upstream_delay))
        result.append({'name': name, 'input_delay_elements': elements,
            'input_delays_raw': delays, 'query_cartesian_m': [x, 0., 0.],
            'observed_ir_left_right': observed[:2], 'observed_delays_raw': observed[2:],
            'independent_expected_ir_left_right': ir,
            'independent_expected_delays_raw': mathematical_delay,
            'ir_interpolation_invariant_passed': ir_ok,
            'delay_interpolation_invariant_passed': delay_ok,
            'mathematical_invariants_passed': ir_ok and delay_ok,
            'known_upstream_expected_delays_raw': upstream_delay,
            'expected_upstream_behavior_observed': upstream_ok,
            'delay_error_from_mathematical_expectation': [a-b for a,b in zip(observed[2:], mathematical_delay)]})
    return result


def run_audit(*, compiler='cc'):
    before = verify_sources()
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError('C compiler is unavailable: ' + compiler)
    version = subprocess.run([executable, '--version'], check=True, capture_output=True,
                             text=True, timeout=60).stdout
    with tempfile.TemporaryDirectory(prefix='ch01-libmysofa-interpolation-') as temporary:
        build = Path(temporary)
        (build / 'probe.c').write_text(HARNESS, encoding='utf-8')
        (build / 'mysofa_export.h').write_text(EXPORT_HEADER, encoding='utf-8')
        upstream = CACHE / 'libmysofa/src/hrtf'
        command = [executable, '-std=c99', '-O0', '-I', str(build), '-I', str(upstream),
                   str(build / 'probe.c'), str(upstream / 'interpolate.c'),
                   str(upstream / 'tools.c'), '-lm', '-o', str(build / 'probe')]
        compiled = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        stdout = subprocess.run([str(build / 'probe')], check=True, capture_output=True,
                                text=True, timeout=60).stdout
    after = verify_sources()
    if before != after:
        raise ValueError('source identity, selection or lock/status changed during the audit')
    cases = evaluate(stdout)
    dependencies = {str(p.relative_to(ROOT)): digest(ordinary_file(p).read_bytes()) for p in (
        Path(__file__), ROOT / 'codes/chapters/ch00/io_contracts.py',
        ROOT / 'codes/chapters/ch00/upstream/fetch_upstreams.py')}
    return {'schema_version': 1, 'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'audit_source': str(Path(__file__).relative_to(ROOT)),
        'audit_source_sha256': dependencies[str(Path(__file__).relative_to(ROOT))],
        'audit_dependencies_sha256': dependencies,
        'scope': 'complete original mysofa_interpolate plus tools.c; artificial caller-controlled structure and neighbor list only',
        'source': before, 'source_before_after_identical': True,
        'source_after': after, 'environment': {'python': sys.version, 'platform': platform.platform()},
        'compiler': {'executable': executable, 'version': version, 'command': command,
                     'stdout': compiled.stdout, 'stderr': compiled.stderr, 'flags': ['-std=c99', '-O0']},
        'scaffold': {'export_header': EXPORT_HEADER, 'export_header_sha256': digest(EXPORT_HEADER.encode()),
            'harness_sha256': digest(HARNESS.encode()), 'upstream_patched': False,
            'temporary_binary_retained': False, 'compiled_original_files': ['src/hrtf/interpolate.c', 'src/hrtf/tools.c'],
            'compiled_original_headers': ['src/hrtf/mysofa.h', 'src/hrtf/tools.h'],
            'additional_identity_files': ['src/resampler/speex_resampler.h', 'LICENSE']},
        'config': {'dimensions': {'I': 1, 'E': 1, 'M': 2, 'N': 1, 'C': 3, 'R': 2},
            'positions_cartesian_m': [[-1.,0.,0.],[1.,0.,0.]], 'input_ir_by_position_left_right': [[2.,4.],[4.,6.]],
            'nearest_index': 0, 'caller_supplied_neighbors': [1,-1,-1,-1,-1,-1],
            'delay_units': 'raw DataDelay numbers; no sample/second conversion is tested',
            'absolute_tolerance': ATOL, 'randomness': 'none; deterministic artificial coefficients',
            'independent_expectation': 'midpoint equal distances 1 m: average IR and directional delays; global delays unchanged'},
        'native_stdout': stdout, 'cases': cases,
        'all_mathematical_invariants_passed': all(c['mathematical_invariants_passed'] for c in cases),
        'mathematical_failure_cases': [c['name'] for c in cases if not c['mathematical_invariants_passed']],
        'expected_upstream_behavior_observed': all(c['expected_upstream_behavior_observed'] for c in cases),
        'not_executed': ['SOFA parser or measurements', 'mysofa_check', 'automatic lookup/neighborhood',
                         'mysofa_open/easy float or short API', 'resampling or delay unit conversion',
                         'audio convolution/rendering', 'device acceptance']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Explicit current strict JSON report destination')
    parser.add_argument('--compiler', default='cc')
    args = parser.parse_args()
    target = report_target(args.report) if args.report is not None else None
    report = run_audit(compiler=args.compiler)
    if target is not None:
        target = report_target(target)
        write_json_report(target, report, forbidden_roots=(CACHE, LOCK, STATUS, Path(__file__)))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report['expected_upstream_behavior_observed']:
        raise SystemExit('Original behavior differs from the independently specified upstream controls')


if __name__ == '__main__':
    main()
