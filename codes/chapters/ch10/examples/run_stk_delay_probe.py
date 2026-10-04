"""Compile a locked STK DelayL probe in a temporary directory; never downloads.

Builds only Stk.cpp and DelayL.cpp, no audio driver, weights or device calls.
This module is inert on import. Output report records component behavior only.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from codes.chapters.ch00.upstream.fetch_upstreams import inspect_project, load_projects
from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch10.examples.run_industrial_interfaces import clean_environment, verify_failure_sources, native_identity, compiled_dependencies

ROOT = Path(__file__).resolve().parents[4]
HARNESS = Path(__file__).with_name('stk_delay_probe.cpp')
CURRENT = ROOT / 'codes/chapters/ch10/reports/stk_delay_current.json'


def validate_measurements(data):
    expected = {'analytic_max_abs_error', 'zero_delay_max_abs_error',
                'one_delay_max_abs_error', 'chunk37_max_abs_error',
                'reset_chunk37_max_abs_difference'}
    if set(data) != expected:
        raise ValueError('unexpected measurement keys')
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid measurement')
        if key.startswith('reset_'):
            if value < .01:
                raise ValueError('reset control did not expose lost history')
        elif value > 1e-12:
            raise ValueError('fixed-delay interface disagrees with independent FIR')


def run_probe(destination=ROOT/'codes/chapters/ch00/upstream/_downloads'):
    identity = native_identity('stk', destination)
    try:
        project = load_projects()['stk']
        source = Path(destination)/'stk'
        checked = inspect_project(project, Path(destination))
        if checked['status'] != 'source_verified':
            raise ValueError('STK source checkout is not verified')
        compiler = shutil.which('c++')
        if compiler is None:
            raise RuntimeError('A C++ compiler is required')
        with tempfile.TemporaryDirectory(prefix='masp-stk-') as temporary:
            work = upstream.work_target(temporary, protected=(destination,))
            binary = work/'delay-probe'
            objects = []
            for index, original in enumerate((HARNESS, source/'src/Stk.cpp', source/'src/DelayL.cpp')):
                obj = work/f'tu{index}.o'
                command = [compiler, '-std=c++17', '-O2', '-I', str(source/'include'),
                    '-MD', '-MF', str(work/f'tu{index}.d'), '-c', str(original), '-o', str(obj)]
                subprocess.run(command, env=clean_environment(), check=True, capture_output=True, text=True, timeout=120)
                objects.append(obj)
            subprocess.run([compiler, *map(str, objects), '-o', str(binary)],
                env=clean_environment(), check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(binary)], env=clean_environment(), check=True, capture_output=True, text=True, timeout=20)
            compilation = compiled_dependencies(work, {"stk": identity}, (HARNESS,))
            binary_sha256 = upstream.sha(binary)
        upstream.check_unchanged(identity)
        data = upstream.strict_json_loads(result.stdout)
        validate_measurements(data)
        return {'project': 'stk', 'revision': project['revision'], 'license': project['license'],
                'status': 'component_numeric_check_passed',
                'source_identity': identity, 'compiled_dependencies': compilation, 'binary_sha256': binary_sha256,
                'actual_dependencies_sha256': upstream.dependencies(__file__, (HARNESS, Path(clean_environment.__code__.co_filename))),
                'environment': {'system': platform.system(), 'machine': platform.machine(),
                                'python': platform.python_version(),
                                'compiler': subprocess.check_output([compiler, '--version'], env=clean_environment(), text=True).splitlines()[0]},
                'source_files_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                       for name in ['include/DelayL.h', 'include/Filter.h', 'include/Stk.h',
                                                    'src/DelayL.cpp', 'src/Stk.cpp', 'LICENSE']},
                'probe_sha256': hashlib.sha256(HARNESS.read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'compile_options': ['-std=c++17', '-O2'],
                'input': {'sample_rate_hz': 16000, 'samples': 512, 'amplitudes': [.18, .18],
                          'frequencies_hz': [500, 6000], 'initial_history': 'zero',
                          'delay_samples': [.5, 0., 1.], 'block_size_samples': 37,
                          'seed': None, 'precision': 'STK default double, no device audio'},
                'measurements': data,
                'limits': 'Scalar tick with preserved state; grouping loop is not StkFrames API coverage. '
                          'Not a device latency, time-varying delay, speech quality or real-time benchmark.'}
    except BaseException as error:
        verify_failure_sources({'stk': identity}, error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report is not None:
        upstream.report_target(args.report, CURRENT, protected=(upstream.CACHE,))
    result = run_probe()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.report:
        upstream.write_report(args.report, result, CURRENT, protected=(upstream.CACHE,))
    print(encoded, end='')
