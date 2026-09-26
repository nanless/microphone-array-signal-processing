"""Compile a locked STK DelayL probe in a temporary directory; never downloads.

Builds only Stk.cpp and DelayL.cpp, no audio driver, weights or device calls.
This module is inert on import. Output report records component behavior only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from codes.upstream.fetch_upstreams import inspect_project, load_projects

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).with_name('stk_delay_probe.cpp')


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


def run_probe(destination=ROOT/'codes/upstream/_downloads'):
    project = load_projects()['stk']
    source = Path(destination)/'stk'
    checked = inspect_project(project, Path(destination))
    if checked['status'] != 'source_verified':
        raise ValueError('STK source checkout is not verified')
    compiler = shutil.which('c++')
    if compiler is None:
        raise RuntimeError('A C++ compiler is required')
    with tempfile.TemporaryDirectory(prefix='masp-stk-') as temporary:
        binary = Path(temporary)/'delay-probe'
        command = [compiler, '-std=c++17', '-O2', '-I', str(source/'include'),
                   str(HARNESS), str(source/'src/Stk.cpp'), str(source/'src/DelayL.cpp'),
                   '-o', str(binary)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=20)
    data = json.loads(result.stdout)
    validate_measurements(data)
    return {'project': 'stk', 'revision': project['revision'], 'license': project['license'],
            'status': 'component_numeric_check_passed',
            'environment': {'system': platform.system(), 'machine': platform.machine(),
                            'python': platform.python_version(),
                            'compiler': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]},
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = run_probe()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end='')
