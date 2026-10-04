"""Read-only SAID original compression module audit, without Torch or weights.

Only artificial JSON and temporary output directories are created. --report
explicitly saves a current execution record; it does not download dependencies.
"""
from __future__ import annotations

# Support documented direct-file commands without changing upstream imports.
if __package__ in (None, ""):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[4]))

from codes.chapters.ch04.core import upstream_contracts as contracts

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys
import tempfile

sys.dont_write_bytecode = True
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
PROJECT = 'said-spatial-imaging'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads' / PROJECT
REVISION = 'cf52ede4f38361cdbb03aa93c8254109583dfe2b'
CURRENT_REPORT = ROOT / 'codes/chapters/ch04/reports/said_compression_contracts.json'
SOURCE_FILES = ('said/utils/compression.py', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
                'LICENSES/SAID-Model-Weights-NonCommercial-1.0.txt',
                'LICENSES/AudioMAE-CC-BY-NC-4.0.txt')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return contracts.git(CACHE, *args)


def verify():
    return contracts.verify_project(PROJECT, CACHE, SOURCE_FILES)['lock_entry']


def build_report():
    identity = contracts.verify_project(PROJECT, CACHE, SOURCE_FILES)
    entry = identity["lock_entry"]
    before = {'head': git('rev-parse', 'HEAD'),
              'status': git('status', '--porcelain', '--untracked-files=no')}
    records = {}
    for relative in SOURCE_FILES:
        records[relative] = identity['used_files'][relative].copy()
    # Import the entire unchanged source file directly; importing said package
    # would pull in model dependencies. No AST edit or compatibility facade.
    name = '_masp_said_original_compression'
    spec = importlib.util.spec_from_file_location(name, CACHE / 'said/utils/compression.py')
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(name)
    sys.modules[name] = module  # dataclass resolves annotations via this entry.
    try:
        spec.loader.exec_module(module)
        high = {'id': 7, 'image_id': 11, 'category_id': 1, 'score': .9,
                'extra': 'preserved', 'segmentation': [[[0,0,1],[2,0,.5],[8,0,.09],[12,0,0]]]}
        low = {'id': 8, 'image_id': 11, 'category_id': 1, 'score': .1,
               'segmentation': [[[359,0,1]]]}
        high_output = module.compress_annotation(high)
        low_output = module.compress_annotation(low)
        failures = {}
        for label, points in [('zero_energy', [[[0,0,0]]]),
                              ('nonfinite_energy', [[[0,0,float('nan')]]])]:
            try:
                module.compress_annotation({**high, 'segmentation': points})
                raise AssertionError(f'expected failure absent: {label}')
            except ValueError as exc:
                failures[label] = {'exception_type': type(exc).__name__, 'message': str(exc)}
        wrapped = module._wrapped_coordinates(np.array([[361.,-1.,1.],[-1.,180.,.5]], dtype=np.float32))
        distances = module._minimum_wrapped_distance_squared(
            np.array([[359.,0.]], np.float32), np.array([[1.,0.]], np.float32))
        payload = {'images': [{'id': 11}], 'annotations': [high, low]}
        with tempfile.TemporaryDirectory(prefix='masp-said-compression-') as temporary:
            temp = pathlib.Path(temporary)
            source = temp / 'input'
            source.mkdir()
            source_file = source / 'artificial_inference.json'
            source_file.write_text(json.dumps(payload, allow_nan=False)+'\n')
            manifest_file = module.compress_prediction_directory(source, temp/'output', workers=1)
            manifest = json.loads(manifest_file.read_text())
            output_file = temp/'output'/source_file.name
            output = json.loads(output_file.read_text())
            file_record = manifest['files'][0]
            # Paths are transient; report hashes and exact JSON preserve input.
            directory_result = {
                'input_payload': payload, 'output_payload': output,
                'file_record': file_record,
                'actual_input_sha256': sha(source_file),
                'actual_output_sha256': sha(output_file),
                'actual_output_bytes': output_file.stat().st_size,
                'maximum_file_bytes': manifest['maximum_file_bytes'],
                'initial_preset': manifest['initial_preset'],
                'fallback_presets': manifest['fallback_presets'],
                'file_count': manifest['file_count'],
                'all_within_limit': manifest['all_prediction_files_within_limit'],
            }
            try:
                module.compress_prediction_directory(source, temp/'too-small',
                                                     workers=1, maximum_file_bytes=1)
                raise AssertionError('one-byte limit unexpectedly succeeded')
            except RuntimeError as exc:
                failures['one_byte_limit'] = {
                    'exception_type': type(exc).__name__, 'message': str(exc),
                    'published_output_exists': (temp/'too-small').exists(),
                    'temporary_leftovers': [p.name for p in temp.glob('.too-small.*')],
                    'scope': 'reduced artificial one-byte limit exercises fallback/exhaustion; not a 20MB real recording',
                }
    finally:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
    contracts.check_unchanged(identity)
    after = {'head': git('rev-parse', 'HEAD'),
             'status': git('status', '--porcelain', '--untracked-files=no')}
    if before != after:
        raise RuntimeError('upstream changed during audit')
    return {
        'run_utc': datetime.now(timezone.utc).isoformat(), 'run_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'audit_source_sha256': sha(pathlib.Path(__file__)), 'lock_sha256': sha(LOCK),
        'lock_entry': entry, 'source_contract': identity,
        'report_source_sha256': contracts.dependencies(__file__), 'before': before, 'after': after, 'sources': records,
        'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': importlib.metadata.version('numpy')},
        'scope': {
            'execution': 'unchanged original compression.py loaded as standalone module; stdlib + NumPy; no said package import',
            'functions': ['compress_annotation', 'compress_prediction_directory(workers=1)',
                          '_wrapped_coordinates', '_minimum_wrapped_distance_squared'],
            'claims_excluded': ['Torch/model execution', 'weights acquisition', 'acoustic prediction accuracy',
                                'paper Table1 reproduction', '20MB large-recording success', 'multithread/process path'],
        },
        'tolerances': {'float32_boundary_absolute': 2e-5, 'serialized_energy_absolute': 1e-4},
        'results': {'high_confidence_input': high, 'high_confidence_output': high_output,
                    'low_confidence_input': low, 'low_confidence_output': low_output,
                    'wrapped_coordinates': wrapped.tolist(), 'wrapped_distance_squared': distances.tolist(),
                    'directory': directory_result, 'failures': failures},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=pathlib.Path, help='explicit current report destination')
    args = parser.parse_args()
    target = contracts.report_target(args.report, CURRENT_REPORT) if args.report else None
    report = build_report()
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.report:
        contracts.write_report(target, report, CURRENT_REPORT)
        print(target)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
