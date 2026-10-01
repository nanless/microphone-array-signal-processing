"""Offline identity and static AST contracts for the fixed author TAC source.

No upstream module is imported, rewritten or executed. This audit describes
shared layers and pooling/reference structure, not trained-network invariance,
separation quality, latency or hardware. Only explicit --report writes a file.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
REVISION = 'e3373b73358a96af6f64fdbe25327def8d6bd973'
ORIGIN = 'https://github.com/yluo42/TAC.git'
LICENSE = 'CC-BY-NC-SA-3.0-US (README declaration; no separate LICENSE)'
FILES = {
    'README.md': 'd7d98dbb02fb58b907def8e0363e3deff639c8ce362bedb1e8fb5d5d0d617566',
    'FaSNet.py': 'd16eb8846f7d70ea7035b891b7f5464972aed27486c863ddb690bd073b73f93e',
    'utility/__init__.py': '01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b',
    'utility/models.py': '9a545100d23ff00fb8b65cfb013f3554bf7d23224706eb4612839ca17e30b523',
}


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
    checkout = ordinary_path(Path(cache) / 'tac', directory=True)
    lock_bytes = ordinary_path(LOCK).read_bytes()
    projects = strict_loads(lock_bytes)['projects']
    matches = [p for p in projects if p['id'] == 'tac']
    if len(matches) != 1:
        raise ValueError('TAC lock entry must be unique')
    entry = matches[0]
    if (entry['revision'] != REVISION or entry['url'] != ORIGIN or entry['license'] != LICENSE or
            entry.get('source_paths') != list(FILES) or
            not all(p in entry['entrypoints'] for p in ('README.md', 'FaSNet.py', 'utility/models.py'))):
        raise ValueError('fixed TAC lock identity or selected source paths changed')
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
    # Check literal selected files, not the requested paths alone. No assets or
    # additional untracked files may silently enter this local source selection.
    actual = {p.relative_to(checkout).as_posix() for p in checkout.rglob('*')
              if '.git' not in p.relative_to(checkout).parts and (p.is_file() or p.is_symlink())}
    if actual != set(FILES):
        raise ValueError('working source selection differs from four literal files')
    sources = []
    for relative, expected in FILES.items():
        content = ordinary_path(checkout / relative).read_bytes()
        if content != git('show', REVISION + ':' + relative, binary=True) or digest(content) != expected:
            raise ValueError('source digest/blob mismatch: ' + relative)
        sources.append({'path': relative, 'sha256': digest(content),
                        'git_blob': git('rev-parse', REVISION + ':' + relative)})
    readme = ordinary_path(checkout / 'README.md').read_text(encoding='utf-8')
    if ('creativecommons.org/licenses/by-nc-sa/3.0/us/' not in readme or
            'Attribution-NonCommercial-ShareAlike 3.0 United States License' not in readme):
        raise ValueError('README license declaration is absent')
    return {'checkout': str(checkout), 'revision': REVISION, 'origin': ORIGIN,
            'source_lock_sha256': digest(lock_bytes), 'source_lock_project_count': len(projects),
            'sources': sources, 'selected_source_files': list(FILES), 'clean': True}


def method(tree, class_name, name):
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError('missing or duplicate class: ' + class_name)
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise ValueError('missing or duplicate method: ' + class_name + '.' + name)
    return methods[0]


def contains(tree, expression):
    """Exact AST predicate only; never compile or execute the inspected source."""
    expected = ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False)
    return any(ast.dump(node, include_attributes=False) == expected for node in ast.walk(tree))


def structure_contracts(models, fasnet):
    trees = [ast.parse(models), ast.parse(fasnet)]
    init = method(trees[0], 'DPRNN_TAC', '__init__')
    forward = method(trees[0], 'DPRNN_TAC', 'forward')
    fa_init = method(trees[1], 'FaSNet_TAC', '__init__')
    fa_forward = method(trees[1], 'FaSNet_TAC', 'forward')
    # Predicates describe a small, fixed source contract. They do not prove a
    # trained network's output invariants or safety for arbitrary num_mic input.
    definitions = [
        ('shared_three_layer_groups', init, [
            'self.ch_transform.append', 'self.ch_average.append', 'self.ch_concat.append',
            'nn.Linear(input_size, hidden_size * 3)',
            'nn.Linear(hidden_size * 3, hidden_size * 3)',
            'nn.Linear(hidden_size * 6, input_size)'],
         'one transform/average/concatenate layer group per network block; not one group per microphone'),
        ('shared_layer_application', forward, [
            'self.ch_transform[i](ch_input)', 'self.ch_average[i](ch_mean)',
            'self.ch_concat[i](ch_output.view(-1, ch_output.shape[-1]))'],
         'the same block-indexed layers process flattened channel features'),
        ('fixed_channel_mean', forward, ['num_mic.max() == 0', 'ch_output.mean(2)'],
         'zero maximum sentinel averages all channels on the original channel axis'),
        ('valid_prefix_mean', forward, ['ch_output[b, :, :num_mic[b]].mean(1)'],
         'variable counts select each batch item valid channel prefix before its mean'),
        ('concatenation', forward, ['torch.cat([ch_output, ch_mean], 2)'],
         'individual channel features are concatenated with the broadcast aggregate'),
        ('residual', forward, ['output + ch_output'],
         'the TAC branch is added to the running channel-indexed representation'),
        ('noncausal_intra_and_group_norm', init, [
            'SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True)',
            'nn.GroupNorm(1, input_size, eps=1e-8)'],
         'intra-segment RNN is bidirectional and the implementation uses GroupNorm; no streaming guarantee'),
        ('single_stage_tac_module', fa_init, [
            "BF_module(self.filter_dim + self.enc_dim, self.feature_dim, self.hidden_dim, self.filter_dim, self.num_spk, self.layer, self.segment_size, model_type='DPRNN_TAC')"],
         'this author wrapper estimates all filters with one TAC module; not the original two-stage FaSNet'),
        ('reference_channel_zero', fa_forward, ['all_seg[:, 0]', 'self.seq_cos_sim(all_context, ref_seg)'],
         'cosine features use center segments of reference channel zero'),
        ('final_valid_channel_mean', fa_forward, [
            'num_mic.max() == 0', 'bf_signal.mean(1)', 'bf_signal[b, :num_mic[b]].mean(0)'],
         'final waveform aggregation also distinguishes all channels from valid prefixes'),
    ]
    rows = []
    for name, node, expressions, interpretation in definitions:
        if not all(contains(node, expression) for expression in expressions):
            raise ValueError('static source contract changed: ' + name)
        text = models if node in (init, forward) else fasnet
        segment = ast.get_source_segment(text, node)
        rows.append({'name': name, 'evidence_level': 'static original AST only',
                     'method': node.name, 'source': 'utility/models.py' if node in (init, forward) else 'FaSNet.py',
                     'line_start': node.lineno, 'line_end': node.end_lineno,
                     'method_source_sha256': digest(segment.encode()), 'matched': True,
                     'interpretation': interpretation})
    return rows


def run_audit(cache=CACHE):
    before = verify_sources(cache)
    checkout = Path(before['checkout'])
    rows = structure_contracts((checkout / 'utility/models.py').read_text(encoding='utf-8'),
                               (checkout / 'FaSNet.py').read_text(encoding='utf-8'))
    after = verify_sources(cache)
    if before != after:
        raise ValueError('upstream or source lock changed during audit')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'tool_sha256': digest(Path(__file__).read_bytes()), **before,
              'status': 'passed_static_contracts', 'before_clean': True, 'after_clean': True,
              'license': {'name': LICENSE, 'declaration_file': 'README.md', 'sha256': FILES['README.md'],
                          'local_source_only': True, 'upstream_code_redistributed': False,
                          'weights_or_audio_obtained': False},
              'execution': {'audit': 'source identity and Python AST inspection',
                            'original_modules_imported': [], 'original_functions_called': [],
                            'ast_extraction_for_execution': False, 'source_patch': False, 'substitutes': [],
                            'not_executed': ['Torch network construction', 'original forward', 'training',
                                             'weights', 'speech separation', 'latency measurement', 'hardware']},
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'torch_available_to_auditor': importlib.util.find_spec('torch') is not None,
                              'torch_imported_by_audit': False},
              'contracts': rows, 'counts': {'static_contracts': len(rows), 'original_runtime_calls': 0},
              'boundaries': ['Channel-indexed TAC output is permutation equivariant under shared operations.',
                             'Channel mean is order invariant; adding or removing channels generally changes its value.',
                             'FaSNet reference channel must be kept consistent; a TAC block alone does not prove wrapper invariance.',
                             'num_mic zero sentinel and valid prefixes are inspected, not validated by network execution.',
                             'Independent teaching NumPy fixtures are not included as original network execution.']}
    strict_loads(json.dumps(report, allow_nan=False))
    return report


def write_report(path, report, cache=CACHE):
    target = report_target(path, cache)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    strict_loads(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.tac-report-', delete=False) as handle:
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
