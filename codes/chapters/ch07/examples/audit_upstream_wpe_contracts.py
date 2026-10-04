"""Read-only fixed WPE contracts; only --report writes a JSON report.

NARA's original complete NumPy module is executed without patches or bytecode.
NeMo is inspected through its AST, never imported or run with a Torch facade.
The separate NumPy examples are book calculations, not NeMo execution.
Run fetch_upstreams --verify --report first; SOURCE_STATUS must bind this lock
and mark both selected sources verified. This tool never acquires dependencies.
"""
from __future__ import annotations

# Preserve both the existing direct-file entry and the module entry.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
STATUS = ROOT / 'codes/chapters/ch00/SOURCE_STATUS.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
CURRENT_REPORT = ROOT / 'codes/chapters/ch07/reports/upstream_wpe_contracts_current.json'
ATOL = 1e-12
SOURCES = {
    'nara_wpe': {
        'revision': 'a166779cca2088817e330481bd20af1a2c598555', 'license': 'MIT',
        'files': {
            'LICENSE': 'f9fad5befd31a5c5540fb671fa56efd59d959959ffe1b6a8156a66549d6410f0',
            'nara_wpe/wpe.py': '385a6f1c67071ba3e243c8a24fe4a041484c55280ad4ed0dbadefa4679a606d2',
        },
    },
    'nemo_wpe': {
        'revision': '2381f42f6979449b5b99538f8f80135831009b51', 'license': 'Apache-2.0',
        'files': {
            'LICENSE': '43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1',
            'README.md': '345642d7b5beda9834aa1fcd175de24cc4da5e9e17b9bf0b3f022a6d8f802579',
            'setup.py': 'cdc64c92e590eab0f1af698ce0f654a4e9f69ca29a87ab48a7bab46a9d614e4a',
            'requirements/requirements_audio.txt': 'afe0013bf8cce9038a81b5163126fea7b05ce270a14e9caaf5079ac6676c67b6',
            'requirements/requirements_common.txt': '7a80b634a857f48cfbe323c8c8237dca5fc8a5eed7231853bf3e713f93a0c0ce',
            'nemo/collections/audio/modules/masking.py': '7f39caa90a0dfabefdc5579ac27a3ac5e5e850b697fc4c057d6ca7da74d69d5e',
            'nemo/collections/audio/modules/transforms.py': '6714e91bae49c2776621dcf94cfe7d06fd93a17aa73733aaf4275336090035eb',
            'nemo/collections/audio/models/enhancement.py': '33018ab99222276c9f247618868be3a6abb31fba8353cfcc2eecd0cade6b020f',
            'nemo/collections/audio/parts/submodules/multichannel.py': '06a47a910a68acdb78f172d03dba1ebc19ebe2cde9e254dc2ee154491e5f2049',
            'nemo/collections/audio/parts/utils/audio.py': 'c692fe097aee1682a000db708f7fb446d738ea2e5ac724abdf397eedf4d475bb',
            'tests/collections/audio/test_audio_modules.py': '35f04245c50b31960099b8d2e8d291fd8aa27a6d910631208d6d4d34f56a1c41',
        },
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def git(directory, *args):
    return contracts.git(directory, *args)


def checkout_state(directory, revision):
    identity = contracts.verify_project('nara_wpe' if revision == SOURCES['nara_wpe']['revision'] else 'nemo_wpe', directory)
    if identity['head'] != revision:
        raise RuntimeError('Fixed source revision differs')
    return {'head': identity['head'], 'tracked_status': '', 'untracked_python': []}


def verify_acquisition(lock, status, lock_sha):
    if status.get('lock_sha256') != lock_sha:
        raise RuntimeError('SOURCE_STATUS is not generated from the current lock')
    entries, states = {}, {}
    for project, spec in SOURCES.items():
        matches = [p for p in lock['projects'] if p['id'] == project]
        rows = [p for p in status['projects'] if p['id'] == project]
        if len(matches) != 1 or len(rows) != 1:
            raise RuntimeError('Exactly one lock/status entry required: '+project)
        entry, row = matches[0], rows[0]
        if entry['revision'] != spec['revision'] or entry['license'] != spec['license']:
            raise RuntimeError('Lock conditions differ: '+project)
        if row.get('revision') != spec['revision'] or row.get('status') != 'source_verified' or \
                row.get('missing_entrypoints') or not row.get('source_selection_verified'):
            raise RuntimeError('Acquisition verification not passed: '+project)
        entries[project], states[project] = entry, row
    return entries, states


def _method(text, class_name, method_name):
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def _code(node):
    return ast.unparse(node)


def _assignment(node, target, value):
    return any(isinstance(n, ast.Assign) and any(_code(t) == target for t in n.targets)
               and _code(n.value) == value for n in ast.walk(node))


def inspect_nemo(mask_text, multichannel_text, model_text):
    """Read original AST facts; no compiling, importing or operator execution."""
    forward = _method(mask_text, 'MaskBasedDereverbWPE', 'forward')
    loop = next(n for n in ast.walk(forward) if isinstance(n, ast.For))
    wforward = _method(multichannel_text, 'WPEFilter', 'forward')
    corr = _method(multichannel_text, 'WPEFilter', 'estimate_correlations')
    solve = _method(multichannel_text, 'WPEFilter', 'estimate_filter')
    conv = _method(multichannel_text, 'WPEFilter', 'convtensor')
    model = _method(model_text, 'EncMaskDecAudioToAudioModel', 'forward')
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and _code(n.func) == 'self.filter']
    call = calls[0] if len(calls) == 1 else None
    keywords = {} if call is None else {k.arg: _code(k.value) for k in call.keywords}
    mask_if = [n for n in ast.walk(loop) if isinstance(n, ast.If)
               and _code(n.test) == 'i == 0 and mask is not None']
    returns = [n for n in ast.walk(forward) if isinstance(n, ast.Return)]
    loading_if = [n for n in ast.walk(solve) if isinstance(n, ast.If)
                  and _code(n.test) == 'self.diag_reg']
    facts = {
        'first_iteration_amplitude_mask': bool(mask_if and _assignment(mask_if[0], 'magnitude', 'mask * magnitude')),
        'power_is_magnitude_squared': _assignment(loop, 'power', 'magnitude ** 2'),
        'previous_output_is_next_regression_input': keywords.get('input') == 'output',
        'returns_spectrum_and_length': any(_code(n.value) == '(output.to(io_dtype), output_length)' for n in returns),
        'inverse_mean_power_plus_eps': _assignment(wforward, 'weight', '1 / (weight + self.eps)')
            and _assignment(wforward, 'weight', 'torch.mean(power, dim=1)'),
        'length_masks_weights_before_correlations': _assignment(corr, 'weight', 'weight.masked_fill(length_mask, 0.0)'),
        'trace_loading_without_dimension_division': _assignment(solve, 'diag_reg',
            'self.diag_reg * torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).real + self.eps'),
        'matrix_loading_conditional_on_diag_reg': bool(loading_if and any(
            isinstance(n, ast.Assign) and any(_code(t) == 'Q' for t in n.targets)
            for n in ast.walk(loading_if[0]))),
        'direct_linear_solve': _assignment(solve, 'G', 'torch.linalg.solve(Q, R)'),
        'history_left_padding': _assignment(conv, 'x', 'torch.nn.functional.pad(x, (filter_length - 1 + delay, 0))'),
        'history_unfold': _assignment(conv, 'tilde_X', 'x.unfold(-1, filter_length, 1)'),
        'model_passes_encoded_length_and_mask': any(isinstance(n, ast.Call)
            and _code(n.func) == 'self.mask_processor'
            and {k.arg: _code(k.value) for k in n.keywords} == {
                'input': 'encoded', 'input_length': 'encoded_length', 'mask': 'mask'} for n in ast.walk(model)),
    }
    nodes = {'mask_forward': forward, 'filter_forward': wforward,
             'correlations': corr, 'filter_estimation': solve, 'history': conv, 'model_forward': model}
    snippets = {name: {'start_line': n.lineno, 'end_line': n.end_lineno,
        'original_text_sha256': hashlib.sha256(('\n'.join(
            (mask_text if name == 'mask_forward' else model_text if name == 'model_forward'
             else multichannel_text).splitlines()[n.lineno-1:n.end_lineno])+'\n').encode()).hexdigest()}
        for name, n in nodes.items()}
    return {'execution_kind': 'static_original_ast_only', 'framework_executed': False,
            'facts': facts, 'snippets': snippets, 'all_facts_verified': all(facts.values()),
            'boundary': 'batch statistics; no Torch, NeMo package, neural weights or waveform transform executed'}


def run_nara(module):
    y = np.asarray([[[10,20,30,40,50,60], [100,200,300,400,500,600]]], dtype=np.complex128)
    filters = np.zeros((1,4,2), dtype=np.complex128)
    filters[0,:,0] = [1,2,3,4]
    offline = module.build_y_tilde(y, taps=2, delay=1)[0,:,5]
    buffer = y[:,:,2:6].transpose(2,0,1).copy()
    captured = {}
    def inspect_return(frame, event, _arg):
        if event == 'return' and frame.f_code is module.online_wpe_step.__code__:
            captured['window'] = frame.f_locals['window'].copy()
    old_profile = sys.getprofile()
    sys.setprofile(inspect_return)
    try:
        pred, inv, updated = module.online_wpe_step(buffer, np.ones(1),
            np.eye(4)[None].astype(complex), filters.copy(), alpha=.95, taps=2, delay=1)
    finally:
        sys.setprofile(old_profile)
    state = module.OnlineWPE(taps=2, delay=1, alpha=.95, channel=2, frequency_bins=1)
    state.buffer = y[:,:,1:5].transpose(2,0,1).copy()
    state.filter_taps = filters.copy()
    class_prediction, window = state._get_prediction(y[:,:,5])
    class_actual = state.step_frame(y[:,:,5])
    zero = module.online_wpe_step(np.zeros((4,1,2), dtype=complex), np.ones(1),
        np.eye(4)[None].astype(complex), filters.copy(), alpha=.5, taps=2, delay=1)
    try:
        state.step_frame(np.zeros((2,2), dtype=complex))
    except Exception as exc:
        invalid = {'type': type(exc).__name__, 'message': str(exc)}
    else:
        invalid = None
    result = {'execution_kind': 'original_complete_numpy_module_and_methods',
        'inputs': {'spectra_real': y.real.tolist(), 'spectra_imag': 'all zero',
            'taps': 2, 'delay': 1, 'alpha': .95, 'current_time_index': 5,
            'first_output_filter': [1,2,3,4], 'second_output_filter': [0,0,0,0],
            'stateless_buffer_times': [2,3,4,5], 'state_old_buffer_times': [1,2,3,4]},
        'offline_window': offline.real.tolist(), 'state_window': window.real.tolist(),
        'stateless_window': captured['window'].real.tolist(),
        'instrumentation': 'Read-only Python return profiling captures the original stateless window local; prior profile restored',
        'stateless_output': pred.real.tolist(), 'class_prediction': class_prediction.real.tolist(),
        'class_step_output': class_actual.real.tolist(),
        'stateless_updated_state_finite': bool(np.isfinite(inv).all() and np.isfinite(updated).all()),
        'zero_input': {'output': zero[0].real.tolist(), 'inverse_covariance': zero[1].real.tolist(),
                       'filter_unchanged': bool(np.array_equal(zero[2], filters))},
        'wrong_frame_shape_exception': invalid,
        'independent_expected': {'offline_window': [50,500,40,400],
            'stateless_window': [40,30,400,300], 'state_window': [20,30,200,300],
            'stateless_first_output': 60-(40+2*30+3*400+4*300),
            'class_first_output': 60-(20+2*30+3*200+4*300)},
        'absolute_tolerance': ATOL,
        'boundary': 'dimensionless labeled spectra; deliberately set coefficients; no speech quality or runtime benchmark'}
    expected = result['independent_expected']
    result['expected_behaviors_verified'] = bool(
        np.array_equal(offline.real, expected['offline_window'])
        and np.array_equal(captured['window'].real[0], expected['stateless_window'])
        and np.array_equal(window.real[0], expected['state_window'])
        and pred[0,0] == expected['stateless_first_output'] and pred[0,1] == 600
        and class_prediction[0,0] == expected['class_first_output']
        and np.array_equal(class_actual, class_prediction) and class_actual[0,1] == 600
        and np.array_equal(zero[0], np.zeros((1,2)))
        and np.array_equal(zero[1], 2*np.eye(4)[None]) and result['zero_input']['filter_unchanged']
        and result['stateless_updated_state_finite']
        and invalid is not None and invalid['type'] == 'AssertionError')
    result['input_sha256'] = digest(result['inputs'])
    return result


def independent_numpy_examples():
    """Explicit book calculations, not an adapter for original Torch operators."""
    # One channel, one frequency, one tap, delay=1, all-frame zero history,
    # no loading and strictly positive powers: isolate changing the regressor.
    y = np.array([1.,2.,3.,4.])
    def one_iteration(observation, power):
        history = np.array([0., *observation[:-1]])
        coefficient = np.sum(history*observation/power)/np.sum(history**2/power)
        return observation-coefficient*history, coefficient
    # Use a separately prescribed first residual to keep the next comparison
    # independent of the first numeric calculation.
    prescribed = np.array([1.,4/7,1/7,-2/7])
    fixed, fixed_g = one_iteration(y, prescribed**2)
    residual, residual_g = one_iteration(prescribed, prescribed**2)
    f = Fraction
    expected = {'first': [1., float(f(4,7)), float(f(1,7)), float(f(-2,7))],
                'fixed_observation': [1., float(f(56,101)), float(f(11,101)), float(f(-34,101))],
                'residual_observation': [1., float(f(216,721)), float(f(-9,721)), float(f(-234,721))]}
    # Fixed initial unit powers isolate the change in the second regressor.
    # This is deliberately not the default initialization of the NeMo wrapper.
    first_unit, first_unit_g = one_iteration(y, np.ones(4))
    verified = bool(np.allclose(first_unit, expected['first'], atol=ATOL, rtol=0)
        and np.allclose(fixed, expected['fixed_observation'], atol=ATOL, rtol=0)
        and np.allclose(residual, expected['residual_observation'], atol=ATOL, rtol=0))
    return {'execution_kind': 'independent_book_numpy_calculations_not_nemo_execution',
        'mask_example': {'magnitude': 2., 'amplitude_mask': .5, 'masked_power': float((2*.5)**2),
                         'power_mask_interpretation': float(2**2*.5), 'unmasked_power': 4.},
        'iteration_example': {'input': y.tolist(), 'initial_power': [1,1,1,1],
            'taps': 1, 'delay': 1, 'history': 'zero left padding; all frames',
            'eps': 0., 'loading': 0., 'first_output': first_unit.tolist(),
            'first_coefficient': float(first_unit_g), 'second_power': (prescribed**2).tolist(),
            'fixed_observation_output': fixed.tolist(), 'fixed_observation_coefficient': float(fixed_g),
            'residual_observation_output': residual.tolist(), 'residual_observation_coefficient': float(residual_g),
            'independent_fraction_outputs': expected,
            'warning': 'Initial unit power is prescribed to isolate dataflow; this is not full MaskBasedDereverbWPE output'},
        'expected_behaviors_verified': verified,
        'boundary': 'No original NeMo function or Torch operator executed; no waveform or neural model'}


def run_audit(cache=CACHE):
    lock, status, lock_sha, status_sha = contracts.source_documents()
    entries, states = verify_acquisition(lock, status, lock_sha)
    before, files, identities = {}, {}, {}
    for project, spec in SOURCES.items():
        identity = contracts.verify_project(project, Path(cache)/project, spec['files'])
        identities[project] = identity
        before[project] = {'head': identity['head'], 'tracked_status': '', 'untracked_python': []}
        files[project] = {}
        for name, expected in spec['files'].items():
            row = identity['used_files'][name]
            if row['sha256'] != expected:
                raise RuntimeError('Original source hash/blob differs: '+project+'/'+name)
            files[project][name] = {'sha256': expected, 'git_blob': row['git_blob']}
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        path = Path(cache)/'nara_wpe/nara_wpe/wpe.py'
        spec = importlib.util.spec_from_file_location('fixed_original_nara_contracts', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nara = run_nara(module)
        nara['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
            'file': str(Path(w.filename).relative_to(Path(cache)/'nara_wpe')), 'line': w.lineno}
            for w in caught]
        text = path.read_text()
        tree = ast.parse(text)
        nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('build_y_tilde', 'online_wpe_step')}
        for name in ('_get_prediction', 'step_frame'):
            nodes['OnlineWPE.'+name] = _method(text, 'OnlineWPE', name)
        nara['original_snippets'] = {name: {'start_line': n.lineno, 'end_line': n.end_lineno,
            'sha256': hashlib.sha256(('\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])+'\n').encode()).hexdigest()}
            for name, n in nodes.items()}
    finally:
        sys.dont_write_bytecode = old_bytecode
    nemo_dir = Path(cache)/'nemo_wpe/nemo/collections/audio'
    nemo = inspect_nemo((nemo_dir/'modules/masking.py').read_text(),
        (nemo_dir/'parts/submodules/multichannel.py').read_text(), (nemo_dir/'models/enhancement.py').read_text())
    examples = independent_numpy_examples()
    for identity in identities.values():
        contracts.check_unchanged(identity)
    after = {p: {'head': i['head'], 'tracked_status': '', 'untracked_python': []}
             for p, i in identities.items()}
    if before != after or sha(LOCK) != lock_sha or sha(STATUS) != status_sha:
        raise RuntimeError('Sources, shared lock or acquisition status changed during audit')
    now = datetime.now(timezone.utc)
    return {'schema_version': 2, 'created_utc': now.isoformat(),
        'verified_date_asia_shanghai': now.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat(),
        'audit_source_sha256': sha(__file__), 'lock_sha256': lock_sha,
        'actual_dependency_sha256': contracts.dependencies(Path(__file__)),
        'source_identities': identities,
        'identity_selection_boundary': 'Used original file identity and complete acquisition selection are independent; no method run upgrades acquisition or full-framework execution',
        'source_status_sha256': status_sha, 'lock_entries': entries, 'acquisition_states': states,
        'original_files': files, 'before': before, 'after': after,
        'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
            'executable': sys.executable, 'upstream_bytecode_writes': False,
            'nemo_or_torch_imported': False},
        'nara': nara, 'nemo': nemo, 'book_examples': examples,
        'scope': {'identity': 'NARA original reference implementation; NeMo maintainer framework implementation',
                  'excluded': 'no full framework, weights, data, acoustic quality, timing or device validation'},
        'status': 'verified_original_nara_static_nemo_and_independent_book_examples' if
            nara['expected_behaviors_verified'] and nemo['all_facts_verified']
            and examples['expected_behaviors_verified'] else 'verification_failed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    target = contracts.report_target(args.report, CURRENT_REPORT, (CACHE,)) if args.report else None
    report = run_audit()
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if target is None:
        print(serialized, end='')
    else:
        contracts.write_report(target, report, CURRENT_REPORT, (CACHE,))
    return 0 if report['status'].startswith('verified_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
