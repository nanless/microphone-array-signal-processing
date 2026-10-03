"""Read-only, bounded numerical contracts for the fixed Acoular sources.

No package import, patch, download, dependency installation or cache is used.
Selected original function/method ASTs retain their mathematical bodies; only
decorators are removed. Explicit protocol objects replace Traits construction,
a local grid driver replaces guvectorize dispatch, and NumPy replaces SciPy FFT.
These are method-body calls, not a complete Acoular pipeline reproduction.

Known original differences remain differences. Only an explicit --report writes
a new current report; old reports and upstream checkouts are protected.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import warnings

import numpy as np

from codes.chapters.ch00.io_contracts import (
    strict_json_loads, validate_parent_chain, validate_report_destination,
    write_json_report,
)
from codes.chapters.ch00.upstream.fetch_upstreams import inspect_project, run_git

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CURRENT_REPORT = ROOT / 'codes/chapters/ch14/reports/upstream_imaging_contracts.json'
REVISION = '13d3d7df74ac1a8135c7ec71da098cbbc03d8652'
ORIGIN = 'https://github.com/acoular/acoular.git'
FILES = {
    'LICENSE': 'b5bc3bfa7c76d388170a8f29f8dc3047bc3ca0abcd3160781f4ec54d3e95f69f',
    'acoular/fbeamform.py': '451f0258cb176d663a57f33377078bb36d6fb1ad074f02033493aafaa5354d11',
    'acoular/fastFuncs.py': '07e8f4b07b5b47ff4703ae360d90b967f8407a8c95238371fb6d9f6ce5f97a5f',
    'acoular/spectra.py': '8ec8532714e3bf42e9fe5dc64d854cad418e4dbb974865d3ec55237ab0778db0',
    'acoular/version.py': '2223d2cbdcea241130dd9d0d52c1192de80ae000cdce1c8263f437aea9567935',
}
ATOL = 2e-14
CLEAN_ATOL = 2e-6  # Original BeamformerCleansc creates a float32 result array.


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ordinary_file(path):
    path = validate_parent_chain(path)
    if not path.is_file():
        raise ValueError('required ordinary file is absent: ' + str(path))
    return path


def report_target(path, cache=CACHE):
    """Reject protected destinations before running any original method.

    Inside this repository only the explicitly current report may be replaced.
    Ordinary external paths (for example test temporary directories) are allowed.
    This finite preflight does not claim race-proof or crash-proof publication.
    """
    forbidden = (CACHE, Path(cache), ROOT / 'reviews', LOCK,
                 Path(__file__), ROOT / 'tests/test_codes_imaging_contracts.py')
    target = validate_report_destination(path, forbidden_roots=forbidden)
    normalized = target.resolve()
    if normalized.is_relative_to(ROOT.resolve()) and normalized != CURRENT_REPORT.resolve():
        raise ValueError('only the current imaging report may be written inside the repository')
    return target


def verify_checkout(checkout, *, revision, origin, files):
    """Independently check origin, complete HEAD, cleanliness and actual blobs.

    Parameterization allows offline temporary-Git tests without changing the
    original cache. The production entry below always supplies fixed constants.
    """
    checkout = validate_parent_chain(checkout)
    if not checkout.is_dir() or not (checkout / '.git').is_dir():
        raise ValueError('an independent ordinary Git checkout is required')
    validate_parent_chain(checkout / '.git')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('a complete lowercase 40-character Git revision is required')
    def git(*args):
        return run_git(list(args), cwd=checkout)
    if git('rev-parse', '--show-toplevel') != str(checkout.resolve()):
        raise ValueError('checkout is not its own Git top level')
    if git('remote', 'get-url', 'origin') != origin:
        raise ValueError('official origin mismatch')
    if git('rev-parse', 'HEAD') != revision:
        raise ValueError('fixed HEAD mismatch')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('upstream worktree is not completely clean')
    sources = {}
    for relative, expected in files.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or '..' in Path(relative).parts):
            raise ValueError('source paths must be ordinary relative checkout members')
        path = ordinary_file(checkout / relative)
        content = path.read_bytes()
        head_blob = git('rev-parse', revision + ':' + relative)
        actual_blob = git('hash-object', '--', str(path))
        if digest(content) != expected or actual_blob != head_blob:
            raise ValueError('source SHA/blob mismatch: ' + relative)
        sources[relative] = {'sha256': digest(content), 'head_blob': head_blob,
                             'actual_blob': actual_blob,
                             'ordinary_file_verified': True}
    return {'origin': origin, 'head': revision, 'worktree_clean': True,
            'required_source_identity_verified': True, 'files': sources}


def verify_sources(cache=CACHE):
    lock_data = ordinary_file(LOCK).read_bytes()
    rows = strict_json_loads(lock_data)['projects']
    matching = [p for p in rows if p['id'] == 'acoular']
    if len(matching) != 1:
        raise ValueError('Acoular source lock entry must be unique')
    project = matching[0]
    if (project['revision'] != REVISION or project['url'] != ORIGIN
            or project['license'] != 'BSD-3-Clause'
            or any(p not in project['entrypoints'] for p in
                   ('LICENSE', 'acoular/fastFuncs.py', 'acoular/fbeamform.py'))):
        raise ValueError('fixed Acoular source lock identity changed')
    checkout = validate_parent_chain(Path(cache) / 'acoular')
    identity = verify_checkout(checkout, revision=REVISION, origin=ORIGIN, files=FILES)
    # Selection is inspected live, not relabeled from an old report or repaired.
    acquisition = inspect_project(project, Path(cache))
    version_tree = ast.parse((checkout / 'acoular/version.py').read_text())
    version = next(ast.literal_eval(n.value) for n in version_tree.body
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    if version != '26.08':
        raise ValueError('fixed source version changed')
    return {**identity, 'source_version': version,
            'source_lock_sha256': digest(lock_data),
            'source_lock_entry': project, 'acquisition_scope': acquisition,
            'license': {'name': 'BSD-3-Clause', 'path': 'LICENSE', 'sha256': FILES['LICENSE']}}


def extract_original(path, names, namespace, records, class_name=None):
    """Compile only identified definitions; retain their bodies and arguments.

    No surrounding module statements/imports/decorators execute. Metadata keeps
    original spans and both AST identities; dependencies are explicit namespace
    entries. There is no mathematical body rewriting or source-file mutation.
    """
    source = ordinary_file(path).read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    parents = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    chosen = {n.name: n for n in parents if isinstance(n, ast.FunctionDef) and n.name in names}
    if set(chosen) != set(names):
        raise ValueError('original definitions missing: ' + ', '.join(names))
    body = []
    for name in names:
        original = chosen[name]
        node = copy.deepcopy(original)
        node.decorator_list = []
        body.append(node)
        records.append({
            'source_file': str(Path(path).relative_to(Path(path).parents[1]))
                if Path(path).parent.name == 'acoular' else str(path),
            'qualified_name': (class_name + '.' if class_name else '') + name,
            'start_line': original.lineno, 'end_line': original.end_lineno,
            'removed_decorators': [ast.unparse(d) for d in original.decorator_list],
            'original_definition_ast_sha256': digest(ast.dump(original).encode()),
            'executed_definition_ast_sha256': digest(ast.dump(node).encode()),
            'mathematical_body_ast_sha256': digest(ast.dump(ast.Module(body=original.body, type_ignores=[])).encode()),
            'mathematical_body_unchanged': True,
        })
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def json_array(value):
    a = np.asarray(value)
    if np.iscomplexobj(a):
        return {'real': a.real.tolist(), 'imag': a.imag.tolist()}
    return a.tolist()


def comparison(name, actual, expected, *, tolerance=ATOL, should_match=True,
               original_expected=None, **details):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid numerical comparison: ' + name)
    matched = bool(np.allclose(a, b, rtol=0, atol=tolerance))
    if matched != should_match:
        raise ValueError('fixed original behavior differs from the recorded contract: ' + name)
    row = {'name': name, 'execution': 'limited original function/method body',
           'actual': json_array(a), 'independent_expected': json_array(b),
           'absolute_tolerance': tolerance,
           'max_absolute_error': float(np.max(np.abs(a - b))) if a.size else 0.,
           'matched_independent_expected': matched,
           'classification': ('matched_independent_expected' if matched
                              else 'observed_original_behavior_difference'), **details}
    if original_expected is not None:
        o = np.asarray(original_expected)
        if o.shape != a.shape or not np.isfinite(o).all() or not np.allclose(a, o, rtol=0, atol=tolerance):
            raise ValueError('independent prediction of original behavior failed: ' + name)
        row.update(independent_prediction_of_original_behavior=json_array(o),
                   matched_prediction_of_original_behavior=True)
    return row


class SteeringProtocol:
    """Known fixed transfer/weights only; no original Traits construction."""
    def __init__(self, transfer):
        self.h = np.asarray(transfer, dtype=complex)
        self.weights = (self.h / np.sum(np.abs(self.h) ** 2, axis=0)).T
        self.grid = SimpleNamespace(size=self.h.shape[1])

    def steer_vector(self, frequency, ind=None):
        return self.weights if ind is None else self.weights[[ind]]

    def transfer(self, frequency, ind=None):
        return self.h.T if ind is None else self.h[:, ind].T


def load_original_bodies(checkout):
    """Return bounded original kernels and explicit protocol/dispatch adapters."""
    records = []
    fast = {'np': np, 'nb': SimpleNamespace(prange=range)}
    names = ['calcCSM', 'damasSolverGaussSeidel',
             '_freqBeamformer_SpecificSteerVec_FullCSM',
             '_freqBeamformer_SpecificSteerVec_CsmRemovedDiag',
             '_freqBeamformer_EigValProb_SpecificSteerVec_FullCSM',
             '_freqBeamformer_EigValProb_SpecificSteerVec_CsmRemovedDiag']
    kernels = extract_original(checkout / 'acoular/fastFuncs.py', names, fast, records)

    def grid_driver(kind, remove_diag, norm, weights, csm):
        if kind != 'custom':
            raise ValueError('only the explicit custom-steering branch is exercised')
        output, normalization = np.zeros(len(weights)), np.zeros(len(weights))
        for i, weight in enumerate(weights):
            result, normal = np.zeros(1), np.zeros(1)
            if isinstance(csm, tuple):
                key = '_freqBeamformer_EigValProb_SpecificSteerVec_'
                fn = kernels[key + ('CsmRemovedDiag' if remove_diag else 'FullCSM')]
                fn(csm[0], csm[1], weight, np.array([norm]), result, normal)
            else:
                key = '_freqBeamformer_SpecificSteerVec_'
                fn = kernels[key + ('CsmRemovedDiag' if remove_diag else 'FullCSM')]
                fn(csm, weight, np.array([norm]), result, normal)
            output[i], normalization[i] = result[0], normal[0]
        return output, normalization

    def gauss_adapter(a, b, n_iter, damp, x):
        return kernels['damasSolverGaussSeidel'](
            a, b, np.atleast_1d(n_iter), np.atleast_1d(damp), x)

    # The sentinel deliberately selects the original custom-steering branches.
    env = {'np': np, 'SteeringVector': type('UnusedTraitsSteeringSentinel', (), {}),
           'beamformerFreq': grid_driver, 'damasSolverGaussSeidel': gauss_adapter}
    path = checkout / 'acoular/fbeamform.py'
    base = extract_original(path, ['sig_loss_norm', '_beamformer_params', '_calc'],
                            env, records, 'BeamformerBase')
    BaseBody = type('OriginalBaseBodyProtocol', (), base)
    psf_methods = extract_original(path, ['_psf_call'], env, records, 'PointSpreadFunction')

    class PSFBodyProtocol:
        _psf_call = psf_methods['_psf_call']

        def __init__(self, steer, **unused_traits):
            self.steer, self.freq = steer, 0.

        @property
        def psf(self):
            return self._psf_call(np.arange(self.steer.grid.size))

    env['PointSpreadFunction'] = PSFBodyProtocol
    damas = extract_original(path, ['_calc'], env, records, 'BeamformerDamas')
    DamasBody = type('OriginalDamasBodyProtocol', (BaseBody,), damas)
    clean = extract_original(path, ['_calc'], env, records, 'BeamformerCleansc')
    CleanBody = type('OriginalCleanscBodyProtocol', (BaseBody,), clean)
    cmf = extract_original(path, ['_get__csm_indices', '_build_dictionary', '_vectorize_csm'],
                           env, records, 'BeamformerCMF')
    CMFBody = type('OriginalCMFDictionaryProtocol', (), cmf)
    spectral_env = {'np': np, 'fft': np.fft, 'calcCSM': kernels['calcCSM']}
    spath = checkout / 'acoular/spectra.py'
    spectral = extract_original(spath, ['_get_source_data'], spectral_env, records, 'BaseSpectra')
    spectral.update(extract_original(spath, ['_get_num_blocks', 'calc_csm'],
                                     spectral_env, records, 'PowerSpectra'))
    SpectraBody = type('OriginalPowerSpectraBodyProtocol', (), spectral)
    return SimpleNamespace(kernels=kernels, gauss=gauss_adapter, driver=grid_driver,
                           Base=BaseBody, PSF=PSFBodyProtocol, Damas=DamasBody,
                           Clean=CleanBody, CMF=CMFBody, Spectra=SpectraBody,
                           extraction_records=records)


def configured_body(cls, steering, csm, *, remove_diag=False, iterations=20):
    obj = cls()
    obj.steer = steering
    obj.freq_data = SimpleNamespace(csm=np.asarray(csm)[None], num_channels=csm.shape[0])
    obj._f = np.array([1000.])
    obj.r_diag, obj.r_diag_norm = remove_diag, 0.
    obj._ac = np.zeros((1, steering.grid.size))
    obj._fr = np.zeros(1, dtype=int)
    obj.n_iter, obj.damp, obj.stopn = iterations, 0.6, 3
    obj.calcmode, obj.psf_precision, obj.cached = 'full', 'float64', False
    return obj


def beamforming_cases(original):
    h = np.array([[1, 1], [1, np.exp(-2j * np.pi / 3)]])
    q = np.array([1., .2])
    c = h @ np.diag(q) @ h.conj().T
    steering = SteeringProtocol(h)
    expected_psf = np.array([[1., .25], [.25, 1.]])
    psf = original.PSF(steering)._psf_call(np.arange(2))
    rows = [comparison('custom_full_csm_psf', psf, expected_psf,
                       transfer=json_array(h), weights=json_array(steering.weights),
                       target='abs(w_i^H a_j)^2; unit diagonal')]
    for remove_diag, expected in ((False, [1.05, .45]), (True, [.9, 0.])):
        obj = configured_body(original.Base, steering, c, remove_diag=remove_diag)
        before = obj.freq_data.csm.copy()
        raw = original.driver('custom', remove_diag, obj.sig_loss_norm(), steering.weights, c)[0]
        obj._calc([0])
        if not np.array_equal(obj.freq_data.csm, before) or obj._fr.tolist() != [1]:
            raise ValueError('original Base modified input or did not mark the result')
        rows.append(comparison('base_' + ('removed_diagonal' if remove_diag else 'full_csm'),
                               obj._ac[0], expected, csm=json_array(c), known_power=q.tolist(),
                               raw_before_original_clipping=raw.tolist(),
                               signal_loss_normalization=obj.sig_loss_norm(),
                               inputs_unchanged=True,
                               target='original Base behavior including default clipping'))
    # Original class starts at the dirty map; the scalar kernel wrapper only
    # replaces the scalar-to-array guvectorize argument boundary.
    for n, expected in ((1, [1.05 - .25 * .45, .45 - .25 * (1.05 - .25 * .45)]),
                        (20, [1., .2])):
        obj = configured_body(original.Damas, steering, c, iterations=n)
        obj.damp = 1.
        obj._calc([0])
        rows.append(comparison('damas_original_class_' + str(n) + '_sweeps', obj._ac[0], expected,
                               iterations=n, initial_solution=[1.05, .45], relax=1.,
                               target='analytic unit-diagonal two-cell iteration from dirty-map initial state'))
    # This intentionally differs from NNLS of ||A x-b||^2.
    a, b = expected_psf.copy(), np.array([1., 0.])
    x = b.copy()
    original.gauss(a, b, 20, 1., x)
    rows.append(comparison('damas_gs_is_not_map_residual_nnls', x, [16/17, 0.],
                           should_match=False, original_expected=[1., 0.],
                           matrix=a.tolist(), dirty_map=b.tolist(), iterations=20,
                           initial_solution=b.tolist(), relax=1.,
                           target='independent NNLS minimizing ||A x-b||^2',
                           original_residual_squared=float(np.sum((a @ x - b) ** 2)),
                           independent_nnls_residual_squared=1/17,
                           reason='clipped GS solves a different quadratic stationarity problem'))
    return rows


def clean_cases(original):
    steering = SteeringProtocol(np.ones((2, 1)))
    c = 2 * np.ones((2, 2), complex)
    rows = []
    for remove_diag in (False, True):
        for n in (1, 2, 4, 20):
            obj = configured_body(original.Clean, steering, c, remove_diag=remove_diag, iterations=n)
            before = obj.freq_data.csm.copy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                obj._calc([0])
            if not np.array_equal(before, obj.freq_data.csm):
                raise ValueError('original CLEAN-SC changed input CSM')
            paper_value = 2 * (1 - (1 - .6) ** n)
            t = np.sqrt(2) - 1
            original_value = paper_value if remove_diag else 2/t * (1 - (1 - .6*t) ** n)
            match = remove_diag or n == 1
            rows.append(comparison(
                'cleansc_' + ('removed_diagonal' if remove_diag else 'full_csm') + '_' + str(n),
                obj._ac[0], [paper_value], tolerance=CLEAN_ATOL, should_match=match,
                original_expected=[original_value], csm=json_array(c), iterations=n,
                r_diag=remove_diag, loop_gain=.6, known_source_power=2.,
                target='single-source paper full-CSM rank-one subtraction, 2*(1-.4**n)',
                original_internal_h_iterations=20, original_result_dtype='float32',
                source_locator='BeamformerCleansc._calc; original D1 always removes the CSM diagonal',
                reason=('diagonal-removed fixed-point matches this symmetric control' if remove_diag
                        else 'original full-CSM path retains diagonal-removed component iteration'),
                warnings=[{'category': w.category.__name__, 'message': str(w.message)} for w in caught],
                inputs_unchanged=True))
    return rows


def cmf_dictionary_cases(original, checkout):
    h = np.array([[1, 1], [1, 1j], [2, -1]], complex)
    q = np.array([2., .5])
    c = h @ np.diag(q) @ h.conj().T
    rows = []
    for remove_diag in (False, True):
        obj = original.CMF()
        obj.steer = SteeringProtocol(h)
        obj.freq_data, obj.r_diag = SimpleNamespace(num_channels=3), remove_diag
        obj._csm_indices = obj._get__csm_indices()
        a, b = obj._build_dictionary(1000.), obj._vectorize_csm(c)
        # Independent pair loops follow column-major upper triangle; no reshape
        # or original masks are used to construct the expected matrix/vector.
        pairs = [(m, n) for n in range(3) for m in range(n+1)]
        rpairs = [(m, n) for m, n in pairs if not remove_diag or m != n]
        ipairs = [(m, n) for m, n in pairs if m != n]
        expected_a = np.array([
            [float((h[m, j]*h[n, j].conjugate()).real) for j in range(2)] for m, n in rpairs]
            + [[float((h[m, j]*h[n, j].conjugate()).imag) for j in range(2)] for m, n in ipairs])
        expected_b = np.array([c[m, n].real for m, n in rpairs] + [c[m, n].imag for m, n in ipairs])[:, None]
        if not np.allclose(a, expected_a, rtol=0, atol=ATOL) or not np.allclose(b, expected_b, rtol=0, atol=ATOL):
            raise ValueError('original CMF dictionary/vectorization differs from independent pair loops')
        rows.append(comparison('cmf_dictionary_' + ('removed_diagonal' if remove_diag else 'full_csm'),
                               a @ q[:, None], expected_b, dictionary=a.tolist(),
                               vectorized_csm=b.tolist(), transfer=json_array(h), csm=json_array(c),
                               real_pairs=[list(p) for p in rpairs], imaginary_pairs=[list(p) for p in ipairs],
                               rank=int(np.linalg.matrix_rank(a)), known_power=q.tolist(),
                               dictionary_matches_independent_pairs=True,
                               vectorization_matches_independent_pairs=True,
                               estimator_execution='not_run',
                               target='original real half-triangle dictionary on a known noncoherent model'))
    obj = original.CMF()
    obj.steer = SteeringProtocol(np.array([[1.], [2.]]))
    obj.freq_data, obj.r_diag = SimpleNamespace(num_channels=2), False
    obj._csm_indices = obj._get__csm_indices()
    a = obj._build_dictionary(1000.)[:, 0]
    b = obj._vectorize_csm(np.diag([1., 10.]))[:, 0]
    reduced_q = float(a @ b / (a @ a))
    centered_a, centered_b = a-a.mean(), b-b.mean()
    intercept_q = float(centered_a @ centered_b / (centered_a @ centered_a))
    intercept = float(b.mean() - intercept_q * a.mean())
    if not np.allclose([reduced_q, intercept_q, intercept], [41/21, 87/35, -8/5], rtol=0, atol=ATOL):
        raise ValueError('independent CMF target counterexample failed')
    # Inspect the factory AST, without replacing a missing sklearn estimator.
    tree = ast.parse((checkout / 'acoular/fbeamform.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BeamformerCMF')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_calc')
    factory = [n for n in ast.walk(method) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == 'LinearRegression']
    if len(factory) != 1 or ast.unparse(factory[0]) != 'LinearRegression(positive=True)':
        raise ValueError('fixed original CMF estimator factory changed')
    rows.append(comparison('cmf_half_triangle_vs_full_frobenius', [reduced_q], [41/25],
                           should_match=False, original_expected=[41/21],
                           execution='original dictionary/vector methods plus independent one-variable optimization',
                           dictionary=a.tolist(), vectorized_csm=b.tolist(),
                           csm=[[1., 0.], [0., 10.]], transfer=[[1.], [2.]],
                           target='full-matrix Frobenius norm requires double off-diagonal weight',
                           reason='original half-triangle real/imag vector has no sqrt(2) weighting',
                           estimator_execution='not_run'))
    estimator = {
        'name': 'original_cmf_nnls_estimator', 'execution': 'not_run',
        'reason': 'no sklearn estimator is invoked; this bounded audit loads dictionary methods only',
        'dependency_present': importlib.util.find_spec('sklearn') is not None,
        'factory_static': ast.unparse(factory[0]), 'factory_line': factory[0].lineno,
        'fit_intercept_argument_explicit': False,
        'official_default_fit_intercept': True,
        'official_interface_source': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html',
        'independent_conditional_intercept_control': {
            'execution': 'independent analytic positive one-column regression; not original estimator',
            'assumption': 'fit_intercept=True default and positive coefficient constraint',
            'power': intercept_q, 'intercept': intercept,
            'through_origin_reduced_power': reduced_q, 'full_frobenius_power': 41/25,
            'original_column_normalization_does_not_remove_intercept': True,
        },
    }
    return rows, estimator


class SampleProtocol:
    def __init__(self, values):
        self.values = np.asarray(values)[:, None]
        self.num_samples, self.num_channels = self.values.shape

    def result(self, block_size):
        for start in range(0, self.num_samples, block_size):
            yield self.values[start:start+block_size]


def spectral_cases(original):
    rows = []
    definitions = [('integer_cosine', 128, 'cosine', .5, True),
                   ('nonintegral_block_count', 200, 'cosine', .32, False),
                   ('dc_endpoint', 128, 'dc', 2., False),
                   ('nyquist_endpoint', 128, 'nyquist', 2., False)]
    for name, n, kind, original_power, match in definitions:
        indices = np.arange(n)
        samples = (np.cos(2*np.pi*8*indices/128) if kind == 'cosine'
                   else np.ones(n) if kind == 'dc' else (-1.) ** indices)
        obj = original.Spectra()
        obj.source = SampleProtocol(samples)
        obj.num_channels, obj.block_size, obj.overlap_, obj.precision = 1, 128, 1., 'complex128'
        obj.window_ = np.ones
        obj.num_blocks = obj._get_num_blocks()
        actual_blocks = sum(1 for _ in obj._get_source_data())
        actual = obj.calc_csm()
        expected = .5 if kind == 'cosine' else 1.
        rows.append(comparison('power_spectra_' + name, [actual[:, 0, 0].real.sum()], [expected],
                               should_match=match, original_expected=[original_power],
                               input_sample_count=n, block_size=128, window='rectangular', overlap_factor=1.,
                               input_definition=('cos(2*pi*8*n/128)' if kind == 'cosine' else kind),
                               actual_complete_blocks=actual_blocks, original_num_blocks=obj.num_blocks,
                               target='analytic one-sided mean-square power of the complete 128-sample cosine/DC/Nyquist block',
                               reason=('nonintegral original divisor despite one complete consumed block' if n == 200
                                       else 'original multiplies all one-sided bins by two, including endpoints' if not match
                                       else 'interior integer-cycle frequency and integer block count'),
                               spectrum_diagonal_real=actual[:, 0, 0].real.tolist(),
                               fft_boundary='NumPy rfft replaces scipy.fft.rfft; original calcCSM body, prange replaced by range'))
    return rows


def run_audit(cache=CACHE):
    before = verify_sources(cache)
    checkout = validate_parent_chain(Path(cache) / 'acoular')
    original = load_original_bodies(checkout)
    rows = beamforming_cases(original) + clean_cases(original)
    cmf_rows, estimator = cmf_dictionary_cases(original, checkout)
    rows += cmf_rows + spectral_cases(original)
    after = verify_sources(cache)
    if before != after:
        raise ValueError('original checkout or source lock changed during the audit')
    matches = sum(r['matched_independent_expected'] for r in rows)
    direct_sources = [Path(__file__), ROOT / 'codes/chapters/ch00/io_contracts.py',
                      ROOT / 'codes/chapters/ch00/upstream/fetch_upstreams.py']
    report = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'audit_completed_with_original_differences',
        'source_identity_before': before, 'source_identity_after': after,
        'before_clean': True, 'after_clean': True,
        'direct_sources': {str(p.relative_to(ROOT)): digest(ordinary_file(p).read_bytes()) for p in direct_sources},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'platform': platform.platform()},
        'original_definitions': original.extraction_records,
        'execution_scope': {
            'package_imported': False, 'source_math_bodies_changed': False,
            'decorators_removed': True, 'traits_construction_replaced_by_protocol_objects': True,
            'custom_grid_driver_replaces_original_beamformerFreq_and_guvectorize_dispatch': True,
            'original_custom_scalar_beamformer_kernels_executed': True,
            'numpy_fft_replaces_scipy_fft': True, 'numba_prange_replaced_by_range': True,
            'psf_branch': 'original custom steering _psf_call, no cache/HDF5 path',
            'damas_class_initialization': 'original x=y.copy(), not tutorial zero initial state',
            'input': 'known deterministic two/three-microphone mathematical controls; no recording or dataset',
            'unexecuted': ['complete Acoular package/Traits graph', 'Numba JIT/parallel dispatch',
                           'HDF5 caching', 'CMF sklearn/scipy/pylops estimators',
                           'wind-tunnel data pipeline', 'GeneralFlow ODE/interpolation',
                           'moving-source or SODIX algorithms'],
        },
        'cases': rows, 'unexecuted_estimator': estimator,
        'primary_reference_locators': {
            'clean_sc_full_csm': {
                'title': 'Sijtsma, CLEAN based on spatial source coherence, NLR-TP-2007-345',
                'url': 'https://reports.nlr.nl/server/api/core/bitstreams/813b0521-b37c-4aff-be61-7a8bceaa06d4/content',
                'locator': 'section 4.6, equation 32: full-CSM component update'},
            'damas_cmf_target': {
                'title': 'Chardon, Picheral and Ollivier, Theoretical analysis of the DAMAS algorithm and efficient implementation of covariance matrix fitting for large-scale acoustic imaging',
                'url': 'https://gilleschardon.fr/papers/damascmf.pdf',
                'locator': 'DAMAS quadratic stationarity and covariance-fitting objective; applicable Gram-model conditions'},
        },
        'counts': {'limited_numerical_cases': len(rows), 'matched_independent_expected': matches,
                   'observed_original_behavior_differences': len(rows)-matches,
                   'unexecuted_estimators': 1},
        'interpretation': 'Audit completion is not algorithm correctness or complete source-selection success. Known differences and unexecuted entries are preserved.',
    }
    strict_json_loads(json.dumps(report, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE, help='existing fixed source-cache parent; never downloads')
    parser.add_argument('--report', type=Path, help='explicit current report or ordinary external path; default stdout only')
    args = parser.parse_args(argv)
    destination = report_target(args.report, args.cache) if args.report is not None else None
    report = run_audit(args.cache)
    if destination is not None:
        # Reuse the ordinary-path/strict-JSON/atomic-publication primitive.
        report_target(destination, args.cache)
        write_json_report(destination, report, forbidden_roots=(CACHE, args.cache, LOCK, ROOT / 'reviews'))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
