"""Read-only fixed-source DOA diagnostics with explicit original/helper/facade scope.

Run in an isolated SciPy + pyroomacoustics==0.10.0 environment. No installation,
network, cache writes, or upstream fixes occur here. --report is the sole opt-in
repository output; original failures are evidence, not successful estimators.
"""
from __future__ import annotations
import argparse
import cmath
import math
import ast
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import types
import warnings
sys.dont_write_bytecode = True
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
files = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()

def extract(project, relative, cls, name, globals_):
    """Execute the unchanged selected AST body after verifying its original blob."""
    record_original_file(project, relative)
    path = CACHE / project / relative
    tree = ast.parse(path.read_text())
    body = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)).body if cls else tree.body
    function = next((node for node in body if isinstance(node, ast.FunctionDef) and node.name == name))
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]

def independent_tops_norms(angles_deg):
    """Scalar, known-steering projection, no eigensolver or upstream spectrum.

    For K=1 each block norm is the length of t_k - a_k(a_k^H t_k).
    Known normalized true steering supplies a_k; the reference bin is excluded.
    """
    positions = [(0.0, 0.0), (0.045, 0.01), (-0.025, 0.05)]
    true_angle = math.pi / 6
    true_projection = [x * math.cos(true_angle) + y * math.sin(true_angle) for (x, y) in positions]
    scale = 2 * math.pi * 16000 / 256 / 343
    answer = []
    for angle_deg in angles_deg:
        angle = math.radians(angle_deg)
        projection = [x * math.cos(angle) + y * math.sin(angle) for (x, y) in positions]
        squared_norm = 0.0
        for frequency_bin in (10, 30):
            target = [cmath.exp(1j * scale * frequency_bin * q) / math.sqrt(3) for q in true_projection]
            transformed = [cmath.exp(1j * scale * (20 * q + (frequency_bin - 20) * p)) / math.sqrt(3) for (q, p) in zip(true_projection, projection)]
            inner = sum((a.conjugate() * t for (a, t) in zip(target, transformed)))
            squared_norm += sum((abs(t - a * inner) ** 2 for (a, t) in zip(target, transformed)))
        answer.append(math.sqrt(squared_norm))
    return answer
REVISIONS = {'pyroomacoustics': '0dd39f2614b7fc44b2cc63dbe7d60f4641068890', 'doatools': '9469db201e0418aef6b97583ef54b6fec2769502'}

def verify_checkout(project):
    directory = CACHE / project
    entry = next((p for p in json.loads(LOCK.read_text())['projects'] if p['id'] == project))
    if entry['revision'] != REVISIONS[project]:
        raise RuntimeError(f'unsupported source revision: {project}')
    if git(directory, 'rev-parse', 'HEAD') != REVISIONS[project]:
        raise RuntimeError(f'checkout HEAD differs: {project}')
    if git(directory, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError(f'modified tracked upstream files: {project}')
    untracked = git(directory, 'ls-files', '--others', '--exclude-standard')
    if any((path.endswith('.py') for path in untracked.splitlines())):
        raise RuntimeError(f'untracked Python source: {project}')
    return entry

def record_original_file(project, relative):
    directory = CACHE / project
    path = directory / relative
    original = subprocess.check_output(['git', '-C', str(directory), 'show', f'HEAD:{relative}'])
    if path.read_bytes() != original:
        raise RuntimeError(f'local source differs from HEAD: {relative}')
    files[f'{project}/{relative}'] = {'sha256': sha(path), 'git_blob': git(directory, 'rev-parse', f'HEAD:{relative}')}

def build_report():
    import pyroomacoustics as pra
    if importlib.metadata.version('pyroomacoustics') != '0.10.0':
        raise RuntimeError('requires pyroomacoustics==0.10.0')
    entries = {project: verify_checkout(project) for project in REVISIONS}
    files.clear()
    state_before = {p: {'head': git(CACHE / p, 'rev-parse', 'HEAD'), 'status': git(CACHE / p, 'status', '--porcelain', '--untracked-files=no')} for p in ('pyroomacoustics', 'doatools')}
    r = {}
    M = 3
    mode = np.ones((129, M, 1), complex)
    beta = np.zeros((2, 1), int)
    Rs = np.stack([np.diag([4, 1, 1]), np.diag([9, 1, 1]), np.diag([16, 1, 1])]).astype(complex)
    a = types.SimpleNamespace(M=M, freq_bins=np.array([20, 30]), mode_vec=mode)
    f = extract('pyroomacoustics', 'pyroomacoustics/doa/cssm.py', 'CSSM', '_coherent_sum', {'np': np})
    actual = f(a, Rs, 20, beta)
    expected = Rs[1] + Rs[2]
    r['cssm_filtered_mapping'] = {'original_diagonal': actual.diagonal().real.tolist(), 'independent_expected_diagonal': [25, 2, 2], 'input_covariance_diagonals': [[4, 1, 1], [9, 1, 1], [16, 1, 1]], 'remaining_bins': [20, 30], 'input_bins': [10, 20, 30], 'frobenius_error': float(np.linalg.norm(actual - expected)), 'scope': 'original _coherent_sum called after a simulated first-bin rejection; not acoustic full _process'}
    w = types.SimpleNamespace(M=M, num_src=1, freq_bins=np.array([20, 30]), mode_vec=mode, Z=np.zeros((3, 2), complex))
    w._subspace_decomposition = lambda R: (np.array([[1.0], [0.0], [0.0]], complex), np.array([[0, 0], [1, 0], [0, 1]], complex), np.array([R[0, 0].real]), np.array([1.0, 1.0]))
    f = extract('pyroomacoustics', 'pyroomacoustics/doa/waves.py', 'WAVES', '_construct_waves_matrix', {'np': np})
    f(w, Rs, 20, beta)
    r['waves_filtered_mapping'] = {'original_Z_first_row': w.Z[0].real.tolist(), 'independent_expected_first_row': [8 / np.sqrt(10), 15 / np.sqrt(17)], 'scope': 'original construction plus explicit eigensubspace adapter for diagonal covariances'}
    wheel_match = {}
    for name in ['doa.py', 'grid.py', 'music.py', 'tops.py', 'srp.py', 'normmusic.py', 'detect_peaks.py']:
        p = CACHE / 'pyroomacoustics/pyroomacoustics/doa' / name
        wp = pathlib.Path(pra.__file__).parent / 'doa' / name
        wheel_match[name] = {'cached_sha': sha(p), 'wheel_sha': sha(wp), 'same': p.read_bytes() == wp.read_bytes()}
        record_original_file('pyroomacoustics', 'pyroomacoustics/doa/' + name)
    if not all((x['same'] for x in wheel_match.values())):
        raise RuntimeError('installed DOA source differs from fixed cache')
    L = np.array([[0, 0.045, -0.025], [0, 0.01, 0.05]])
    fs = 16000
    nfft = 256
    bins = np.array([10, 20, 30])
    theta = np.deg2rad(30)
    dirs = np.deg2rad(np.arange(360))
    X = np.zeros((3, 129, 8), complex)
    Q = np.exp(2j * np.pi * np.arange(4)[:, None] * np.arange(8)[None, :] / 8)
    for (k, power) in zip(bins, [1.0, 4.0, 2.0]):
        av = np.exp(2j * np.pi * (fs * k / nfft) / 343 * (np.array([np.cos(theta), np.sin(theta)]) @ L))
        X[:, k, :] = np.sqrt(power) * av[:, None] * Q[0] + 0.1 * Q[1:4]
    t = pra.doa.TOPS(L, fs, nfft, num_src=1, azimuth=dirs)
    t.locate_sources(X, freq_bins=bins)
    expected = independent_tops_norms(range(360))
    orig = 1 / t.grid.values
    r['tops'] = {'original_peak_deg': float(np.rad2deg(t.azimuth_recon[0])), 'independent_peak_deg': float(np.argmin(expected)), 'original_min_singular_value': float(np.min(orig)), 'independent_at_true_deg': float(expected[30]), 'max_absolute_norm_difference': float(np.max(np.abs(orig - np.array(expected)))), 'bins': bins.tolist(), 'reference_bin': 20, 'geometry_m': L.tolist(), 'fs_hz': fs, 'nfft': nfft, 'sound_speed_m_per_s': 343, 'snapshots': 8, 'angles_deg': list(range(360)), 'selected_input_real': X[:, bins, :].real.tolist(), 'selected_input_imag': X[:, bins, :].imag.tolist(), 'all_original_singular_values': orig.tolist(), 'independent_projection_norms': expected, 'input': '3 noncollinear microphones; exact orthogonal 8 snapshot channels; 30deg target; signal powers 1,4,2 and noise covariance .01I'}
    try:
        s0 = pra.doa.SRP(L, fs, nfft, mode='near', r=np.array([0.2, 0.5]), azimuth=np.deg2rad([0, 90, 180]))
    except Exception as exc:
        r['srp_near_two_distances'] = {'exception_type': type(exc).__name__, 'message': str(exc)}
    s0 = pra.doa.SRP(L, fs, nfft, mode='near', r=np.array([0.2]), azimuth=np.deg2rad([0, 90, 180]))
    sf = pra.doa.SRP(L, fs, nfft, mode='far', azimuth=np.deg2rad([0, 90, 180]))
    r['srp_near_one_distance'] = {'n_points': s0.grid.n_points, 'candidate_radius': [0.2], 'grid_cartesian_norms': np.linalg.norm(s0.grid.cartesian, axis=0).tolist(), 'mode_vector_near_vs_far_max_difference': float(np.max(np.abs(s0.mode_vec[20, :, :] - sf.mode_vec[20, :, :])))}
    sys.path.insert(0, str(CACHE / 'doatools'))
    from doatools.model.sources import FarField1DSourcePlacement
    from doatools.estimation.core import get_noise_subspace, ensure_n_resolvable_sources
    f = extract('doatools', 'doatools/estimation/music.py', 'RootMUSIC1D', 'estimate', {'np': np, 'get_noise_subspace': get_noise_subspace, 'ensure_n_resolvable_sources': ensure_n_resolvable_sources, 'FarField1DSourcePlacement': FarField1DSourcePlacement})
    a = np.exp(1j * np.pi * np.arange(6) * 0.5)
    R = a[:, None] * a[None, :].conj() + 0.1 * np.eye(6)
    model = types.SimpleNamespace(_wavelength=1.0)
    try:
        f(model, R, 1)
        r['root_music_original'] = {'unexpected': 'success'}
    except Exception as exc:
        r['root_music_original'] = {'exception_type': type(exc).__name__, 'message': str(exc)}

    class Facade:

        def __getattr__(self, k):
            return np.complex128 if k == 'complex_' else getattr(np, k)
    g = extract('doatools', 'doatools/estimation/music.py', 'RootMUSIC1D', 'estimate', {'np': Facade(), 'get_noise_subspace': get_noise_subspace, 'ensure_n_resolvable_sources': ensure_n_resolvable_sources, 'FarField1DSourcePlacement': FarField1DSourcePlacement})
    (resolved, ans) = g(model, R, 1, unit='deg')
    r['root_music_facade'] = {'resolved': bool(resolved), 'angles_deg': ans.locations.tolist(), 'independent_expected_deg': [30.0], 'input_covariance_real': R.real.tolist(), 'input_covariance_imag': R.imag.tolist(), 'input_steering_real': a.real.tolist(), 'input_steering_imag': a.imag.tolist(), 'white_noise_power': 0.1, 'wavelength_m': 1, 'spacing_m': 0.5, 'facade': 'only np.complex_ -> np.complex128; original method AST body unchanged'}
    smooth = extract('doatools', 'doatools/estimation/preprocessing.py', None, 'spatial_smooth', {'np': np})
    try:
        smooth(np.array([[4, 0, 4], [0, 0, 0], [4, 0, 4]], dtype=int), 2)
    except Exception as exc:
        r['spatial_smooth_integer_input'] = {'exception_type': type(exc).__name__, 'message': str(exc)}
    v = np.array([2.0, 0.0, 2.0])
    coh = v[:, None] * v[None, :]
    r['spatial_smooth'] = {'original_forward': smooth(coh, 2).tolist(), 'independent_expected': [[2, 0], [0, 2]], 'original_fb': smooth(coh.astype(complex), 2, True).real.tolist()}
    # Shared E04-20 input; compute expectation with scalar log sums, not products.
    ld = extract('doatools', 'doatools/estimation/source_number.py', None,
                 'ld_stat', {'np': np, 'log': math.log})
    aic = extract('doatools', 'doatools/estimation/source_number.py', None,
                  'aic', {'np': np, 'ld_stat': ld})
    mdl = extract('doatools', 'doatools/estimation/source_number.py', None,
                  'mdl', {'np': np, 'ld_stat': ld, 'log': math.log})
    eigenvalues = [9.0, 4.0, 1.2, .8]
    sample_count = 100
    independent_F = []
    for k in range(4):
        tail = eigenvalues[k:]
        independent_F.append(sample_count * len(tail) *
                             (math.log(sum(tail)/len(tail)) -
                              sum(math.log(x) for x in tail)/len(tail)))
    original_ld = [float(ld(np.array(eigenvalues[::-1]), k, sample_count))
                   for k in range(4)]
    r['source_counts'] = {
        'descending_eigenvalues': eigenvalues, 'snapshots': sample_count,
        'original_ld': original_ld, 'independent_F': independent_F,
        'independent_AIC': [2*f + 2*k*(8-k) for k, f in enumerate(independent_F)],
        'independent_MDL': [f + .5*k*(8-k)*math.log(sample_count)
                            for k, f in enumerate(independent_F)],
        'original_AIC_choice': int(aic(np.array(eigenvalues[::-1]), sample_count)),
        'original_MDL_choice': int(mdl(np.array(eigenvalues[::-1]), sample_count)),
        'original_AIC_is_half_book_score': True,
        'original_MDL_common_offset': .5*math.log(sample_count),
    }
    with np.errstate(all='ignore'):
        underflow = float(ld(np.array(eigenvalues[::-1])*1e-300, 0, sample_count))
    r['ld_product_underflow'] = {
        'classification': 'finite' if math.isfinite(underflow) else
                          'positive_infinity' if underflow > 0 else 'nonfinite',
        'value': underflow if math.isfinite(underflow) else None,
        'input_scale': 1e-300,
        'independent_scale_invariant_F': independent_F[0],
        'scope': 'original eigenvalue product underflow; not a valid finite likelihood score',
    }
    state_after = {p: {'head': git(CACHE / p, 'rev-parse', 'HEAD'), 'status': git(CACHE / p, 'status', '--porcelain', '--untracked-files=no')} for p in state_before}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and pathlib.Path(filename).is_relative_to(CACHE / 'doatools'):
            record_original_file('doatools', str(pathlib.Path(filename).relative_to(CACHE / 'doatools')))
    for (project, license_file) in (('pyroomacoustics', 'LICENSE'), ('doatools', 'LICENSE.md')):
        record_original_file(project, license_file)
        verify_checkout(project)
    if state_before != state_after:
        raise RuntimeError('upstream checkout changed during execution')
    return {'run_utc': datetime.now(timezone.utc).isoformat(), 'run_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'audit_source_sha256': sha(pathlib.Path(__file__)), 'lock_sha256': sha(LOCK), 'lock_entries': entries, 'scope': {'TOPS': 'installed 0.10.0 original full class; seven relevant Python DOA files exactly equal fixed cache', 'CSSM_WAVES': 'original AST helper bodies with simulated first-frequency rejection; WAVES diagonal eigenspace adapter; original _process not run', 'RootMUSIC': 'original estimate method and fixed helper imports; original NumPy failure, then local complex_ alias facade', 'SRP': 'original constructor and mode-vector inspection only; no complete near-field renderer', 'spatial_smooth': 'original AST preprocessing method, float/complex success and integer dtype failure', 'source_counts': 'original AST ld_stat/aic/mdl; ascending inputs, scalar log-sum reference and classified product underflow', 'claims_excluded': ['fixed upstream patch', 'real audio accuracy', 'hardware validation', 'paper benchmark reproduction', 'FRIDA', 'sparse solvers', 'full package compatibility']}, 'tolerances': {'root_angle_deg': 1e-06, 'smoothing_absolute': 1e-14, 'independent_TOPS_true_norm': 1e-12, 'wheel_source_equal': 'byte exact'}, 'python': sys.version, 'executable': sys.executable, 'environment': {x: importlib.metadata.version(x) for x in ['numpy', 'scipy', 'pyroomacoustics', 'Cython', 'pybind11']}, 'before': state_before, 'after': state_after, 'sources': files.copy(), 'wheel_python_sources': wheel_match, 'results': r}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=pathlib.Path, help='explicitly write the current execution report')
    args = parser.parse_args()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = build_report()
    report['warnings'] = [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught]
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content, encoding='utf-8')
        print(args.report)
    else:
        print(content, end='')
if __name__ == '__main__':
    main()
