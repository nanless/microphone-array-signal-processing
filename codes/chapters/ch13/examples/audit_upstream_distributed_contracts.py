"""Bounded original-source contracts for DANSE and synchronization helpers.

The fixed MATLAB source is inspected only, never executed or rewritten. Three
original Python function ASTs are executed with NumPy: no package import,
DWACD/online-WACD, resampler, network transport or GEVD-DANSE execution occurs.
Known short-input and unobserved-peak behavior remains visible. Mathematical
and update-order controls are independent rewrites, not MATLAB execution.
Default output is stdout; only an explicit ordinary --report writes a new
current or external report. Finite pre/post checks do not eliminate races or
provide crash-persistent publication guarantees.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import numpy as np

from codes.chapters.ch00.io_contracts import (
    strict_json_loads, validate_parent_chain, validate_report_destination,
    write_json_report,
)
from codes.chapters.ch00.core import source_history
from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch12.examples.audit_upstream_imaging_contracts import (
    digest, ordinary_file, verify_checkout, extract_original, comparison,
)

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
HISTORICAL_REPORT = ROOT / 'codes/chapters/ch13/reports/upstream_distributed_contracts.json'
CURRENT_REPORT = HISTORICAL_REPORT.with_name('upstream_distributed_contracts_current.json')
WOLA_REVISION = 'a24b73fcd2dc028659535d07bb08068b06108616'
PADER_REVISION = 'cd7054fcf72da637e4a5e11f035e8979691faf70'
WOLA_ORIGIN = 'https://github.com/AlexanderBertrandLab/Old_Code.git'
PADER_ORIGIN = 'https://github.com/fgnt/paderwasn.git'
WOLA_FILES = {'WOLA_DANSE1.m': '54fd630bd7f34ac3d9a97db2a30cee6c7a6a1bc496274b53f9ba9f756ce0772b'}
PADER_FILES = {
    'LICENSE': '60be4f44a5baa0db1ed1c9f1bc3ea5f7274b2032bb57bd87c054fa27efaa81fe',
    'README.md': '710dbf1ff641b3117dc5dd5a9a94efb913c044d5da427fd747c81364217e6bb0',
    'setup.py': '8fc652f65a3422f4a069a9ab783d429b58190c17e6691c35671f8e94011acfea',
    'paderwasn/synchronization/sync.py': '5b1f6e7f04c6587d5b7b9534d82bed9b7b1b85a7e395750f071457d0dc36bfb7',
    'paderwasn/synchronization/time_shift_estimation.py': '175289ea782c8bdbb7ec24247a61cdeb9fcba77079ec00cde89f73400a171713',
    'paderwasn/synchronization/utils.py': 'cae9707fef4e680930a42b4d54804773d5e1db8a5f5a780ed8ece300e34552dd',
    'paderwasn/synchronization/sro_estimation.py': '4f56cc299f3e86b46e343c398c2a95d45800e0aa1661bf080dda6de7323cf5dd',
}
GOLDEN_TOLERANCE = 1e-4  # Original helper default: interval width, in samples.


def report_target(path, cache=CACHE):
    target = validate_report_destination(path, forbidden_roots=(
        CACHE, Path(cache), LOCK, upstream.STATUS, source_history.SNAPSHOT_ROOT,
        HISTORICAL_REPORT, ROOT / 'reviews', Path(__file__),
        ROOT / 'tests/test_codes_distributed_contracts.py'))
    if target.resolve().is_relative_to(ROOT.resolve()) and target.resolve() != CURRENT_REPORT.resolve():
        raise ValueError('only the current distributed report may be written inside the repository')
    return target


def verify_sources(cache=CACHE):
    """Separate required-file identity from the entire locked sparse selection."""
    lock_bytes = ordinary_file(LOCK).read_bytes()
    projects = strict_json_loads(lock_bytes)['projects']
    identities = {}
    for identifier, revision, origin, files in (
            ('danse-wola', WOLA_REVISION, WOLA_ORIGIN, WOLA_FILES),
            ('paderwasn', PADER_REVISION, PADER_ORIGIN, PADER_FILES)):
        matches = [p for p in projects if p['id'] == identifier]
        if len(matches) != 1:
            raise ValueError('source lock entry must be unique: ' + identifier)
        project = matches[0]
        if project['revision'] != revision or project['url'] != origin:
            raise ValueError('fixed source lock identity changed: ' + identifier)
        if identifier == 'paderwasn' and project['license'] != 'MIT':
            raise ValueError('fixed paderwasn license changed')
        if identifier == 'danse-wola' and project['license'] != 'LicenseRef-Bertrand-DANSE-3-Conditions':
            raise ValueError('fixed custom MATLAB file-header license changed')
        if any(p not in project['entrypoints'] for p in files):
            raise ValueError('required source absent from locked entrypoints: ' + identifier)
        checkout = validate_parent_chain(Path(cache) / identifier)
        identity = upstream.verify_project(identifier, checkout, relatives=tuple(files))
        for relative, expected in files.items():
            if identity['used_files'][relative]['sha256'] != expected:
                raise ValueError('fixed used-source SHA changed: ' + relative)
        identities[identifier] = {
            **identity, 'source_lock_entry': project,
            'required_source_identity_verified': True,
            'acquisition_scope': identity['live_complete_selection'],
            'license': {'name': ('MIT' if identifier == 'paderwasn' else 'custom file header; three redistribution conditions; no standard BSD disclaimer'),
                        'path': ('LICENSE' if identifier == 'paderwasn' else 'WOLA_DANSE1.m'),
                        'sha256': files['LICENSE' if identifier == 'paderwasn' else 'WOLA_DANSE1.m']},
        }
    if any(row['lock_sha256'] != digest(lock_bytes) for row in identities.values()):
        raise ValueError('source lock changed during preflight')
    return {'source_lock_sha256': digest(lock_bytes),
            'source_status_sha256': identities['paderwasn']['status_sha256'],
            'projects': identities}


def actual_dependencies():
    return upstream.dependencies(__file__, extras=(
        ROOT / 'codes/chapters/ch12/examples/audit_upstream_imaging_contracts.py',
        ROOT / 'codes/chapters/ch00/core/source_history.py'))


def numpy_entry_identity():
    """Identify the actual namespace entry file, not the whole NumPy package."""
    path = ordinary_file(np.__file__)
    return {'path': str(path), 'sha256': digest(path.read_bytes()), 'version': np.__version__,
            'scope': 'actual imported NumPy entry file only; compiled modules and full installed dependency closure not verified'}


def historical_provenance():
    """Verify old bytes without assigning today's tool or status to old results."""
    payload = ordinary_file(HISTORICAL_REPORT).read_bytes()
    if digest(payload) != '8a8ef88ae9660ecb2fe8ea99d726652ae9aa44f91790e3f1f02d4d736ee8da4b':
        raise ValueError('historical distributed report bytes changed')
    report = strict_json_loads(payload)
    bindings = {}
    for relative, expected in report['direct_sources'].items():
        revision = ('a215b4630c0c21a8744cf27436a2c9ffa9c00053'
                    if relative == 'codes/chapters/ch14/examples/audit_upstream_imaging_contracts.py'
                    else '6c1f1448dc0efdeb2ea964b9a6323643406ab209')
        # The shared Git text interface strips whitespace. Read this one binary
        # object without altering any bytes, with the same injected-env isolation.
        environment = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                           GIT_TERMINAL_PROMPT='0', GIT_NO_LAZY_FETCH='1', GIT_NO_REPLACE_OBJECTS='1')
        original = subprocess.run(
            ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false',
             'show', revision + ':' + relative], cwd=ROOT, env=environment,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
        actual = digest(original)
        if actual != expected:
            raise ValueError('historical source differs from original Git bytes: ' + relative)
        bindings[relative] = {'commit': revision, 'sha256': actual}
    lock = source_history.verify_lock_binding(
        report['source_identity_before']['source_lock_sha256'], ('danse-wola', 'paderwasn'))
    return {'report_sha256': digest(payload), 'direct_source_git_bindings': bindings,
            'source_lock_binding': lock,
            'scope': 'historical bytes and lock provenance only; no historical rerun or current dependency substitution; old report did not bind SOURCE_STATUS'}


def load_original_bodies(checkout):
    records, namespace = [], {'np': np}
    directory = Path(checkout) / 'paderwasn/synchronization'
    for filename, name in (('utils.py', 'golden_section_max_search'),
                           ('time_shift_estimation.py', 'max_time_lag_search'),
                           ('sync.py', 'coarse_sync')):
        extract_original(directory / filename, [name], namespace, records)
    for record in records:
        record['source_file'] = str(Path(record['source_file']).relative_to(checkout))
    return SimpleNamespace(**{name: namespace[name] for name in
        ('golden_section_max_search', 'max_time_lag_search', 'coarse_sync')}, extraction_records=records)


def original_cases(original):
    rows = []
    for delay, polarity in ((-3, 1), (0, 1), (4, 1), (4, -1)):
        ref, sig = np.zeros(32), np.zeros(32)
        ref[8], sig[8 + delay] = 1., float(polarity)
        saved_sig, saved_ref = sig.copy(), ref.copy()
        aligned, aligned_ref, actual = original.coarse_sync(sig, ref, 32)
        if not np.array_equal(sig, saved_sig) or not np.array_equal(ref, saved_ref):
            raise ValueError('original coarse_sync mutated its input')
        if (len(aligned) != 32 - abs(delay) or len(aligned_ref) != 32 - abs(delay)
                or np.argmax(abs(aligned)) != np.argmax(abs(aligned_ref))):
            raise ValueError('coarse synchronization retained support differs')
        rows.append(comparison(f'coarse_delay_{delay}_polarity_{polarity}', [actual], [delay],
            tolerance=0, retained_lengths=[len(aligned), len(aligned_ref)],
            target='known finite impulse location difference, not an SRO estimate',
            input_peak_indices=[8 + delay, 8], inputs_unchanged=True))
    short = np.array([0., 1., 0., 0.])
    aligned, ref, offset = original.coarse_sync(short, short.copy(), 8)
    if len(aligned) != 0 or len(ref) != 0:
        raise ValueError('fixed short-input trimming behavior changed')
    rows.append(comparison('coarse_short_input_offset_baseline', [offset], [0],
        tolerance=0, should_match=False, original_expected=[-4],
        actual_input_length=4, requested_len_sync=8, retained_lengths=[0, 0],
        reason='original subtracts len_sync-1 although available correlation has length 2*4-1'))
    aligned, ref, offset = original.coarse_sync(np.zeros(8), np.zeros(8), 8)
    if offset != -7 or len(aligned) != 1 or len(ref) != 1:
        raise ValueError('fixed silence coarse behavior changed')
    rows.append({'name': 'coarse_silence_no_observable_peak', 'execution': 'limited original function body',
        'actual_offset': offset, 'retained_lengths': [len(aligned), len(ref)],
        'independent_identifiable_offset': None, 'peak_observed': False,
        'classification': 'unreliable_numeric_output_without_observation',
        'reason': 'all correlation values tie at zero; argmax returns first index; no confidence gate'})
    for lag in (0., 3., -5., 2.25, 31.75, 32., 33.):
        size = 64
        signed_bins = np.fft.fftfreq(size) * size
        spectrum = np.exp(-2j * np.pi * signed_bins * lag / size)
        spectrum[size // 2] = 0  # Real-signal Nyquist ambiguity excluded explicitly.
        saved = spectrum.copy()
        actual = float(original.max_time_lag_search(spectrum))
        # The search is periodic. Match modulo N, including near the signed boundary.
        error = (actual - lag + size / 2) % size - size / 2
        if abs(error) > GOLDEN_TOLERANCE or not np.array_equal(spectrum, saved):
            raise ValueError('original periodic lag search disagrees with analytic phase slope')
        rows.append(comparison('gcc_phase_lag_' + str(lag), [error], [0],
            tolerance=GOLDEN_TOLERANCE, actual_lag=actual, known_lag=lag,
            signed_wrapped_lag=(lag + size / 2) % size - size / 2,
            periodic_lag_error=error, fft_size=size, nyquist_bin=0,
            spectrum={'real': spectrum.real.tolist(), 'imag': spectrum.imag.tolist()},
            target='analytic spectral linear phase modulo 64 samples', inputs_unchanged=True,
            tolerance_interpretation='golden-section interval tolerance; not measured clock or device accuracy'))
    for peak in (-.3, .6):
        actual = original.golden_section_max_search(lambda x: -(x - peak) ** 2, (-1., 1.))
        rows.append(comparison('golden_quadratic_peak_' + str(peak), [actual], [peak],
            tolerance=GOLDEN_TOLERANCE, target='independent concave quadratic vertex'))
    actual = float(original.max_time_lag_search(np.zeros(64)))
    if not (-32.5 < actual < -31.5):
        raise ValueError('fixed unobserved GCC search tie behavior changed')
    rows.append({'name': 'gcc_zero_spectrum_no_observation', 'execution': 'limited original function body',
        'actual_lag': actual, 'independent_identifiable_lag': None, 'peak_observed': False,
        'classification': 'unreliable_numeric_output_without_observation',
        'search_interval': [-32.5, -31.5],
        'reason': 'coarse real IFFT and fractional magnitude objectives are identically zero; no validity result'})
    return rows


def matlab_static_contracts(path):
    source = ordinary_file(path).read_text()
    lines = source.splitlines()
    definitions = [
        ('frame_loop_no_tail_flush', 'for iter=1:L/2:lengthsignal-L',
         'MATLAB inclusive 1-based frames stop at lengthsignal-L; no terminal flush'),
        ('broadcast_before_filter_updates', "Zblock(k,u)=Wext{k}(:,u)'*Yblock{k}(:,u);",
         'old external filter broadcasts current local spectrum'),
        ('frame_start_vad', 'if onoff(iter)==1', 'only first sample label determines frame class'),
        ('noise_inverse_without_loading', 'Rnninv{k}(:,:,u)=inv(Rnn{k}(:,:,u));',
         'sample-count threshold does not prove positive definiteness'),
        ('all_internal_nodes_update', '%compute internal optimal filters at each node',
         'internal loop is outside sequential external-target token selection'),
        ('ordinary_evd', '[X,D]=eig(Rxx{k}(:,:,u));', 'ordinary EVD, not GEVD'),
        ('algebraic_maximum_then_absolute', '[Dmax,maxind]=max(diag(D));',
         'select algebraically greatest eigenvalue before applying abs, not greatest absolute eigenvalue'),
        ('absolute_selected_eigenvalue', "Rxx{k}(:,:,u)=X(:,maxind)*abs(Dmax)*X(:,maxind)';",
         'negative selected eigenvalue is turned positive'),
        ('internal_rank_one_mwf', 'Wint{k}(:,u)= (1/(mu+trace(P)))*P(:,1);',
         'rank-one SDW-MWF; not general full-rank MWF'),
        ('external_smoothing_before_target_event', 'Wext{k}(:,u)= lambda_ext*Wext{k}(:,u)+(1-lambda_ext)*Wext_target{k}(:,u);',
         'target event affects first broadcast two frames later'),
        ('both_classes_event_gate', 'if Ryysamples>=min_nb_samples && Rnnsamples>=min_nb_samples',
         'counts are class-specific, not a fixed wall-clock three-second interval'),
        ('sequential_external_target_only', 'Wext_target{updatetoken}=(1-alpha)*Wext_target{updatetoken}+alpha*Wint{updatetoken}(1:nbmicsnode(updatetoken),:);',
         'sequential token does not freeze all internal filters'),
        ('output_after_internal_update', "Yest(u)=Wint{k}(:,u)'*In{k}(:,u);",
         'new internal weights act on same-frame stacked spectra, without real transport'),
        ('symmetric_hann_without_cola_denominator', 'sqrt(hann(L)).*blockest;',
         'default symmetric MATLAB Hann, half hop; no COLA normalization'),
    ]
    rows = []
    for name, snippet, meaning in definitions:
        found = [j + 1 for j, line in enumerate(lines) if snippet in line]
        if not found:
            raise ValueError('fixed MATLAB static contract missing: ' + name)
        rows.append({'name': name, 'execution': 'static source inspection; MATLAB not run',
                     'line_numbers': found, 'snippet': snippet, 'interpretation': meaning})
    # Explicit dependence checks avoid treating substring presence as a parser proof.
    ordered_names = ('broadcast_before_filter_updates', 'frame_start_vad', 'ordinary_evd',
                     'internal_rank_one_mwf', 'external_smoothing_before_target_event',
                     'both_classes_event_gate', 'sequential_external_target_only', 'output_after_internal_update')
    snippets = {name: snippet for name, snippet, _meaning in definitions}
    positions = [source.index(snippets[name]) for name in ordered_names]
    if positions != sorted(positions):
        raise ValueError('MATLAB source statement order changed')
    return rows


def matlab_configuration(path):
    source = ordinary_file(path).read_text()
    required = ('L_default=512*fs/16000;', 'simultaneous_default=1;',
                'alpha_default=0.7;', 'mu_default=1;',
                'lambda=exp(log(0.5)/(2*nbsamples_per_sec));',
                'lambda_ext=exp(log(0.5)/(0.2*nbsamples_per_sec));',
                'min_nb_samples=3*nbsamples_per_sec;', 'plotresults=1;',
                "fname='results';", '%%  Version: 1.4')
    if any(text not in source for text in required):
        raise ValueError('fixed MATLAB default configuration changed')
    return {'execution': 'static source inspection; MATLAB not run',
            'version': '1.4', 'sample_rate': 'required fs argument; no sample-rate default',
            'default_fft_length_expression': '512*fs/16000',
            'default_simultaneous': 1, 'default_alpha': .7, 'default_mu': 1,
            'scm_half_life_seconds': 2., 'external_smoothing_half_life_seconds': .2,
            'event_gate': 'three seconds worth of frames collected separately in EACH VAD class',
            'plotting_hardcoded': True, 'save_basename': 'results',
            'paper_configuration': 'IWAENC2010 section IV uses fs=32000, L=512, mu=5, alpha=.5; not this source default',
            'paper_url': 'https://homes.esat.kuleuven.be/~abertran/reports/IWAENC10.pdf'}


def independent_matlab_controls():
    """Independent NumPy mathematics and copied scheduling, not original calls."""
    size = 512
    n = np.arange(size)
    hann = .5 - .5 * np.cos(2 * np.pi * n / (size - 1))
    cola = hann[:size // 2] + hann[size // 2:]
    if np.allclose(cola, 1., rtol=0, atol=1e-12):
        raise ValueError('symmetric Hann control unexpectedly meets unit COLA')
    scm = np.diag([-9., -1.])
    eigenvalues, vectors = np.linalg.eigh(scm)
    index = int(np.argmax(eigenvalues))
    reconstructed = abs(eigenvalues[index]) * np.outer(vectors[:, index], vectors[:, index])
    if not np.array_equal(reconstructed, np.diag([0., 1.])):
        raise ValueError('algebraic maximum then abs control differs')
    # An event after smoothing in frame zero changes target to three. Broadcast
    # occurs before smoothing in every frame; all internal node updates continue.
    external, target, timeline = 1., 1., []
    for frame in range(4):
        broadcast = external
        external = .5 * external + .5 * target
        if frame == 0:
            target = 3.
        timeline.append({'frame': frame, 'broadcast_weight': broadcast,
                         'post_smoothing_weight': external, 'post_event_target': target,
                         'internal_nodes_updated': [0, 1], 'external_target_token': 0 if frame == 0 else None})
    if [r['broadcast_weight'] for r in timeline] != [1., 1., 2., 2.5]:
        raise ValueError('independent two-frame update-order control differs')
    repeated_snapshots = np.ones((2, 8))
    singular_noise = repeated_snapshots @ repeated_snapshots.T / 8
    if np.linalg.matrix_rank(singular_noise) != 1:
        raise ValueError('repeated-snapshot singular covariance control differs')
    # MATLAB 1:L/2:N-L inclusive, translated explicitly to zero-based starts.
    loop_starts = list(range(0, 24 - 8, 4))
    if loop_starts != [0, 4, 8, 12]:
        raise ValueError('independent final-frame support control differs')
    return [
        {'name': 'symmetric_hann_cola', 'execution': 'independent NumPy mathematical control; not MATLAB execution',
         'window_length': size, 'hop': size // 2, 'cola_min': float(cola.min()),
         'cola_max': float(cola.max()), 'max_unit_error': float(abs(cola - 1).max()), 'unit_cola': False},
        {'name': 'negative_scm_algebraic_max_then_abs',
         'execution': 'independent NumPy mathematical control; not MATLAB execution',
         'input_scm': scm.tolist(), 'eigenvalues': eigenvalues.tolist(),
         'selected_algebraic_eigenvalue': float(eigenvalues[index]),
         'reconstructed_scm': reconstructed.tolist(), 'positive_part_projection': np.zeros((2, 2)).tolist(),
         'differs_from_positive_part_projection': True},
        {'name': 'external_target_two_frame_effect',
         'execution': 'independent scheduling rewrite; not original MATLAB execution',
         'smoothing': .5, 'event_frame': 0, 'first_affected_broadcast_frame': 2, 'timeline': timeline},
        {'name': 'many_noise_frames_do_not_imply_spd',
         'execution': 'independent NumPy mathematical control; not MATLAB execution',
         'noise_frame_count': 8, 'dimension': 2, 'noise_scm': singular_noise.tolist(),
         'rank': 1, 'inverse_exists': False},
        {'name': 'unprocessed_terminal_support',
         'execution': 'independent MATLAB-loop index rewrite; not original MATLAB execution',
         'input_length': 24, 'frame_length': 8, 'hop': 4,
         'zero_based_processed_starts': loop_starts, 'last_processed_exclusive_end': 20,
         'complete_frame_start_excluded': 16, 'terminal_samples_unprocessed': 4},
    ]


def dwacd_static_gate_control(path):
    """Inspect the original slice and evaluate a separate activity-only rewrite.

    No DWACD object, STFT, coherence estimator or resampler executes here.
    """
    payload = ordinary_file(path).read_bytes()
    source = payload.decode('utf-8')
    tree = ast.parse(source)
    assignments = {n.targets[0].id: n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in ('activity_seg_delayed', 'seg_delayed')}
    expected = {
        'activity_seg_delayed': 'activity_sig[start_delayed + shift:start + shift + self.seg_len]',
        'seg_delayed': 'sig[start_delayed + shift:start_delayed + shift + self.seg_len]',
    }
    if set(assignments) != set(expected) or any(
            ast.unparse(assignments[name].value) != value for name, value in expected.items()):
        raise ValueError('fixed DWACD delayed-window static contract changed')
    constructor = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == '__init__'
                       and {'seg_len', 'temp_dist', 'src_activity_th'} <= {a.arg for a in n.args.args})
    defaults = dict(zip([a.arg for a in constructor.args.args][-len(constructor.args.defaults):],
                        [ast.literal_eval(d) for d in constructor.args.defaults]))
    if (defaults['seg_len'], defaults['temp_dist'], defaults['src_activity_th']) != (8192, 8192, .75):
        raise ValueError('fixed DWACD default window contract changed')
    length, distance = defaults['seg_len'], defaults['temp_dist']
    activity = np.concatenate((np.zeros(length), np.ones(distance)))
    ref = np.ones(length + distance)
    start_delayed, shift, start = 0, 0, distance
    original_delayed = activity[start_delayed+shift:start+shift+length]
    matching_delayed = activity[start_delayed+shift:start_delayed+shift+length]
    current = activity[start+shift:start+shift+length]
    ref_current, ref_delayed = ref[start:start+length], ref[:length]
    threshold = defaults['src_activity_th'] * length
    original_counts = [int(np.sum(a)) for a in (ref_delayed, ref_current, original_delayed, current)]
    matching_counts = [int(np.sum(a)) for a in (ref_delayed, ref_current, matching_delayed, current)]
    original_gate = all(n > threshold for n in original_counts)
    matching_gate = all(n > threshold for n in matching_counts)
    if original_counts != [8192, 8192, 8192, 8192] or matching_counts != [8192, 8192, 0, 8192]:
        raise ValueError('independent delayed activity counts changed')
    if not original_gate or matching_gate:
        raise ValueError('independent delayed activity gate difference changed')
    return {
        'static_contract': {'source_file': 'paderwasn/synchronization/sro_estimation.py',
            'source_sha256': digest(payload), 'original_dwacd_executed': False,
            'assignments': {k: {'start_line': n.lineno, 'end_line': n.end_lineno,
                                'expression': ast.unparse(n.value)} for k, n in assignments.items()},
            'default_segment_length': length, 'default_distance': distance},
        'independent_control': {
            'execution': 'independent activity-slice and threshold rewrite only; not original DWACD execution',
            'segment_index': 0, 'shift': 0, 'microphone_activity': '8192 zeros followed by 8192 ones',
            'reference_activity': '16384 ones', 'threshold': threshold, 'comparison': 'strictly greater',
            'count_order': ['reference_delayed', 'reference_current', 'microphone_delayed', 'microphone_current'],
            'original_slice_length': len(original_delayed), 'matching_signal_slice_length': len(matching_delayed),
            'original_slice_counts': original_counts, 'matching_signal_slice_counts': matching_counts,
            'original_slice_gate': original_gate, 'matching_signal_slice_gate': matching_gate,
            'scope': 'one fixed-default activity control; no SRO estimate, coherence product or acoustic observation'},
    }


def run_audit(cache=CACHE):
    direct_before = actual_dependencies()
    external_before = numpy_entry_identity()
    history_before = historical_provenance()
    before = verify_sources(cache)
    wola = Path(cache) / 'danse-wola/WOLA_DANSE1.m'
    original = load_original_bodies(Path(cache) / 'paderwasn')
    rows = original_cases(original)
    static = matlab_static_contracts(wola)
    controls = independent_matlab_controls()
    dwacd = dwacd_static_gate_control(Path(cache) / 'paderwasn/paderwasn/synchronization/sro_estimation.py')
    after = copy.deepcopy(before)
    for identifier, identity in after['projects'].items():
        after['projects'][identifier] = upstream.check_unchanged(identity)
    if before != verify_sources(cache):
        raise ValueError('fixed sources or source lock changed during the audit')
    direct_after, external_after = actual_dependencies(), numpy_entry_identity()
    history_after = historical_provenance()
    if direct_before != direct_after or external_before != external_after or history_before != history_after:
        raise ValueError('actual dependencies or historical provenance changed during execution')
    report = {
        'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'audit_completed_with_original_differences_and_unobserved_peaks',
        'source_identity_before': before, 'source_identity_after': after,
        'direct_sources': direct_after,
        'actual_dependencies_before': direct_before, 'actual_dependencies_after': direct_after,
        'external_namespace_before': external_before, 'external_namespace_after': external_after,
        'actual_dependencies_unchanged': True, 'historical_provenance': history_after,
        'source_lock_sha256': before['source_lock_sha256'],
        'source_status_sha256': before['source_status_sha256'],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
        'original_definitions': original.extraction_records,
        'execution_scope': {
            'python_original_functions_executed': ['coarse_sync', 'max_time_lag_search', 'golden_section_max_search'],
            'package_imported': False, 'original_function_bodies_unchanged': True,
            'module_imports_executed': False, 'namespace_dependencies': ['numpy'],
            'decorators': 'none present in these three selected definitions; extraction removes any decorators',
            'matlab_executed': False, 'octave_executed': False,
            'matlab_status': 'not executed in this audit; runtime availability not probed; no installation attempted',
            'matlab_runtime_probed': False,
            'dwacd_executed': False, 'online_wacd_executed': False,
            'sro_resampler_executed': False, 'gevd_danse_executed': False,
            'network_transport_executed': False, 'datasets_or_recordings_used': False,
            'unexecuted_module_dependencies': {
                'sync.py': ['scipy.signal.windows.hann'],
                'time_shift_estimation.py': ['scipy.signal.windows.blackman', 'paderbox.array.segment.segment_axis'],
                'utils.py': ['paderbox.array.segment_axis'],
                'sro_estimation.py': ['paderbox.transform.STFT', 'max_time_lag_search'],
            },
            'setup_install_requires': 'empty list in fixed setup.py; README supplies pinned paderbox/lazy_dataset separately',
        },
        'cases': rows, 'matlab_static_contracts': static,
        'matlab_configuration': matlab_configuration(wola), 'independent_matlab_controls': controls,
        'additional_dwacd_static_activity_control': dwacd,
        'counts': {'original_case_invocations': len(rows),
                   'matched_independent_expected': sum(r.get('matched_independent_expected', False) for r in rows),
                   'observed_original_behavior_differences': sum(r.get('classification') == 'observed_original_behavior_difference' for r in rows),
                   'unobserved_peaks_with_numeric_output': sum(r.get('peak_observed') is False for r in rows)},
        'interpretation': 'Completion verifies bounded helper contracts and source identity. It does not certify all algorithm correctness, source selection, MATLAB execution, device SRO accuracy or distributed network operation.',
    }
    strict_json_loads(json.dumps(report, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE, help='existing fixed checkout parent; never downloads')
    parser.add_argument('--report', type=Path, help='explicit current report or ordinary external path; default stdout')
    args = parser.parse_args(argv)
    destination = report_target(args.report, args.cache) if args.report is not None else None
    report = run_audit(args.cache)
    if destination is not None:
        report_target(destination, args.cache)
        write_json_report(destination, report, forbidden_roots=(
            CACHE, args.cache, LOCK, upstream.STATUS, source_history.SNAPSHOT_ROOT,
            HISTORICAL_REPORT, ROOT / 'reviews'))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
