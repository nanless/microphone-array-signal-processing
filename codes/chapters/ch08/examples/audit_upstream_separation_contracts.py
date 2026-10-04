"""Fixed separation contracts: static source evidence and bounded original calls.

No downloads, patches, model weights, Torch/CuPy facades or bytecode writes.
Only --report writes a report. Run using the existing PRA 0.10.0 environment
for the original PRA calls; unavailable packages remain explicit failures.
The historical audit/report is not rewritten. Current helper dependencies are bound to their actual source hashes. NeMo examples below are independent NumPy
calculations, never executions of NeMo or Torch methods.
"""
from __future__ import annotations
# Preserve the direct-file and module entries.
if __name__ == "__main__" and not __package__:
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[4]))

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo
import numpy as np

from codes.chapters.ch04.core import upstream_contracts as contracts

CURRENT_REPORT = Path(__file__).resolve().parents[4] / "codes/chapters/ch08/reports/upstream_separation_contracts_current.json"
ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
CACHE = ROOT / 'codes/chapters/ch00/upstream/_downloads'
LEGACY = ROOT / 'codes/chapters/ch08/examples/audit_separation_upstream_interfaces.py'
ATOL = 1e-12
LEGACY_SHA = 'cb4dd83db4ead5c9973bf723deb372073f717f9fe4290c71fcbfd34d8c4e439b'
SOURCES = {'arraydps': {'files': {'LICENSE': '5d40b048a5fb04cc63482634406ddc2d9df2b14eddfe600c4f40a66872a4de0e',
                        'README.md': 'e4137cef5d18159114ab699d87b78ec76ef9426815b3d0cd45bbd186a214aad0',
                        'separate.py': 'd6dd09e1128189ca3cb77367c8e5d4a0a09c21a56364b8b72dcfd9502b074117',
                        'src/FCP.py': 'd93a4ac29cd934e5562665ee2bb525dc27452df85ca7b7a0ab2c53b191f3ff14',
                        'src/IVA.py': '7819ee4c1a111e666dac27241d0a268389cac4b6294ec4bfaa9ae6786d4d5eb9',
                        'src/sampler.py': 'ddaa9d7c77446e507f1275bdf5e81d8e8d7b2f9478ba65ce80c9349104cd294f',
                        'src/sampler_spatial_v1_reverb_iva_8kHz.py': 'e1ec8f304b342c91b2a8db5363253307df837a4f60ddc4b535dd9da1b0b8e62e',
                        'src/stft.py': '5f2ea16d6566b8162deae149870cee69061af3cd5d529941c6adad311b4d827f'},
              'license': 'MIT',
              'revision': '750ac2b7c75458f4ca5bad203dafda528f575e55'},
 'asteroid': {'files': {'LICENSE': 'c12aebc7a4eeeec2e482414004fd5d68275d7608552031cd48f4088b403f902d',
                        'asteroid/masknn/norms.py': '80bc9d54d9bbae5b3a110516cb927b5fc9a8ff72e10a257acb6ed4569c1e605b',
                        'asteroid/models/conv_tasnet.py': '29e0decce888c967d22c83a69b1d75d784854f8d345b9f62581568beba37eb3f'},
              'license': 'MIT',
              'revision': 'fce87469132760fbab41c20616ea0f0e079aad38'},
 'espnet': {'files': {'LICENSE': '4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5',
                      'espnet2/enh/separator/tfgridnet_separator.py': 'e08616b3e1964c7b0019320fff47110a4708fff44752173faff95aefed50e1f6'},
            'license': 'Apache-2.0',
            'revision': 'be79590bb2ff26ffb01bc825c5f68cb9418b7f0d'},
 'gss': {'files': {'LICENSE': '5658d3e38fcd75f27608d9ba7ee83a2c1a04cc7a76334509907b41c87b92fc8b',
                   'gss/beamformer/beamform.py': 'c62cd33cd2277935a4cbd66c0a8dfbc9ad815892e40a2d79f5d55010470c0865',
                   'gss/beamformer/souden_mvdr.py': '0b48b9cc30e4075bb168c0ba903a18ba359062ca3d9b683c7ba064f532a01a1b',
                   'gss/cacgmm/cacgmm.py': 'dcc46c325d71fe1861828d5330a919155e480f60a79273553e98deaed89b1e9b',
                   'gss/cacgmm/cacgmm_trainer.py': 'a3ecae7552203930392b6b5a75a26608aa238e3ae8b7ced2fcd30021f4c50c11',
                   'gss/cacgmm/utils.py': '065fea8ce2271a7ccb04f58faddb34ab6d94531ccaecd9b908a0f977a48f615e',
                   'gss/core/enhancer.py': '2e4ce14a9bd82e15471fa38e702aceb126db381d6642797f77b29bdaba208099',
                   'gss/core/gss.py': '22a56720db745c65be9cedf7c25980f4a3eac12a84ad1729e710071dd2f8a669'},
         'license': 'MIT',
         'revision': '10fad18cae85e2e4342c77421abc70c9c5da23ed'},
 'mamba_tasnet': {'files': {'LICENSE': '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986',
                            'README.md': '77d52b7bafb3d64c9d8811f962e7e468a93af144483975e7b468822fe408d20e'},
                  'license': 'GPL-3.0',
                  'revision': 'a35c692f27213781a11b1606c375cda1e1f0fb62'},
 'nemo_wpe': {'files': {'LICENSE': '43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1',
                        'README.md': '345642d7b5beda9834aa1fcd175de24cc4da5e9e17b9bf0b3f022a6d8f802579',
                        'nemo/collections/audio/modules/masking.py': '7f39caa90a0dfabefdc5579ac27a3ac5e5e850b697fc4c057d6ca7da74d69d5e'},
              'license': 'Apache-2.0',
              'revision': '2381f42f6979449b5b99538f8f80135831009b51'},
 'notsofar1': {'files': {'DATA_LICENSE': '9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94',
                         'LICENSE': 'c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383',
                         'css/css.py': 'f5994bd884d5ecf19ae043ef756ac7b3200072eb231f94f296b3a6694655559e',
                         'css/css_with_conformer/separate.py': 'fd5587483bbd4368e8c40c372b9e2a914a7bcae9cf006a55ef3dc899055defeb'},
               'license': 'MIT code; DATA_LICENSE and dataset-version restrictions separate',
               'revision': '6f58e08b008f7530ba4141f0aeb02447c70b6fd7'},
 'pyroomacoustics': {'files': {'LICENSE': '0922c9a0c1f5bb35a1e0df7b56954864e27cfc625fc3b041acde0707ce797e46',
                               'pyroomacoustics/bss/auxiva.py': '61016f0b27b8b4f1cfecdddb99c3f11f038552695abb299d10b6855d3c24ad6a',
                               'pyroomacoustics/bss/common.py': 'd142a27bdbd80e8a4f0c31e78759c9884f6ed44c0609b89e38902d86e50ee11c',
                               'pyroomacoustics/bss/fastmnmf.py': 'a13c253bc971d4dab7ae8f3d87ab7c309eb6b560b44e2b4c09b7018b50540518',
                               'pyroomacoustics/bss/fastmnmf2.py': '378ff43327cb7968e1ae08e0acf7b7099ad0267e11ae0559cf4d82a18b44093c',
                               'pyroomacoustics/bss/ilrma.py': '1bed68575f0bbfc899aade4f2f17f5210004b30ea7f0737b8acba3ccb5bd15b3',
                               'pyroomacoustics/bss/trinicon.py': '4202bb2b0834f982e246db37131c902fb078b533989512f81a6e162742b663fe'},
                     'license': 'MIT',
                     'revision': '0dd39f2614b7fc44b2cc63dbe7d60f4641068890'},
 'ssspy': {'files': {'LICENSE': '809ba860a2750092fd00a8ca4cb4dc459ae4f9f6e538f574ea52b8b1daae91b6',
                     'ssspy/algorithm/projection_back.py': '5cf34568d63971e296caafad84dd5ccf24cc09d220cb1e20569772ad972b17b6',
                     'ssspy/bss/cacgmm.py': '4cc823e115d8073043ed22cd597697b259f0b1f0ed23b02086d232af168c6219'},
           'license': 'Apache-2.0',
           'revision': '38b9389e8b1914422561f1936d9b28d042d62d2c'}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def git(directory, *args, binary=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    out = subprocess.check_output(['git', '-C', str(directory), *args], env=env,
                                  text=not binary, stderr=subprocess.PIPE)
    return out if binary else out.strip()


def checkout_state(directory, revision):
    if not (directory / '.git').exists() or \
            Path(git(directory, 'rev-parse', '--show-toplevel')).resolve() != directory.resolve():
        raise RuntimeError('Independent fixed checkout required')
    state = {'head': git(directory, 'rev-parse', 'HEAD'),
             'tracked_status': git(directory, 'status', '--porcelain', '--untracked-files=no'),
             'untracked_python': [s for s in git(directory, 'ls-files', '--others',
                 '--exclude-standard').splitlines() if s.endswith('.py')]}
    if state['head'] != revision or state['tracked_status'] or state['untracked_python']:
        raise RuntimeError('Revision or clean source condition differs')
    return state


def verify_sources(cache):
    identities = {}
    for name, spec in SOURCES.items():
        files = list(spec['files'])
        if name == 'ssspy':
            files.extend(p for p in contracts.git(cache/name, 'ls-files', '--', '*.py').splitlines()
                         if (cache/name/p).is_file())
        identity = contracts.verify_project(name, cache/name, files)
        for filename, expected in spec['files'].items():
            if identity['used_files'][filename]['sha256'] != expected:
                raise ValueError('Source/blob SHA mismatch: '+name+'/'+filename)
        identities[name] = identity
    return identities


def method(text, dotted):
    nodes = ast.parse(text).body
    node = None
    for name in dotted.split('.'):
        node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and n.name == name)
        nodes = getattr(node, 'body', [])
    return node


def snippet(text, dotted):
    node = method(text, dotted)
    segment = ast.get_source_segment(text, node)
    return {'symbol': dotted, 'line': node.lineno, 'end_line': node.end_lineno,
            'segment_sha256': hashlib.sha256(segment.encode()).hexdigest()}


def defaults(node):
    args = node.args.args
    return {arg.arg: ast.literal_eval(value) for arg, value in
            zip(args[-len(node.args.defaults):], node.args.defaults)
            if isinstance(value, (ast.Constant, ast.UnaryOp))}


def cli_defaults(text):
    result = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr == 'add_argument' and node.args:
            name = ast.literal_eval(node.args[0])
            for kw in node.keywords:
                if kw.arg == 'default':
                    result[name.removeprefix('--')] = ast.literal_eval(kw.value)
    return result


def readme_recipe(text):
    tokens = shlex.split(text.split('python separate.py', 1)[1].split('```', 1)[0].replace('\\\n', ' '))
    return {tokens[i][2:]: tokens[i+1] for i in range(len(tokens)-1) if tokens[i].startswith('--')}


def static_contracts(cache):
    def read(project, file):
        return (cache / project / file).read_text()
    driver = read('arraydps', 'separate.py')
    sampler = read('arraydps', 'src/sampler_spatial_v1_reverb_iva_8kHz.py')
    generic = read('arraydps', 'src/sampler.py')
    imports = [{'module': n.module, 'names': [a.name for a in n.names], 'line': n.lineno}
               for n in ast.walk(ast.parse(driver)) if isinstance(n, ast.ImportFrom)
               and any(a.name == 'Sampler' for a in n.names)]
    scalar_calls = [{'line': n.lineno, 'function': ast.unparse(n.func),
                     'arguments': [ast.unparse(a) for a in n.args]}
                    for n in ast.walk(ast.parse(driver)) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'batch_SDR_torch']
    stop_tests = [{'line': n.lineno, 'test': ast.unparse(n.test)}
                  for n in ast.walk(ast.parse(driver)) if isinstance(n, ast.If)
                  and ('max_snr' in ast.unparse(n.test) or 'max_trials' in ast.unparse(n.test))]
    relevant = ['n_fft', 'hop_length', 'fcp_epsilon', 'n_frames_past', 'n_frames_future',
                'max_trials', 'max_trials2', 'max_trials3', 'snr_stop', 'snr_stop2', 'snr_stop3']
    cls = defaults(method(sampler, 'Sampler.__init__'))
    cli = cli_defaults(driver)
    recipe = readme_recipe(read('arraydps', 'README.md'))
    array = {'execution_kind': 'static_AST_and_source_only', 'imports': imports,
             'sampler_symbols': [n.name for n in method(sampler, 'Sampler').body
                                 if isinstance(n, ast.FunctionDef)],
             'generic_sampler_symbols': [n.name for n in method(generic, 'Sampler').body
                                         if isinstance(n, ast.FunctionDef)],
             'truth_metric_calls': scalar_calls, 'stopping_conditions': stop_tests,
             'parameter_layers': {'class_defaults': {k: cls[k] for k in relevant if k in cls},
                                  'CLI_defaults': {k: cli[k] for k in relevant if k in cli},
                                  'README_recipe': {k: recipe[k] for k in relevant if k in recipe}},
             'paper_configuration': {'url': 'https://arxiv.org/html/2505.05657v3',
                 'location': 'Appendix C.3', 'n_fft': 512, 'hop_length': 64,
                 'fcp_epsilon': .001, 'kind': 'primary_text_read_not_code_execution'},
             'torch_weights_sampling_or_SDR_function_executed': False,
             'evidence': [snippet(sampler, x) for x in ['Sampler.__init__',
                          'Sampler.get_score_rec_guidance', 'Sampler.separate']]}
    mask = read('nemo_wpe', 'nemo/collections/audio/modules/masking.py')
    forward = method(mask, 'MaskEstimatorGSS.forward')
    nemo = {'execution_kind': 'static_AST_and_source_only',
            'forward_arguments': [a.arg for a in forward.args.args],
            'returned_expressions': [ast.unparse(n.value) for n in ast.walk(forward)
                                     if isinstance(n, ast.Return)],
            'component_count_assignment': [ast.unparse(n) for n in ast.walk(forward)
                                          if isinstance(n, ast.Assign) and
                                          any(isinstance(t, ast.Name) and t.id == 'num_outputs'
                                              for t in n.targets)],
            'background_component_added_by_this_class': False,
            'torch_or_NeMo_executed': False,
            'evidence': [snippet(mask, x) for x in ['MaskEstimatorGSS.normalize',
                         'MaskEstimatorGSS.update_masks', 'MaskEstimatorGSS.update_weights',
                         'MaskEstimatorGSS.update_pdf', 'MaskEstimatorGSS.forward',
                         'MaskBasedBeamformer.forward']]}
    enhancement = read('gss', 'gss/core/enhancer.py')
    gss_node = next(n for n in ast.walk(ast.parse(enhancement)) if isinstance(n, ast.FunctionDef)
                    and n.name == 'enhance_batch')
    observation = {'execution_kind': 'static_call_chain_only',
                   'ordered_relevant_calls': [{'line': n.lineno, 'source': ast.unparse(n)}
                       for n in sorted(ast.walk(gss_node), key=lambda n: getattr(n, 'lineno', 0))
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in ['wpe_block', 'gss_block', 'bf_block']],
                   'waveform_chain_or_CuPy_executed': False}
    demo = read('notsofar1', 'css/css_with_conformer/separate.py')
    demo_init = method(demo, 'Separator.__init__')
    demo_calls = [{'line': n.lineno, 'keywords': [k.arg for k in n.keywords]}
                  for n in ast.walk(ast.parse(demo)) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'Separator']
    identity_line = read('mamba_tasnet', 'README.md').splitlines()[0]
    identity_urls = re.findall(r'https://arxiv.org/abs/[0-9.]+', identity_line)
    if len(identity_urls) != 2:
        raise RuntimeError('Fixed README method identities differ')
    return {'arraydps': array, 'nemo_gss': nemo, 'gss_observation_chain': observation,
            'notsofar_independent_demo': {'execution_kind': 'static_signature_only',
                'constructor_arguments': [a.arg for a in demo_init.args.args], 'calls': demo_calls,
                'main_css_pipeline_executed_or_implied_failed': False},
            'mamba_identity': {'execution_kind': 'fixed_README_only',
                'README_line': identity_line,
                'Mamba_TasNet_paper': identity_urls[0],
                'Dual_Path_Mamba_paper': identity_urls[1]}}


def gss_power_calls(cache, legacy):
    text = (cache / 'gss/gss/beamformer/beamform.py').read_text()
    fn = legacy.extract_function(text, 'get_power_spectral_density_matrix', {'cp': np})
    x = np.array([[[1., 2., 0.], [0., 0., 3.]]], dtype=complex)
    direction = np.array([[[1., 1., 0.], [0., 0., 1.]]], dtype=complex)
    mask = np.array([[1., 1., 0.]])
    cases = [('power', x, mask, [[2.5, 0], [0, 0]]),
             ('direction', direction, mask, [[1, 0], [0, 0]]),
             ('double_amplitude', 2*x, mask, [[10, 0], [0, 0]]),
             ('below_mask_floor', x, mask*1e-12, [[.05, 0], [0, 0]]),
             ('empty_mask', x, np.zeros_like(mask), [[0, 0], [0, 0]])]
    rows = []
    for name, value, weight, oracle in cases:
        before = weight.copy()
        observed = fn(value, weight)
        expected = np.array(oracle)[None]
        error = float(np.max(abs(observed-expected)))
        if error > ATOL or not np.array_equal(weight, before):
            raise RuntimeError('Independent SCM expectation failed: '+name)
        rows.append({'case': name, 'observation': legacy.complex_record(value),
                     'mask': weight.tolist(), 'observed': legacy.complex_record(observed),
                     'independent_expected': legacy.complex_record(expected),
                     'max_abs_error': error, 'mask_unchanged': True})
    return {'execution_kind': 'unchanged_AST_function_with_NumPy_instead_of_CuPy',
            'evidence': snippet(text, 'get_power_spectral_density_matrix'),
            'facade': 'cp is NumPy; only floating masks; no asfarray/asnumpy compatibility shim',
            'CuPy_GSS_package_or_beamformer_chain_executed': False,
            'axes': 'frequency,sensor,time; mask frequency,time', 'cases': rows}


def nemo_book_calculations():
    alpha = np.array([.2, .3, .5])
    activity = np.array([0., 1., 1.])
    eps = 1e-8
    rows = []
    for log_pdf in [np.array([0., 0., 0.]), np.array([1000., 0., 0.])]:
        numerator = alpha*activity*np.exp(log_pdf-np.max(log_pdf))
        out = numerator/(sum(numerator)+eps)
        rows.append({'log_pdf': log_pdf.tolist(), 'observed_book_calculation': out.tolist(),
                     'sum': float(out.sum()), 'active_support_oracle': [0., .375, .625]})
    return {'execution_kind': 'independent_book_NumPy_calculation_not_upstream_execution',
            'alpha': alpha.tolist(), 'activity': activity.tolist(), 'eps': eps,
            'explanation': 'All-component maximum before activity; additive epsilon makes mass below one. '
                           'Inactive dominant density causes exponential underflow.', 'cases': rows}


def ssspy_package_attempt(cache):
    code = r'''
import json, sys, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
try:
    from pathlib import Path
    import importlib.machinery
    spec=importlib.machinery.PathFinder.find_spec('ssspy',sys.path)
    if spec is None or Path(spec.origin).resolve().parent != (Path(sys.argv[1])/'ssspy').resolve():
        raise ValueError('ssspy package location differs before import')
    from codes.chapters.ch08.examples.audit_separation_upstream_interfaces import source_imports, package_preflight, package_unchanged
    installed = package_preflight('ssspy', Path(sys.argv[1]), Path(sys.argv[1])/'ssspy')
    with source_imports('ssspy'):
        from ssspy.bss.cacgmm import CACGMM
    x=np.array([[[1,0,2,0]],[[0,1,0,2]]],dtype=complex)
    m=CACGMM(n_sources=2,normalization=False,permutation_alignment=False,
             record_loss=False,reference_id=0,rng=np.random.default_rng(3))
    y=m(x,n_iter=0)
    zero_iter={'shape':list(y.shape),'mask_reference_error':float(np.max(abs(y-m.posterior*x[0]))),
               'posterior':m.posterior.tolist(), 'output_real':y.real.tolist(), 'output_imag':y.imag.tolist()}
    m.unit_input=np.array([[[1,0]],[[0,1]]],dtype=complex)
    m.covariance=np.array([np.eye(2)[None],np.eye(2)[None]],dtype=complex)
    m.posterior=np.array([[[.75,.25]],[[.25,.75]]]); m.n_frames=2
    m.update_parameters()
    expected=np.array([[[[1.5,0],[0,.5]]],[[[.5,0],[0,1.5]]]])
    print(json.dumps({'status':'executed','zero_iterations':zero_iter,
                     'package_python_identity':package_unchanged(installed),
                     'shape_step':{'observed_real':m.covariance.real.tolist(),
                                   'observed_imag':m.covariance.imag.tolist(),
                                   'independent_expected':expected.tolist(),
                                   'error':float(np.max(abs(m.covariance-expected)))},
                     'n_iter_for_separation_quality':0},allow_nan=False))
except Exception as exc:
    print(json.dumps({'status':'failed','exception_type':type(exc).__name__,'message':str(exc)}))
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GIT_', 'PYTHONPATH'))}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    out = subprocess.run([sys.executable, '-B', '-c', code, str(cache/'ssspy')], env=env,
                         capture_output=True, text=True, timeout=30, check=True)
    result = contracts.strict_json_loads(out.stdout)
    result.update({'execution_kind': 'original_package_import_and_bounded_call_attempt',
                   'stderr': out.stderr, 'dependencies_installed': False,
                   'source_patched': False, 'bytecode_writes': False, 'python_loading': 'direct checked source; package pyc bypassed and preserved'})
    return result


def build_report(cache=CACHE):
    sys.dont_write_bytecode = True
    cache = Path(cache)
    cache = contracts.validate_parent_chain(cache)
    lock_sha = sha(LOCK)
    actual_dependencies = contracts.dependencies(Path(__file__), (LEGACY,))
    before = verify_sources(cache)
    spec = importlib.util.spec_from_file_location('ch08_original_helpers', LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    exec(compile(contracts.ordinary_file(LEGACY).read_bytes(), str(LEGACY), 'exec', dont_inherit=True), legacy.__dict__)
    legacy.verify_sources(cache)
    try:
        pra = legacy.run_pra(cache)
    except ModuleNotFoundError as exc:
        pra = {'execution_kind': 'original_package_unavailable',
               'exception_type': type(exc).__name__, 'message': str(exc)}
    now = datetime.now(timezone.utc)
    report = {'schema_version': 2, 'created_at': now.isoformat(),
              'verified_at_local': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
              'tool_sha256': sha(__file__), 'source_lock_sha256': lock_sha,
              'current_helper_sha256': sha(LEGACY),
              'historical_helper_sha256': LEGACY_SHA,
              'actual_dependency_sha256': actual_dependencies,
              'source_status_sha256': before['gss']['status_sha256'],
              'source_identities': before, 'source_contract_sha256': digest(SOURCES),
              'sources': SOURCES, 'verification': before,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'executable': sys.executable, 'platform': platform.platform(),
                              'bytecode_writes': False, 'upstream_patches': False},
              'static': static_contracts(cache),
              'executed': {'pra_original_methods': pra,
                           'projection_back_original_modules': legacy.run_projection_back(cache),
                           'gss_original_posterior_arithmetic': legacy.run_gss_arithmetic(cache),
                           'gss_original_power_SCM': gss_power_calls(cache, legacy)},
              'attempted': {'ssspy_cacgmm_package': ssspy_package_attempt(cache)},
              'book_calculations': {'nemo_mask_arithmetic': nemo_book_calculations()},
              'tolerance': {'SCM_abs': ATOL},
              'not_executed': ['Torch or CuPy', 'ArrayDPS sampling/SDR/weights',
                               'NeMo operators', 'NOTSOFAR pipeline', 'ASR',
                               'natural speech separation quality', 'hardware or latency']}
    for identity in before.values():
        contracts.check_unchanged(identity)
    if sha(LOCK) != lock_sha or sha(__file__) != report['tool_sha256']:
        raise RuntimeError('Audit inputs changed during execution')
    if contracts.dependencies(Path(__file__), (LEGACY,)) != actual_dependencies:
        raise ValueError('Current tool dependencies changed during audit')
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=CACHE)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    target = contracts.report_target(args.report, CURRENT_REPORT, protected=(args.source_root,)) if args.report else None
    report = build_report(args.source_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if target:
        contracts.write_report(target, report, CURRENT_REPORT, protected=(args.source_root,))
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
