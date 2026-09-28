"""Bounded WPE interface audit, without downloads, patches, models or audio.

NARA calls execute its hash-checked original module. Other implementations are
read statically. An optional MetaAF function is extracted unchanged from its AST
and called with the genuine NARA state class and a synthetic STFT boundary.
This is not an audio transform or a JAX/MetaAF package run.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

CODES = Path(__file__).resolve().parents[1]
SOURCES = {'btk20': {'files': {'LICENSE': '1dfe95044a48d8c90dde9fe0d32fa75beba51633b866cc91f10b803489edeec0',
                     'btk20_src/dereverberation/dereverberation.cc': '56dd5e81a2da21f9ab1edd8a0169473233c68a188112534fa706cebbfeac2cad'},
           'revision': 'feff19ec8bcb770f6530fe280dc3ccafc2f5984a'},
 'espnet': {'files': {'LICENSE': '4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5',
                      'espnet2/enh/layers/dnn_wpe.py': 'a71db01cb84be5b42096f926c800d5f358c1b7f45b021ca1147ce78816ed1736',
                      'espnet2/enh/layers/mask_estimator.py': '267b76279f58bf5d6c0054f30b1866ea5adb120231dac6f58e411a7873eaddd7',
                      'espnet2/enh/layers/wpe.py': '1648649392558463d27d727d0a0a17cbba02efce46ed3b897f85864bb1dda6a1'},
            'revision': 'be79590bb2ff26ffb01bc825c5f68cb9418b7f0d'},
 'gss': {'files': {'LICENSE': '5658d3e38fcd75f27608d9ba7ee83a2c1a04cc7a76334509907b41c87b92fc8b',
                   'gss/core/enhancer.py': '2e4ce14a9bd82e15471fa38e702aceb126db381d6642797f77b29bdaba208099',
                   'gss/wpe/wpe.py': '37d4ba884a341a86238e9e0d0e85bea27d07380206959fbe341651fdda9a8be2'},
         'revision': '10fad18cae85e2e4342c77421abc70c9c5da23ed'},
 'metaaf': {'files': {'metaaf/LICENSE': '4281265bd0d7692b781b56f3395f9673e9a67e74b10348e470a6c43a4259d3fb',
                      'zoo/LICENSE': '24d3e5484e1fa549be9d7b4ea8c1534a6f0227c34f6e7ab602f0e32a80728d9c',
                      'zoo/wpe/nara_wpe_eval.py': 'c42b71b4d77cc564c40965e8612569cbac1decce62dceccc688fd2ce8b4a0b3b',
                      'zoo/wpe/wpe.py': '47e7f846b09f85c44a76134f504bdb1fa781bf7c394740c29d4f2196cc03ec37',
                      'zoo/wpe/wpe_eval.py': '18924db865c36d6898b4f748ae9ff7d6f8d64bd801293f694c8d8a4628fb1b8e'},
            'revision': '56c4665bdc51c2e0595a7c0cd9b1266408adceff'},
 'nara_wpe': {'files': {'LICENSE': 'f9fad5befd31a5c5540fb671fa56efd59d959959ffe1b6a8156a66549d6410f0',
                        'nara_wpe/tf_wpe.py': '54ca54c9b1e10d1188142063caf54fa9322f30da5f897a0841f1f60d96762fcf',
                        'nara_wpe/utils.py': '7ec579df10b18f1eca38a0d9c797d242a89f0344fa98e2fc2b0cc51fe0332570',
                        'nara_wpe/wpe.py': '385a6f1c67071ba3e243c8a24fe4a041484c55280ad4ed0dbadefa4679a606d2'},
              'revision': 'a166779cca2088817e330481bd20af1a2c598555'},
 'speexdsp': {'files': {'COPYING': '2654a4264b2bfe298dedc508748d140111840c315cc8eb646a3a68c13fa75b01',
                        'libspeexdsp/preprocess.c': '5f7143d00f12af60759ea1b9249fba3615fd92fcead2671e9350fe6cf530ecbd'},
              'revision': '8e29a256ef0235ebbe7fcb8417b5ac7731eb8307'},
 'tso_vace_wpe': {'files': {'LICENSE': '664fb260188108b8c9b3e21ab0c8c8f57c94b421f8134dc1544ede4627d2713b',
                            'README.md': '816199b90fee133396dbab2c76e432d5ff16e6968d531fc4be0051691b404422',
                            'bldnn_4M62.py': '038fa1948d05cf548f1cc48bf5905f457b1f8288560cd2583e02d0f80aeb08b1',
                            'gcunet4c_4M4390.py': 'fff0244234a5137b095f02889a059f83bd360e54f735ebf0ab611e936125f69a',
                            'run.py': '3315e9ff2e3f6f9edda995d24a1e18548a6d0233ed894b96257fad8aad21e6d5',
                            'torch_custom/custom_layers.py': 'f41ac5e906317b0148a40efcb900be83ab91641fd6886f5ca20e3d4cc9bb8837',
                            'torch_custom/iterative_wpe.py': 'ebcde972a551d38129092add41c032782f54f032693a25f65220d95d71bdef16',
                            'torch_custom/math_utils.py': 'dfa13868bcecd57572fb6de42337d279057f37e6dff60c4eb8545f6325dfca73',
                            'torch_custom/neural_wpe.py': '03b4bb34f58719b401b09c3105184c4e423a8f68007b46d093b19445621761aa',
                            'torch_custom/signal_utils.py': 'c1ea6a5f4ee1e72e034de60e89331979c2eef4f3df3018d03dbf8efc49eba125',
                            'torch_custom/spectral_ops.py': 'a4803ae54739a0a02b692c533961237866767d110e5e6446f69380a55a00ce0a',
                            'torch_custom/stft_helper.py': '9ececfe942834f543e86f6f584923ba529869e5788fa06993c7bd0dead62a47b',
                            'torch_custom/torch_utils.py': '7bb348ca3fed5fd7fb8d392a589211ac39e71d917dc26c8e939f7d4209a2851d',
                            'torch_custom/wpe_th_utils.py': 'f10e2323dc0737232669fe21755192c0885b40506439643b887cf6bb646d171c',
                            'vace_wpe.py': 'bb64a67de37257d94beaa59cc3f04203c9f6b0a9a394867de2eff285d5b63e29'},
                  'revision': '10ee77dd020d58af508feb77251f9353208cb33a'}}
CONFIG = {
    "input_real": [[[1., 2., 1., 4., 2., 1.]],
                   [[1e-8, 3e-8, 2e-8, 1e-8, 5e-8, 2e-8]]],
    "input_imag": "all zero", "dtype": "complex128", "axis_order": "frequency,channel,time",
    "taps": 1, "delay": 1, "iterations": 1, "statistics_mode": "valid",
    "psd_context": 0, "helper_input": "arange(1,13).reshape(2,1,6)",
    "helper_contexts": [1, [0, 2]], "randomness": "none",
    "scope": "dimensionless synthetic spectra; no acoustic quality or speed claim",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_sha256():
    return hashlib.sha256(json.dumps({"sources": SOURCES, "config": CONFIG},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source(path, spec):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], env=env,
                                       text=True, stderr=subprocess.PIPE).strip()
    if not (path / ".git").exists() or Path(git("rev-parse", "--show-toplevel")).resolve() != path.resolve():
        raise ValueError("expected independent checkout; fetch separately")
    if git("rev-parse", "HEAD") != spec["revision"]:
        raise ValueError("revision mismatch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("upstream tracked files modified")
    for name, digest in spec["files"].items():
        if sha256(path / name) != digest:
            raise ValueError("source hash mismatch: " + name)


def load_original(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_nara(module):
    y = np.asarray(CONFIG["input_real"], dtype=CONFIG["dtype"])
    kwargs = {k: CONFIG[k] for k in ("taps", "delay", "iterations", "statistics_mode", "psd_context")}
    variants = {}
    for name in ("wpe_v6", "wpe_v7", "wpe_v8"):
        z = getattr(module, name)(y, **kwargs)
        variants[name] = {"output_real": z.real.tolist(), "output_imag": z.imag.tolist(),
                          "finite": bool(np.isfinite(z).all())}
    helper = np.arange(1., 13.).reshape(2, 1, 6)
    try:
        module.get_power(helper, psd_context=(0, 2))
    except Exception as exc:
        tuple_case = {"exception_type": type(exc).__name__, "message": str(exc)}
    else:
        tuple_case = {"exception_type": None}
    return {
        "execution_kind": "hash_checked_original_module_functions",
        "default_alias_is_v7": module.wpe is module.wpe_v7,
        "variants": variants,
        "inverse_power_batched": module.get_power_inverse(y).tolist(),
        "inverse_power_frequency1_alone": module.get_power_inverse(y[1]).tolist(),
        "get_power_context1": module.get_power(helper, psd_context=1).tolist(),
        "get_power_context_0_2": tuple_case,
        "note": "get_power helper is not the get_power_inverse path used by offline variants",
    }


def inspect_espnet(dnn_text, mask_text, low_text):
    """Bounded AST facts, not framework execution or a full dataflow proof."""
    dnn = ast.parse(dnn_text)
    forward = next(n for n in ast.walk(dnn) if isinstance(n, ast.FunctionDef) and n.name == "forward")
    calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "wpe_one_iteration"]
    keyword_names = sorted({k.arg for c in calls for k in c.keywords})
    reads = sorted({n.attr for n in ast.walk(forward) if isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name) and n.value.id == "self"})
    ignored = [x for x in ("diagonal_loading", "diag_eps", "use_torch_solver") if x not in reads]
    mask = ast.parse(mask_text)
    dropped = [n.lineno for n in ast.walk(mask) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "masked_fill"]
    low = ast.parse(low_text)
    solver = next(n for n in ast.walk(low) if isinstance(n, ast.FunctionDef)
                  and n.name == "get_filter_matrix_conj")
    defaults = dict(zip([a.arg for a in solver.args.args][-len(solver.args.defaults):],
                        [ast.unparse(d) for d in solver.args.defaults]))
    return {"execution_kind": "static_ast_only", "wpe_call_keywords": keyword_names,
            "public_controls_not_read_in_forward": ignored,
            "discarded_masked_fill_lines": dropped, "solver_defaults": defaults,
            "inverse_call_lines": [n.lineno for n in ast.walk(solver) if isinstance(n, ast.Call)
                                   and isinstance(n.func, ast.Attribute) and n.func.attr == "inverse"],
            "framework_or_network_executed": False}


def extract_function(text, name, globals_):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.Module(body=[node], type_ignores=[])
    namespace = dict(globals_)
    exec(compile(code, "<unchanged-original-function>", "exec"), namespace)
    return namespace[name]


def run_metaaf(root, nara):
    text = (root / "zoo/wpe/nara_wpe_eval.py").read_text()
    spectra = np.arange(1., 73.).reshape(1, 8, 9).astype(complex)
    captured = {}
    def inverse_boundary(z, **_kwargs):
        captured["spectra"] = z.copy()
        return np.zeros((1, 64))  # Only satisfy wrapper return contract, not audio.
    fn = extract_function(text, "fit_nara", {"np": np, "OnlineWPE": nara.OnlineWPE,
                                             "stft": lambda *_a, **_k: spectra.copy(),
                                             "istft": inverse_boundary})
    config = {"window_size": 16, "hop_size": 4, "n_taps": 1, "delay": 1,
              "alpha": .95, "n_in_chan": 1}
    out = fn(np.zeros((64, 1)), config)
    state = nara.OnlineWPE(taps=1, delay=1, alpha=.95, channel=1, frequency_bins=9)
    direct = np.stack([state.step_frame(frame) for frame in spectra.transpose(1,2,0)])
    actual = captured["spectra"].transpose(1,2,0)
    return {"execution_kind": "unchanged_function_ast_with_original_nara_state_and_transform_stubs",
            "config": config, "waveform_argument": "64 by 1 zeros; ignored by STFT stub",
            "stft_stub": "arange(1,73).reshape(1,8,9).astype(complex128)",
            "istft_stub": "captures input spectra; returns 1 by 64 zeros, not computed audio",
            "real_stft_or_istft_executed": False, "jax_or_metaaf_package_executed": False,
            "captured_spectra_real": actual.real.tolist(),
            "captured_spectra_imag": actual.imag.tolist(),
            "max_difference_from_original_single_frame_calls": float(np.max(np.abs(actual-direct))),
            "finite": bool(np.isfinite(actual).all()), "dummy_return_shape": list(out.shape),
            "status": "window_input_supported_by_actual_state_implementation",
            "note": "step_frame docstring specifies F,D, but implementation accepts a full 3D buffer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=CODES / "upstream/_downloads")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    for project, spec in SOURCES.items():
        verify_source(args.source_root / project, spec)
    nara_root = args.source_root / "nara_wpe"
    nara = load_original(nara_root / "nara_wpe/wpe.py", "wpe_fixed_audit")
    esp = args.source_root / "espnet/espnet2/enh/layers"
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
        "sources": SOURCES, "config": CONFIG,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "bytecode_writes": False},
        "nara": run_nara(nara),
        "espnet": inspect_espnet((esp / "dnn_wpe.py").read_text(),
                                 (esp / "mask_estimator.py").read_text(), (esp / "wpe.py").read_text()),
        "static_scope": {
            "tensorflow": "read tf_wpe.py; no TensorFlow runtime; block .7 weights current statistics",
            "gss": "read frequency chunking and global power floor; no CuPy/GPU execution",
            "btk20": "read C++ per-output powers and loading; no GSL compilation or execution",
            "tso_vace_wpe": "read 15 source/notice files; no PyTorch or weights executed",
            "speexdsp": "read disabled dereverb update/control code; no new preprocess execution",
        },
        "silence_boundary": "Separate experiment: codes/reports/chapter07_online_wpe_silence.json",
        "not_executed": ["speech quality", "ASR", "hardware timing", "neural weights", "training"],
    }
    if "metaaf" in SOURCES:
        report["metaaf_nara_wrapper"] = run_metaaf(args.source_root / "metaaf", nara)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
