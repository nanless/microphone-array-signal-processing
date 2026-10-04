"""Read-only, fixed-source industrial contracts; no models or audio devices.

Original C translation units / extracted bodies and a Python main AST are
executed with explicitly named scaffolds. This is not an ARM, VAD classifier,
RNNoise network, Speex preprocessor, or ONNX Runtime performance experiment.
Only --report writes a report; compilation and raw PCM stay in a temporary dir.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch10.examples.run_industrial_interfaces import clean_environment, verify_failure_sources

import argparse
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
DEFAULT_CACHE = upstream.CACHE
CURRENT = ROOT / "codes/chapters/ch10/reports/industrial_contracts_current.json"
SOURCES = {
    "cmsis_dsp": {
        "revision": "83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b",
        "license": "Apache-2.0",
        "files": {
            "Source/FilteringFunctions/arm_fir_q15.c": "6016f7f884cd0c0bd45e7aed223c76e3c48f1779421fcc1ecf0b52dc2eff8186",
            "Source/FilteringFunctions/arm_fir_init_q15.c": "3dc74f28dbda071b199749018dea9ddec3f286a8f32a673b994a3704b34eaf22",
            "LICENSE": "b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1",
        },
    },
    "speexdsp": {
        "revision": "8e29a256ef0235ebbe7fcb8417b5ac7731eb8307",
        "license": "BSD-3-Clause; inspected source/header notices retained upstream",
        "files": {
            "libspeexdsp/preprocess.c": "5f7143d00f12af60759ea1b9249fba3615fd92fcead2671e9350fe6cf530ecbd",
            "libspeexdsp/arch.h": "102f6a14a95f8ae0bcfc69270f3a9e3fbba08b63bba688ca524f71e3faa48c23",
            "COPYING": "2654a4264b2bfe298dedc508748d140111840c315cc8eb646a3a68c13fa75b01",
        },
    },
    "webrtc": {
        "revision": "0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e",
        "license": "BSD-3-Clause and PATENTS",
        "files": {
            "common_audio/vad/webrtc_vad.c": "1b494b4fd4bd3fd9eff03d4aadac2157a54859020a5c6a2a4dec15a871f539dd",
            "LICENSE": "ab00a482b6a3902e40211b43c5d0441962ea99b6cc7c25c0f243fa270b78d482",
            "PATENTS": "01462e2068d1a04c2274f3389773014c14ed9bc3446b28303543bd3e3c064145",
        },
    },
    "rnnoise": {
        "revision": "70f1d256acd4b34a572f999a05c87bf00b67730d",
        "license": "COPYING BSD-3-Clause; demo file header BSD-2-Clause",
        "files": {
            "examples/rnnoise_demo.c": "644064fcf3c1762e4d2e6341d59dc2db2e0cea4ba5d42e62767411e387014c98",
            "COPYING": "45d37ca1cdb278c088e1aa85e0e65ca3a534ed86a28dcc96ca16810248a61d35",
        },
    },
    "fastenhancer": {
        "revision": "f85223bd546b27f39dc0744e0310dcd246f750a4",
        "license": "MIT code; weights/data not obtained or exercised",
        "files": {
            "scripts/test_onnx.py": "b6a93660a5e777677ae978bfc477dba09feefd8cba6e0f36e5746dd3caf1d344",
            "LICENSE": "38164c22720cdccceb5f0dbb8f7218dcf17bac97ac536cad2e77c33dfbc45a18",
        },
    },
}
SOURCE_URLS = {
    "cmsis_dsp": "https://github.com/ARM-software/CMSIS-DSP.git",
    "speexdsp": "https://gitlab.xiph.org/xiph/speexdsp.git",
    "webrtc": "https://webrtc.googlesource.com/src",
    "rnnoise": "https://gitlab.xiph.org/xiph/rnnoise.git",
    "fastenhancer": "https://github.com/aask1357/fastenhancer.git",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def strict_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _git_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def verify_sources(cache):
    records = {}
    for name, spec in SOURCES.items():
        identity = upstream.verify_project(name, Path(cache) / name,
            relatives=tuple(spec["files"]))
        for relative, expected in spec["files"].items():
            if identity["used_files"][relative]["sha256"] != expected:
                raise ValueError("source digest mismatch: " + name + "/" + relative)
        records[name] = identity
    return records


def c_fragment(path, name):
    """Copy an original complete body verbatim, never replace its expressions."""
    source = Path(path).read_text(encoding="utf-8-sig")
    match = re.search(r"(?m)^(?:static inline )?(?:int|void) " + re.escape(name) + r"\(", source)
    if match is None:
        raise ValueError("original function declaration not found: " + name)
    start = match.start()
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    # These fixed bodies contain no brace-bearing string/comment. Their whole
    # file hashes are verified first; this is deliberately not a general parser.
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise ValueError("unclosed original body: " + name)
    fragment = source[start:end]
    return fragment, {"name": name, "first_line": source.count("\n", 0, start) + 1,
                      "last_line": source.count("\n", 0, end) + 1,
                      "sha256": digest(fragment.encode())}


def _compile(compiler, name, files, directory, flags=()):
    target = directory / name
    command = [compiler, "-std=c11", "-O2", *flags, *map(str, files), "-o", str(target)]
    run = subprocess.run(command, env=clean_environment(), capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise RuntimeError("compiler failed: " + run.stderr)
    return target, {"command": command, "returncode": run.returncode,
                    "stdout": run.stdout, "stderr": run.stderr}


def _execute(target, *args):
    run = subprocess.run([str(target), *map(str, args)], env=clean_environment(), capture_output=True,
                         text=True, timeout=30)
    if run.returncode:
        raise RuntimeError("C contract process failed: " + run.stderr)
    return run


def fir_integer_oracle(samples, impulse):
    """Direct causal convolution, integer floor and Q15 saturation; no CMSIS."""
    result = []
    for n in range(len(samples)):
        accumulator = sum(impulse[k] * samples[n-k]
                          for k in range(min(n+1, len(impulse))))
        result.append(max(-32768, min(32767, accumulator // 32768)))
    return result


def probe_cmsis(cache, directory, compiler):
    include = directory / "dsp"
    include.mkdir()
    (directory / "arm_compiler_specific.h").write_text("#define ARM_DSP_ATTRIBUTE\n")
    header = include / "filtering_functions.h"
    header.write_text("""#include <stdint.h>
#include <string.h>
typedef int16_t q15_t; typedef int32_t q31_t; typedef int64_t q63_t;
typedef enum {ARM_MATH_SUCCESS=0, ARM_MATH_ARGUMENT_ERROR=-1} arm_status;
typedef struct {uint16_t numTaps; q15_t *pState; const q15_t *pCoeffs;} arm_fir_instance_q15;
static inline int32_t __SSAT(int64_t x,int bits){(void)bits;return x>32767?32767:x< -32768?-32768:(int32_t)x;}
void arm_fir_q15(const arm_fir_instance_q15*,const q15_t*,q15_t*,uint32_t);
arm_status arm_fir_init_q15(arm_fir_instance_q15*,uint16_t,const q15_t*,q15_t*,uint32_t);
""")
    # Reversed coefficients are passed explicitly; all taps satisfy the
    # documented even >=4 restriction. The narrow saturation facade is only
    # exercised with post-shift values representable by the real int32 operand.
    cases = [
        ("half_lsb", [16384,16384,0,0], [1,0,-1,0,1], [5]),
        ("saturation", [-32768,-32768,0,0], [-32768]*4, [4]),
        ("asymmetric_whole", [8192,-4096,2048,1024], [8,16,-8,24,-16,0,32], [7]),
        ("asymmetric_chunks", [8192,-4096,2048,1024], [8,16,-8,24,-16,0,32], [2,1,4]),
    ]
    body = ['#include <stdio.h>\n#include "dsp/filtering_functions.h"\nint main(void){']
    for index, (_, impulse, samples, chunks) in enumerate(cases):
        body.append('{ q15_t c[4]={' + ','.join(map(str, reversed(impulse))) + '};')
        body.append('q15_t x[]={' + ','.join(map(str, samples)) + '},s[16],y[16]; arm_fir_instance_q15 a;')
        body.append('printf("%d",arm_fir_init_q15(&a,4,c,s,' + str(max(chunks)) + '));')
        offset = 0
        for chunk in chunks:
            body.append(f'arm_fir_q15(&a,x+{offset},y+{offset},{chunk});')
            offset += chunk
        body.append(f'for(int i=0;i<{len(samples)};i++)printf(" %d",y[i]);printf("\\n"); }}')
    body.append('return 0;}')
    harness = directory / "cmsis_contract.c"
    harness.write_text('\n'.join(body))
    origin = cache / "cmsis_dsp/Source/FilteringFunctions"
    target, compile_record = _compile(compiler, "cmsis_contract", [harness, origin / "arm_fir_q15.c",
                                       origin / "arm_fir_init_q15.c"], directory, ["-I", str(directory)])
    execution = _execute(target)
    outputs = execution.stdout.splitlines()
    if len(outputs) != len(cases):
        raise AssertionError("unexpected CMSIS output count")
    records = []
    for case, line in zip(cases, outputs):
        name, impulse, samples, chunks = case
        status, *observed = map(int, line.split())
        expected = fir_integer_oracle(samples, impulse)
        if status != 0 or observed != expected:
            raise AssertionError("CMSIS direct integer oracle mismatch: " + name)
        records.append({"name": name, "input_q15": samples, "impulse_q15": impulse,
                        "stored_coefficients_q15": list(reversed(impulse)), "chunks": chunks,
                        "state_allocation_q15": 16, "init_status": status,
                        "observed_q15": observed, "expected_q15": expected, "exact": True})
    return {"execution": "two complete original C translation units, scalar branch",
            "substitutions": ["q15/q31/q63 typedefs and FIR struct/prototypes", "empty ARM_DSP_ATTRIBUTE",
                              "__SSAT: explicit signed-16 saturation for the tested int32 post-shift domain"],
            "compile_defines": [], "not_executed": ["MVE/DSP/LOOPUNROLL kernels", "ARM hardware/cycle timing"],
            "scaffold_sha256": {p.name: file_sha(p) for p in [header, harness, directory / "arm_compiler_specific.h"]},
            "compile": compile_record, "stdout": execution.stdout, "stderr": execution.stderr, "cases": records}


def probe_speex(cache, directory, compiler):
    names = ["preproc_min_track_swap", "preproc_min_track", "preproc_update_prob", "preproc_noise_update"]
    fragments = [c_fragment(cache / "speexdsp/libspeexdsp/preprocess.c", n) for n in names]
    harness = directory / "speex_contract.c"
    harness.write_text('#include <stdio.h>\n#define FLOATING_POINT\n#define OUTSIDE_SPEEX\n#include "arch.h"\n#define NOISE_SHIFT 0\n' +
                      '\n'.join(f[0] for f in fragments) + """
int main(void){
 float S[2]={10,10},m[2]={4,3.99},noise[2]={2,2},ps[2]={6,6}; int flag[2];
 preproc_update_prob(S,m,flag,2);preproc_noise_update(flag,ps,noise,.25,.75,2);
 printf("%d %d %.9g %.9g\\n",flag[0],flag[1],noise[0],noise[1]);
 ps[0]=ps[1]=1;preproc_noise_update(flag,ps,noise,.25,.75,2);
 printf("%.9g %.9g\\n",noise[0],noise[1]);
 float lo[2]={5,7},temp[2]={6,4},current[2]={3,8};
 preproc_min_track(lo,temp,current,2);printf("%.9g %.9g %.9g %.9g\\n",lo[0],lo[1],temp[0],temp[1]);
 current[0]=9;current[1]=2;preproc_min_track_swap(lo,temp,current,2);
 printf("%.9g %.9g %.9g %.9g\\n",lo[0],lo[1],temp[0],temp[1]);
 return 0;
}
""")
    target, compilation = _compile(compiler, "speex_contract", [harness], directory,
                                   ["-I", str(cache / "speexdsp/libspeexdsp")])
    execution = _execute(target)
    observed = [[float(v) for v in line.split()] for line in execution.stdout.splitlines()]
    # Hand expectations: 0.4*10 == 4 does not satisfy >; a guarded bin still
    # tracks a falling ps. Running min and window restart have distinct inputs.
    expected = [[0.,1.,3.,2.], [2.5,1.75], [3.,7.,3.,4.], [3.,2.,9.,2.]]
    if len(observed) != len(expected) or any(len(a) != len(b) or any(
            not math.isfinite(x) or abs(x-y) > 1e-6 for x,y in zip(a,b))
            for a,b in zip(observed, expected)):
        raise AssertionError("Speex helper hand oracle mismatch")
    return {"execution": "four unchanged original helper bodies and original floating arch.h macros",
            "fragments": [f[1] for f in fragments], "defines": ["FLOATING_POINT", "OUTSIDE_SPEEX", "NOISE_SHIFT=0"],
            "substitutions": ["own deterministic two-bin driver; no preprocess state"],
            "not_executed": ["complete speex_preprocess", "FFT", "speech classification", "MCRA/IMCRA algorithm"],
            "inputs": {"S": [10,10], "Smin": [4,3.99], "noise": [2,2], "rising_ps": [6,6],
                       "falling_ps": [1,1], "beta": .25, "beta_1": .75,
                       "min_track": {"Smin": [5,7], "Stmp": [6,4], "S": [3,8]}, "swap_S": [9,2]},
            "observed_rows": observed, "expected_rows": expected, "absolute_tolerance": 1e-6,
            "scaffold_sha256": file_sha(harness), "compile": compilation, "stderr": execution.stderr}


def probe_vad(cache, directory, compiler):
    fragments = [c_fragment(cache / "webrtc/common_audio/vad/webrtc_vad.c", n)
                 for n in ["WebRtcVad_ValidRateAndFrameLength", "WebRtcVad_Process"]]
    harness = directory / "vad_contract.c"
    harness.write_text("""#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef struct {int init_flag;} VadInstT;typedef VadInstT VadInst;
static const int kInitCheck=42, kValidRates[]={8000,16000,32000,48000};
static const size_t kRatesSize=4;static const int kMaxFrameLengthMs=30;
static int calls[4];
int WebRtcVad_ValidRateAndFrameLength(int,size_t);
#define FN(rate,j) int WebRtcVad_CalcVad##rate(VadInstT*a,const int16_t*b,size_t c){(void)a;(void)b;(void)c;calls[j]++;return j;}
FN(8khz,0) FN(16khz,1) FN(32khz,2) FN(48khz,3)
""" + '\n'.join(f[0] for f in fragments) + """
int main(void){int rates[]={8000,16000,32000,48000};VadInst a={42},bad={0};int16_t x[1440]={0};
 for(int j=0;j<4;j++)for(int ms=10;ms<=30;ms+=10){size_t n=rates[j]/1000*ms;
 printf("%d %d %d %d\\n",rates[j],ms,WebRtcVad_ValidRateAndFrameLength(rates[j],n),WebRtcVad_Process(&a,rates[j],x,n));}
 printf("negative %d %d %d %d %d %d\\n",WebRtcVad_Process(&a,44100,x,441),WebRtcVad_Process(&a,16000,x,240),
 WebRtcVad_Process(NULL,16000,x,160),WebRtcVad_Process(&bad,16000,x,160),WebRtcVad_Process(&a,16000,NULL,160),
 WebRtcVad_Process(&a,16000,x,0));
 printf("calls %d %d %d %d\\n",calls[0],calls[1],calls[2],calls[3]);return 0;}
""")
    target, compilation = _compile(compiler, "vad_contract", [harness], directory)
    execution = _execute(target)
    lines = execution.stdout.splitlines()
    observed = [list(map(int, s.split())) for s in lines[:12]]
    expected = [[rate, ms, 0, 0 if rate == 8000 else 1]
                for rate in [8000,16000,32000,48000] for ms in [10,20,30]]
    negative = list(map(int, lines[12].split()[1:]))
    counts = list(map(int, lines[13].split()[1:]))
    if observed != expected or negative != [-1]*6 or counts != [3]*4:
        raise AssertionError("VAD validation/router oracle mismatch")
    return {"execution": "two unchanged original C bodies; classifier functions replaced by route counters",
            "fragments": [f[1] for f in fragments],
            "substitutions": ["minimal VadInstT.init_flag struct and fixed constants", "WebRtcVad_CalcVad8khz returns 0",
                              "WebRtcVad_CalcVad16khz returns 1", "WebRtcVad_CalcVad32khz returns 2", "WebRtcVad_CalcVad48khz returns 3"],
            "not_executed": ["original CalcVad classifiers", "actual speech decision", "APM"],
            "input_pcm": "zero signed-16 mono; amplitude not used by counter substitutes",
            "observed_legal_rows": observed, "expected_legal_rows": expected,
            "negative_names": ["44100Hz", "15ms", "null_handle", "uninitialized", "null_audio", "zero_length"],
            "negative_outputs": negative, "classifier_route_calls": counts,
            "scaffold_sha256": file_sha(harness), "compile": compilation, "stderr": execution.stderr}


def probe_rnnoise(cache, directory, compiler):
    header = directory / "rnnoise.h"
    header.write_text("typedef struct {int x;} DenoiseState;typedef struct {int x;} RNNModel;\n"
                      "DenoiseState *rnnoise_create(RNNModel*);void rnnoise_destroy(DenoiseState*);\n"
                      "float rnnoise_process_frame(DenoiseState*,float*,const float*);\n")
    adapter = directory / "rnnoise_adapter.c"
    adapter.write_text("""#include <stdio.h>
#include "rnnoise.h"
_Static_assert(sizeof(short)==2,"demo fixture requires two-byte short PCM");
DenoiseState *rnnoise_create(RNNModel*m){static DenoiseState s;(void)m;return &s;}
void rnnoise_destroy(DenoiseState*s){(void)s;}
float rnnoise_process_frame(DenoiseState*s,float*out,const float*in){(void)s;
 fprintf(stderr,"first=%.0f\\n",in[0]);for(int i=0;i<480;i++)out[i]=in[i];return .5f;}
""")
    target, compilation = _compile(compiler, "rnnoise_demo_contract",
        [cache / "rnnoise/examples/rnnoise_demo.c", adapter], directory, ["-I", str(directory)])
    cases = []
    for length in [1440,1457,479,0]:
        samples = [1000+i//480 for i in range(length)]
        input_bytes = struct.pack("=" + "h"*length, *samples)
        input_path, output_path = directory / f"input-{length}.pcm", directory / f"output-{length}.pcm"
        input_path.write_bytes(input_bytes)
        execution = _execute(target, input_path, output_path)
        output_bytes = output_path.read_bytes()
        complete_frames = length // 480
        expected = samples[480:complete_frames*480]
        observed = list(struct.unpack("=" + "h"*(len(output_bytes)//2), output_bytes))
        trace = [float(line.split("=")[1]) for line in execution.stderr.splitlines()]
        if observed != expected or trace != [1000.+k for k in range(complete_frames)]:
            raise AssertionError("RNNoise original demo wrapper oracle mismatch")
        cases.append({"input_samples": length, "input_native_pcm16_sha256": digest(input_bytes),
                      "complete_input_frames": complete_frames, "dropped_partial_samples": length % 480,
                      "output_samples": len(observed), "expected_output_samples": max(0,complete_frames-1)*480,
                      "output_native_pcm16_sha256": digest(output_bytes), "exact_identity_wrapper_match": True,
                      "process_first_values": trace, "first_output": observed[0] if observed else None,
                      "stderr": execution.stderr})
    return {"execution": "complete original demo main compiled with identity denoiser substitutes",
            "substitutions": ["own minimal rnnoise.h", "rnnoise_create static state", "rnnoise_destroy no-op",
                              "rnnoise_process_frame identity copy and input trace; returns .5"],
            "not_executed": ["RNNoise original neural/DSP core", "weights", "model lifecycle APIs", "quality evaluation"],
            "raw_pcm_format": "native-endian signed short, sizeof(short)=2 required by scaffold environment",
            "amplitude_unit": "raw PCM integer magnitude represented as float, not divided by 32768",
            "scaffold_sha256": {p.name: file_sha(p) for p in [header, adapter]}, "compile": compilation, "cases": cases}


def probe_fast(cache):
    import numpy as np
    path = cache / "fastenhancer/scripts/test_onnx.py"
    parsed = ast.parse(path.read_text())
    main = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    source_fragment = ast.get_source_segment(path.read_text(), main)
    cases = []
    for length in [1000,16000,256,1,0]:
        calls, saved = [], []
        class Session:
            def __init__(self, *args, **kwargs):
                self.options = kwargs
            def get_inputs(self):
                return [types.SimpleNamespace(name="cache_in_0", shape=(1,))]
            def run(self, unused, feed):
                calls.append({"new_samples": int(feed["wav_in"].shape[-1]),
                              "cache_input": feed["cache_in_0"].tolist()})
                # A deterministic artificial output ramp distinguishes the
                # crop's source interval. It is not generated by a network.
                start = 256*(len(calls)-1)
                output = (np.arange(start, start+256, dtype=np.float32)/65536).reshape(1,256)
                return [output, feed["cache_in_0"]+1]
        def capture_write(name, rate, data):
            saved.append({"requested_path": name, "sample_rate_hz": rate,
                          "samples": data.tolist(), "dtype": str(data.dtype)})
        times = iter([0.,.01])
        namespace = {"np": np, "time": types.SimpleNamespace(perf_counter=lambda: next(times)),
            "librosa": types.SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(length,np.float32),16000)),
            "onnxruntime": types.SimpleNamespace(SessionOptions=types.SimpleNamespace,
                ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
                GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=0), InferenceSession=Session),
            "scipy": types.SimpleNamespace(io=types.SimpleNamespace(wavfile=types.SimpleNamespace(write=capture_write))),
            "tqdm": lambda values: values}
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
        args = types.SimpleNamespace(audio_path="controlled-memory-input", sr=16000, n_fft=512,
                                     hop_size=256, onnx_path="substitute-session", save_output=True)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            namespace["main"](args)
        # Independently count H-point increments for the padded flush range.
        count = (length + 256 + 255) // 256
        denominator = count*256
        printed_rtf = float(re.search(r"RTF: ([^\n]+)", printed.getvalue()).group(1))
        expected_crop = [(i+256)/65536 for i in range(length)]
        if len(calls) != count or [v["cache_input"] for v in calls] != [[float(i)] for i in range(count)]:
            raise AssertionError("FastEnhancer state/call-count oracle mismatch")
        if any(c["new_samples"] != 256 for c in calls) or len(saved) != 1 or saved[0]["samples"] != expected_crop:
            raise AssertionError("FastEnhancer output crop oracle mismatch")
        expected_rtf = .01*16000/denominator
        if printed_rtf != expected_rtf:
            raise AssertionError("FastEnhancer wrapper denominator mismatch")
        cases.append({"source_samples": length, "sample_rate_hz": 16000, "n_fft": 512, "hop_size": 256,
                      "fake_elapsed_seconds": .01, "calls": calls, "wrapper_denominator_samples": denominator,
                      "wrapper_printed_rtf": printed_rtf, "expected_wrapper_rtf": expected_rtf,
                      "source_time_rtf": .01*16000/length if length else None,
                      "source_time_classification": "positive_source_duration" if length else "undefined_zero_source_duration",
                      "saved_output_samples": len(saved[0]["samples"]), "crop_start": 256,
                      "first_saved_value": saved[0]["samples"][0] if length else None,
                      "last_saved_value": saved[0]["samples"][-1] if length else None,
                      "saved_dtype": saved[0]["dtype"], "saved_rate_hz": saved[0]["sample_rate_hz"],
                      "saved_samples_sha256": digest(np.asarray(saved[0]["samples"],dtype='<f4').tobytes()),
                      "stdout": printed.getvalue()})
    return {"execution": "unchanged original main AST with named boundary substitutes",
            "fragment": {"name": "main", "first_line": main.lineno, "last_line": main.end_lineno,
                         "sha256": digest(source_fragment.encode())},
            "substitutions": ["librosa.load memory zeros", "ONNX session fake output ramp/state+1", "SessionOptions/enums",
                              "tqdm identity", "perf_counter returns 0 and .01", "scipy.io.wavfile.write in-memory receiver"],
            "not_executed": ["librosa resampling", "ONNXRuntime inference", "model or weights", "real timing or enhancement"],
            "numpy_version": np.__version__, "cases": cases}


def run_audit(cache=DEFAULT_CACHE):
    cache = upstream.validate_parent_chain(cache)
    before = verify_sources(cache)
    try:
        preflight = copy.deepcopy(before)
        compiler = shutil.which("cc")
        if compiler is None:
            raise RuntimeError("C compiler cc is required; no installation is attempted")
        compiler_version = subprocess.check_output([compiler, "--version"], env=clean_environment(), text=True).strip()
        with tempfile.TemporaryDirectory(prefix="masp-industrial-contracts-") as temporary:
            directory = Path(temporary)
            contracts = {"cmsis_q15": probe_cmsis(cache, directory, compiler),
                         "speex_noise_helpers": probe_speex(cache, directory, compiler),
                         "webrtc_vad_router": probe_vad(cache, directory, compiler),
                         "rnnoise_demo": probe_rnnoise(cache, directory, compiler),
                         "fastenhancer_wrapper": probe_fast(cache)}
        after = {name: upstream.check_unchanged(identity) for name, identity in before.items()}
        report = {"schema": "industrial-contracts-v1", "status": "passed_limited_controlled_contracts",
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "tool_sha256": file_sha(__file__), "source_lock_sha256": file_sha(LOCK),
                  "actual_dependencies_sha256": upstream.dependencies(__file__, (Path(clean_environment.__code__.co_filename),)),
                  "source_cache": str(cache), "sources_before": preflight, "sources_after": after,
                  "scope": "fixed-source control flow and numerical helpers; substitutes are not actual model/audio/hardware execution",
                  "environment": {"python": sys.version, "executable": sys.executable,
                                  "platform": platform.platform(), "machine": platform.machine(),
                                  "byteorder": sys.byteorder, "compiler": compiler_version,
                                  "numpy": contracts["fastenhancer_wrapper"]["numpy_version"]},
                  "contracts": contracts}
        strict_json(report)
        return report
    except BaseException as error:
        verify_failure_sources(before, error)
        raise


def write_report(path, report, cache=DEFAULT_CACHE):
    upstream.write_report(path, report, CURRENT, protected=(DEFAULT_CACHE, cache))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, help="explicit new report; default only prints JSON")
    args = parser.parse_args()
    if args.report is not None:
        upstream.report_target(args.report, CURRENT, protected=(DEFAULT_CACHE, args.cache))
    report = run_audit(args.cache)
    if args.report is not None:
        write_report(args.report, report, args.cache)
    print(strict_json(report), end="")


if __name__ == "__main__":
    main()
