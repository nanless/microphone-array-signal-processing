"""Read-only fixed tracking contracts; only --report writes a current report.

ODAS: three unchanged C translation units and original headers. SAF: unchanged
tracker3d_step body with named counter/state substitutes, not RBMCDA execution.
FilterPy Q and Stone Soup isvalid: unchanged AST definitions with named adapters.
The sibling interface tool is a new current execution; historical reports are never edited.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from codes.chapters.ch04.core import upstream_contracts as contracts

CURRENT = Path(__file__).resolve().parents[1] / "reports/upstream_tracking_contracts_current.json"
INTERFACE = Path(__file__).with_name("audit_tracking_upstream_interfaces.py")

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
CACHE = ROOT / "codes/chapters/ch00/upstream/_downloads"
REVISIONS = {
    "odas": "bcb845434495e293df3d48f1203b7a86e1852449",
    "spatial-audio-framework": "18fd5aba46e20787b51f28f7197a68506c965c07",
    "filterpy": "3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33",
    "stonesoup": "8d1edeb07ef8505ed065cbef435cfb5e517d9bdc",
}
FILES = {
    'odas': {
        'LICENSE': 'd80055b1eac9fdb001f429f262c79ce6a7619b85064a759f17b0b33bbb9c8b2a',
        'src/system/kalman2kalman.c': 'c6dc032c43504c0a98a5015ebebfe1e74be63f72a2484f79550c80a56f3d9b65',
        'src/signal/kalman.c': 'b3cbd730ea5a466ec2924b055efada63da260cc805462c02db16c6ee6cbac102',
        'src/utils/matrix.c': 'bf61ae5e49840d0ad4d978cf1934b6bb7d96ff38451aeb5382d55c11dd2abd1a',
        'include/odas/system/kalman2kalman.h': '2c37c6fd1a92cf70a9030a7538924b9c9bbce24f4a5ff39d0cd5fcbfb9037442',
        'include/odas/signal/kalman.h': '6ed7d7f5b5040ad7fb000c555fa2a5c48eeed70a7b1dced0a6722b3fcda0789e',
        'include/odas/signal/postprob.h': 'e528c3b6da2ca3790ed589b95405ef8d095c043998db4a8175207adbb2afac9d',
        'include/odas/signal/pot.h': '56e632d71e7e2f0653a4f71fe7fcaafb1aaf936044c2c855d3bb664f28d39e31',
        'include/odas/signal/target.h': 'cd8f18a5e923c187875d847951c69c2679c2ddef5f858c7e875f93a7d75d7ea6',
        'include/odas/utils/matrix.h': 'acd4892c480ed07d7aba398cc0f3026e62d5914540f35a4098f95856a7adf602',
    },
    'spatial-audio-framework': {
        'LICENSE.md': '0148c9bfe5f2093cd06cecd3359bc16bc29a2200863a59af81a90b7922c67af3',
        'framework/modules/saf_tracker/saf_tracker.c': 'dd36f1878cddf0f54e16f9ac6e742afc46de2654d5af1cc288d3f3813adf7b10',
        'framework/modules/saf_tracker/saf_tracker.h': '68425a5b74d1ba545c9dba583492aed00c24b77cb6aa49df6dcdcec25a230c2b',
    },
    'filterpy': {
        'LICENSE': '8ffce1097f1b1c0fba42e4449ef49aa05cb7b45566dd0989407839dba07af99c',
        'filterpy/common/discretization.py': 'b3e625792e777bfef93f8a7d9020624c8750f66fc482eeec7b3b2c5ae2fdea96',
    },
    'stonesoup': {
        'LICENSE': '2462f3d8a857f601e266f048a4ff051366c1e05f098cf3a1524eaf922f879815',
        'stonesoup/dataassociator/probability.py': 'd472902c229095cdbb20c7feee2a2ea89aa7bd3e77fdb84147f8a60cfaa3be36',
        'stonesoup/hypothesiser/probability.py': 'ab9832c864d130429ded5515ed588995d3729a321f49587c28504494ef034455',
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())


def unique_projects(rows):
    """Do not let duplicate identities silently overwrite provenance."""
    result = {}
    for row in rows:
        name = row["id"]
        if not isinstance(name, str) or name in result:
            raise ValueError("duplicate or invalid source identity: " + str(name))
        result[name] = row
    return result


def verify_sources(cache=CACHE):
    identities = {}
    for name, revision in REVISIONS.items():
        identity = contracts.verify_project(name, Path(cache) / name,
            relatives=tuple(FILES[name]), lock_path=LOCK,
            status_path=LOCK.with_name("SOURCE_STATUS.json"))
        if identity["head"] != revision:
            raise ValueError("fixed HEAD mismatch: " + name)
        acquisition = identity["recorded_complete_selection"]
        selection = acquisition.get("status") == "source_verified"
        if (acquisition.get("status") not in ("source_verified", "source_selection_mismatch")
                or acquisition.get("revision") != identity["head"]
                or acquisition.get("missing_entrypoints") != []
                or acquisition.get("source_selection_verified") is not selection):
            raise ValueError("source identity unavailable: " + name)
        for relative, expected in FILES[name].items():
            if identity["used_files"][relative]["sha256"] != expected:
                raise ValueError("source bytes mismatch: " + name + "/" + relative)
        # Keep familiar compatibility fields while storing the full contract.
        identity.update(acquisition_record=identity["recorded_complete_selection"],
            source_selection_verified=identity["live_complete_selection"]["source_selection_verified"],
            required_source_identity_verified=True, worktree_clean=True,
            tracked_clean=True, files=identity["used_files"],
            license=identity["lock_entry"]["license"],
            lock_entry_sha256=stable_digest(identity["lock_entry"]))
        identities[name] = identity
    return identities


def extract_python(path, names, class_name=None):
    tree = ast.parse(path.read_text())
    parent = tree if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in names]
    fragments = {n.name: digest(ast.get_source_segment(path.read_text(), n).encode()) for n in nodes}
    return nodes, fragments


def python_contracts(cache):
    source = cache / "filterpy/filterpy/common/discretization.py"
    names = ["Q_discrete_white_noise", "Q_continuous_white_noise"]
    nodes, fragments = extract_python(source, names)
    # Only the one-block branch is used. This adapter preserves a sole matrix,
    # rather than claiming to execute SciPy's general block_diag implementation.
    def one_block_diag(*blocks):
        if len(blocks) != 1:
            raise ValueError("only the audited single block is supported")
        return np.asarray(blocks[0]).copy()
    namespace = {"np": np, "array": np.array, "eye": np.eye, "dot": np.dot,
                 "block_diag": one_block_diag}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    discrete = namespace[names[0]](2, dt=.5, var=2.)
    continuous = namespace[names[1]](2, dt=.5, spectral_density=2.)
    expected_discrete = np.array([[1/32, 1/8], [1/8, 1/2]])
    expected_continuous = np.array([[1/12, 1/4], [1/4, 1.]])
    np.testing.assert_allclose(discrete, expected_discrete, rtol=0, atol=1e-15)
    np.testing.assert_allclose(continuous, expected_continuous, rtol=0, atol=1e-15)
    source_j = cache / "stonesoup/stonesoup/dataassociator/probability.py"
    methods, jf = extract_python(source_j, ["isvalid"], "JPDA")
    # Retain the staticmethod decorator and original method body in a minimal
    # class shell. No Stone Soup base classes, predictions or probabilities run.
    shell = ast.ClassDef(name="JPDAContract", bases=[], keywords=[], body=methods,
                         decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    scope = {}
    exec(compile(module, str(source_j), "exec"), scope)
    isvalid = scope["JPDAContract"].isvalid
    labels = [object(), object(), object()]
    records = []
    for a, b in product(range(-1, 3), repeat=2):
        measurements = [None if i < 0 else labels[i] for i in (a, b)]
        result = bool(isvalid([SimpleNamespace(measurement=m) for m in measurements]))
        expected = a < 0 or b < 0 or a != b
        if result != expected:
            raise AssertionError("JPDA event contract changed")
        records.append({"assignment": [a, b], "valid": result})
    return {"execution": "unchanged AST definitions; not full packages",
            "filterpy_q": {"dt": .5, "discrete_acceleration_variance": 2.,
                           "continuous_density": 2., "discrete": discrete.tolist(),
                           "continuous": continuous.tolist(), "absolute_tolerance": 1e-15,
                           "fragment_sha256": fragments,
                           "adapters": ["NumPy array/eye/dot/np namespace", "single-block block_diag adapter"],
                           "not_called": ["dim3/4", "general SciPy block_diag", "order_by_derivative"]},
            "jpda_isvalid": {"events": records, "valid_count": sum(r["valid"] for r in records),
                             "fragment_sha256": jf,
                             "adapters": ["minimal class shell", "SimpleNamespace hypothesis", "identity-labelled object detections", "None missed detections"],
                             "not_called": ["JPDA.associate", "enumerate_JPDA_hypotheses", "PDAHypothesiser", "updater", "metric"]}}


ODAS_HARNESS = r'''#include <system/kalman2kalman.h>
int main(void){
 kalman2kalman_obj *o=kalman2kalman_construct(.5f,.2f,.1f,1.e-10f);
 kalman_obj *k=kalman_construct_zero();
 k->x_lm1lm1->array[0]=1; k->x_lm1lm1->array[3]=1;
 for(int i=0;i<6;i++)k->P_lm1lm1->array[i*6+i]=1;
 kalman2kalman_predict(o,k);
 printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",o->F->array[3],o->Q->array[0],o->Q->array[21],o->R->array[0],k->x_llm1->array[0],k->x_llm1->array[3],k->P_llm1->array[0],k->P_llm1->array[3],k->P_llm1->array[21]);
 kalman_destroy(k);kalman2kalman_destroy(o);return 0;
}'''

SAF_ADAPTER = r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TRACKER3D_MAX_NUM_PARTICLES 100
typedef struct {float m0,m1,m2;} Mean;
typedef struct {float p00,p11,p22;} Cov;
typedef struct {int nTargets,targetIDs[1]; Mean M[1]; Cov P[1]; float W,W0,W_prev;} MCS_data;
typedef struct {int Np;float W_avg_coeff;} Params;
typedef struct {int incrementTime; Params tpars; void *SS[4],*SS_resamp[4];} tracker3d_data;
int predicts=0, updates=0, ticks[8], force_low=0;
void tracker3d_predict(void *h,int t){tracker3d_data*d=h;predicts++;for(int i=0;i<d->tpars.Np;i++)((MCS_data*)d->SS[i])->M[0].m0+=1;}
void tracker3d_update(void*h,float*x,int t){ticks[updates++]=t;}
float eff_particles(void**s,int n){return force_low?0:n;}
int tracker3d_getMaxParticleIdx(void*h){tracker3d_data*d=h;int k=0;for(int i=1;i<d->tpars.Np;i++)if(((MCS_data*)d->SS[i])->W>((MCS_data*)d->SS[k])->W)k=i;return k;}
void tracker3d_particleCopy(void*a,void*b){memcpy(b,a,sizeof(MCS_data));}
void* realloc1d(void*p,size_t n){return realloc(p,n);}
'''

SAF_MAIN = r'''int main(void){
 tracker3d_data d={0}; MCS_data a[4]={0},b[4]={0};
 float *pos=NULL,*var=NULL,obs[6]={1,0,0,0,1,0};int *ids=NULL,n=0;
 d.tpars.Np=1; a[0].nTargets=1;a[0].targetIDs[0]=101;a[0].M[0].m0=9;a[0].P[0].p00=1;a[0].W=a[0].W0=1;d.SS[0]=&a[0];d.SS_resamp[0]=&b[0];
 tracker3d_step(&d,NULL,0,&pos,&var,&ids,&n);printf("empty1 %d %d %.9g %.9g %d\n",d.incrementTime,predicts,pos[0],var[0],ids[0]);
 tracker3d_step(&d,NULL,0,&pos,&var,&ids,&n);printf("empty2 %d %d %.9g %.9g %d\n",d.incrementTime,predicts,pos[0],var[0],ids[0]);
 tracker3d_step(&d,obs,1,&pos,&var,&ids,&n);printf("resume %d %d %d %.9g %d\n",d.incrementTime,predicts,updates,pos[0],ticks[0]);
 tracker3d_step(&d,obs,2,&pos,&var,&ids,&n);printf("twoobs %d %d %d %.9g %d %d\n",d.incrementTime,predicts,updates,pos[0],ticks[1],ticks[2]);
 free(pos);free(var);free(ids);pos=var=NULL;ids=NULL;
 memset(&d,0,sizeof(d));memset(a,0,sizeof(a));memset(b,0,sizeof(b));predicts=updates=0;force_low=1;d.tpars.Np=4;
 for(int i=0;i<4;i++){a[i].nTargets=1;a[i].targetIDs[0]=101*(i+1);a[i].M[0].m0=10*i;a[i].W=i==3?.7f:.1f;a[i].W0=.25f;d.SS[i]=&a[i];d.SS_resamp[i]=&b[i];}
 tracker3d_step(&d,obs,1,&pos,&var,&ids,&n);
 printf("copy %d %d %d %d %.9g %.9g %.9g %.9g %d\n",a[0].targetIDs[0],a[1].targetIDs[0],a[2].targetIDs[0],a[3].targetIDs[0],a[0].W,a[1].W,a[2].W,a[3].W,ids[0]);
 free(pos);free(var);free(ids);return 0;
}'''


def extract_c_step(path):
    source = path.read_text()
    begin = source.index("void tracker3d_step\n")
    end = source.index("\n#endif /* SAF_ENABLE_TRACKER_MODULE */", begin)
    # This fixed file ends with step; keep every body byte including disabled
    # alternatives. Its source SHA is checked before extraction.
    fragment = source[begin:end]
    if not fragment.rstrip().endswith("}"):
        raise ValueError("unexpected fixed C function boundary")
    return fragment


def c_contracts(cache):
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("cc unavailable; no native execution claimed")
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    results = {}
    with tempfile.TemporaryDirectory(prefix="masp-tracking-") as directory:
        temp = Path(directory)
        def build(name, harness, extra):
            source = temp / (name + ".c")
            binary = temp / name
            source.write_text(harness)
            command = [compiler, "-std=c11", "-O0", *extra, str(source), "-lm", "-o", str(binary)]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError("native compile failed: " + completed.stderr)
            output = subprocess.check_output([str(binary)], text=True)
            return output, {"command": command, "compiler": version,
                            "compile_stdout": completed.stdout, "compile_stderr": completed.stderr,
                            "harness_sha256": digest(harness.encode()), "raw_output": output}
        odas = cache / "odas"
        extras = ["-I", str(odas / "include/odas"), *[str(odas / p) for p in (
            "src/system/kalman2kalman.c", "src/signal/kalman.c", "src/utils/matrix.c")]]
        output, record = build("odas_contract", ODAS_HARNESS, extras)
        actual = list(map(float, output.split()))
        expected = [.5, 0., .04, .01, 1., 0., 1.25, .5, 1.04]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        record.update({"execution": "original three C translation units; handwritten inputs",
                       "input": {"dt": .5, "sigmaQ": .2, "sigmaR": .1, "epsilon": 1e-10,
                                 "state": [1,0,0,1,0,0], "covariance": "identity6"},
                       "output_order": ["F03", "Q00", "Q33", "R00", "direction_x", "velocity_x", "P00", "P03", "P33"],
                       "actual": actual, "independent_expected": expected, "absolute_tolerance": 1e-6,
                       "not_run": ["mod_sst", "association", "update", "ODAS application", "audio", "hardware"]})
        results["odas"] = record
        saf_source = cache / "spatial-audio-framework/framework/modules/saf_tracker/saf_tracker.c"
        fragment = extract_c_step(saf_source)
        output, record = build("saf_step_contract", SAF_ADAPTER + fragment + SAF_MAIN, [])
        lines = {r.split()[0]: list(map(float, r.split()[1:])) for r in output.splitlines()}
        expected = {"empty1": [1,0,9,1,101], "empty2": [2,0,9,1,101],
                    "resume": [0,3,1,12,3], "twoobs": [0,4,3,13,1,0],
                    "copy": [404,404,404,404,.25,.25,.25,.25,404]}
        if lines != expected:
            raise AssertionError("SAF fixed step control flow changed")
        record.update({"execution": "unchanged original tracker3d_step body with explicit dependency substitutes",
                       "fragment_sha256": digest(fragment.encode()), "actual": lines,
                       "independent_expected": expected, "absolute_tolerance": 0,
                       "adapters": ["minimal tracker/particle/mean/covariance structs", "tracker3d_predict counter and +1 marker", "tracker3d_update tick counter only", "eff_particles forced branch selector", "tracker3d_getMaxParticleIdx weight argmax", "tracker3d_particleCopy struct memcpy", "realloc1d C realloc"],
                       "not_run": ["tracker3d_create", "original prediction/update", "birth/death", "Gaussian likelihood", "actual ESS", "RBMCDA", "BLAS", "MEX", "LOCATA", "hardware"],
                       "inputs": {"empty_then_resume": [0,0,1,2], "dt": "one caller tick per step; no real elapsed-time clock in substitute", "one_particle": {"id": 101, "marker_x": 9, "variance_x": 1},
                                  "copy": {"particle_ids": [101,202,303,404], "weights": [.1,.1,.1,.7], "forced_ess": 0, "Np": 4, "W_avg_coeff": 0}}})
        results["saf"] = record
    return results


def historical_recheck(cache):
    # This is a new current execution of the sibling interface tool. Its SHA
    # is a real dependency; neither historical JSON/tool binding is rewritten.
    from codes.chapters.ch09.examples import audit_tracking_upstream_interfaces as module
    report = module.run(cache)
    return {"current_interface_tool_sha256": digest(INTERFACE.read_bytes()),
            "scope": "new current execution; the two original historical reports remain unchanged",
            "source_identities": report["source_identities"],
            "filterpy": report["filterpy"], "stonesoup_reducer": report["stonesoup_reducer"]}


def run(cache=CACHE):
    sys.dont_write_bytecode = True
    deps = contracts.dependencies(Path(__file__), (INTERFACE,))
    before = verify_sources(cache)
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "tool_sha256": digest(Path(__file__).read_bytes()),
              "source_lock_sha256": contracts.sha(LOCK),
              "source_status_sha256": contracts.sha(LOCK.with_name("SOURCE_STATUS.json")),
              "actual_dependencies_sha256": deps,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "numpy": np.__version__, "platform": platform.platform()},
              "source_identities": before, "python": python_contracts(cache),
              "native": c_contracts(cache), "current_interface_recheck": historical_recheck(cache)}
    for identity in before.values():
        contracts.check_unchanged(identity)
    if contracts.dependencies(Path(__file__), (INTERFACE,)) != deps:
        raise ValueError("Actual audit dependencies changed during execution")
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=CACHE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.suffix != ".json":
        raise ValueError("Report destination must use .json")
    target = None if args.report is None else contracts.report_target(
        args.report, CURRENT, (CACHE, args.source_root))
    report = run(args.source_root)
    if target is None:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        contracts.write_report(target, report, CURRENT, (CACHE, args.source_root))
        print(target)


if __name__ == "__main__":
    main()
