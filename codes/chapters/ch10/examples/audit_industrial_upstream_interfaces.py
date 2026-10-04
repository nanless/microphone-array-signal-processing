"""Pinned industrial boundary audit; no downloads, models or audio devices.

pystoi is called as the original package. DeepFilterNet is only read statically;
the queue example below is independent Python arithmetic, not a Rust execution.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch10.examples.run_industrial_interfaces import verify_failure_sources

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

CODES = Path(__file__).resolve().parents[3]
CURRENT = upstream.ROOT / "codes/chapters/ch10/reports/industrial_upstream_interfaces_current.json"
SOURCES = {
    "pystoi": {
        "revision": "74872b000753a7a42ff51aa0868af8c82c7f9053",
        "files": {
            "pystoi/__init__.py": "48ce9882b6ba42f643f8305481a48902ddee97aa4f07bca13e44fea1b7d104ea",
            "pystoi/stoi.py": "482b30bfa8e60f1453200315afcd5564fc9e9aa1d14aeb9710a58fd7a9f52658",
            "pystoi/utils.py": "0222df0354a248ccc7f8525eda9bda9da815e5d60106f611a5f4750d1ba7f321",
            "LICENSE": "35c25f6087c4e1857ba6d4bc0b7957b7bc523c9ff190a9e2cccb4b5ecba018ab",
        },
    },
    "deepfilternet": {
        "revision": "d375b2d8309e0935d165700c91da9de862a99c31",
        "files": {
            "ladspa/src/lib.rs": "d8fd5bfb0432609f3225648e6389505a8fcf0568d54da2bd9cb3cb30ed70ede8",
            "LICENSE-MIT": "24e6bb09c928af8d8e56268082f87413247ce36b39dd5d33add2f9893968065e",
            "LICENSE-APACHE": "1eaee808c5fb6b4e895ba30425285a5cdc5dd25bba2cd230f264c2200c331aec",
        },
    },
}
CONFIG = {
    "pystoi": {"sample_rate_hz": 10000, "zero_lengths": [256, 1024, 10000],
               "self_comparison_length": 10000, "input_rng": "default_rng(4).normal",
               "normalization_seeds": [0, 1], "alignment_samples": 0},
    "dfn_queue": {"channels": 2, "frame_size": 480,
                  "initial_queue_length_per_channel": 960, "proc_delay_before": 960},
    "scope": "synthetic numerical inputs; no speech quality, hardware or model inference",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_sha256():
    return hashlib.sha256(json.dumps({"sources": SOURCES, "config": CONFIG},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sources(root):
    identities = {}
    for name, spec in SOURCES.items():
        identity = upstream.verify_project(name, Path(root) / name,
            relatives=tuple(spec["files"]))
        for relative, expected in spec["files"].items():
            if identity["used_files"][relative]["sha256"] != expected:
                raise ValueError("source digest mismatch: " + name + "/" + relative)
        identities[name] = identity
    return identities


def loaded_pystoi_modules(identity):
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if name == "pystoi" or name.startswith("pystoi."):
            filename = getattr(module, "__file__", None)
            if not filename:
                raise ValueError("pystoi module has no ordinary source: " + name)
            path = upstream.ordinary_file(filename)
            try:
                relative = path.relative_to(identity["checkout"]).as_posix()
            except ValueError as error:
                raise ValueError("foreign preloaded pystoi module: " + name) from error
            if relative not in identity["used_files"]:
                raise ValueError("pystoi module was not preflighted: " + relative)
            loaded[name] = {"path": str(path), "sha256": upstream.sha(path)}
    return loaded


def callback_clock_example():
    """Independent arithmetic of the fixed Rust counter, not Rust execution."""
    rows = []
    for initial, label in ((0, "initial_counter_zero"), (1, "after_reset_then_unconditional_increment")):
        counter = initial
        calls = 0
        while True:
            calls += 1
            if counter > 1000:
                break
            counter += 1
        rows.append({"initial_counter": initial, "initial_condition": label,
            "first_eligible_callback": calls,
            "input_audio_seconds_by_host_block": {str(n): calls * n / 48000 for n in (128, 480, 1024)}})
    return {"execution": "independent Python integer clock arithmetic", "sample_rate_hz": 48000,
        "model_hop_samples": 480, "threshold": 1000, "strict_test": "counter > 10*sr/model_hop",
        "other_branch_conditions_assumed": "rtf < .5 and sufficient processing delay/output queue",
        "rows": rows, "not_measured": "Rust host, scheduling and actual elapsed wall time"}


def queue_semantics_example():
    """Model only the inspected branch; no worker, mutex, model or Rust package."""
    cfg = CONFIG["dfn_queue"]
    queues = [deque(range(cfg["initial_queue_length_per_channel"]))
              for _ in range(cfg["channels"])]
    for channel in queues[:cfg["frame_size"]]:
        channel.popleft()
    return {
        "execution": "independent Python queue semantics example, not upstream execution",
        "queue_lengths_after": [len(q) for q in queues],
        "first_remaining_sample_indices": [q[0] for q in queues],
        "actual_removed_per_channel": [cfg["initial_queue_length_per_channel"] - len(q) for q in queues],
        "proc_delay_after": cfg["proc_delay_before"] - cfg["frame_size"],
        "intended_frame_drop_lengths": [cfg["initial_queue_length_per_channel"] - cfg["frame_size"]] * cfg["channels"],
    }


def inspect_dfn(root):
    source = (root / "deepfilternet/ladspa/src/lib.rs").read_text()
    patterns = {
        "channel_iteration": "for o_q_ch in o_q.iter_mut().take(self.frame_size)",
        "one_pop_per_channel": "o_q_ch.pop_front().unwrap();",
        "metadata_frame_decrement": "self.proc_delay -= self.frame_size;",
        "blocking_wait": "sleep(self.sleep_duration);",
        "delay_limit": "if self.proc_delay >= self.sr",
        "callback_counter": "self.t_proc_change += 1;",
        "hop_based_counter_threshold": "self.t_proc_change > 10 * self.sr / self.frame_size",
    }
    matches = {}
    for name, pattern in patterns.items():
        matches[name] = [i for i, line in enumerate(source.splitlines(), 1) if pattern in line]
        if not matches[name]:
            raise ValueError("expected static evidence missing: " + name)
    return {"execution": "static source inspection only", "rust_plugin_executed": False,
            "evidence_lines": matches, "queue_example": queue_semantics_example(),
            "callback_clock_example": callback_clock_example()}


def run_pystoi(root):
    sys.dont_write_bytecode = True
    identity = verify_sources(root)["pystoi"]
    loaded_pystoi_modules(identity)
    import numpy as np
    import scipy
    sys.path.insert(0, str((root / "pystoi").resolve()))
    import pystoi
    from pystoi import stoi
    if Path(pystoi.__file__).resolve().parent != (root / "pystoi/pystoi").resolve():
        raise ValueError("imported unexpected pystoi package")
    loaded_pystoi_modules(identity)
    function_path = upstream.ordinary_file(stoi.__code__.co_filename)
    if function_path != Path(identity['checkout'])/'pystoi/stoi.py':
        raise ValueError('imported foreign pystoi callable')
    cfg = CONFIG["pystoi"]
    inputs = [("zeros_" + str(n), np.zeros(n)) for n in cfg["zero_lengths"]]
    inputs += [("random_self", np.random.default_rng(4).normal(size=cfg["self_comparison_length"]))]
    records = []
    previous_rng_state = np.random.get_state()
    try:
        for name, x in inputs:
            for extended in (False, True):
                for seed in (cfg["normalization_seeds"] if extended else [0]):
                    np.random.seed(seed)
                    row = {"input": name, "samples": len(x), "extended": extended,
                           "normalization_seed": seed, "input_sha256": hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
                           "reference_equals_degraded": True, "reference_has_energy": bool(np.any(x != 0))}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            value = float(stoi(x.copy(), x.copy(), cfg["sample_rate_hz"], extended=extended))
                            row.update(value=value if np.isfinite(value) else None, finite=bool(np.isfinite(value)))
                            row["classification"] = ("nonfinite" if not row["finite"] else
                                "warning_sentinel" if caught else "finite_zero_energy_invalid" if not row["reference_has_energy"]
                                else "finite_self_comparison")
                        except Exception as exc:
                            row.update(value=None, finite=False, classification="exception",
                                       exception_type=type(exc).__name__, exception_message=str(exc))
                    row["warnings"] = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
                    records.append(row)
    finally:
        np.random.set_state(previous_rng_state)
    return {"execution": "original pystoi package functions", "numpy": np.__version__,
            "scipy": scipy.__version__, "records": records,
            "loaded_original_modules": loaded_pystoi_modules(identity),
            "environment_dependencies": {m.__name__: {"version": m.__version__, "path": str(upstream.ordinary_file(m.__file__)), "sha256": upstream.sha(m.__file__)} for m in (np, scipy)},
            "not_executed": ["MATLAB/Octave reference", "natural speech corpus", "resampling path (all inputs already 10 kHz)"]}


def run(root):
    sys.dont_write_bytecode = True
    root = upstream.validate_parent_chain(root)
    identities = verify_sources(root)
    try:
        result = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                  "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
                  "sources": SOURCES, "config": CONFIG,
                  "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
                  "pystoi": run_pystoi(root), "deepfilternet": inspect_dfn(root)}
        result["source_identities"] = {name: upstream.check_unchanged(identity) for name, identity in identities.items()}
        result["actual_dependencies_sha256"] = upstream.dependencies(__file__, (Path(verify_failure_sources.__code__.co_filename),))
        return result
    except BaseException as error:
        verify_failure_sources(identities, error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=CODES / "chapters/ch00/upstream/_downloads")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None:
        upstream.report_target(args.report, CURRENT, protected=(upstream.CACHE, args.upstream_root))
    result = run(args.upstream_root)
    if args.report is not None:
        upstream.write_report(args.report, result, CURRENT, protected=(upstream.CACHE, args.upstream_root))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
