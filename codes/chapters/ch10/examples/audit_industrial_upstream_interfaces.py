"""Pinned industrial boundary audit; no downloads, models or audio devices.

pystoi is called as the original package. DeepFilterNet is only read statically;
the queue example below is independent Python arithmetic, not a Rust execution.
"""
from __future__ import annotations

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
    locks = {p["id"]: p for p in json.loads((CODES / "chapters/ch00/SOURCES.lock.json").read_text())["projects"]}
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for name, spec in SOURCES.items():
        checkout = root / name
        def git(*args):
            return subprocess.check_output(["git", "-C", str(checkout), *args], env=env,
                                           text=True, stderr=subprocess.PIPE).strip()
        if locks[name]["revision"] != spec["revision"] or git("rev-parse", "HEAD") != spec["revision"]:
            raise ValueError("revision mismatch: " + name)
        if Path(git("rev-parse", "--show-toplevel")).resolve() != checkout.resolve():
            raise ValueError("not an independent checkout: " + name)
        if git("status", "--porcelain", "--untracked-files=all"):
            raise ValueError("upstream worktree is not clean: " + name)
        for relative, digest in spec["files"].items():
            if sha256(checkout / relative) != digest:
                raise ValueError("source digest mismatch: " + name + "/" + relative)


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
    }
    matches = {}
    for name, pattern in patterns.items():
        matches[name] = [i for i, line in enumerate(source.splitlines(), 1) if pattern in line]
        if not matches[name]:
            raise ValueError("expected static evidence missing: " + name)
    return {"execution": "static source inspection only", "rust_plugin_executed": False,
            "evidence_lines": matches, "queue_example": queue_semantics_example()}


def run_pystoi(root):
    import numpy as np
    import scipy
    sys.path.insert(0, str((root / "pystoi").resolve()))
    import pystoi
    from pystoi import stoi
    if Path(pystoi.__file__).resolve().parent != (root / "pystoi/pystoi").resolve():
        raise ValueError("imported unexpected pystoi package")
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
            "not_executed": ["MATLAB/Octave reference", "natural speech corpus", "resampling path (all inputs already 10 kHz)"]}


def run(root):
    sys.dont_write_bytecode = True
    verify_sources(root)
    result = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
              "sources": SOURCES, "config": CONFIG,
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
              "pystoi": run_pystoi(root), "deepfilternet": inspect_dfn(root)}
    verify_sources(root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=CODES / "chapters/ch00/upstream/_downloads")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.upstream_root)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
