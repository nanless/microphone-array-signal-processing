"""Compare the teaching WPE solver with unmodified nara-wpe 0.0.11.

Run from the repository root with:
    .venv/bin/python codes/chapters/ch07/examples/compare_wpe_reference.py

This optional experiment requires NumPy, installed nara-wpe==0.0.11, and
the existing fixed NARA Git source checkout with current lock/status records.
It does not download inputs, install dependencies, or evaluate speech quality.
Both solvers use valid statistics, no power smoothing and no diagonal loading.
Only frames with a complete regression history are compared: nara-wpe filters
startup frames using zero padding, whereas the teaching solver passes them
through. Random complex inputs keep the power floors inactive in this test.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.chapters.ch07.core.dereverberation import offline_wpe
from codes.chapters.ch04.core import upstream_contracts as contracts


def compare() -> dict:
    from codes.chapters.ch07.examples.compare_online_wpe_reference import (
        _load_locked_module, _check_locked_module, NARA_WPE_MODULE_SHA256,
    )
    module, identity, installed = _load_locked_module()
    wpe_v6, version = module.wpe_v6, installed['version']
    reference_sha256 = NARA_WPE_MODULE_SHA256

    seed = 20260922
    rng = np.random.default_rng(seed)
    observed = rng.standard_normal((4, 2, 192)) + 1j * rng.standard_normal((4, 2, 192))
    # A fixed lagged component makes the prediction problem nontrivial.
    observed[..., 3:] += (0.4 + 0.2j) * observed[..., :-3].copy()
    taps, delay = 2, 3
    first = taps + delay - 1
    cases = {
        "two_channels": observed,
        "one_channel": observed[:, :1, :],
        "channel_permutation": observed[:, ::-1, :],
        "scaled": 0.25 * observed,
    }
    checks = []
    for name, signal in cases.items():
        for iterations in (1, 3):
            result = offline_wpe(
                signal, taps=taps, delay=delay, iterations=iterations,
                diagonal_loading=0.0, power_floor=1e-10,
            )
            reference = wpe_v6(
                signal, taps=taps, delay=delay, iterations=iterations,
                psd_context=0, statistics_mode="valid",
            )
            actual_valid = result[..., first:]
            reference_valid = reference[..., first:]
            difference = actual_valid - reference_valid
            max_abs = float(np.max(np.abs(difference)))
            relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(reference_valid))
            np.testing.assert_allclose(actual_valid, reference_valid, rtol=1e-10, atol=1e-11)
            # The teaching solver's startup behavior is an intentional contract.
            np.testing.assert_array_equal(result[..., :first], signal[..., :first])
            checks.append({"case": name, "iterations": iterations,
                           "max_absolute_complex_error": max_abs,
                           "relative_l2_error": relative_l2})
    _check_locked_module(identity, installed)
    return {
        "source_identity": identity, "installed_identity": installed,
        "actual_dependency_sha256": contracts.dependencies(Path(__file__),
            (Path(__file__).with_name("compare_online_wpe_reference.py"),
             ROOT/"codes/chapters/ch07/core/dereverberation.py",
             ROOT/"codes/chapters/ch02/core/conventions.py")),
        "reference": f"nara-wpe {version} / wpe_v6",
        "reference_module_sha256": reference_sha256,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "system": platform.system(), "machine": platform.machine(),
                        "dtype": str(observed.dtype)},
        "seed": seed,
        "shape_FCT": list(observed.shape),
        "taps": taps,
        "delay_frames": delay,
        "first_compared_frame_zero_based": first,
        "statistics_mode": "valid",
        "power_context": 0,
        "diagonal_loading": 0.0,
        "checks": checks,
        "scope": "Complex regression equivalence on synthetic STFT arrays; not speech quality or real-time validation.",
    }


if __name__ == "__main__":
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
