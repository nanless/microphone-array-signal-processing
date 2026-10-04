"""Offline SOF TDFB source audit and independent scalar counterexamples.

This does not execute MATLAB, Octave, FIR design or firmware. It verifies the
locked unmodified source, records three static inconsistencies, and evaluates scalar
mathematical consequences. No upstream code is patched or imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
from codes.chapters.ch04.core import upstream_contracts as contracts

ROOT = Path(__file__).resolve().parents[3]
CURRENT_REPORT = ROOT / 'chapters/ch05/reports/sof_tdfb_design_current.json'
REVISION = "b6c6a05d52536313fe8e8752b1c4e069b1cc4002"
SOURCE_HASHES = {
    "LICENCE": "cb6a9aa5baac3ea573feebcdc298526f701e2b7cd005a22208799778e26f0068",
    "src/audio/tdfb/tune/sof_bf_design.m": "d5c7863bc1088df42231622fb525ecd0344324aecccf3dd1da7afa2debe03cf9",
    "src/audio/tdfb/tune/sof_bf_defaults.m": "07297c12736170f60cc3724fa66cf5e1876790f0e8a7de198a119eabb50b9438",
    "src/audio/tdfb/tune/sof_bf_paths.m": "574612f29672fffdd05fc1950ee62ccc206ad62221a42512ab2d326580b09d24",
    "src/audio/tdfb/tdfb_generic.c": "73216798217a1fa4012bb4fe90ba7fda091809b6f7580f1d6c05f2e004344909",
    "src/audio/tdfb/tdfb.c": "704dbd9c5b30ae2011be9d6cbc78b809f7b82947669ba99d8a15b4001d7873c5",
    "src/audio/tdfb/tdfb.h": "8f29aec9e1b80bd74330a0d86f03ddefe1357ba905bb7e97ba2673a0fa3f9ccc",
}


def independent_examples() -> dict:
    """Deterministic, dimensioned examples; these are not SOF filter outputs."""
    frequency, distance, sound_speed = 1000., .05, 343.
    phase = 2 * math.pi * frequency * distance / sound_speed
    spherical_coherence = math.sin(phase) / phase
    # MATLAB/Octave sinc(t) = sin(pi*t)/(pi*t), not sin(t)/t.
    source_expression_value = float(np.sinc(phase))
    weights = np.array([[.5, .5], [1., 0.]])
    steering = np.ones(2)
    last_diffuse = np.array([[1., .2], [.2, 1.]])
    target_power = np.abs(weights @ steering)**2
    correct_denominators = np.sum(np.abs(weights)**2, axis=1)
    stale_denominator = float(weights[-1] @ last_diffuse @ weights[-1])
    return {
        "coherence": {
            "frequency_hz": frequency, "distance_m": distance,
            "sound_speed_m_s": sound_speed,
            "phase_radians": phase,
            "correct_normalized_sinc_argument": 2 * frequency * distance / sound_speed,
            "spherical_coherence_sin_phase_over_phase": spherical_coherence,
            "fixed_source_expression_with_standard_sinc": source_expression_value,
            "randomness": "none",
        },
        "wng_loop_fixture": {
            "identity": "independently chosen two-frequency weights; not designed SOF filters",
            "weights_rows": weights.tolist(), "steering": steering.tolist(),
            "last_frequency_diffuse_coherence": last_diffuse.tolist(),
            "target_powers": target_power.tolist(),
            "white_noise_denominators": correct_denominators.tolist(),
            "stale_last_di_denominator": stale_denominator,
            "correct_wng_linear": (target_power / correct_denominators).tolist(),
            "stale_denominator_wng_linear": (target_power / stale_denominator).tolist(),
        },
        "loading_conversion": {
            "fixed_default_mu_db": -40.,
            "fixed_source_10_power_mu_db_over_20": 10**(-40/20),
            "if_interpreted_as_power_db_10_power_mu_db_over_10": 10**(-40/10),
            "interpretation": "record implementation convention; do not silently replace it",
        },
    }


def verify_source(source_dir: Path) -> dict:
    identity = contracts.verify_project('sof', source_dir, SOURCE_HASHES)
    source_dir = Path(identity['checkout'])
    # An unexpected local sinc implementation would invalidate our premise.
    if any(path.name.lower() == "sinc.m" for path in source_dir.rglob("*.m")):
        raise ValueError("local sinc.m requires a separate shadowing audit")
    for name, expected in SOURCE_HASHES.items():
        if identity['used_files'][name]['sha256'] != expected:
            raise ValueError("fixed source hash mismatch: " + name)
    return identity


def run_audit(source_dir: Path | None = None) -> dict:
    source_dir = contracts.validate_parent_chain(source_dir or ROOT / "chapters/ch00/upstream/_downloads/sof")
    identity = verify_source(source_dir)
    created = datetime.now(timezone.utc)
    report = {
        "schema_version": 1, "verified_at": created.date().isoformat(),
        "created_utc": created.isoformat(),
        "status": "static_source_inconsistencies_with_independent_scalar_examples",
        "execution_scope": {
            "source_verification": True, "python_scalar_examples": True,
            "matlab_or_octave_design": False, "fir_filter_generation": False,
            "firmware_or_hardware": False, "upstream_modified": False,
        },
        "provenance": {
            "url": "https://github.com/thesofproject/sof", "revision": REVISION,
            "source_sha256": SOURCE_HASHES,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "source_identity": identity,
        "report_source_sha256": contracts.dependencies(Path(__file__)),
        "findings": [
            {"id": "SOF-TDFB-COHERENCE", "file": "src/audio/tdfb/tune/sof_bf_design.m",
             "line": 123, "expression": "sinc(2*pi*f*lnm/bf.c)",
             "condition": "standard MATLAB/Octave normalized sinc; no local sinc.m",
             "effect": "extra pi in the spherical diffuse-coherence argument"},
            {"id": "SOF-TDFB-WNG", "file": "src/audio/tdfb/tune/sof_bf_design.m",
             "line": 272, "expression": "wng = num / denom2",
             "condition": "denom computed at line 271; denom2 retained from last DI iteration",
             "effect": "white-noise gain uses the last diffuse denominator, not current white-noise denominator"},
            {"id": "SOF-TDFB-DIFFUSE-RESET", "file": "src/audio/tdfb/tune/sof_bf_design.m",
             "line": 412, "expression": "nmi = zeros(nti, bf.mic_n)",
             "condition": "optional create_simulation_data; azimuth loop begins at line 405, reset occurs inside each angle",
             "effect": "post-loop output retains only the last azimuth contribution; this branch was not executed",
             "execution": "static_source_only; no MATLAB, audio generation or firmware execution"},
        ],
        "fixed_source_defaults": {
            "design_fft_length": 1024, "nonnegative_frequency_bins": 513,
            "sample_rate_hz": 16000, "fir_taps": 64, "window": "hann",
            "filter_type": "SDB", "mu_db": -40, "track_doa": 0,
        },
        "documentation_boundary": {
            "url": "https://thesofproject.github.io/latest/developer_guides/algorithms/tdfb/time_domain_fixed_beamformer.html",
            "observed": "design description mentions old tools/tune/tdfb/bf_design.m, 512 bins, Kaiser and mu -50",
            "interpretation": "floating documentation is not the parameter authority for the locked source",
        },
        "mathematical_examples": independent_examples(),
    }
    contracts.check_unchanged(identity)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", "--report", dest='output', type=Path)
    args = parser.parse_args()
    protected = (contracts.CACHE, args.source_dir) if args.source_dir else (contracts.CACHE,)
    target = contracts.report_target(args.output, CURRENT_REPORT, protected) if args.output else None
    report = run_audit(args.source_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if target is not None:
        contracts.write_report(target, report, CURRENT_REPORT, protected)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
