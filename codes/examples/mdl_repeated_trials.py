"""E04-06: independently resampled narrowband MDL boundary experiments.

No speech, STFT, audio generation or downloads are involved. Each trial draws
new zero-mean circular complex Gaussian snapshots and scores the uncentered
sample covariance without loading. Counts are experimental observations, not
guarantees for rooms or recordings. Run this file for a JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.array_tutorial.doa import mdl_source_count

SEED = 20260922


def wilson_interval(successes: int, trials: int) -> list[float]:
    """Two-sided 95% Wilson score interval for independent Bernoulli trials.

    Formula: NIST/SEMATECH e-Handbook, section 7.2.4.1 (no continuity correction).
    It quantifies a repeated-trial match probability, not a single MDL decision.
    """
    if (isinstance(trials, (bool, np.bool_)) or not isinstance(trials, (int, np.integer))
            or trials < 1 or isinstance(successes, (bool, np.bool_))
            or not isinstance(successes, (int, np.integer)) or not 0 <= successes <= trials):
        raise ValueError("require integer 0 <= successes <= trials and trials > 0")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [0. if successes == 0 else max(0., center-radius),
            1. if successes == trials else min(1., center+radius)]


def _complex_normal(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)) / math.sqrt(2)


def run_experiment(trials: int = 200, seed: int = SEED) -> dict:
    """Seven conditions with separate PCG64 streams, all trials retained."""
    for name, value, minimum in (("trials", trials, 1), ("seed", seed, 0)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    trials, seed = int(trials), int(seed)
    steering = np.exp(1j*np.pi*np.arange(4)[:, None]*np.array([-.5, .5])[None, :])
    specifications = (
        ("independent_n100_p1", 100, 1., "independent", [1., 1., 1., 1.]),
        ("independent_n16_p1", 16, 1., "independent", [1., 1., 1., 1.]),
        ("independent_n100_p01", 100, .1, "independent", [1., 1., 1., 1.]),
        ("independent_n16_p01", 16, .1, "independent", [1., 1., 1., 1.]),
        ("coherent_n100_p1", 100, 1., "coherent", [1., 1., 1., 1.]),
        ("white_noise_only", 100, 0., "none", [1., 1., 1., 1.]),
        ("colored_noise_only", 100, 0., "none", [9., 4., 1., 1.]),
    )
    cases = []
    for index, (name, count, power, source_model, noise_variances) in enumerate(specifications):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
        noise_diagonal = np.asarray(noise_variances)
        source_covariance = power * (np.ones((2, 2)) if source_model == "coherent" else np.eye(2))
        signal_covariance = steering @ source_covariance @ steering.conj().T
        population_covariance = signal_covariance + np.diag(noise_diagonal)
        physical_count = 0 if source_model == "none" else 2
        histogram = np.zeros(4, dtype=int)
        for _ in range(trials):
            noise = np.sqrt(noise_diagonal[:, None]) * _complex_normal(rng, (4, count))
            if source_model == "none":
                snapshots = noise
            else:
                sources = _complex_normal(rng, (1 if source_model == "coherent" else 2, count))
                if source_model == "coherent":
                    sources = np.repeat(sources, 2, axis=0)
                snapshots = math.sqrt(power) * (steering @ sources) + noise
            sample_covariance = snapshots @ snapshots.conj().T / count
            selected, _ = mdl_source_count(np.linalg.eigvalsh(sample_covariance), count)
            histogram[selected] += 1
        matches = int(histogram[physical_count])
        snr = (float(10*np.log10(np.trace(signal_covariance).real / noise_diagonal.sum()))
               if physical_count else None)
        cases.append({
            "name": name, "seed_sequence": [seed, index], "snapshots": count,
            "source_model": source_model, "per_source_power": power,
            "noise_variances": noise_variances, "physical_source_count": physical_count,
            "signal_covariance_rank": 0 if physical_count == 0 else (1 if source_model == "coherent" else 2),
            "mdl_physical_source_model_valid": source_model != "coherent" and noise_variances == [1., 1., 1., 1.],
            "array_mean_snr_db": snr,
            "population_eigenvalues_descending": np.linalg.eigvalsh(population_covariance)[::-1].tolist(),
            "selected_count_histogram_k0_to_k3": histogram.tolist(),
            "physical_count_match_rate": matches/trials,
            "physical_count_match_wilson95": wilson_interval(matches, trials),
        })
    return {
        "seed": seed, "trials_per_condition": trials, "channels": 4,
        "frequency_hz": 1000., "sound_speed_m_s": 343., "spacing_m": .1715,
        "source_azimuths_deg": [-30., 30.], "random_generator": "PCG64",
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "model": "iid known-zero-mean circular complex Gaussian narrowband snapshots; no STFT or loading",
        "interval": "per-condition two-sided Wilson 95%, not simultaneous coverage or MDL posterior confidence",
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    print(json.dumps(run_experiment(arguments.trials, arguments.seed), indent=2, allow_nan=False))
