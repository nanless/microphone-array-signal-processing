"""E04-08: repeated two-source narrowband DOA resolution experiment.

Run with ``.venv/bin/python -m codes.examples.doa_resolution_trials``.
All samples are mathematical complex-Gaussian snapshots. This is a controlled
resolution comparison, not an evaluation on speech or measured rooms.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from codes.array_tutorial.doa import bartlett_spectrum, capon_spectrum, music_spectrum
from codes.examples.mdl_repeated_trials import wilson_interval


DEFAULT_SEED = 20260924
SEPARATIONS_DEG = (50.0, 15.0, 8.0)
GRID_DEG = np.arange(-60.0, 60.0 + 0.5, 0.5)
METHODS = ("Bartlett", "Capon", "MUSIC")


def steering_rows(angles_deg: np.ndarray, channels: int = 8) -> np.ndarray:
    """Rows use a_m(theta)=exp(+j*pi*m*sin(theta)), d=lambda/2."""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1 or channels < 2 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be finite 1-D values and channels >= 2")
    return np.exp(1j * np.pi * np.sin(np.deg2rad(angles))[:, None]
                  * np.arange(channels)[None, :])


def two_peaks(scores: np.ndarray, grid_deg: np.ndarray = GRID_DEG,
              minimum_separation_deg: float = 3.0) -> list[float]:
    """Pick two highest *local* peaks subject to a predeclared separation."""
    values = np.asarray(scores, dtype=float)
    grid = np.asarray(grid_deg, dtype=float)
    if (values.ndim != 1 or grid.ndim != 1 or values.size != grid.size
            or values.size < 3 or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
            or not np.isfinite(minimum_separation_deg)
            or minimum_separation_deg <= 0):
        raise ValueError("finite scores and increasing grid of length >= 3 required")
    local = np.flatnonzero((values[1:-1] > values[:-2])
                           & (values[1:-1] >= values[2:])) + 1
    ranked = sorted(local, key=lambda i: (-values[i], grid[i]))
    chosen: list[float] = []
    for i in ranked:
        angle = float(grid[i])
        if all(abs(angle-other) >= minimum_separation_deg for other in chosen):
            chosen.append(angle)
        if len(chosen) == 2:
            break
    return sorted(chosen)


def classify_peaks(peaks_deg: list[float], truth_deg: tuple[float, float],
                   tolerance_deg: float = 3.0) -> str:
    """One-to-one ordered matching for two disjoint truth neighborhoods."""
    if len(peaks_deg) < 2:
        return "fewer_than_two_peaks"
    if len(peaks_deg) != 2 or not np.isfinite(tolerance_deg) or tolerance_deg <= 0:
        raise ValueError("require exactly two peaks and positive tolerance")
    predicted = sorted(peaks_deg)
    truth = sorted(truth_deg)
    if all(abs(a-b) <= tolerance_deg for a, b in zip(predicted, truth)):
        return "matched"
    return "wrong_location"


def _complex_normal(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)) / math.sqrt(2)


def run_experiment(trials: int = 200, seed: int = DEFAULT_SEED,
                   snapshots: int = 400, array_snr_db: float = 20.0) -> dict:
    """Independent trials; all methods/angles share each trial's source/noise draw.

    The two sources each have unit variance. Array SNR is total two-source
    signal power per microphone divided by its independent noise variance.
    MUSIC is given the true source count 2; the other methods also select
    exactly two peaks, so this does not evaluate automatic source counting.
    """
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in (trials, seed, snapshots)):
        raise ValueError("trials, seed and snapshots must be integers")
    if trials < 1 or seed < 0 or snapshots < 8 or not np.isfinite(array_snr_db):
        raise ValueError("require trials>=1, seed>=0, snapshots>=8, finite SNR")
    trials, seed, snapshots = int(trials), int(seed), int(snapshots)
    noise_power = 2.0 / (10.0 ** (array_snr_db / 10.0))
    if not np.isfinite(noise_power) or noise_power <= 0.0:
        raise ValueError("SNR gives nonrepresentable noise power")
    candidates = steering_rows(GRID_DEG)
    counts = {separation: {method: {category: 0 for category in
              ("matched", "fewer_than_two_peaks", "wrong_location", "algorithm_error")}
              for method in METHODS} for separation in SEPARATIONS_DEG}

    for trial in range(trials):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, trial])))
        sources = _complex_normal(rng, (2, snapshots))
        noise = math.sqrt(noise_power) * _complex_normal(rng, (8, snapshots))
        for separation in SEPARATIONS_DEG:
            truth = (-separation/2.0, separation/2.0)
            steering = steering_rows(np.asarray(truth)).T
            x = steering @ sources + noise
            covariance = x @ x.conj().T / snapshots
            for method in METHODS:
                try:
                    if method == "Bartlett":
                        spectrum = bartlett_spectrum(covariance, candidates)
                    elif method == "Capon":
                        spectrum = capon_spectrum(covariance, candidates,
                                                  relative_diagonal_loading=1e-6)
                    else:
                        spectrum = music_spectrum(covariance, candidates, source_count=2)
                    category = classify_peaks(two_peaks(spectrum), truth)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    category = "algorithm_error"
                counts[separation][method][category] += 1

    cases = []
    for separation in SEPARATIONS_DEG:
        for method in METHODS:
            histogram = counts[separation][method]
            matched = histogram["matched"]
            cases.append({"separation_deg": separation,
                          "truth_deg": [-separation/2, separation/2],
                          "method": method, "outcome_counts": histogram,
                          "success_rate": matched/trials,
                          "wilson_95_interval": wilson_interval(matched, trials)})
    return {
        "model": "two independent unit-power circular complex Gaussian narrowband sources",
        "array": "8-element ULA, half-wavelength spacing, broadside zero, positive toward +x",
        "trials": trials, "trial_seed_sequence": "PCG64(SeedSequence([seed, trial_index]))",
        "seed": seed, "snapshots_per_trial": snapshots,
        "array_input_snr_db": array_snr_db, "noise_power_per_channel": noise_power,
        "shared_random_inputs_across_methods_and_separations": True,
        "candidate_grid_deg": [-60.0, 60.0, 0.5],
        "peak_selection": "two highest local peaks, at least 3 degrees apart",
        "success_definition": "one-to-one angle errors <= 3 degrees for both sources",
        "music_source_count_supplied": 2,
        "capon_relative_diagonal_loading": 1e-6,
        "interval": "two-sided Wilson 95% for independent trial events, NIST 7.2.4.1",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--snapshots", type=int, default=400)
    parser.add_argument("--snr-db", type=float, default=20.0)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.trials, args.seed, args.snapshots, args.snr_db),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
