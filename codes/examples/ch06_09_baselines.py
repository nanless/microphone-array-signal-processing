"""Run compact chapter 6--9 NumPy baselines from the repository root.

The AEC input is noiseless and exactly matched, so its very high ERLE only
checks arithmetic convergence; it is not a real-recording performance claim.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codes.array_tutorial.aec import erle_db, nlms  # noqa: E402
from codes.array_tutorial.dereverberation import offline_wpe  # noqa: E402
from codes.array_tutorial.separation import pit_permutation, si_sdr  # noqa: E402
from codes.array_tutorial.tracking import CircularParticleFilter, ConstantVelocityKalman  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)

    reference = rng.normal(size=3000)
    path = np.array([0.8, -0.25, 0.1])
    microphone = np.convolve(reference, path, mode="full")[: reference.size]
    residual, _, weights = nlms(reference, microphone, 3, step_size=0.5)
    print(
        "AEC weights:",
        np.round(weights, 3),
        "matched-noiseless ERLE:",
        round(erle_db(microphone[1000:], residual[1000:]), 1),
        "dB (arithmetic check only)",
    )

    predictor = 0.5 + 0.25j
    spectrum = np.empty((1, 30), dtype=np.complex128)
    spectrum[0, 0] = 1.0 + 0.3j
    for frame in range(1, spectrum.shape[-1]):
        spectrum[0, frame] = np.conj(predictor) * spectrum[0, frame - 1]
    dereverberated = offline_wpe(spectrum, taps=1, delay=1, iterations=1, power_floor=1e-8)
    before_power = float(np.mean(np.abs(spectrum[0, 1:]) ** 2))
    after_power = float(np.mean(np.abs(dereverberated[0, 1:]) ** 2))
    print(
        "WPE known one-tap predictor:",
        predictor,
        "valid-frame power",
        f"{before_power:.3e} -> {after_power:.3e}",
    )

    references = rng.normal(size=(2, 400))
    estimates = references[::-1] + 0.01 * rng.normal(size=references.shape)
    permutation, score = pit_permutation(estimates, references)
    print("PIT permutation:", permutation, "mean SI-SDR:", round(score, 1), "dB")
    print("single-source SI-SDR:", round(si_sdr(estimates[0], references[1]), 1), "dB")

    kalman = ConstantVelocityKalman([179.0, 2.0], np.eye(2), np.diag([0.1, 0.01]))
    kalman.predict(1.0)
    kalman.update(-178.0, 4.0)
    print("Kalman state:", np.round(kalman.state, 2))

    particle_filter = CircularParticleFilter(np.linspace(-180.0, 180.0, 200, endpoint=False))
    particle_filter.predict(2.0, 1.0, 1.0, rng)
    particle_filter.update(-178.0, 5.0, clutter_probability=0.1)
    particle_filter.resample_if_needed(rng)
    print("PF angle:", round(particle_filter.estimate(), 1), "degrees")


if __name__ == "__main__":
    main()
