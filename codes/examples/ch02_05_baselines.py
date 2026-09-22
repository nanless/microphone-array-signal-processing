"""Run the book's small Chapter 2--5 NumPy baselines.

Execute from the repository root:

    .venv/bin/python codes/examples/ch02_05_baselines.py

The script creates no files.  Its printed values are checks, not performance
benchmarks or claims about real recordings.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codes.array_tutorial.beamforming import (  # noqa: E402
    blocking_matrix,
    dsb_weights,
    lcmv_weights,
    mvdr_weights,
    wiener_gain,
)
from codes.array_tutorial.covariance import spatial_covariance  # noqa: E402
from codes.array_tutorial.doa import (  # noqa: E402
    bartlett_spectrum,
    capon_spectrum,
    gcc_phat,
    music_spectrum,
)
from codes.array_tutorial.geometry import plane_wave_steering  # noqa: E402
from codes.array_tutorial.spectral import istft, stft  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20260922)

    waveform = rng.standard_normal((2, 1000))
    spectra = stft(waveform, n_fft=128, hop_length=32)
    reconstructed = istft(spectra, n_fft=128, hop_length=32, length=1000)
    covariance = spatial_covariance(spectra)
    print("STFT shape (channels, frequency, frames):", spectra.shape)
    print("STFT round-trip maximum error:", np.max(np.abs(waveform - reconstructed)))
    print("SCM shape (frequency, channels, channels):", covariance.shape)

    sample_rate = 16_000
    x2 = np.zeros(64)
    x1 = np.zeros(64)
    x2[20] = 1.0
    x1[23] = 1.0
    tau, _, _, _ = gcc_phat(x1, x2, sample_rate)
    print("GCC-PHAT tau12 (samples):", tau * sample_rate)

    scan_covariance = np.array([[1.25, 1.0], [1.0, 1.25]], dtype=complex)
    scan_vectors = np.array([[1.0, 1.0], [1.0, 1.0j]])
    print("Bartlett [0 deg, 30 deg]:", bartlett_spectrum(scan_covariance, scan_vectors))
    print("Capon [0 deg, 30 deg]:", capon_spectrum(scan_covariance, scan_vectors))
    print("MUSIC [0 deg, 30 deg]:", music_spectrum(scan_covariance, scan_vectors, source_count=1))

    positions = np.array([[0.0, 0.0], [0.04, 0.0]])
    steering = plane_wave_steering(positions, [1000.0], np.deg2rad(30.0))[0]
    fixed = dsb_weights(steering)
    print("DSB target response:", np.vdot(fixed, steering))

    target = np.array([1.0, 1.0], dtype=complex)
    interference = np.array([1.0, 1.0j], dtype=complex)
    noise_covariance = 10.0 * np.outer(interference, interference.conj()) + np.eye(2)
    adaptive = mvdr_weights(noise_covariance, target)
    print("MVDR weights:", adaptive)
    print("MVDR target response:", np.vdot(adaptive, target))

    constraints = np.column_stack((target, interference))
    constrained = lcmv_weights(np.eye(2), constraints, np.array([1.0, 0.0]))
    blocker = blocking_matrix(target)
    print("LCMV responses [target, interference]:", constraints.conj().T @ constrained)
    print("GSC blocking residual:", blocker.conj().T @ target)
    print("Wiener gain for output power 4 and noise power 0.7:", wiener_gain(4.0, 0.7))


if __name__ == "__main__":
    main()
