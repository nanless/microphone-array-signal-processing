"""Run a small blind AuxIVA experiment against the pinned ssspy source tree.

The source signals and mixing matrix are used only to generate and score the
mixture. AuxIVA receives the two observed microphone STFTs, with no source,
mixing matrix, activity label, or true demixing filter passed to it.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np

from codes.array_tutorial.separation import pit_permutation, si_sdr
from codes.array_tutorial.spectral import istft, stft


ROOT = Path(__file__).resolve().parents[2]
SOURCE_TREE = ROOT / "codes/upstream/_downloads/ssspy"
SOURCE_REVISION = "38b9389e8b1914422561f1936d9b28d042d62d2c"
SEED = 2468
SAMPLE_RATE_HZ = 8000
SAMPLES = 32000
N_FFT = 256
HOP = 64
ITERATIONS = 30
MIXING = np.array([[1.0, 0.9], [0.8, 1.0]])


def pinned_auxiva_class():
    """Load only the checked-out version recorded in SOURCES.lock.json."""

    if not (SOURCE_TREE / "ssspy/bss/iva.py").is_file():
        raise RuntimeError("Fetch the locked ssspy source tree before this experiment")
    revision = subprocess.run(
        ["git", "-C", str(SOURCE_TREE), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"ssspy revision {revision} differs from lock {SOURCE_REVISION}")
    changed = subprocess.run(
        ["git", "-C", str(SOURCE_TREE), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if changed:
        raise RuntimeError("ssspy checkout has local modifications; inspect them before reproduction")
    sys.path.insert(0, str(SOURCE_TREE))
    from ssspy.bss.iva import AuxLaplaceIVA

    if not Path(inspect.getfile(AuxLaplaceIVA)).resolve().is_relative_to(SOURCE_TREE.resolve()):
        raise RuntimeError("Imported AuxLaplaceIVA from a different source tree")
    return AuxLaplaceIVA


def independent_sources() -> np.ndarray:
    """Two independent seeded carriers with distinct deterministic envelopes."""

    rng = np.random.default_rng(SEED)
    carriers = rng.standard_normal((2, SAMPLES))
    envelope_levels = np.array([
        [1.0, 0.08, 0.5, 0.1, 1.0, 0.15, 0.7, 0.2],
        [0.08, 1.0, 0.2, 0.7, 0.12, 1.0, 0.15, 0.8],
    ])
    envelope = np.repeat(envelope_levels, SAMPLES // envelope_levels.shape[1], axis=1)
    sources = carriers * envelope
    return sources / np.std(sources, axis=1, keepdims=True)


def _score(observation: np.ndarray, estimate: np.ndarray, references: np.ndarray) -> dict:
    permutation, mean_output = pit_permutation(estimate, references)
    input_scores = [si_sdr(observation[0], source) for source in references]
    output_scores = [si_sdr(estimate[index], references[reference])
                     for index, reference in enumerate(permutation)]
    gain_ratios = [float(np.linalg.norm(estimate[index]) / np.linalg.norm(references[reference]))
                   for index, reference in enumerate(permutation)]
    return {
        "output_to_reference": list(permutation),
        "input_si_sdr_db_per_reference": input_scores,
        "output_si_sdr_db_per_output": output_scores,
        "mean_input_si_sdr_db": float(np.mean(input_scores)),
        "mean_output_si_sdr_db": mean_output,
        "mean_si_sdri_db": mean_output - float(np.mean(input_scores)),
        "output_to_reference_rms_ratios": gain_ratios,
    }


def _fit(observation: np.ndarray, auxiva_class) -> tuple[np.ndarray, list[float]]:
    coefficients = stft(observation, n_fft=N_FFT, hop_length=HOP)
    model = auxiva_class(
        spatial_algorithm="IP", scale_restoration=True,
        record_loss=True, reference_id=0,
    )
    separated = model(coefficients, n_iter=ITERATIONS)
    output = istft(separated, n_fft=N_FFT, hop_length=HOP, length=observation.shape[1])
    return output, [float(value) for value in model.loss]


def _read_mono_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as sound:
        if sound.getnchannels() != 1 or sound.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16: {path}")
        sample_rate = sound.getframerate()
        samples = np.frombuffer(sound.readframes(sound.getnframes()), dtype="<i2")
    return samples.astype(np.float64) / 32768.0, sample_rate


def run_experiment(*, harmonic_counterexample: bool = True) -> dict:
    auxiva_class = pinned_auxiva_class()
    sources = independent_sources()
    observation = MIXING @ sources
    coefficients = stft(observation, n_fft=N_FFT, hop_length=HOP)
    roundtrip = istft(coefficients, n_fft=N_FFT, hop_length=HOP, length=SAMPLES)
    output, loss = _fit(observation, auxiva_class)
    references = MIXING[0, :, None] * sources
    report = {
        "scope": "single seeded mathematical-signal example, not speech or paper-result reproduction",
        "upstream": {"id": "ssspy", "revision": SOURCE_REVISION,
                     "algorithm": "AuxLaplaceIVA", "spatial_update": "IP"},
        "input": {"sample_rate_hz": SAMPLE_RATE_HZ, "samples": SAMPLES,
                  "seed": SEED, "source": "independent standard Gaussian carriers times fixed eight-segment envelopes, each normalized by its standard deviation",
                  "mixing_matrix_for_generation_and_scoring_only": MIXING.tolist(),
                  "mixing_matrix_condition_number": float(np.linalg.cond(MIXING)),
                  "noise": "none", "reverberation": "none"},
        "analysis": {"n_fft": N_FFT, "hop_samples": HOP, "window": "periodic Hann",
                     "center_padding": True, "iterations": ITERATIONS,
                     "projection_back_reference_microphone": 0,
                     "padded_stft_roundtrip_max_abs_error": float(np.max(np.abs(roundtrip - observation)))},
        "score": _score(observation, output, references),
        "objective": {"initial": loss[0], "final": loss[-1], "recorded_points": len(loss)},
    }

    if harmonic_counterexample:
        first, first_rate = _read_mono_pcm16(ROOT / "codes/audio/separation_source1.wav")
        second, second_rate = _read_mono_pcm16(ROOT / "codes/audio/separation_source2.wav")
        if first_rate != second_rate or first.shape != second.shape:
            raise ValueError("Existing harmonic source assets have incompatible sampling or length")
        harmonic_sources = np.stack((first, second))
        harmonic_mix = np.array([[1.0, 0.5], [0.2, 1.0]]) @ harmonic_sources
        harmonic_output, harmonic_loss = _fit(harmonic_mix, auxiva_class)
        harmonic_reference = np.stack((first, 0.5 * second))
        report["harmonic_counterexample"] = {
            "source_assets": ["codes/audio/separation_source1.wav", "codes/audio/separation_source2.wav"],
            "sample_rate_hz": first_rate,
            "scope": "one fixed mathematical harmonic input; not a general failure rate",
            "score": _score(harmonic_mix, harmonic_output, harmonic_reference),
            "objective": {"initial": harmonic_loss[0], "final": harmonic_loss[-1]},
        }

    # A rank-one observation cannot identify two independent source images.
    # Run the pinned method, but do not score an arbitrary output as success.
    singular = np.array([[1.0, 0.9], [1.0, 0.9]])
    report["rank_deficient_boundary"] = {
        "mixing_matrix": singular.tolist(),
        "rank": int(np.linalg.matrix_rank(singular)),
        "sources": 2,
        "independent_observations": 1,
    }
    try:
        _fit(singular @ sources, auxiva_class)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        report["rank_deficient_boundary"]["upstream_run"] = {
            "status": "failed_numerically", "error_type": type(error).__name__,
            "message": str(error),
        }
    else:
        report["rank_deficient_boundary"]["upstream_run"] = {
            "status": "returned_without_identifiability_guarantee",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-harmonic-counterexample", action="store_true")
    args = parser.parse_args()
    result = run_experiment(harmonic_counterexample=not args.without_harmonic_counterexample)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
