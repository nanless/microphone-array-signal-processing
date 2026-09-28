"""Check known near-end injection through NLMS and SpeexDSP AEC.

The first fixture is fully synthetic with exact PCM-domain additivity.  The
second overlays a separately recorded real near-end microphone clip on a
real far-end pair; it is semi-synthetic, not simultaneous real double talk.
No third-party recordings are redistributed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np

from codes.array_tutorial.aec import nlms
from codes.examples.aec_real_pair_experiment import (
    FRAME, PAIR_DIR, RATE, ROOT, SOURCE_COMMIT, load_pinned_pair,
    speex_linear_aec,
)


SYNTH_SHA256 = {
    "aec_far.wav": "e4e5cd0e98f709fb326feae82157d08c60e422178ce6089f85c6f3442082cfca",
    "aec_near.wav": "3325c8296ad9c1c1316bba7250c895ee52eb098d07385681a1b380516681a1fb",
    "aec_microphone.wav": "772befe7256b195c486d34a80c23a5377304680bba135261679506cba6c218d6",
}
REAL_NEAR_NAME = "-0AcvGNEdEK-DQGxWmtq2Q_nearend_singletalk_mic.wav"
REAL_NEAR_SHA256 = "192291e973b7dee519102f30ae75e4209712ce55d02f55a9020a7c3b37a1d0e0"


def read_pinned_pcm(path: Path, sha256: str) -> np.ndarray:
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError(f"{path.name}: SHA-256 differs from pinned WAV (or is an LFS pointer)")
    with wave.open(io.BytesIO(blob), "rb") as wav:
        fmt = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
        if fmt != (1, 2, RATE, "NONE"):
            raise ValueError(f"{path.name}: expected mono PCM16 16 kHz, got {fmt}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    return samples


def checked_pcm_add(base: np.ndarray, injection: np.ndarray) -> np.ndarray:
    if base.dtype != np.int16 or injection.dtype != np.int16 or base.shape != injection.shape:
        raise ValueError("same-shape mono PCM16 arrays required")
    summed = base.astype(np.int32) + injection.astype(np.int32)
    if np.any((summed < -32768) | (summed > 32767)):
        raise ValueError("known near-end injection would clip PCM16")
    return summed.astype(np.int16)


def interval_mask(length: int, start: int, stop: int) -> np.ndarray:
    """Known activity covers the full declared interval, including zero crossings."""
    if not (0 <= start < stop <= length):
        raise ValueError("invalid known activity interval")
    mask = np.zeros(length, dtype=bool)
    mask[start:stop] = True
    return mask


def increment_metrics(base_output: np.ndarray, injected_output: np.ndarray,
                      known_near: np.ndarray, start: int, stop: int,
                      *, sample_unit: str = "pcm_counts") -> dict:
    """No fitted delay or gain: compare same-sample output increment to input."""
    for name, array in (("base_output", base_output), ("injected_output", injected_output),
                        ("known_near", known_near)):
        if array.ndim != 1 or np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite real 1-D array")
    if not (0 <= start < stop <= len(known_near)):
        raise ValueError("invalid fixed score interval")
    if base_output.shape != injected_output.shape or base_output.shape != known_near.shape:
        raise ValueError("output and known injection shapes differ")
    base = base_output[start:stop].astype(np.float64)
    injected = injected_output[start:stop].astype(np.float64)
    target = known_near[start:stop].astype(np.float64)
    scale = max(float(np.max(np.abs(base))), float(np.max(np.abs(injected))),
                float(np.max(np.abs(target))))
    if scale <= 0:
        raise ValueError("known near-end signal is silent in the score interval")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            delta_scaled = injected / scale - base / scale
            target_scaled = target / scale
            target_energy = float(np.dot(target_scaled, target_scaled))
            if target_energy <= 0:
                raise ValueError("known near-end energy is zero at the common numeric scale")
            error_scaled = delta_scaled - target_scaled
            projection = float(np.dot(delta_scaled, target_scaled) / target_energy)
            increment_ratio = float(np.dot(delta_scaled, delta_scaled) / target_energy)
            error_ratio = float(np.dot(error_scaled, error_scaled) / target_energy)
            near_rms = float(scale * np.sqrt(np.mean(target_scaled ** 2)))
    except FloatingPointError as exc:
        raise ValueError("increment metric exceeds the finite float64 range") from exc
    if not np.all(np.isfinite((projection, increment_ratio, error_ratio, near_rms))):
        raise ValueError("increment metric exceeds the finite float64 range")
    return {
        "score_interval_half_open": [start, stop],
        "sample_unit": sample_unit,
        "known_near_rms": near_rms,
        "increment_projection_gain_no_delay_fit": projection,
        "increment_to_known_near_power_ratio_db": float(10 * np.log10(increment_ratio))
            if increment_ratio > 0 else None,
        "fixed_sample_relative_squared_error": error_ratio,
        "fixed_sample_relative_error_db": float(10 * np.log10(error_ratio))
            if error_ratio > 0 else None,
        "exact_match_within_float_roundoff": bool(error_ratio <= 1e-24),
    }


def speex_pair(library: Path, reference: np.ndarray, base: np.ndarray,
               mixed: np.ndarray, near: np.ndarray, score: tuple[int, int],
               recovery: tuple[int, int]) -> dict:
    out_base, rate_base = speex_linear_aec(library, reference, base)
    out_mixed, rate_mixed = speex_linear_aec(library, reference, mixed)
    zero = np.zeros_like(reference)
    zero_base, rate_zero_base = speex_linear_aec(library, zero, base)
    zero_mixed, rate_zero_mixed = speex_linear_aec(library, zero, mixed)
    if {rate_base, rate_mixed, rate_zero_base, rate_zero_mixed} != {RATE}:
        raise RuntimeError("Speex sampling-rate readback differed")
    a, b = recovery
    if not (0 <= a < b <= len(reference)) or np.any(near[a:b]):
        raise ValueError("recovery interval must contain no injected near-end samples")
    def recovery_rms(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sqrt(np.mean((right[a:b].astype(float) - left[a:b].astype(float)) ** 2)))
    return {
        "algorithm": "SpeexDSP synchronous core, fresh state per run; no RES preprocessor",
        "paired_reference": increment_metrics(out_base, out_mixed, near, *score),
        "zero_reference": increment_metrics(zero_base, zero_mixed, near, *score),
        "recovery_interval_half_open": [a, b],
        "recovery_output_difference_rms_pcm_counts": recovery_rms(out_base, out_mixed),
        "zero_reference_recovery_difference_rms_pcm_counts": recovery_rms(zero_base, zero_mixed),
        "paired_output_sha256": hashlib.sha256(out_mixed.astype("<i2").tobytes()).hexdigest(),
        "zero_reference_output_sha256": hashlib.sha256(zero_mixed.astype("<i2").tobytes()).hexdigest(),
    }


def synthetic_fixture(library: Path, audio_dir: Path = ROOT / "codes/audio") -> dict:
    far = read_pinned_pcm(audio_dir / "aec_far.wav", SYNTH_SHA256["aec_far.wav"])
    near = read_pinned_pcm(audio_dir / "aec_near.wav", SYNTH_SHA256["aec_near.wav"])
    mixed = read_pinned_pcm(audio_dir / "aec_microphone.wav", SYNTH_SHA256["aec_microphone.wav"])
    if far.shape != near.shape or far.shape != mixed.shape or len(far) != 2 * RATE:
        raise ValueError("synthetic fixture shape changed")
    base32 = mixed.astype(np.int32) - near.astype(np.int32)
    if np.any((base32 < -32768) | (base32 > 32767)):
        raise ValueError("removing known near-end signal overflows PCM16")
    base = base32.astype(np.int16)
    if not np.array_equal(checked_pcm_add(base, near), mixed):
        raise AssertionError("PCM-domain addition identity failed")
    score, recovery = (19200, 28800), (28800, 32000)
    freeze = interval_mask(len(near), *score)
    x = far.astype(float) / 32768
    d0 = base.astype(float) / 32768
    d1 = mixed.astype(float) / 32768
    nlms_results = {}
    for label, mask in (("oracle_freeze", freeze), ("no_freeze", None)):
        out0, _, _ = nlms(x, d0, 32, step_size=0.4, epsilon=1e-8, freeze=mask)
        out1, _, _ = nlms(x, d1, 32, step_size=0.4, epsilon=1e-8, freeze=mask)
        nlms_results[label] = increment_metrics(out0, out1, d1 - d0, *score,
                                                sample_unit="digital_full_scale")
    return {
        "fixture": "fully synthetic harmonic near-end + sparse 3-tap echo; not speech or measured room",
        "source_wav_sha256": SYNTH_SHA256,
        "sample_rate_hz": RATE,
        "near_active_interval_half_open": [*score],
        "far_end_single_talk_interval_half_open": [9600, 17600],
        "pcm_identity": "mixed = (mixed - near) + near exactly in int16 counts; no clipping",
        "nlms": nlms_results,
        "speex": speex_pair(library, far, base, mixed, near, score, recovery),
    }


def real_hybrid_fixture(library: Path, pair_dir: Path = PAIR_DIR) -> dict:
    far, base, files = load_pinned_pair(pair_dir)
    near_source = read_pinned_pcm(pair_dir / REAL_NEAR_NAME, REAL_NEAR_SHA256)
    usable = min(len(far), len(base), len(near_source)) // FRAME * FRAME
    if usable < 11 * RATE:
        raise ValueError("real pair too short for fixed windows")
    far, base, near_source = far[:usable], base[:usable], near_source[:usable]
    score, recovery = (3 * RATE, 8 * RATE), (8 * RATE, 11 * RATE)
    injection = np.zeros_like(base)
    injection[score[0]:score[1]] = near_source[score[0]:score[1]]
    mixed = checked_pcm_add(base, injection)
    return {
        "fixture": "semi-synthetic overlay: real far-end pair plus separately recorded real near-end microphone waveform; not simultaneous double talk",
        "source_commit": SOURCE_COMMIT,
        "far_pair_files": files,
        "near_file": {"name": REAL_NEAR_NAME, "sha256": REAL_NEAR_SHA256,
                      "meaning": "separate near-end-single-talk mic recording, not clean speech"},
        "sample_rate_hz": RATE,
        "injection_gain": 1.0,
        "score_interval_half_open": [*score],
        "input_peak_pcm_counts": int(np.max(np.abs(mixed.astype(np.int32)))),
        "actual_near_rms_pcm_counts": float(np.sqrt(np.mean(injection[score[0]:score[1]].astype(float) ** 2))),
        "speex": speex_pair(library, far, base, mixed, injection, score, recovery),
    }


def run(library: Path, *, include_real_hybrid: bool = False,
        audio_dir: Path = ROOT / "codes/audio", pair_dir: Path = PAIR_DIR) -> dict:
    result = {
        "speex_source_commit": "8e29a256ef0235ebbe7fcb8417b5ac7731eb8307",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "metric_warning": "Output increment from two separately adapted runs includes state changes and preprocessing. It is not isolated near-end speech, perceptual quality or double-talk ERLE; no gain or delay is fitted in the fixed-sample error.",
        "synthetic": synthetic_fixture(library, audio_dir),
    }
    if include_real_hybrid:
        result["real_hybrid"] = real_hybrid_fixture(library, pair_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speex-library", type=Path, required=True)
    parser.add_argument("--include-real-hybrid", action="store_true",
                        help="also require the pinned third-party WAVs in ignored cache")
    args = parser.parse_args()
    print(json.dumps(run(args.speex_library, include_real_hybrid=args.include_real_hybrid),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
