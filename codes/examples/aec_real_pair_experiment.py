"""Run a pinned, real loopback/microphone pair through SpeexDSP's linear AEC.

The Microsoft AEC Challenge recordings stay in the ignored upstream cache.
This script does not download, publish, or relabel them as clean echo truth.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAIR_DIR = ROOT / "codes/upstream/_downloads/aec-challenge/datasets/real"
PREFIX = "-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_"
SOURCE_COMMIT = "6c633d0a9d2a143a0e364899b91b06f127315b18"
SOURCE_SHA256 = {
    "lpb": "9b204ad5473726526d14830103e53647897699ef89d49624964b7f2449040426",
    "mic": "6b4c3e01b969c5cad91f248ff967cfa03df6554f3a060d6e5b06b7d20341bba6",
}
RATE = 16000
FRAME = 160
FILTER = 4096
SET_SAMPLING_RATE = 24
GET_SAMPLING_RATE = 25


def load_pinned_pair(directory: Path = PAIR_DIR) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reject LFS pointers, altered bytes, non-PCM16 or changed channel layouts."""
    arrays: dict[str, np.ndarray] = {}
    files: dict[str, dict] = {}
    for kind in ("lpb", "mic"):
        path = directory / f"{PREFIX}{kind}.wav"
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != SOURCE_SHA256[kind]:
            raise ValueError(f"{kind} SHA-256 differs from pinned original (or is an LFS pointer)")
        with wave.open(io.BytesIO(blob), "rb") as wav:
            fmt = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                   wav.getcomptype())
            if fmt != (1, 2, RATE, "NONE"):
                raise ValueError(f"{kind} must be mono, PCM16, 16 kHz WAV: got {fmt}")
            frames = wav.getnframes()
            pcm = np.frombuffer(wav.readframes(frames), dtype="<i2").copy()
        if len(pcm) != frames:
            raise ValueError(f"{kind} WAV ended early")
        arrays[kind] = pcm
        files[kind] = {"name": path.name, "sha256": digest, "samples": frames,
                       "duration_s": frames / RATE}
    return arrays["lpb"], arrays["mic"], files


def peak_lag(reference: np.ndarray, microphone: np.ndarray, *, start: int,
             stop: int, max_lag: int) -> tuple[int, float]:
    """Diagnostic normalized correlation; positive lag means mic follows loopback.

    This is not a physical device-delay measurement or an alignment operation.
    The denominator is recomputed over the overlap at each candidate lag.
    """
    if not (0 <= start < stop <= min(len(reference), len(microphone))):
        raise ValueError("invalid diagnostic interval")
    if not 0 <= max_lag < stop - start:
        raise ValueError("invalid max_lag")
    x = reference.astype(np.float64) / 32768.0
    d = microphone.astype(np.float64) / 32768.0
    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        low, high = max(start, start + lag), min(stop, stop + lag)
        xs, ds = x[low - lag:high - lag], d[low:high]
        norm = float(np.linalg.norm(xs) * np.linalg.norm(ds))
        corr = float(np.dot(xs, ds) / norm) if norm > 0 else 0.0
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
    return best_lag, best_corr


def speex_linear_aec(library: Path, reference: np.ndarray,
                     microphone: np.ndarray) -> tuple[np.ndarray, int]:
    """Drive the synchronous 16-bit API with exact 10 ms, paired frames."""
    if reference.dtype != np.int16 or microphone.dtype != np.int16:
        raise ValueError("Speex inputs must be PCM16")
    if reference.ndim != 1 or microphone.ndim != 1 or reference.shape != microphone.shape:
        raise ValueError("Speex inputs must be equal-length mono arrays")
    if len(reference) % FRAME:
        raise ValueError("Speex inputs must contain whole 10 ms frames")
    dsp = ctypes.CDLL(str(library.resolve(strict=True)))
    dsp.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    dsp.speex_echo_state_init.restype = ctypes.c_void_p
    dsp.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dsp.speex_echo_ctl.restype = ctypes.c_int
    dsp.speex_echo_cancellation.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p]
    dsp.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    state = dsp.speex_echo_state_init(FRAME, FILTER)
    if not state:
        raise RuntimeError("SpeexDSP could not create an echo state")
    try:
        requested = ctypes.c_int(RATE)
        if dsp.speex_echo_ctl(state, SET_SAMPLING_RATE, ctypes.byref(requested)) != 0:
            raise RuntimeError("SpeexDSP rejected 16 kHz sampling rate")
        actual = ctypes.c_int()
        if dsp.speex_echo_ctl(state, GET_SAMPLING_RATE, ctypes.byref(actual)) != 0:
            raise RuntimeError("SpeexDSP could not report sampling rate")
        if actual.value != RATE:
            raise RuntimeError(f"SpeexDSP used {actual.value} Hz, not {RATE} Hz")
        output = np.empty_like(microphone)
        for start in range(0, len(reference), FRAME):
            play = reference[start:start + FRAME]
            rec = microphone[start:start + FRAME]
            out = output[start:start + FRAME]
            dsp.speex_echo_cancellation(state, rec.ctypes.data, play.ctypes.data,
                                         out.ctypes.data)
    finally:
        dsp.speex_echo_state_destroy(state)
    return output, actual.value


def power_ratio_db(before: np.ndarray, after: np.ndarray) -> float:
    input_power = float(np.mean((before.astype(float) / 32768.0) ** 2))
    output_power = float(np.mean((after.astype(float) / 32768.0) ** 2))
    if input_power <= 0 or output_power <= 0:
        raise ValueError("nonzero input and output powers required")
    return float(10.0 * np.log10(input_power / output_power))


def run(library: Path, directory: Path = PAIR_DIR) -> tuple[dict, np.ndarray]:
    reference, microphone, files = load_pinned_pair(directory)
    common = min(len(reference), len(microphone))
    usable = common - common % FRAME
    reference, microphone = reference[:usable], microphone[:usable]
    lag, corr = peak_lag(reference, microphone, start=RATE, stop=4 * RATE,
                         max_lag=2000)
    output, actual_rate = speex_linear_aec(library, reference, microphone)
    zero_reference = np.zeros_like(reference)
    zero_output, zero_actual_rate = speex_linear_aec(library, zero_reference, microphone)
    late_reference = np.zeros_like(reference)
    late_reference[RATE:] = reference[:-RATE]
    late_output, late_actual_rate = speex_linear_aec(library, late_reference, microphone)
    if zero_actual_rate != actual_rate or late_actual_rate != actual_rate:
        raise RuntimeError("SpeexDSP sampling-rate readback changed between controls")
    # The first 3 s are a declared convergence interval, not chosen by score.
    score_start, score_stop = 3 * RATE, usable
    windows = []
    for start in range(score_start, score_stop - RATE + 1, RATE):
        windows.append({"sample_interval_half_open": [start, start + RATE],
                        "input_output_power_change_db": power_ratio_db(
                            microphone[start:start + RATE], output[start:start + RATE])})
    result = {
        "dataset": "Microsoft AEC Challenge real/farend_singletalk",
        "source_commit": SOURCE_COMMIT, "files": files,
        "license_note": "Real crowd-recording redistribution terms not established; WAVs stay in ignored cache.",
        "algorithm": "SpeexDSP synchronous speex_echo_cancellation; core AUMDF output only",
        "speex_source_commit": "8e29a256ef0235ebbe7fcb8417b5ac7731eb8307",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "pcm_format": "mono 16-bit 16000 Hz; no resampling, gain scaling or time shift",
        "frame_samples": FRAME, "filter_samples": FILTER, "filter_duration_ms": 1000 * FILTER / RATE,
        "speex_rate_readback_hz": actual_rate,
        "common_samples": usable, "discarded_tail_samples": {
            "lpb": files["lpb"]["samples"] - usable,
            "mic": files["mic"]["samples"] - usable},
        "diagnostic_lag": {"training_interval_half_open": [RATE, 4 * RATE],
                           "search_lag_samples": [-2000, 2000],
                           "peak_lag_samples_mic_after_lpb": lag,
                           "signed_normalized_correlation": corr,
                           "used_to_shift_input": False},
        "convergence_interval_half_open": [0, score_start],
        "score_interval_half_open": [score_start, score_stop],
        "input_output_power_change_db": power_ratio_db(
            microphone[score_start:score_stop], output[score_start:score_stop]),
        "zero_reference_input_output_power_change_db": power_ratio_db(
            microphone[score_start:score_stop], zero_output[score_start:score_stop]),
        "late_reference_input_output_power_change_db": power_ratio_db(
            microphone[score_start:score_stop], late_output[score_start:score_stop]),
        "late_reference_shift_samples": RATE,
        "paired_output_relative_to_zero_reference_db": power_ratio_db(
            zero_output[score_start:score_stop], output[score_start:score_stop]),
        "score_input_rms_digital": float(np.sqrt(np.mean(
            (microphone[score_start:score_stop].astype(float) / 32768.0) ** 2))),
        "score_output_rms_digital": float(np.sqrt(np.mean(
            (output[score_start:score_stop].astype(float) / 32768.0) ** 2))),
        "output_pcm_sha256": hashlib.sha256(output.astype("<i2").tobytes()).hexdigest(),
        "zero_reference_output_pcm_sha256": hashlib.sha256(
            zero_output.astype("<i2").tobytes()).hexdigest(),
        "late_reference_output_pcm_sha256": hashlib.sha256(
            late_output.astype("<i2").tobytes()).hexdigest(),
        "whole_second_windows": windows,
        "metric_warning": "Input/output power change on one official far-end-single-talk clip. Zero-reference processing also changes power; a 1 s late reference is a deliberately invalid pairing. None is clean-component ERLE or near-end preservation.",
    }
    return result, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speex-library", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    parser.add_argument("--output-wav", type=Path,
                        help="Optional ignored local PCM16 output for listening; never commit it")
    args = parser.parse_args()
    result, output = run(args.speex_library, args.pair_dir)
    if args.output_wav is not None:
        destination = args.output_wav.resolve()
        if not destination.is_relative_to((ROOT / "codes/upstream/_downloads").resolve()):
            raise ValueError("output WAV must stay inside the Git-ignored upstream cache")
        if destination.exists():
            raise FileExistsError("refusing to overwrite an existing output WAV")
        with wave.open(str(destination), "wb") as wav:
            wav.setparams((1, 2, RATE, len(output), "NONE", "not compressed"))
            wav.writeframes(output.astype("<i2").tobytes())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
