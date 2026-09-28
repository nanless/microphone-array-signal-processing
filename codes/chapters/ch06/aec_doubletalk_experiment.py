"""Reproduce a real double-talk AEC control experiment; no clean near-end truth.

The Microsoft AEC Challenge WAVs stay in the ignored upstream cache.  The
reported power ratios are descriptive, not ERLE or near-end preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np

from codes.examples.aec_real_pair_experiment import (
    FRAME, PAIR_DIR, RATE, ROOT, SOURCE_COMMIT, power_ratio_db,
    speex_linear_aec,
)


PREFIX = "-2jLGNCgf0WDpKMY2iup7g_doubletalk_"
SHA256 = {
    "lpb": "790203e68cc8cc9b02becc4efefbde0ddf962fff4d223608b00ffc8b2682ed71",
    "mic": "4e7e35290d763d2d22173976c2cfd46739c9afe76729a0c283c1b69609f22041",
}
SCORE_START = 3 * RATE
SCORE_STOP = 8 * RATE


def load_doubletalk(directory: Path = PAIR_DIR) -> tuple[np.ndarray, np.ndarray, dict]:
    arrays = {}
    files = {}
    for kind in ("lpb", "mic"):
        path = directory / f"{PREFIX}{kind}.wav"
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != SHA256[kind]:
            raise ValueError(f"{kind} SHA-256 differs from pinned original (or is an LFS pointer)")
        with wave.open(io.BytesIO(blob), "rb") as wav:
            fmt = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
            if fmt != (1, 2, RATE, "NONE"):
                raise ValueError(f"{kind} must be mono PCM16 16 kHz: got {fmt}")
            count = wav.getnframes()
            pcm = np.frombuffer(wav.readframes(count), dtype="<i2").copy()
        if len(pcm) != count or count < SCORE_STOP:
            raise ValueError(f"{kind} is too short for the prespecified [3,8) s interval")
        if np.max(np.abs(pcm.astype(np.int32))) == 0:
            raise ValueError(f"{kind} is silent")
        arrays[kind] = pcm
        files[kind] = {
            "name": path.name, "sha256": digest, "samples": count,
            "duration_s": count / RATE,
            "full_scale_samples": int(np.count_nonzero(np.abs(pcm.astype(np.int32)) >= 32767)),
            "rms_digital": float(np.sqrt(np.mean((pcm.astype(float) / 32768) ** 2))),
        }
    return arrays["lpb"], arrays["mic"], files


def run(library: Path, directory: Path = PAIR_DIR) -> tuple[dict, np.ndarray]:
    reference, microphone, files = load_doubletalk(directory)
    usable = min(len(reference), len(microphone)) // FRAME * FRAME
    reference, microphone = reference[:usable], microphone[:usable]
    late_reference = np.zeros_like(reference)
    late_reference[RATE:] = reference[:-RATE]
    outputs = {
        "paired": speex_linear_aec(library, reference, microphone)[0],
        "zero_reference": speex_linear_aec(library, np.zeros_like(reference), microphone)[0],
        "late_reference_1s": speex_linear_aec(library, late_reference, microphone)[0],
    }
    sl = slice(SCORE_START, SCORE_STOP)
    result = {
        "dataset": "Microsoft AEC Challenge real/doubletalk",
        "source_commit": SOURCE_COMMIT,
        "files": files,
        "license_note": "Real crowd-recording redistribution terms not established; WAVs stay in ignored cache.",
        "algorithm": "SpeexDSP synchronous speex_echo_cancellation; core AUMDF output only",
        "speex_source_commit": "8e29a256ef0235ebbe7fcb8417b5ac7731eb8307",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "format": "mono PCM16 16 kHz; 160-sample frames; no input alignment or resampling",
        "usable_samples": usable,
        "discarded_tail_samples": {k: files[k]["samples"] - usable for k in files},
        "convergence_interval_half_open": [0, SCORE_START],
        "score_interval_half_open": [SCORE_START, SCORE_STOP],
        "input_rms_digital": float(np.sqrt(np.mean((microphone[sl].astype(float) / 32768) ** 2))),
        "controls": {
            name: {
                "input_output_power_change_db": power_ratio_db(microphone[sl], output[sl]),
                "output_rms_digital": float(np.sqrt(np.mean((output[sl].astype(float) / 32768) ** 2))),
                "output_pcm_sha256": hashlib.sha256(output.astype("<i2").tobytes()).hexdigest(),
            } for name, output in outputs.items()
        },
        "paired_output_relative_to_zero_reference_db": power_ratio_db(
            outputs["zero_reference"][sl], outputs["paired"][sl]),
        "metric_warning": "The microphone contains near-end speech, echo and noise with no component truth. Total input/output power change is NOT ERLE, echo leakage or near-end preservation. A different output proves only that the reference affects this implementation; listen or use a separately justified perceptual protocol for quality.",
    }
    return result, outputs["paired"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speex-library", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    parser.add_argument("--output-wav", type=Path, help="Optional output only inside ignored cache")
    args = parser.parse_args()
    result, output = run(args.speex_library, args.pair_dir)
    if args.output_wav:
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
