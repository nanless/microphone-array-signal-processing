"""Generate or strictly read-check the three independent E03-17 WAV assets.

Run from the repository root with ``python -m
codes.chapters.ch03.examples.generate_geometry_audio [--check]``.
Checking reads actual PCM and recomputes measurements; it never repairs files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import wave
import io

import numpy as np

from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch03.core.geometry_audio import (
    FILE_NAMES, LIMITS, SAMPLE_RATE, SAMPLES, generate_signals, geometry_parameters, measure_signal,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1]/"geometry_audio"
SOURCE_PATHS = (
    "codes/chapters/ch03/core/geometry_audio.py",
    "codes/chapters/ch03/examples/generate_geometry_audio.py",
    "codes/chapters/ch03/core/geometry.py",
    "codes/chapters/ch02/core/conventions.py",
    "codes/chapters/ch00/core/audio_samples.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_assets() -> tuple[dict[str, bytes], dict]:
    """Prepare deterministic assets in memory; neither import nor call writes."""
    signals = generate_signals()
    blobs, files, samples = {}, {}, {}
    for name, x in signals.items():
        filename = FILE_NAMES[name]
        blob = pcm16_bytes(x, SAMPLE_RATE)
        rate, actual = read_pcm16(blob)
        blobs[filename] = blob
        files[filename] = {"sample_rate_hz": rate, "channels": len(actual),
                           "samples_per_channel": actual.shape[1], "sha256": _sha(blob)}
        samples[name] = {"file": filename, "float_measurements": measure_signal(x),
                         "pcm_measurements": measure_signal(actual),
                         "quantization_max_abs_error": float(np.max(abs(actual-x)))}
    manifest = {"schema_version": 1, "sample_rate_hz": SAMPLE_RATE,
                "samples_per_channel": SAMPLES, "common_export_gain": 1.0,
                "origin": "original deterministic mathematical plane-wave synthesis",
                "source_sha256": {path: _sha((ROOT/path).read_bytes()) for path in SOURCE_PATHS},
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "system": platform.system(), "machine": platform.machine()},
                "pcm": "signed little-endian PCM16, round-to-nearest-even, no dither or per-file gain",
                "parameters": geometry_parameters(), "files": files, "samples": samples,
                "limits": LIMITS, "listening": "Start at low volume; no automatic playback. Six-channel playback is device-dependent."}
    return blobs, manifest


def check_assets(output: Path) -> dict:
    """Read-check exact file set, current sources, actual WAVs and new scores."""
    output = Path(output)
    expected_names = set(FILE_NAMES.values()) | {"MANIFEST.json"}
    if not output.is_dir() or {p.name for p in output.iterdir()} != expected_names:
        raise ValueError("geometry_audio must contain exactly three WAVs and MANIFEST.json")
    manifest = json.loads((output/"MANIFEST.json").read_text())
    current_sources = {path: _sha((ROOT/path).read_bytes()) for path in SOURCE_PATHS}
    if manifest.get("source_sha256") != current_sources:
        raise ValueError("geometry_audio source set or SHA is stale")
    expected_blobs, expected_manifest = prepare_assets()
    if manifest != expected_manifest:
        raise ValueError("geometry_audio manifest parameters or numerical measurements are stale")
    for name, filename in FILE_NAMES.items():
        blob = (output/filename).read_bytes()
        info = manifest["files"][filename]
        if _sha(blob) != info["sha256"]:
            raise ValueError("geometry_audio WAV SHA mismatch: "+filename)
        with wave.open(io.BytesIO(blob), "rb") as reader:
            expected_channels = 1 if name == "reference" else 6
            if (reader.getframerate(), reader.getnchannels(), reader.getnframes(),
                reader.getsampwidth(), reader.getcomptype()) != (SAMPLE_RATE, expected_channels, SAMPLES, 2, "NONE"):
                raise ValueError("geometry_audio PCM format mismatch: "+filename)
        rate, pcm = read_pcm16(blob)
        actual_measurements = measure_signal(pcm)
        if actual_measurements != manifest["samples"][name]["pcm_measurements"]:
            raise ValueError("geometry_audio actual PCM measurements mismatch: "+filename)
        # Re-encoding is supplementary; it cannot replace the preceding readback.
        if blob != expected_blobs[filename]:
            raise ValueError("geometry_audio PCM no longer matches the continuous source: "+filename)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.check:
        manifest = check_assets(args.output_dir)
    else:
        blobs, manifest = prepare_assets()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, blob in blobs.items():
            (args.output_dir/filename).write_bytes(blob)
        (args.output_dir/"MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        check_assets(args.output_dir)
    print(json.dumps({"status": "checked" if args.check else "generated",
                      "directory": str(args.output_dir), "files": list(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
