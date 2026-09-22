"""Prepare R01 from a fixed licensed archive, or check published files offline.

No arguments and --check are read-only and never access the network.
--prepare requires the fixed local archive. Only --download permits networking.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import wave
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codes.array_tutorial.real_recordings import (  # noqa: E402
    CHANNELS, FILENAMES, SAMPLE_RATE, SAMPLES, experiment, read_pcm_wav,
)

URL = "https://zenodo.org/records/1227121/files/NRIVER_16k.zip?download=1"
ARCHIVE_BYTES = 98689565
ARCHIVE_MD5 = "54264db61d3fe073fb81f2e40e0d19b5"
ARCHIVE_SHA256 = "98e3a3f05a7234ea07619d965bafeea68a100bbce5c9269b7487393e3b559660"
EXCERPT_PCM_SHA256 = "cff2df04123f7eb01a5ee724bf594f4f42b68388e4b49c2d58fa3a0ddd91f12c"
DEFAULT_ARCHIVE = ROOT / "codes/upstream/_downloads/demand/NRIVER_16k.zip"
DEFAULT_OUTPUT = ROOT / "codes/real_audio"
INPUTS = ("codes/array_tutorial/real_recordings.py", "codes/examples/prepare_real_recordings.py")


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def verify_archive(path: Path) -> None:
    """Refuse a different existing archive; never overwrite it to repair a check."""
    if not path.is_file() or path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("fixed archive missing or byte length differs")
    digest, legacy = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            legacy.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256 or legacy.hexdigest() != ARCHIVE_MD5:
        raise ValueError("archive checksum differs; existing file was not changed")


def download_archive(path: Path) -> None:
    """Download only the fixed 99 MB archive, verify both hashes, then publish."""
    if path.exists():
        verify_archive(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".demand-download-", dir=path.parent)
    tmp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as target, urllib.request.urlopen(URL, timeout=60) as source:
            count = 0
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > ARCHIVE_BYTES:
                    raise ValueError("download exceeds fixed archive size")
                target.write(chunk)
        verify_archive(tmp)
        # Exclusive link publication prevents overwriting a concurrently created cache.
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_excerpt(path: Path) -> np.ndarray:
    verify_archive(path)
    channels = []
    with zipfile.ZipFile(path) as archive:
        for number in range(1, CHANNELS + 1):
            member = f"NRIVER/ch{number:02d}.wav"
            with archive.open(member) as source, wave.open(source, "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(),
                    wav.getcomptype()) != (1, 2, SAMPLE_RATE, 4800064, "NONE"):
                    raise ValueError(f"unexpected source WAV format: {member}")
                blob = wav.readframes(SAMPLES)
            if len(blob) != SAMPLES * 2:
                raise ValueError(f"truncated source excerpt: {member}")
            channels.append(np.frombuffer(blob, dtype="<i2"))
    result = np.stack(channels, axis=1)
    verify_excerpt(result)
    return result


def verify_excerpt(pcm: np.ndarray) -> None:
    if pcm.shape != (SAMPLES, CHANNELS) or sha256(pcm.astype("<i2").tobytes()) != EXCERPT_PCM_SHA256:
        raise ValueError("excerpt differs from the pinned original PCM samples")


def artifacts(pcm: np.ndarray) -> tuple[dict[str, bytes], dict]:
    verify_excerpt(pcm)
    files, metrics = experiment(pcm)
    records = []
    for filename, blob in files.items():
        decoded, rate = read_pcm_wav(blob)
        normalized = decoded.astype(float) / 32768.0
        records.append({"file": filename, "sha256": sha256(blob),
                        "sample_rate_hz": rate, "channels": decoded.shape[1],
                        "samples": decoded.shape[0], "duration_s": decoded.shape[0] / rate,
                        "peak": float(np.max(np.abs(normalized))),
                        "rms": np.sqrt(np.mean(normalized**2, axis=0)).tolist(),
                        "common_export_gain": 1.0})
    row_step = float(np.sqrt(.05**2 - .025**2))
    geometry = [[row * row_step, col * .05 + (.025 if row % 2 else 0), 0.0]
                for row in range(4) for col in range(4)]
    manifest = {
        "schema_version": 1,
        "origin": "real simultaneously acquired 16-microphone environmental recording; not synthesized speech",
        "source": {"dataset": "DEMAND", "version": "1.0", "scene": "NRIVER",
                   "record_url": "https://zenodo.org/records/1227121", "archive_url": URL,
                   "archive_bytes": ARCHIVE_BYTES, "archive_md5": ARCHIVE_MD5,
                   "archive_sha256": ARCHIVE_SHA256, "checked_on": "2026-09-22",
                   "members": [f"NRIVER/ch{i:02d}.wav" for i in range(1, 17)],
                   "source_frames": 4800064, "sample_interval_half_open": [0, SAMPLES],
                   "excerpt_interleaved_pcm_sha256": EXCERPT_PCM_SHA256,
                   "original_rate_hz": 48000, "upstream_resampled_rate_hz": SAMPLE_RATE,
                   "upstream_resampler": "MATLAB R2012a resample; DEMAND.pdf section 3.2"},
        "license": {"id": "CC-BY-SA-3.0", "url": "https://creativecommons.org/licenses/by-sa/3.0/",
                    "authors": ["Joachim Thiemann", "Nobutaka Ito", "Emmanuel Vincent"],
                    "scope": "source recording, excerpt, channel extraction and averaged audio; not repository code"},
        "geometry": {"unit": "m", "channel_order": list(range(1, 17)), "xyz": geometry,
                     "coordinate_source": "official scripts.zip arraypos.m; DEMAND.pdf section 3.1",
                     "scripts_zip_sha256": "3415d1c54620b5ba30004943e3515f63db76930640f27aa2c30a21a718677dad",
                     "position_tolerance_m": .002, "plane": "parallel to ground",
                     "microphones": "16 Sony ECM-C10; gains not calibrated relative to one another",
                     "recorder": "one Inrevium/Tokyo Electron Device TD-BD-16ADUSB",
                     "source_position_ground_truth": None},
        "processing": {"common_export_gain": 1.0, "dc_removed": False, "time_shifts_samples": [0] * 16,
                       "per_file_normalization": False, "channel_calibration": False,
                       "weights": {"mean02": [.5, .5], "mean16": [1 / 16] * 16},
                       "quantization": "PCM16 little endian; nearest-even; no dither; no clipping",
                       "clean_speech_reference": None, "snr_or_si_sdr_claim": False,
                       "listening": "No formal listening evaluation; lower device volume before playback."},
        "generator_inputs": {p: sha256((ROOT / p).read_bytes()) for p in INPUTS},
        "metrics": metrics,
        "files": records,
    }
    return files, manifest


def check(output: Path, archive: Path | None = None) -> dict:
    """Read-only, offline verification including regenerated bytes and metadata."""
    pcm, rate = read_pcm_wav((output / FILENAMES[0]).read_bytes())
    if rate != SAMPLE_RATE:
        raise ValueError("wrong excerpt sample rate")
    if archive is not None and archive.exists() and not np.array_equal(pcm, load_excerpt(archive)):
        raise ValueError("published input differs from archive")
    files, expected = artifacts(pcm)
    if json.loads((output / "MANIFEST.json").read_text()) != expected:
        raise ValueError("manifest missing or stale; run --prepare explicitly")
    if {p.name for p in output.glob("*.wav")} != set(files):
        raise ValueError("unexpected WAV file set")
    for name, blob in files.items():
        if (output / name).read_bytes() != blob:
            raise ValueError(f"audio differs: {name}")
    return {"files": len(files), "bytes": sum(map(len, files.values())), "checked": True,
            "archive_checked": archive is not None and archive.exists()}


def prepare(output: Path, archive: Path) -> dict:
    files, manifest = artifacts(load_excerpt(archive))
    output.mkdir(parents=True, exist_ok=True)
    # Stage and verify every artifact first. Manifest is published last; an
    # interrupted replacement cannot pass --check until --prepare is rerun.
    with tempfile.TemporaryDirectory(prefix=".real-audio-", dir=output.parent) as directory:
        stage = Path(directory)
        for name, blob in files.items():
            (stage / name).write_bytes(blob)
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + "\n")
        check(stage)
        for name in (*files, "MANIFEST.json"):
            os.replace(stage / name, output / name)
    return check(output, archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--download", action="store_true", help="download fixed archive and prepare")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    result = prepare(args.output, args.archive) if args.prepare or args.download else check(args.output, args.archive)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
