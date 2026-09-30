"""Generate/check the independent E02-16 finite-window convolution WAVs.

Run ``python -m codes.chapters.ch02.examples.generate_stft_convolution``.
``--check`` reconstructs in memory and checks the exact published set without
writing or repairing anything. Byte comparisons require the recorded runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import wave

import numpy as np

from codes.chapters.ch00.core.audio_samples import pcm16_bytes, read_pcm16
from codes.chapters.ch02.core.stft_convolution import (
    BURST_INTERVALS, CASE_NAMES, DELAY_SAMPLES, FREQUENCIES, HOP_LENGTH, N_FFT,
    OUTPUT_SAMPLES, SAMPLE_RATE, SOURCE_SAMPLES, build_convolution_audio, measure_waveform,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIRECTORY = ROOT / "codes/chapters/ch02/stft_audio"
SOURCE_PATHS = ("codes/chapters/ch02/core/stft_convolution.py",
                "codes/chapters/ch02/examples/generate_stft_convolution.py",
                "codes/chapters/ch02/core/spectral.py",
                "codes/chapters/ch02/core/conventions.py",
                "codes/chapters/ch00/core/audio_samples.py")


def wav_metadata(payload: bytes) -> dict:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        return {"sample_rate_hz": wav.getframerate(), "channels": wav.getnchannels(),
                "samples_per_channel": wav.getnframes(), "sample_width_bytes": wav.getsampwidth(),
                "compression": wav.getcomptype()}


def expected_assets() -> tuple[dict[str, bytes], dict]:
    """Reconstruct in memory, use the sole shared PCM encoder, then decode."""
    case = build_convolution_audio()
    signals = case["signals"]
    payloads = {name+".wav": pcm16_bytes(signals[name]) for name in CASE_NAMES}
    decoded = {name: read_pcm16(payloads[name+".wav"])[1][0] for name in CASE_NAMES}
    denominator = case["synthesis_denominator"]
    if np.min(denominator) <= 1e-12:
        raise ValueError("Retained samples lack WOLA support")
    manifest = {
        "schema_version": 1, "exercise_id": "E02-16", "sample_rate_hz": SAMPLE_RATE,
        "samples_per_channel": OUTPUT_SAMPLES, "channels": 1, "common_export_gain": 1.,
        "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                          for path in SOURCE_PATHS},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "byteorder": sys.byteorder, "platform": platform.platform()},
        "parameters": {
            "source_samples": SOURCE_SAMPLES, "output_samples": OUTPUT_SAMPLES,
            "source_zero_padding_samples": DELAY_SAMPLES,
            "source_hash_samples": OUTPUT_SAMPLES,
            "source_duration_s": SOURCE_SAMPLES/SAMPLE_RATE,
            "output_duration_s": OUTPUT_SAMPLES/SAMPLE_RATE,
            "source": "sum of four 0.06*sin(2*pi*f*n/fs) tones, multiplied by intermittent linear-faded envelopes",
            "frequencies_hz": list(FREQUENCIES), "amplitudes": [.06]*4,
            "burst_intervals_samples": [list(pair) for pair in BURST_INTERVALS],
            "edge_fade_samples": 320,
            "fade": "each half-open burst [a,b): min(1,(n-a)/320,(b-1-n)/320)",
            "seed": None, "rir_nonzero_indices": [0, DELAY_SAMPLES], "rir_values": [1., .5],
            "reflection_delay_s": DELAY_SAMPLES/SAMPLE_RATE,
            "n_fft": N_FFT, "window": "DFT-periodic Hann", "hop_length": HOP_LENGTH,
            "center": True, "endpoint_padding": "256 zeros at each side; complete final frame",
            "convolution": "full time-domain linear convolution, including every nonzero tail sample",
            "framewise_mtf": "rfft(h,512)*STFT(padded source), then unchanged real WOLA",
            "normalization": "NumPy backward FFT; a single fixed export gain 1 for every file",
            "initial_history": "zero; no fitted delay or gain compensation",
            "source_float64_le_sha256": hashlib.sha256(np.asarray(case["source"], dtype="<f8").tobytes()).hexdigest(),
            "rir_float64_le_sha256": hashlib.sha256(np.asarray(case["rir"], dtype="<f8").tobytes()).hexdigest(),
        },
        "synthesis_support": {
            "retained_interval_samples": [0, OUTPUT_SAMPLES], "denominator_samples": OUTPUT_SAMPLES,
            "analysis_frames": len(case["frame_starts"]),
            "frame_starts_first_last": [case["frame_starts"][0], case["frame_starts"][-1]],
            "window_squared_sum_min": float(np.min(denominator)),
            "window_squared_sum_max": float(np.max(denominator)),
            "window_squared_sum_first_last_8": [denominator[:8].tolist(), denominator[-8:].tolist()],
            "window_squared_sum_float64_le_sha256": hashlib.sha256(np.asarray(denominator, dtype="<f8").tobytes()).hexdigest(),
        },
        "quantization": {"format": "signed little-endian PCM16", "scale": 32768,
                         "rounding": "nearest, ties to even", "sample_width_bytes": 2,
                         "max_rounding_error_bound": .5/32768, "clipping": "rejected"},
        "comparison": "Uncentered full-support digital energy and error, N=32320; no gain/delay fitting. PCM scores compare actual PCM references; float scores compare float references.",
        "limits": "Mathematical tones and a sparse filter, not speech, a measured RIR, T60, noise-removal or human listening scores. Framewise multiplication approximates filtering; its errors do not demonstrate failure of unmodified STFT inversion. The 3-file continuous example is distinct from the four-point FFT8 control.",
        "listening": "Start with low playback volume; never autoplay or normalize individual files.",
        "files": {}, "samples": {},
    }
    for name in CASE_NAMES:
        filename = name+".wav"
        error = float(np.max(np.abs(decoded[name]-signals[name])))
        if error > .5/32768 + 1e-15:
            raise ValueError("PCM quantization exceeds the rounding bound")
        manifest["files"][filename] = {
            **wav_metadata(payloads[filename]), "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
            "peak": float(np.max(np.abs(decoded[name]))), "quantization_max_abs_error": error,
        }
        measurements = {}
        for domain, arrays in (("float", signals), ("pcm", decoded)):
            measurements[domain+"_measurements"] = {
                "against_unmodified_source": measure_waveform(arrays[name], arrays["stft_roundtrip"]),
                "against_full_convolution": measure_waveform(arrays[name], arrays["full_convolution"]),
            }
        manifest["samples"][name] = {"file": filename, "common_export_gain": 1., **measurements}
    # Keep the genuine roundtrip error independent of the roundtrip's self-reference.
    manifest["unmodified_roundtrip_against_input"] = measure_waveform(signals["stft_roundtrip"], case["source"])
    quantized_input = read_pcm16(pcm16_bytes(case["source"]))[1][0]
    manifest["unmodified_roundtrip_pcm_against_quantized_input"] = measure_waveform(decoded["stft_roundtrip"], quantized_input)
    return payloads, manifest


def check_assets(directory: Path = DEFAULT_DIRECTORY) -> dict:
    """Strictly read-only: extra/missing/stale files or metadata cause failure."""
    payloads, expected = expected_assets()
    directory = Path(directory)
    required = set(payloads) | {"MANIFEST.json"}
    actual = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if actual != required:
        raise ValueError(f"Asset set differs: missing={sorted(required-actual)}, extra={sorted(actual-required)}")
    published = json.loads((directory/"MANIFEST.json").read_text(encoding="utf-8"))
    if published != expected:
        raise ValueError("Manifest differs from current sources, runtime, parameters or measurements")
    actual_signals = {}
    for name in CASE_NAMES:
        filename = name+".wav"
        payload = (directory/filename).read_bytes()
        if payload != payloads[filename]:
            raise ValueError(f"Stale or modified WAV: {filename}")
        metadata = wav_metadata(payload)
        recorded = published["files"][filename]
        if any(metadata[key] != recorded[key] for key in metadata):
            raise ValueError(f"PCM metadata mismatch: {filename}")
        actual_signals[name] = read_pcm16(payload)[1][0]
    for name in CASE_NAMES:
        scores = {"against_unmodified_source": measure_waveform(actual_signals[name], actual_signals["stft_roundtrip"]),
                  "against_full_convolution": measure_waveform(actual_signals[name], actual_signals["full_convolution"])}
        if scores != published["samples"][name]["pcm_measurements"]:
            raise ValueError(f"Actual PCM measurement mismatch: {name}")
    return published


def generate_assets(directory: Path = DEFAULT_DIRECTORY) -> dict:
    payloads, manifest = expected_assets()
    directory = Path(directory)
    required = set(payloads) | {"MANIFEST.json"}
    if directory.exists() and ({path.name for path in directory.iterdir()} - required):
        raise ValueError("Refusing to overwrite a directory containing unrelated assets")
    directory.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        (directory/filename).write_bytes(payload)
    (directory/"MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                    allow_nan=False)+"\n", encoding="utf-8")
    return check_assets(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="strictly read-only verification")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    try:
        manifest = check_assets(args.output) if args.check else generate_assets(args.output)
    except (OSError, ValueError, wave.Error) as error:
        parser.exit(1, f"STFT convolution asset verification failed: {error}\n")
    print(f"{'Checked' if args.check else 'Generated and checked'} {len(manifest['files'])} STFT convolution WAVs")


if __name__ == "__main__":
    main()
