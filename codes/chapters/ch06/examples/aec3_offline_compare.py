"""Run an already-built, pinned WebRTC audioproc_f on the two AEC Challenge pairs.

This is an adapter, not an AEC3 reimplementation.  It rejects missing binaries,
stages only pinned PCM in an external empty directory, and keeps linear/final taps separate.
"""

from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np

from codes.chapters.ch06.aec_doubletalk_experiment import load_doubletalk
from codes.chapters.ch06.examples.aec_real_pair_experiment import (
    FRAME, PAIR_DIR, RATE, ROOT, load_pinned_pair, power_ratio_db,
    contracts, output_destination, write_output_wav,
)


def command(binary: Path, mic: Path, lpb: Path, final: Path, linear: Path,
            call_order: Path) -> list[str]:
    """Pin the APM knobs; `rc` is needed because WAV-simulator default is `cr`."""
    return [
        str(binary), f"--i={mic}", f"--ri={lpb}", f"--o={final}",
        f"--linear_aec_output={linear}", "--fixed_interface=true",
        "--aec=1", "--agc=0", "--agc2=0", "--ns=0", "--hpf=0", "--ts=0",
        "--stream_delay=0", f"--custom_call_order_file={call_order}",
    ]


def write_wav(path: Path, samples: np.ndarray) -> None:
    write_output_wav(path, samples)


def preflight_members(directory, names, *, protected=()):
    """Check all producer outputs before any binary execution or file write."""
    directory = contracts.work_target(directory, (contracts.CACHE, *protected))
    for name in names:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError("Expected ordinary output leaf names")
        output_destination(directory / name, protected=protected)
    return directory


def read_wav(path: Path, expected: int) -> np.ndarray:
    path = contracts.ordinary_file(path)
    with wave.open(str(path), "rb") as wav:
        fmt = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
        if fmt != (1, 2, RATE, "NONE"):
            raise ValueError(f"unexpected AEC3 output format {fmt} for {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    if len(samples) != expected:
        raise ValueError(f"AEC3 output length {len(samples)} != input {expected} for {path}")
    return samples


def run(binary: Path, outdir: Path, pair_dir: Path = PAIR_DIR) -> dict:
    binary = contracts.validate_parent_chain(binary)
    if not binary.exists():
        raise FileNotFoundError(binary)
    binary = contracts.ordinary_file(binary)
    if not binary.stat().st_mode & 0o111:
        raise ValueError("audioproc_f must be an executable ordinary file")
    names = ["call_order_rc.txt", *[f"{name}_{suffix}.wav"
             for name in ("farend_singletalk", "doubletalk")
             for suffix in ("mic_input", "lpb_input", "aec3_final", "aec3_linear")]]
    outdir = preflight_members(outdir, names, protected=(binary, pair_dir))
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("AEC3 output directory must be empty before execution")
    # Input validation precedes writes. Both cases are ready before the first call.
    inputs = {"farend_singletalk": load_pinned_pair(pair_dir),
              "doubletalk": load_doubletalk(pair_dir)}
    for name, (_, microphone, _) in inputs.items():
        if len(microphone) < (3 * RATE if name == "farend_singletalk" else 8 * RATE):
            raise ValueError(f"{name}: score interval extends beyond input")
    outdir.mkdir(parents=True, exist_ok=True)
    order = outdir / "call_order_rc.txt"
    with order.open("x") as stream:
        stream.write("rc\n")

    results: dict = {
        "webrtc_source_commit_expected": "0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "actual_dependency_sha256": contracts.dependencies(Path(__file__), (
            Path(__file__).with_name('aec_real_pair_experiment.py'),
            ROOT / 'codes/chapters/ch06/aec_doubletalk_experiment.py')),
        "binary_provenance_warning": "A hash does not prove the binary was built from the expected commit; record build log and GN arguments separately.",
        "call_order": "render then capture within each 10 ms frame (rc)",
        "stream_delay_ms": 0,
        "metric_warning": "Input/output total power ratios are descriptive; double-talk has no clean near-end or echo truth. Final AEC3 output includes further processing; compare linear and final taps separately.",
        "pairs": {},
    }
    for name, loader, start, stop_s in (
        ("farend_singletalk", load_pinned_pair, 3 * RATE, None),
        ("doubletalk", load_doubletalk, 3 * RATE, 8 * RATE),
    ):
        reference, microphone, files = inputs[name]
        usable = min(len(reference), len(microphone)) // FRAME * FRAME
        reference, microphone = reference[:usable], microphone[:usable]
        stop = usable if stop_s is None else stop_s
        if stop > usable:
            raise ValueError(f"{name}: score interval extends beyond input")
        mic_path = outdir / f"{name}_mic_input.wav"
        lpb_path = outdir / f"{name}_lpb_input.wav"
        write_wav(mic_path, microphone)
        write_wav(lpb_path, reference)
        final_path = outdir / f"{name}_aec3_final.wav"
        linear_path = outdir / f"{name}_aec3_linear.wav"
        # Recheck outputs immediately before the external producer writes them.
        preflight_members(outdir, (final_path.name, linear_path.name), protected=(binary, pair_dir))
        args = command(binary, mic_path, lpb_path, final_path, linear_path, order)
        subprocess.run(args, check=True, cwd=outdir)
        score_slice = slice(start, stop)
        taps = {}
        for label, path in (("linear", linear_path), ("final", final_path)):
            output = read_wav(path, usable)
            taps[label] = {
                "input_output_power_change_db": power_ratio_db(
                    microphone[score_slice], output[score_slice]),
                "output_wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        results["pairs"][name] = {
            "files": files,
            "score_interval_half_open": [start, stop],
            "common_samples": usable,
            "call": args,
            "taps": taps,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audioproc", type=Path, required=True,
                        help="Official audioproc_f executable built from the pinned WebRTC commit")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty ordinary directory outside the repository, source cache and input files")
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.audioproc, args.output_dir, args.pair_dir),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
