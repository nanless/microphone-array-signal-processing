"""Reproduce a small anechoic two-speaker activity-guided cACGMM/MVDR chain.

The simulator knows separate synthetic sources; the estimator receives only
two-channel mixture STFT and externally provided activity. WPE is bypassed
because this particular fixture has no reverberation. This is not GPU-GSS.
"""

from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[4]))

import hashlib
import argparse
import io
import json
import platform
from pathlib import Path

import numpy as np

from codes.chapters.ch00.core.audio_samples import delay_samples, pcm16_bytes, read_pcm16
from codes.chapters.ch08.core.gss_teaching import guided_cacgmm_mvdr
from codes.chapters.ch08.core.separation import si_sdr
from codes.chapters.ch02.core.spectral import istft, stft


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "codes/chapters/ch08/gss_audio"
ASSET_NAMES = {"source_1.wav", "source_2.wav", "mixture.wav", "enhanced_correct.wav",
               "enhanced_missed.wav", "STATE.npz", "MANIFEST.json"}


def _check_output_members(out_dir: Path, *, check: bool) -> None:
    """Preflight before model execution or writes; never follow asset links."""
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        raise ValueError("GSS output must be an ordinary directory")
    if not out_dir.exists():
        if check:
            raise ValueError("GSS output directory is missing")
        return
    members = list(out_dir.iterdir())
    if any(member.is_symlink() or not member.is_file() for member in members):
        raise ValueError("GSS members must be ordinary files without symlinks")
    names = {member.name for member in members}
    if names != ASSET_NAMES and (check or names):
        raise ValueError("GSS asset set must contain exactly five WAVs, STATE.npz and MANIFEST.json")


def fixture(seed: int = 20260924) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fs, n = 16000, 32000
    times = np.arange(n) / fs
    rng = np.random.default_rng(seed)

    def envelope(start: float, stop: float) -> np.ndarray:
        result = np.zeros(n)
        active = (times >= start) & (times < stop)
        segment = times[active]
        result[active] = np.maximum(0, np.minimum(1, (segment-start)/.03) *
                                    np.minimum(1, (stop-segment)/.03))
        return result

    first = .2 * rng.standard_normal(n) * envelope(.1, 1.5)
    second = .2 * rng.standard_normal(n) * envelope(.5, 1.9)
    noise = .003 * rng.standard_normal((2, n))
    mixture = np.vstack((first + delay_samples(second, 2),
                         delay_samples(first, 2) + second)) + noise
    return first, second, mixture, times


def run_experiment() -> tuple[dict, dict[str, np.ndarray]]:
    first, second, mixture, times = fixture()
    n_fft, hop = 256, 128
    spectrum = stft(mixture, n_fft=n_fft, hop_length=hop).transpose(1, 0, 2)
    centers = np.arange(spectrum.shape[2]) * hop / 16000
    activity = np.column_stack(((centers >= .1) & (centers < 1.5),
                                (centers >= .5) & (centers < 1.9))).astype(np.int8)
    fitted = guided_cacgmm_mvdr(spectrum, activity, iterations=8, target_speaker=0)
    enhanced = istft(fitted["output"][None], n_fft=n_fft, hop_length=hop,
                     length=times.size)[0]
    missed = activity.copy()
    missed[:, 0] = 0  # controlled complete loss of target activity
    wrong = guided_cacgmm_mvdr(spectrum, missed, iterations=8, target_speaker=0)
    wrong_output = istft(wrong["output"][None], n_fft=n_fft, hop_length=hop,
                         length=times.size)[0]
    score = (times >= .2) & (times < 1.45)
    data = {
        "scope": "original NumPy activity-guided cACG fixed-count updates, target/non-target SCM and mask-MVDR on synthetic anechoic noise-like sources; not exact conditional-likelihood EM, official GPU-GSS or speech",
        "sample_rate_hz": 16000,
        "seed": 20260924,
        "stft": {"n_fft": n_fft, "hop": hop, "center": True, "window": "periodic Hann", "padding": "128 zeros at each end; complete frames", "synthesis": "weighted overlap-add, trim leading 128 and return 32000 samples"},
        "sources": "two independent seeded Gaussian sequences with 30 ms onset/offset; first active [0.1,1.5) s, second [0.5,1.9) s",
        "array": "two microphones; source 1 direct at mic 0 and delayed two samples at mic 1; source 2 reversed; independent microphone noise RMS 0.003",
        "wpe": "bypassed because the fixture contains no reverberation; code can optionally call the independent offline WPE teaching implementation",
        "cacgmm_iterations": 8,
        "shape_loading": 0.02,
        "correct_shape_resets": fitted["shape_reset_count"],
        "low_energy_bins": fitted["low_energy_bins"],
        "energy_gate": {"absolute_norm_floor": fitted["absolute_energy_floor"], "relative_to_bin_peak": fitted["relative_energy_floor"]},
        "state_phase": fitted["state_phase"],
        "stopping_rule": fitted["stopping_rule"],
        "beam_routes": fitted["beam_diagnostics"]["routes"],
        "missed_beam_routes": wrong["beam_diagnostics"]["routes"],
        "correct_mask_sum_max_error": float(np.max(np.abs(fitted["posterior"].sum(axis=1) - 1))),
        "inactive_target_max_posterior": float(np.max(fitted["posterior"][:, 0, activity[:, 0] == 0])),
        "silent_frame_background_min_posterior": float(np.min(fitted["posterior"][:, -1, activity.sum(axis=1) == 0])),
        "scored_interval_seconds": [0.2, 1.45],
        "si_sdr_db": {"reference_mic0": si_sdr(mixture[0, score], first[score]),
                       "correct_activity_output": si_sdr(enhanced[score], first[score])},
        "activity_error": {"missed_target_interval_seconds": [0.1, 1.5],
                           "correct_output_scored_si_sdr_db": si_sdr(enhanced[score], first[score]),
                           "missed_output_scored_si_sdr_db": si_sdr(wrong_output[score], first[score]),
                           "missed_target_max_posterior": float(np.max(wrong["posterior"][:, 0, missed[:, 0] == 0]))},
        "limits": "one deterministic room-free mixture, external near-oracle activity, reference microphone 0, no diarization, room, real meeting, ASR or WER; single-case SI-SDR is not a general performance claim",
    }
    arrays = {"source_1": first[None], "source_2": second[None],
              "mixture": mixture, "enhanced_correct": enhanced[None],
              "enhanced_missed": wrong_output[None],
              "posterior": fitted["posterior"], "target_scm": fitted["target_scm"],
              "other_scm": fitted["other_scm"], "weights": fitted["weights"],
              "activity": activity, "mixture_stft": spectrum}
    for key in ("e_step_shapes", "e_step_priors", "shape_matrices", "post_update_priors", "valid_points", "actual_iterations_per_frequency"):
        arrays[key] = fitted[key]
    return data, arrays


def generate(out_dir: Path = OUT, *, check: bool = False) -> dict:
    if type(check) is not bool:
        raise ValueError("check must be boolean")
    out_dir = Path(out_dir)
    _check_output_members(out_dir, check=check)
    result, arrays = run_experiment()
    wav_names = ["source_1", "source_2", "mixture", "enhanced_correct", "enhanced_missed"]
    peak = max(float(np.max(np.abs(arrays[name]))) for name in wav_names)
    gain = 0.7 / peak
    result["common_export_gain"] = gain
    result["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system(), "machine": platform.machine()}
    sources = ["codes/chapters/ch08/examples/gss_teaching_demo.py", "codes/chapters/ch08/core/gss_teaching.py", "codes/chapters/ch08/core/separation.py", "codes/chapters/ch02/core/spectral.py", "codes/chapters/ch02/core/conventions.py", "codes/chapters/ch00/core/audio_samples.py", "codes/chapters/ch07/core/dereverberation.py"]
    result["generator_inputs"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    result["pcm"] = "little-endian signed PCM16, nearest-even rounding, no dither; common gain for all files"
    result["score"] = "reference mic0 source1, [3200,23200) samples, centered SI-SDR; no time alignment, float and decoded PCM reported separately"
    files = {}
    contents = {}
    decoded = {}
    for name in wav_names:
        path = out_dir / f"{name}.wav"
        content = pcm16_bytes(arrays[name] * gain)
        contents[path.name] = content
        decoded[name] = read_pcm16(content)[1]
        files[path.name] = {"channels": int(arrays[name].shape[0]),
                            "samples_per_channel": int(arrays[name].shape[1]),
                            "sha256": hashlib.sha256(content).hexdigest()}
    result["pcm_si_sdr_db"] = {name: si_sdr(decoded[name][0, 3200:23200], decoded["source_1"][0, 3200:23200]) for name in ("mixture", "enhanced_correct", "enhanced_missed")}
    state = io.BytesIO()
    np.savez_compressed(state, **{name: value for name, value in arrays.items() if name not in wav_names})
    contents["STATE.npz"] = state.getvalue()
    files["STATE.npz"] = {"sha256": hashlib.sha256(contents["STATE.npz"]).hexdigest()}
    result["files"] = files
    contents["MANIFEST.json"] = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if check:
        for name, content in contents.items():
            if not (out_dir / name).is_file() or (out_dir / name).read_bytes() != content:
                raise ValueError(f"GSS asset missing or stale: {name}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            (out_dir / name).write_bytes(content)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.output, check=args.check), ensure_ascii=False, indent=2))
