"""Reproduce a small anechoic two-speaker activity-guided cACGMM/MVDR chain.

The simulator knows separate synthetic sources; the estimator receives only
two-channel mixture STFT and externally provided activity. WPE is bypassed
because this particular fixture has no reverberation. This is not GPU-GSS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from codes.array_tutorial.audio_samples import delay_samples, pcm16_bytes
from codes.array_tutorial.gss_teaching import guided_cacgmm_mvdr
from codes.array_tutorial.separation import si_sdr
from codes.array_tutorial.spectral import istft, stft


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "codes" / "gss_audio"


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
        "scope": "original NumPy two-microphone cACGMM EM, activity gating, target/non-target SCM and mask-MVDR on synthetic anechoic noise-like sources; not official GPU-GSS or speech",
        "sample_rate_hz": 16000,
        "seed": 20260924,
        "stft": {"n_fft": n_fft, "hop": hop, "center": True},
        "sources": "two independent seeded Gaussian sequences with 30 ms onset/offset; first active [0.1,1.5) s, second [0.5,1.9) s",
        "array": "two microphones; source 1 direct at mic 0 and delayed two samples at mic 1; source 2 reversed; independent microphone noise RMS 0.003",
        "wpe": "bypassed because the fixture contains no reverberation; code can optionally call the independent offline WPE teaching implementation",
        "cacgmm_iterations": 8,
        "shape_loading": 0.02,
        "correct_shape_resets": fitted["shape_reset_count"],
        "low_energy_bins": fitted["low_energy_bins"],
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
    return data, arrays


def generate(out_dir: Path = OUT) -> dict:
    result, arrays = run_experiment()
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_names = ["source_1", "source_2", "mixture", "enhanced_correct", "enhanced_missed"]
    peak = max(float(np.max(np.abs(arrays[name]))) for name in wav_names)
    gain = 0.7 / peak
    result["common_export_gain"] = gain
    files = {}
    for name in wav_names:
        path = out_dir / f"{name}.wav"
        content = pcm16_bytes(arrays[name] * gain)
        path.write_bytes(content)
        files[path.name] = {"channels": int(arrays[name].shape[0]),
                            "samples_per_channel": int(arrays[name].shape[1]),
                            "sha256": hashlib.sha256(content).hexdigest()}
    state_path = out_dir / "STATE.npz"
    np.savez_compressed(state_path, **{name: arrays[name] for name in
                                        ("posterior", "target_scm", "other_scm", "weights", "activity", "mixture_stft")})
    files[state_path.name] = {"sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
    result["files"] = files
    (out_dir / "MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
