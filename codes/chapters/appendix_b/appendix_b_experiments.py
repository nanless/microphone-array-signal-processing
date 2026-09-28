"""Appendix B E13-03..10: fixed-room calculations and evidence boundaries.

This module does not execute pyroomacoustics or create assets. E13-03,
E13-07 and E13-09 read checked-in synthetic room results or PCM; the other
exercises use small analytic fixtures with independent answers.
"""

from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np

from codes.chapters.ch02.core.spectral import stft
from codes.chapters.ch04.core.doa import srp_phat


# codes/chapters/appendix_b/<module>.py is three directory levels below the repo.
ROOT = Path(__file__).resolve().parents[3]
ROOM = ROOT / "codes/chapters/appendix_b/room_audio"


def _read_results() -> dict:
    result = json.loads((ROOM / "RESULTS.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or len(result.get("results", [])) != 6:
        raise ValueError("six-position room results are missing or malformed")
    paths = ((ROOT / result["generator"]["path"], result["generator"]["sha256"]),
             (ROOM / result["assets"]["figure"]["file"],
              result["assets"]["figure"]["sha256"]),
             (ROOM / result["assets"]["audio_manifest"]["file"],
              result["assets"]["audio_manifest"]["sha256"]))
    for path, expected in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"room result provenance digest differs: {path.name}")
    return result


def paired_room_comparison(rows: list[dict]) -> dict:
    """Compare only the two fixed ±30° positions at each distance."""
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != 6:
        raise ValueError("six distinct room cases are required")
    pairs = {}
    for side in ("left", "right"):
        near = by_name[f"fixed_near_{side}"]
        far = by_name[f"fixed_far_{side}"]
        if (not math.isclose(near["true_azimuth_deg"], far["true_azimuth_deg"])
                or not near["distance_m"] < far["distance_m"]):
            raise ValueError("fixed pair must retain angle and increase distance")
        pairs[side] = {
            "near_distance_m": near["distance_m"],
            "far_distance_m": far["distance_m"],
            "report_unrounded_drr_far_minus_near_db":
                far["drr_db_median"] - near["drr_db_median"],
            "published_3dp_drr_far_minus_near_db":
                round(far["drr_db_median"], 3) - round(near["drr_db_median"], 3),
            "doa_error_far_minus_near_deg": (far["absolute_doa_error_deg"]
                                               - near["absolute_doa_error_deg"]),
        }
    seeded = [{"name": by_name[name]["name"],
               "distance_m": by_name[name]["distance_m"],
               "absolute_doa_error_deg": by_name[name]["absolute_doa_error_deg"]}
              for name in ("seeded_1", "seeded_2")]
    return {"fixed_pairs": pairs, "seeded_positions": seeded,
            "scope": "one synthetic room; distance also moves the source relative to walls"}


def four_mic_drr_and_convolution() -> dict:
    """Keep median-of-ratios separate from ratio-of-pooled-energies."""
    direct_energy = np.array([1.0, 4.0, 1.0, 4.0])
    reflected_energy = np.array([1.0, 1.0, 4.0, 1.0])
    per_mic_db = 10.0 * np.log10(direct_energy / reflected_energy)
    direct_rir = np.array([1.0, 0.0])
    reflected_rir = np.array([0.0, 0.5])
    excitation = np.array([1.0, 1.0])
    direct = np.convolve(excitation, direct_rir)
    reflected = np.convolve(excitation, reflected_rir)
    full = direct + reflected
    direct_power = float(np.dot(direct, direct))
    reflected_power = float(np.dot(reflected, reflected))
    cross_term = float(2.0 * np.dot(direct, reflected))
    return {"per_mic_drr_db": per_mic_db.tolist(),
            "median_drr_db": float(np.median(per_mic_db)),
            "pooled_energy_drr_db": float(10.0 * np.log10(
                np.sum(direct_energy) / np.sum(reflected_energy))),
            "direct_output": direct.tolist(), "reflected_output": reflected.tolist(),
            "full_output": full.tolist(), "direct_output_energy": direct_power,
            "reflected_output_energy": reflected_power,
            "cross_term_in_full_energy": cross_term,
            "full_output_energy": float(np.dot(full, full))}


def t20_from_edc_points(times_s, decay_db) -> dict:
    """Fit an exercise fixture with sampled -5 and -25 dB endpoints.

    A measured RIR generally needs crossing interpolation, noise-floor checks
    and a stated fit policy; this small calculation intentionally assumes its
    three input points already include both interval endpoints.
    """
    times = np.asarray(times_s, dtype=float)
    decay = np.asarray(decay_db, dtype=float)
    if (times.ndim != 1 or decay.ndim != 1 or len(times) != len(decay)
            or len(times) < 3 or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(decay)) or np.any(np.diff(times) <= 0)
            or np.any(np.diff(decay) >= 0)):
        raise ValueError("EDC points require three finite times and strictly falling dB")
    indices = np.flatnonzero((decay >= -25.0) & (decay <= -5.0))
    if (len(indices) < 3 or decay[indices[0]] != -5.0
            or decay[indices[-1]] != -25.0):
        raise ValueError("-5 to -25 dB fit interval is unavailable")
    slope = float(np.polyfit(times[indices], decay[indices], 1)[0])
    if not math.isfinite(slope) or slope >= 0:
        raise ValueError("EDC slope must be finite and negative")
    return {"slope_db_per_s": slope, "t20_s": -20.0 / slope,
            "t60_extrapolated_s": -60.0 / slope}


def two_mic_srp_phase() -> dict:
    """Compare the library score with an independent one-pair phase calculation."""
    speed = 343.0
    spacing = 0.04
    frequency = 1000.0
    true_azimuth_deg = 30.0
    tau_12 = spacing * math.sin(math.radians(true_azimuth_deg)) / speed
    phase = 2.0 * math.pi * frequency * tau_12
    positions = np.array([[0.0, 0.0], [spacing, 0.0]])
    spectrum = np.array([[[1.0 + 0.0j]], [[np.exp(1j * phase)]]])
    candidates = np.array([-30.0, 0.0, 30.0])
    scores = srp_phat(spectrum, np.array([frequency]), positions,
                      np.deg2rad(candidates), sound_speed=speed)
    return {"tau_12_us": tau_12 * 1e6,
            "cross_spectrum_phase_rad": -phase,
            "candidate_azimuth_deg": candidates.tolist(),
            "srp_scores": scores.tolist(),
            "analytic_scores": [math.cos(2 * phase), math.cos(phase), 1.0]}


def _read_pcm(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
            raise ValueError("room exercise requires uncompressed PCM16")
        channels, rate, frames = (stream.getnchannels(), stream.getframerate(),
                                  stream.getnframes())
        data = np.frombuffer(stream.readframes(frames), dtype="<i2")
    return rate, data.reshape(frames, channels).astype(float) / 32768.0


def room_pcm_readback(manifest: dict, results: dict) -> dict:
    """Use the existing near-left source/direct/full files; no new simulation."""
    name = "fixed_near_left"
    records = {item["role"]: item for item in manifest["files"] if item["case"] == name}
    if set(records) != {"source", "direct", "full"}:
        raise ValueError("near-left room triplet is incomplete")
    streams = {}
    for role, record in records.items():
        path = ROOM / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"room PCM digest differs: {path.name}")
        rate, pcm = _read_pcm(path)
        if rate != manifest["sample_rate_hz"] or pcm.shape != (
                record["frames"], record["channels"]):
            raise ValueError("room PCM format differs from manifest")
        streams[role] = pcm
    source = streams["source"][:, 0]
    direct = streams["direct"]
    n_fft = 1 << (len(source) + len(direct) - 2).bit_length()
    source_spectrum = np.fft.rfft(source, n_fft)
    lags = []
    for channel in range(4):
        correlation = np.fft.irfft(
            np.fft.rfft(direct[:, channel], n_fft) * source_spectrum.conj(), n_fft)
        lags.append(int(np.argmax(np.abs(correlation[:256]))))
    source_position = np.asarray(next(case["source_m"] for case in manifest["cases"]
                                      if case["name"] == name))
    microphone_positions = np.asarray(manifest["microphones_m"])
    physical_lags = np.linalg.norm(microphone_positions - source_position,
                                   axis=1) * manifest["sample_rate_hz"] / manifest["sound_speed_m_s"]
    return {"case": name,
            "shapes_frames_channels": {role: list(pcm.shape) for role, pcm in streams.items()},
            "pcm_peak_absolute": {role: float(np.max(np.abs(pcm)))
                                  for role, pcm in streams.items()},
            "common_export_gain_from_manifest": manifest["common_gain"],
            "measured_source_to_direct_lag_samples": lags,
            "geometric_propagation_samples": physical_lags.tolist(),
            "library_fractional_delay_filter_length":
                results["fractional_delay_filter_length_samples"],
            "scope": "the correlation peak includes the fixed library interpolation delay; it is not propagation delay alone"}


def room_pcm_srp_windows(manifest: dict, results: dict) -> dict:
    """Score both time spans of all six *published PCM* room triplets.

    These estimates are read back after PCM16 quantization. They are kept
    separate from the unquantized simulation estimates in RESULTS.json.
    """
    sample_rate = manifest["sample_rate_hz"]
    if sample_rate != results["sample_rate_hz"] or len(manifest["files"]) != 18:
        raise ValueError("room manifest and result sample rate or file count disagree")
    settings = results["doa"]
    n_fft, hop = settings["n_fft"], settings["hop_length"]
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    low, high = settings["frequency_band_hz"]
    selected = (frequencies >= low) & (frequencies <= high)
    first_angle, last_angle, step = settings["azimuth_grid_deg"]
    grid = np.arange(first_angle, last_angle + step / 2, step, dtype=float)
    microphones = np.asarray(manifest["microphones_m"], dtype=float)
    records = {(item["case"], item["role"]): item for item in manifest["files"]}
    if len(records) != 18 or len(manifest["cases"]) != 6:
        raise ValueError("room manifest needs six distinct source/direct/full triplets")
    report_rows = {row["name"]: row for row in results["results"]}
    rows = []
    for case in manifest["cases"]:
        name = case["name"]
        streams = {}
        for role, channels in (("source", 1), ("direct", 4), ("full", 4)):
            record = records[(name, role)]
            path = ROOM / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"room PCM digest differs: {path.name}")
            rate, pcm = _read_pcm(path)
            if rate != sample_rate or pcm.shape != (record["frames"], channels):
                raise ValueError(f"room PCM format differs from manifest: {path.name}")
            streams[role] = pcm
        if (len(streams["source"]) != sample_rate or
                streams["direct"].shape != streams["full"].shape):
            raise ValueError(f"room triplet duration differs: {name}")
        full = streams["full"].T
        scores = {}
        for label, samples in (("first_second", full[:, :sample_rate]),
                               ("entire_file", full)):
            spectra = stft(samples, n_fft=n_fft, hop_length=hop, center=True)
            score = srp_phat(spectra[:, selected], frequencies[selected],
                             microphones, np.deg2rad(grid),
                             sound_speed=manifest["sound_speed_m_s"])
            index = int(np.argmax(score))
            scores[label] = {"samples": int(samples.shape[1]),
                             "estimated_azimuth_deg": float(grid[index]),
                             "peak_srp_score": float(score[index])}
        rows.append({"name": name,
                     "source_frames": int(len(streams["source"])),
                     "direct_frames": int(len(streams["direct"])),
                     "full_frames": int(len(streams["full"])),
                     "pcm_scores": scores,
                     "unquantized_report_estimate_deg":
                         report_rows[name]["estimated_azimuth_deg"]})
    return {"input": "existing published PCM16 WAV, SHA-256 checked against MANIFEST.json",
            "pcm_scale_divisor": 32768, "stft_center": True,
            "stft_window": "periodic Hann", "n_fft": n_fft, "hop_length": hop,
            "frequency_band_hz": [low, high],
            "azimuth_grid_deg": [first_angle, last_angle, step],
            "first_second_interval_samples": [0, sample_rate],
            "cases": rows,
            "scope": "the PCM readback estimates are separate from the unquantized room simulation report"}


def equal_drr_different_spectra() -> dict:
    """Two equal-energy reflection RIRs need not have the same spectrum."""
    direct = np.array([1.0, 0.0, 0.0])
    reflections = (np.array([0.0, 0.5, 0.5]),
                   np.array([0.0, 0.5, -0.5]))
    cases = []
    for label, reflected in zip(("same_sign", "opposite_sign"), reflections):
        full = direct + reflected
        direct_energy = float(np.dot(direct, direct))
        reflected_energy = float(np.dot(reflected, reflected))
        cases.append({"name": label, "direct_rir": direct.tolist(),
                      "reflected_rir": reflected.tolist(), "full_rir": full.tolist(),
                      "direct_energy": direct_energy,
                      "reflected_energy": reflected_energy,
                      "drr_db": 10.0 * math.log10(direct_energy / reflected_energy),
                      "reflected_dc_response": float(sum(reflected)),
                      "reflected_nyquist_response":
                          float(sum(value * (-1) ** index
                                    for index, value in enumerate(reflected))),
                      "full_dc_response": float(sum(full)),
                      "full_nyquist_response":
                          float(sum(value * (-1) ** index
                                    for index, value in enumerate(full)))})
    return {"cases": cases,
            "scope": "equal scalar RIR energy ratios do not determine frequency response or localization"}


def evidence_claims(*, locked: bool, license_checked: bool, obtained: bool,
                    executed: bool, scored: bool) -> list[str]:
    """Grant claims only at or below the supplied hypothetical evidence level."""
    flags = (locked, license_checked, obtained, executed, scored)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("evidence flags must be booleans")
    if license_checked and not locked or obtained and not (locked and license_checked) or (
            executed and not obtained) or (scored and not executed):
        raise ValueError("evidence stages must not skip a prerequisite")
    names = ("fixed source and version identified", "license checked", "source obtained",
             "fixed input executed", "specified scoring completed")
    return [name for name, flag in zip(names, flags) if flag]


def run_exercises() -> dict:
    results = _read_results()
    manifest = json.loads((ROOM / "MANIFEST.json").read_text(encoding="utf-8"))
    edc = t20_from_edc_points([0.05, 0.15, 0.25], [-5.0, -15.0, -25.0])
    shifted = t20_from_edc_points([0.15, 0.25, 0.35], [-5.0, -15.0, -25.0])
    return {
        "E13-03": paired_room_comparison(results["results"]),
        "E13-04": four_mic_drr_and_convolution(),
        "E13-05": {**edc, "shifted_time_t60_s": shifted["t60_extrapolated_s"]},
        "E13-06": two_mic_srp_phase(),
        "E13-07": room_pcm_readback(manifest, results),
        "E13-08": {"cases_are_hypothetical": True,
                    "A": evidence_claims(locked=True, license_checked=True,
                                         obtained=False, executed=False, scored=False),
                    "B": evidence_claims(locked=True, license_checked=True,
                                         obtained=True, executed=True, scored=False),
                    "C": evidence_claims(locked=True, license_checked=True,
                                         obtained=True, executed=True, scored=True)},
        "E13-09": room_pcm_srp_windows(manifest, results),
        "E13-10": equal_drr_different_spectra(),
    }


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
