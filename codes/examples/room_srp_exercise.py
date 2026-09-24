"""Appendix B, exercise 16: reproducible shoebox RIR and SRP-PHAT study.

``--check`` needs only NumPy and verifies the fixed geometry and the Sabine
input calculation. ``--run`` additionally needs pyroomacoustics 0.10.0.
Neither mode downloads data or claims that a simulated room is a measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.array_tutorial.doa import srp_phat
from codes.array_tutorial.spectral import stft

PRA_VERSION = "0.10.0"
ROOM_DIM_M = (12.0, 10.0, 6.0)
ARRAY_CENTER_M = (6.0, 5.0, 3.0)
MIC_OFFSETS_M = ((-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03))
SAMPLE_RATE_HZ = 16000
SOUND_SPEED_M_S = 343.0
TARGET_T60_S = 0.6
SEED = 20260924
N_FFT = 512
HOP_LENGTH = 128
FREQUENCY_BAND_HZ = (300.0, 2000.0)
AZIMUTH_GRID_DEG = tuple(range(-80, 81))


def sabine_inputs() -> tuple[float, int, float, float]:
    """Mirror pyroomacoustics 0.10.0 ``inverse_sabine`` for a 3-D shoebox.

    This independent, dependency-free calculation is only an input check;
    the RIR-based T20 extrapolation in ``--run`` measures achieved decay.
    """
    dimensions = ROOM_DIM_M
    volume = math.prod(dimensions)
    pairs = [dimensions[i] * dimensions[j] for i in range(3) for j in range(i + 1, 3)]
    surface = 2.0 * sum(pairs)
    alpha = 24.0 * math.log(10.0) * volume / (SOUND_SPEED_M_S * surface * TARGET_T60_S)
    radii = [dimensions[i] * dimensions[j] / math.hypot(dimensions[i], dimensions[j])
             for i in range(3) for j in range(i + 1, 3)]
    order = math.ceil(SOUND_SPEED_M_S * TARGET_T60_S / min(radii) - 1.0)
    return alpha, order, volume, surface


def configuration() -> dict:
    """Return all geometry and random draws without importing pyroomacoustics."""
    alpha, order, volume, surface = sabine_inputs()
    rng = np.random.Generator(np.random.PCG64(SEED))
    positions = [
        ("fixed_near_left", 1.0, -30.0),
        ("fixed_near_right", 1.0, 30.0),
        ("fixed_far_left", 2.5, -30.0),
        ("fixed_far_right", 2.5, 30.0),
        ("seeded_1", float(rng.uniform(1.0, 2.5)), float(rng.uniform(-60.0, 60.0))),
        ("seeded_2", float(rng.uniform(1.0, 2.5)), float(rng.uniform(-60.0, 60.0))),
    ]
    microphones = [[ARRAY_CENTER_M[0] + x, ARRAY_CENTER_M[1] + y, ARRAY_CENTER_M[2]]
                   for x, y in MIC_OFFSETS_M]
    cases = []
    for index, (name, distance, azimuth) in enumerate(positions):
        angle = math.radians(azimuth)
        source = [ARRAY_CENTER_M[0] + distance * math.sin(angle),
                  ARRAY_CENTER_M[1] + distance * math.cos(angle), ARRAY_CENTER_M[2]]
        cases.append({"name": name, "source_m": source, "distance_m": distance,
                      "true_azimuth_deg": azimuth, "excitation_seed_sequence": [SEED, index + 1]})
    config = {
        "status": "geometry_checked_only", "pyroomacoustics_version_required": PRA_VERSION,
        "room_dimensions_m": list(ROOM_DIM_M), "room_volume_m3": volume,
        "room_surface_m2": surface, "target_t60_s": TARGET_T60_S,
        "sabine_energy_absorption": alpha, "inverse_sabine_suggested_max_order": order,
        "sample_rate_hz": SAMPLE_RATE_HZ, "sound_speed_m_s": SOUND_SPEED_M_S,
        "microphones_m": microphones, "array_center_m": list(ARRAY_CENTER_M),
        "cases": cases, "source_position_seed": SEED,
        "source_signal": "1 s independent PCG64 white Gaussian noise per case; no additive noise",
        "doa": {"method": "far-field SRP-PHAT", "n_fft": N_FFT,
                "hop_length": HOP_LENGTH, "frequency_band_hz": list(FREQUENCY_BAND_HZ),
                "azimuth_grid_deg": [AZIMUTH_GRID_DEG[0], AZIMUTH_GRID_DEG[-1], 1]},
        "t60": "Schroeder T20 (-5 to -25 dB), linear dB fit extrapolated to 60 dB",
        "drr": ("10 log10(sum(h_direct^2)/sum((h_full-h_direct)^2)), per microphone; "
                "direct RIR uses fully absorbing walls at the same image order and RIR length"),
    }
    validate_configuration(config)
    return config


def validate_configuration(config: dict) -> None:
    """Reject source/microphone overlap, wall contact, or invalid Sabine input."""
    if not 0.0 < config["sabine_energy_absorption"] < 1.0:
        raise ValueError("Sabine absorption must be strictly between zero and one")
    dimensions = np.asarray(config["room_dimensions_m"], dtype=float)
    center = np.asarray(config["array_center_m"], dtype=float)
    microphones = np.asarray(config["microphones_m"], dtype=float)
    if microphones.shape != (4, 3) or np.any(microphones <= 0) or np.any(microphones >= dimensions):
        raise ValueError("all four microphones must be strictly inside the room")
    if not np.allclose(microphones.mean(axis=0), center, atol=1e-12):
        raise ValueError("array center and microphone coordinates disagree")
    for case in config["cases"]:
        source = np.asarray(case["source_m"], dtype=float)
        if source.shape != (3,) or np.any(source <= 0) or np.any(source >= dimensions):
            raise ValueError("every source must be strictly inside the room")
        if np.min(np.linalg.norm(microphones - source, axis=1)) < 0.2:
            raise ValueError("a source is too close to a microphone")
        distance = float(np.linalg.norm(source - center))
        azimuth = math.degrees(math.atan2(source[0] - center[0], source[1] - center[1]))
        if not math.isclose(distance, case["distance_m"], abs_tol=1e-12):
            raise ValueError("source distance and coordinates disagree")
        if not math.isclose(azimuth, case["true_azimuth_deg"], abs_tol=1e-12):
            raise ValueError("source azimuth and coordinates disagree")


def drr_db(full_rir: np.ndarray, direct_rir: np.ndarray) -> float:
    """Decompose same-clock RIRs into direct path and reflected remainder."""
    full = np.asarray(full_rir, dtype=float)
    direct = np.asarray(direct_rir, dtype=float)
    if full.ndim != 1 or direct.ndim != 1 or not full.size or not direct.size:
        raise ValueError("RIRs must be nonempty one-dimensional arrays")
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(direct)):
        raise ValueError("RIRs must be finite")
    length = max(full.size, direct.size)
    direct_padded = np.pad(direct, (0, length - direct.size))
    full_padded = np.pad(full, (0, length - full.size))
    direct_power = float(np.dot(direct_padded, direct_padded))
    reflected = full_padded - direct_padded
    reflected_power = float(np.dot(reflected, reflected))
    if direct_power <= 0.0 or reflected_power <= 0.0:
        raise ValueError("DRR needs nonzero direct and reflected energy")
    return 10.0 * math.log10(direct_power / reflected_power)


def measured_t60_from_t20(rir: np.ndarray) -> float:
    """Fit Schroeder decay from -5 to -25 dB, extrapolate slope to -60 dB.

    Unlike a helper that silently shortens the fit interval, this raises when
    the specified interval is unavailable. No tail trimming or noise-floor
    correction is used; synthetic noiseless RIRs are the intended input.
    """
    impulse = np.asarray(rir, dtype=float)
    if impulse.ndim != 1 or impulse.size < 2 or not np.all(np.isfinite(impulse)):
        raise ValueError("RIR must be a finite one-dimensional vector")
    energy = np.cumsum(np.square(impulse[::-1]))[::-1]
    if energy[0] <= 0.0:
        raise ValueError("RIR has no energy")
    with np.errstate(divide="ignore"):
        decay_db = 10.0 * np.log10(energy / energy[0])
    start = np.flatnonzero(decay_db <= -5.0)
    stop = np.flatnonzero(decay_db <= -25.0)
    if not start.size or not stop.size or stop[0] - start[0] < 3:
        raise ValueError("RIR does not contain a usable -5 to -25 dB decay interval")
    sample_indices = np.arange(start[0], stop[0] + 1, dtype=float)
    slope_db_s = np.polyfit(sample_indices / SAMPLE_RATE_HZ,
                            decay_db[start[0]:stop[0] + 1], 1)[0]
    if not math.isfinite(slope_db_s) or slope_db_s >= 0.0:
        raise ValueError("T20 fit must have a negative finite slope")
    return -60.0 / slope_db_s


def _make_room(pra: object, source_m: list[float], microphones_m: list[list[float]],
               absorption: float, max_order: int):
    room = pra.ShoeBox(ROOM_DIM_M, fs=SAMPLE_RATE_HZ,
                       materials=pra.Material(absorption), max_order=max_order,
                       air_absorption=False, ray_tracing=False, use_rand_ism=False)
    room.set_sound_speed(SOUND_SPEED_M_S)
    room.add_source(source_m)
    room.add_microphone_array(np.asarray(microphones_m).T)
    room.compute_rir()
    return room


def _fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    length = signal.size + impulse.size - 1
    n_fft = 1 << (length - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(signal, n_fft) * np.fft.rfft(impulse, n_fft), n_fft)[:length]


def _excitation(case: dict) -> np.ndarray:
    seed = np.random.SeedSequence(case["excitation_seed_sequence"])
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(SAMPLE_RATE_HZ)


def _room_metrics(pra: object, config: dict, case: dict, max_order: int,
                  *, with_doa: bool) -> dict:
    started = time.perf_counter()
    full = _make_room(pra, case["source_m"], config["microphones_m"],
                      config["sabine_energy_absorption"], max_order)
    # pyroomacoustics 0.10.0 applies a zero-phase high-pass filter to each
    # complete RIR. Its boundary response depends on RIR length. A zero-order
    # RIR is much shorter than the full RIR, so subtracting their filtered
    # outputs would count filter-boundary differences as reflections. Keeping
    # the same image order and setting wall absorption to 1.0 leaves only the
    # direct path while preserving the time axis and filter boundary.
    direct = _make_room(pra, case["source_m"], config["microphones_m"],
                        1.0, max_order)
    if any(len(full.rir[m][0]) != len(direct.rir[m][0]) for m in range(4)):
        raise RuntimeError("full and direct-only RIRs have different lengths")
    drr = [drr_db(full.rir[m][0], direct.rir[m][0]) for m in range(4)]
    t60 = np.asarray([measured_t60_from_t20(full.rir[m][0]) for m in range(4)])
    row = {"name": case["name"], "source_m": case["source_m"],
           "distance_m": case["distance_m"], "true_azimuth_deg": case["true_azimuth_deg"],
           "max_order": max_order, "image_source_count": int(full.sources[0].images.shape[1]),
           "rir_length_samples_per_mic": [len(full.rir[m][0]) for m in range(4)],
           "drr_db_per_mic": drr,
           "drr_db_median": float(np.median(drr)), "t60_s_per_mic": t60.tolist(),
           "t60_s_median": float(np.median(t60))}
    if with_doa:
        excitation = _excitation(case)
        waveforms = np.stack([_fft_convolve(excitation, np.asarray(full.rir[m][0]))[:SAMPLE_RATE_HZ]
                              for m in range(4)])
        spectra = stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH)
        frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE_HZ)
        selected = (frequencies >= FREQUENCY_BAND_HZ[0]) & (frequencies <= FREQUENCY_BAND_HZ[1])
        grid = np.asarray(AZIMUTH_GRID_DEG, dtype=float)
        score = srp_phat(spectra[:, selected], frequencies[selected],
                         np.asarray(config["microphones_m"]), np.deg2rad(grid),
                         sound_speed=SOUND_SPEED_M_S)
        estimate = float(grid[int(np.argmax(score))])
        row.update({"estimated_azimuth_deg": estimate,
                    "absolute_doa_error_deg": abs(estimate - case["true_azimuth_deg"]),
                    "excitation_seed_sequence": case["excitation_seed_sequence"]})
    row["elapsed_s"] = time.perf_counter() - started
    return row


def run_experiment(max_order: int | None = None) -> dict:
    """Run actual pyroomacoustics RIRs; return observations, never cached values."""
    started = time.perf_counter()
    config = configuration()
    try:
        import pyroomacoustics as pra
    except ImportError as error:
        raise RuntimeError("--run requires pyroomacoustics==0.10.0; --check works without it") from error
    if pra.__version__ != PRA_VERSION:
        raise RuntimeError(f"pyroomacoustics {PRA_VERSION} required; found {pra.__version__}")
    expected_alpha, suggested_order = pra.inverse_sabine(TARGET_T60_S, ROOM_DIM_M, c=SOUND_SPEED_M_S)
    if not math.isclose(expected_alpha, config["sabine_energy_absorption"], rel_tol=1e-12):
        raise RuntimeError("local Sabine check disagrees with pyroomacoustics")
    if suggested_order != config["inverse_sabine_suggested_max_order"]:
        raise RuntimeError("local image-order check disagrees with pyroomacoustics")
    if max_order is None:
        max_order = suggested_order
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 2:
        raise ValueError("max_order must be an integer >= 2")
    previous_order = max(1, max_order - 8)
    anchor = config["cases"][0]
    previous = _room_metrics(pra, config, anchor, previous_order, with_doa=False)
    rows = [_room_metrics(pra, config, case, max_order, with_doa=True) for case in config["cases"]]
    current = rows[0]
    t60_change = abs(current["t60_s_median"] - previous["t60_s_median"])
    drr_change = abs(current["drr_db_median"] - previous["drr_db_median"])
    return {
        **config, "status": "pyroomacoustics_simulation_executed",
        "pyroomacoustics_version_installed": pra.__version__, "actual_max_order": max_order,
        "rir_highpass_enabled": bool(pra.constants.get("rir_hpf_enable")),
        "rir_highpass_cutoff_hz": (float(pra.constants.get("rir_hpf_fc"))
                                   if pra.constants.get("rir_hpf_enable") else None),
        "elapsed_s": time.perf_counter() - started,
        "convergence_check": {
            "case": anchor["name"], "previous_max_order": previous_order,
            "previous_t60_s_median": previous["t60_s_median"],
            "previous_drr_db_median": previous["drr_db_median"],
            "t60_change_s": t60_change, "drr_change_db": drr_change,
            "threshold_t60_s": 0.02, "threshold_drr_db": 0.5,
            "within_threshold": t60_change <= 0.02 and drr_change <= 0.5,
            "scope": "one source position, median of four microphones; rerun at higher order if false",
        },
        "results": rows,
        "interpretation_limit": "one fixed synthetic room and six positions; no real-recording or population claim",
    }


def plot_results(report: dict, path: Path) -> None:
    """Save aligned acoustic and localization bars for all six positions."""
    import matplotlib.pyplot as plt

    rows = report["results"]
    labels = [f"{r['name']}\n{r['distance_m']:.2f} m, {r['true_azimuth_deg']:+.1f}°"
              for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.2, 7.2),
                             constrained_layout=True)
    values = ([r["drr_db_median"] for r in rows],
              [r["t60_s_median"] for r in rows],
              [r["absolute_doa_error_deg"] for r in rows])
    ylabels = ("Median microphone DRR (dB)", "T20-extrapolated T60 (s)",
               "Absolute SRP-PHAT error (°)")
    hatches = ("///", "\\\\", "..", "xx", "//", "--")
    for ax, data, ylabel in zip(axes, values, ylabels):
        bars = ax.bar(x, data, width=0.68, color="#96b4cf", edgecolor="#182c3a")
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].axhline(0, color="#182c3a", linewidth=0.8)
    axes[1].axhline(report["target_t60_s"], color="#b3453d", linestyle="--",
                    linewidth=1.5, label="0.6 s design target")
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].set_ylim(0, max(report["target_t60_s"] * 1.14,
                             max(values[1]) * 1.1))
    axes[2].set_ylim(0, max(values[2]) * 1.3)
    axes[2].set_xticks(x, labels, fontsize=10)
    fig.suptitle("Six positions in one fixed synthetic room (pyroomacoustics 0.10.0)",
                 fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_pcm16(path: Path, samples: np.ndarray) -> str:
    """Write one- or four-channel PCM without hidden per-file normalization."""
    frames = np.asarray(samples, dtype=float)
    if (frames.ndim != 2 or frames.shape[0] == 0 or frames.shape[1] not in (1, 4)
            or not np.all(np.isfinite(frames))):
        raise ValueError("PCM input must be finite frames x 1 or 4 channels")
    if np.max(np.abs(frames)) > 1.0:
        raise ValueError("PCM input would clip")
    pcm = np.rint(frames * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(frames.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE_HZ)
        stream.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_audio(report: dict, directory: Path) -> dict:
    """Export optional synthetic A/B files with one gain shared by every file."""
    import pyroomacoustics as pra

    if report.get("status") != "pyroomacoustics_simulation_executed":
        raise ValueError("audio export requires an executed room simulation")
    if pra.__version__ != report["pyroomacoustics_version_installed"]:
        raise ValueError("installed pyroomacoustics version differs from the report")
    if directory.exists():
        raise ValueError(f"audio output directory already exists: {directory}")
    prepared = []
    peak = 0.0
    for case in report["cases"]:
        full = _make_room(pra, case["source_m"], report["microphones_m"],
                          report["sabine_energy_absorption"], report["actual_max_order"])
        direct = _make_room(pra, case["source_m"], report["microphones_m"],
                            1.0, report["actual_max_order"])
        excitation = _excitation(case)
        waveforms = {}
        for label, room in (("full", full), ("direct", direct)):
            channels = [_fft_convolve(excitation, np.asarray(room.rir[m][0]))
                        for m in range(4)]
            length = max(len(channel) for channel in channels)
            waveforms[label] = np.stack([np.pad(channel, (0, length - len(channel)))
                                         for channel in channels], axis=1)
        if waveforms["full"].shape != waveforms["direct"].shape:
            raise RuntimeError("full and direct-only audio lengths differ")
        peak = max(peak, float(np.max(np.abs(excitation))),
                   *(float(np.max(np.abs(v))) for v in waveforms.values()))
        prepared.append((case, excitation, waveforms))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("audio export has zero or non-finite peak")
    common_gain = 0.8 / peak
    directory.mkdir(parents=True)
    files = []
    for case, excitation, waveforms in prepared:
        names = {"source": excitation[:, None], **waveforms}
        for label, values in names.items():
            path = directory / f"{case['name']}_{label}.wav"
            scaled = values * common_gain
            files.append({"case": case["name"], "role": label, "file": path.name,
                          "channels": int(values.shape[1]), "frames": int(values.shape[0]),
                          "peak_before_gain": float(np.max(np.abs(values))),
                          "peak_after_gain": float(np.max(np.abs(scaled))),
                          "sha256": _write_pcm16(path, scaled)})
    manifest = {
        "provenance": "mathematically synthesized white Gaussian noise, not recorded speech",
        "pyroomacoustics_version": report["pyroomacoustics_version_installed"],
        "numpy_version": np.__version__,
        "sample_rate_hz": SAMPLE_RATE_HZ, "pcm": "signed little-endian 16-bit, interleaved",
        "source_duration_s": 1.0, "max_order": report["actual_max_order"],
        "room_dimensions_m": report["room_dimensions_m"],
        "microphones_m": report["microphones_m"],
        "sound_speed_m_s": report["sound_speed_m_s"],
        "target_t60_s": report["target_t60_s"],
        "wall_energy_absorption_full": report["sabine_energy_absorption"],
        "wall_energy_absorption_direct": 1.0,
        "rir_highpass_enabled": report["rir_highpass_enabled"],
        "rir_highpass_cutoff_hz": report["rir_highpass_cutoff_hz"],
        "source_and_microphone_clocks_aligned": True,
        "propagation_delay_and_gain_compensated": False,
        "channel_length_handling": "trailing zeros pad each channel to the longest convolution",
        "normalization": "one common gain for all source/full/direct files; no per-file peak scaling",
        "common_gain": common_gain, "maximum_pre_gain_peak": peak,
        "maximum_post_gain_peak": 0.8,
        "excitation": "1 s independent PCG64 standard-normal float64 per case",
        "cases": [{"name": c["name"], "source_m": c["source_m"],
                   "excitation_seed_sequence": c["excitation_seed_sequence"],
                   "source_float64_le_sha256": hashlib.sha256(
                       np.asarray(signal, dtype="<f8").tobytes()).hexdigest()}
                  for c, signal, _ in prepared],
        "files": files,
    }
    (directory / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                               + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate geometry and inputs without PRA")
    mode.add_argument("--run", action="store_true", help="run pyroomacoustics 0.10.0 simulation")
    parser.add_argument("--max-order", type=int, help="actual image order (default: inverse_sabine suggestion)")
    parser.add_argument("--plot", type=Path, help="save a three-panel DRR/T60/DOA chart after --run")
    parser.add_argument("--audio-dir", type=Path,
                        help="write optional synthetic source/direct/full PCM files and manifest")
    args = parser.parse_args()
    if args.check and (args.max_order is not None or args.plot is not None or args.audio_dir is not None):
        parser.error("--max-order, --plot, and --audio-dir require --run")
    try:
        report = configuration() if args.check else run_experiment(args.max_order)
        if args.plot is not None:
            plot_results(report, args.plot)
        if args.audio_dir is not None:
            manifest = export_audio(report, args.audio_dir)
            report["audio_export"] = {"directory": str(args.audio_dir),
                                      "file_count": len(manifest["files"]),
                                      "common_gain": manifest["common_gain"]}
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
