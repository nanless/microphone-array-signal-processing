"""Chapter 1: finite records, alignment error and two distinct head models.

Run ``.venv/bin/python -m codes.chapters.ch01.chapter01_experiments``. E01-04--09
use deterministic mathematical inputs and actual published mathematical WAVs,
not device or listener measurements.
No files are written. Powers are dimensionless digital mean squares, without
subtracting sample means; dB values are relative levels, never dB SPL.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

from codes.chapters.ch00.core.audio_samples import alignment_error_case, read_pcm16
from codes.chapters.ch01.examples.generate_binaural_cues import check_assets, wav_metadata

ROOT = Path(__file__).resolve().parents[3]


def _tone_amplitudes(signal: np.ndarray, fs: int, frequencies: tuple[int, ...]) -> list[float]:
    """Real sine/cosine least squares on the published integer-period window."""
    index = np.arange(1600, 30400)
    basis = np.column_stack([function(2 * np.pi * frequency * index / fs)
                             for frequency in frequencies for function in (np.sin, np.cos)])
    coefficients = np.linalg.lstsq(basis, signal[index], rcond=None)[0]
    return [float(np.hypot(coefficients[i], coefficients[i + 1]))
            for i in range(0, len(coefficients), 2)]


def published_alignment_measurements() -> dict:
    """Validate and measure the four committed alignment WAVs, without writing."""
    manifest_path = ROOT / "codes/chapters/ch00/audio/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["files"] if row["group"] == "alignment_error"]
    stems = {"alignment_reference", "alignment_array", "alignment_unaligned", "alignment_aligned"}
    if {Path(row["file"]).stem for row in records} != stems or len(records) != 4:
        raise ValueError("Alignment manifest must contain exactly the four expected WAVs")
    if manifest["groups"]["alignment_error"]["common_export_gain"] != 1.:
        raise ValueError("Alignment fixtures require the published common export gain 1")
    for path, recorded_hash in manifest["generator_inputs"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != recorded_hash:
            raise ValueError(f"Alignment generating source is stale: {path}")
    decoded_signals, assets = {}, {}
    for row in records:
        relative_path = f"codes/chapters/{row['chapter']}/audio/{row['file']}"
        payload = (ROOT / relative_path).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        metadata = wav_metadata(payload)
        channels = 2 if row["file"] == "alignment_array.wav" else 1
        required = {"sample_rate_hz": 16000, "channels": channels, "samples_per_channel": 32000,
                    "sample_width_bytes": 2, "compression": "NONE"}
        if actual_hash != row["sha256"] or metadata != required:
            raise ValueError(f"Published alignment SHA/PCM format mismatch: {relative_path}")
        if (row["sample_rate_hz"], row["channels"], row["samples"], row["common_export_gain"]) != (16000, channels, 32000, 1.):
            raise ValueError(f"Alignment manifest format/gain mismatch: {relative_path}")
        fs, decoded = read_pcm16(payload)
        stem = Path(row["file"]).stem
        decoded_signals[stem] = decoded
        assets[stem] = {"path": relative_path, "sha256": actual_hash, **metadata,
                        "tone_amplitudes_by_channel": [_tone_amplitudes(channel, fs, (1000, 4000))
                                                        for channel in decoded]}
    if not np.array_equal(decoded_signals["alignment_reference"], decoded_signals["alignment_aligned"]):
        raise ValueError("Published alignment reference and aligned output differ")
    floats = alignment_error_case()["signals"]
    scores = {}
    for domain, signals in (("float", floats), ("pcm", decoded_signals)):
        amplitudes = {stem: _tone_amplitudes(signals[stem][0] if signals[stem].ndim == 2 else signals[stem],
                                           16000, (1000, 4000))
                      for stem in ("alignment_reference", "alignment_unaligned", "alignment_aligned")}
        reference = amplitudes["alignment_reference"]
        scores[domain] = {"cases": [
            {"frequency_hz": frequency, "reference_amplitude": reference[i],
             "unaligned_amplitude": amplitudes["alignment_unaligned"][i],
             "aligned_amplitude": amplitudes["alignment_aligned"][i],
             "unaligned_amplitude_ratio": amplitudes["alignment_unaligned"][i] / reference[i],
             "unaligned_level_change_db": float(20 * np.log10(amplitudes["alignment_unaligned"][i] / reference[i])),
             "aligned_amplitude_ratio": amplitudes["alignment_aligned"][i] / reference[i]}
            for i, frequency in enumerate((1000, 4000))]}
    return {"manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_sha256": manifest["generator_inputs"], "assets": assets,
            "scoring_interval_samples": [1600, 30400], "scoring_samples": 28800,
            "measurement": "Real sine/cosine least squares, unchanged common export gain 1; no mean removal",
            "float_measurements": scores["float"], "pcm_measurements": scores["pcm"]}


def finite_record_cross_terms() -> dict:
    """E01-04: expand the square before interpreting a short-record gain."""
    noise = np.array([[1., 1., -1., -1.], [1., -1., 1., -1.]])
    rows = []
    for count in (4, 3):
        first, second = noise[:, :count]
        mean = (first + second) / 2
        p1, p2 = float(np.mean(first**2)), float(np.mean(second**2))
        cross = float(np.mean(first * second))
        power = float(np.mean(mean**2))
        rows.append({
            "samples": count, "noise_1": first.tolist(), "noise_2": second.tolist(),
            "products": (first * second).tolist(), "average": mean.tolist(),
            "average_squared": (mean**2).tolist(),
            "input_mean_squares": [p1, p2], "cross_second_moment": cross,
            "diagonal_contribution": (p1 + p2) / 4,
            "cross_contribution": cross / 2, "output_mean_square": power,
            "gain_relative_to_channel_1_db": float(10 * np.log10(p1 / power)),
        })
    return {"cases": rows,
            "measurement": "Uncentered finite-record mean square; no sample-mean removal",
            "limits": "Fixed sequences demonstrate the identity, not statistical independence. "
                      "Cropping shares samples, so these are not independent trials."}


def residual_sample_delay() -> dict:
    """E01-05: target-only loss when two equal channels differ by one sample.

    Compare sinusoidal amplitudes, not pointwise error against an unmatched
    time reference. Integer-period projection is independent of the analytic
    cosine identity. After alignment both channels equal s[n-1].
    """
    fs = 16000
    index = np.arange(16000)
    rows = []
    for frequency in (1000, 4000):
        phase = 2 * np.pi * frequency * index / fs
        first = np.sin(phase)
        second = np.sin(phase - 2 * np.pi * frequency / fs)
        averaged = (first + second) / 2
        amplitude = float(2 * abs(np.mean(averaged * np.exp(-1j * phase))))
        ratio = float(np.cos(np.pi * frequency / fs))
        rows.append({"frequency_hz": frequency,
                     "interchannel_phase_rad": float(2 * np.pi * frequency / fs),
                     "target_amplitude_ratio": ratio,
                     "projected_target_amplitude_ratio": amplitude,
                     "target_power_ratio": ratio**2,
                     "target_level_change_db": float(20 * np.log10(ratio)),
                     "aligned_target_amplitude_ratio": 1.0})
    return {"sample_rate_hz": fs, "residual_delay_samples": 1,
            "residual_delay_seconds": 1 / fs, "cases": rows,
            "unaligned_common_delay_samples": .5,
            "aligned_common_delay_samples": 1,
            "published_audio": published_alignment_measurements(),
            "limits": "No noise is present: these are target level changes, not SNR gains. "
                      "Exact alignment is known, not estimated."}


def woodworth_comparison() -> dict:
    """E01-06: compare a ray model around a sphere with unobstructed points.

    Woodworth: Aaronson & Hartmann (2014), JASA 135, 817--823,
    DOI 10.1121/1.4861243. Antipodal ears, distant source, frontal hemisphere;
    the signed formula extends the unsigned ray length by left/right symmetry.
    It is a high-frequency approximation, not a listener threshold or HRTF.
    """
    radius, speed, fs = .0875, 343., 16000
    rows = []
    for angle in (-90, -30, -1, 0, 1, 30, 90):
        theta = np.deg2rad(angle)
        sphere = float(radius * (theta + np.sin(theta)) / speed)
        free = float(2 * radius * np.sin(theta) / speed)
        rows.append({"azimuth_deg": angle, "azimuth_rad": float(theta),
                     "sphere_itd_us": sphere * 1e6,
                     "free_point_itd_us": free * 1e6,
                     "sphere_itd_samples": sphere * fs,
                     "free_point_itd_samples": free * fs})
    return {"head_radius_m": radius, "sound_speed_m_s": speed,
            "sample_rate_hz": fs, "cases": rows,
            "angle_convention": "Zero is front; positive angles point right",
            "itd_convention": "left arrival minus right arrival; positive means right arrives first",
            "limits": "The sphere and free points are different physical models; "
                      "neither predicts a universal human discrimination threshold."}


def mixture_power_and_identifiability() -> dict:
    """E01-07: finite-record cross terms and nonunique signal/noise decomposition."""
    def measure(signal, noise):
        signal, noise = np.asarray(signal, dtype=float), np.asarray(noise, dtype=float)
        ps, pv = float(np.mean(signal**2)), float(np.mean(noise**2))
        cross = float(np.mean(signal * noise))
        mixture = signal + noise
        return {"signal": signal.tolist(), "noise": noise.tolist(), "mixture": mixture.tolist(),
                "signal_mean_square": ps, "noise_mean_square": pv,
                "cross_second_moment": cross, "mixture_mean_square": float(np.mean(mixture**2)),
                "sum_of_component_mean_squares": ps + pv, "cross_contribution": 2 * cross,
                "snr_db": float(10 * np.log10(ps / pv))}
    signal = [1, -1, 1, -1]
    return {"illustrative_cross_terms": [measure(signal, noise) for noise in
                                          ([1, 1, -1, -1], signal, [-1, 1, -1, 1])],
            "cases": [measure([1, 0, -1, 0], [1, 0, -1, 0]),
                      measure([1.5, 0, -1.5, 0], [.5, 0, -.5, 0])],
            "limits": "Uncentered finite-record identity. Observing x alone does not specify which decomposition is true; "
                      "these constructed noise components are not assumed independent of the target."}


def binaural_level_and_delay() -> dict:
    """E01-08: read actual stereo PCM and report amplitude and delay separately."""
    manifest = check_assets()
    return {"sample_rate_hz": manifest["sample_rate_hz"],
            "common_export_gain": manifest["common_export_gain"],
            "channel_order": manifest["channel_order"], "conventions": manifest["conventions"],
            "samples": manifest["samples"], "files": manifest["files"],
            "source_sha256": manifest["source_sha256"], "limits": manifest["limits"]}


def white_noise_gain() -> dict:
    """E01-09: target preservation alone does not bound weight noise amplification."""
    noise = np.array([[1., 1., -1., -1.], [1., -1., 1., -1.]])
    rows = []
    for values in ((.5, .5), (2., -1.)):
        weights = np.array(values)
        target_gain = float(np.sum(weights))
        noise_power = float(np.mean((weights @ noise)**2))
        gain = target_gain**2 / float(np.sum(weights**2))
        rows.append({"weights": list(values), "target_amplitude_gain": target_gain,
                     "squared_weight_norm": float(np.sum(weights**2)),
                     "output_noise_mean_square": noise_power,
                     "white_noise_gain_linear": gain, "white_noise_gain_db": float(10 * np.log10(gain))})
    return {"noise_channels": noise.tolist(), "input_noise_mean_squares": [1., 1.], "cases": rows,
            "limits": "The finite sequences have zero cross moment and equal unit mean square. "
                      "White-noise gain concerns the equal-variance uncorrelated-noise model, not arbitrary noise or directivity."}


def run_exercises() -> dict:
    """Stable exercise IDs; metadata lives inside each individual result."""
    return {"E01-04": finite_record_cross_terms(),
            "E01-05": residual_sample_delay(), "E01-06": woodworth_comparison(),
            "E01-07": mixture_power_and_identifiability(),
            "E01-08": binaural_level_and_delay(), "E01-09": white_noise_gain()}


if __name__ == "__main__":
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
