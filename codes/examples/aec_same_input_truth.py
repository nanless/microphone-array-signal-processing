"""Apply three pinned AEC interfaces to one known-component PCM fixture.

This is a synthetic interface and metric experiment, not an industrial AEC
ranking. SpeexDSP's synchronous core, WebRTC AEC3's exported linear/final
outputs, and an oracle-frozen teaching NLMS process different internal chains.
AEC3 input/output WAVs are temporary and are not redistributed.

Run from the repository root with the already-built pinned binaries:
    .venv/bin/python -m codes.examples.aec_same_input_truth \
      --speex-library /private/tmp/speexdsp-aec-20260923/libspeexdsp.dylib \
      --audioproc codes/upstream/_downloads/webrtc_aec3_checkout/src/out/aec3/audioproc_f
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from codes.array_tutorial.aec import nlms
from codes.examples.aec3_offline_compare import command, read_wav, write_wav
from codes.examples.aec_controlled_doubletalk import checked_pcm_add, increment_metrics
from codes.examples.aec_real_pair_experiment import speex_linear_aec


RATE = 16000
FRAME = 160
SAMPLES = 6 * RATE
SEED = 20260924
PATH = (0.70, 0.0, -0.25, 0.0, 0.12)
FAR_SCORE = (2 * RATE, 3 * RATE)
NEAR_SCORE = (3 * RATE, 4 * RATE)
SPEEX_SHA256 = "c3e70172a3a9bf60b60bfd0bddfd58550a1899fef09078a9c7d5270d4a12105d"
AEC3_SHA256 = "82f8eebaf574eb10fdee695ee7ea05a54418300a045a3ffb0de8f4c78f4592d3"


def _pcm16(values: np.ndarray) -> np.ndarray:
    scaled = np.rint(values * 32768.0)
    if not np.all(np.isfinite(scaled)) or np.max(np.abs(scaled)) > 32767:
        raise ValueError("synthetic input would clip PCM16")
    return scaled.astype("<i2")


def known_components() -> dict[str, np.ndarray]:
    """Use one quantized reference and an exactly additive PCM near-end arm."""

    rng = np.random.default_rng(SEED)
    index = np.arange(SAMPLES)
    render = _pcm16(
        0.12 * rng.standard_normal(SAMPLES)
        + 0.04 * np.sin(2 * np.pi * 310 * index / RATE)
    )
    render_float = render.astype(np.float64) / 32768.0
    echo = _pcm16(np.convolve(render_float, np.asarray(PATH))[:SAMPLES])
    near = np.zeros(SAMPLES, dtype="<i2")
    start, stop = NEAR_SCORE
    near[start:stop] = _pcm16(
        0.04 * rng.standard_normal(stop - start)
        + 0.03 * np.sin(2 * np.pi * 503 * np.arange(stop - start) / RATE)
    )
    mixed = checked_pcm_add(echo, near)
    if SAMPLES % FRAME or np.any(near[:start]) or np.any(near[stop:]):
        raise AssertionError("known near-end interval or frame count changed")
    return {"render": render, "echo": echo, "near": near, "mixed": mixed}


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def echo_to_output_ratio_db(echo: np.ndarray, output: np.ndarray) -> float | None:
    """Power ratio on one far-only interval; null means ideal infinity."""

    first = np.asarray(echo, dtype=np.float64)
    second = np.asarray(output, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("equal-length, nonempty 1-D arrays required")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("finite values required")
    scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))))
    if scale <= 0:
        raise ValueError("echo and output are both silent")
    with np.errstate(over="raise", invalid="raise"):
        before = float(np.mean((first / scale) ** 2))
        after = float(np.mean((second / scale) ** 2))
    if before <= 0:
        raise ValueError("echo has zero power")
    return float(10.0 * np.log10(before / after)) if after > 0 else None


def _score(base: np.ndarray, injected: np.ndarray, known_near: np.ndarray,
           echo: np.ndarray, *, output_kind: str, lag_samples: int) -> dict:
    if base.shape != injected.shape or base.shape != known_near.shape or base.shape != echo.shape:
        raise ValueError("outputs and components must have the same shape")
    if type(lag_samples) is not int or not 0 <= lag_samples < SAMPLES - NEAR_SCORE[1]:
        raise ValueError("invalid independently measured output lag")
    a, b = FAR_SCORE
    c, d = NEAR_SCORE
    aligned_base = base[lag_samples:] if lag_samples else base
    aligned_injected = injected[lag_samples:] if lag_samples else injected
    aligned_near = known_near[:-lag_samples] if lag_samples else known_near
    aligned_echo = echo[:-lag_samples] if lag_samples else echo
    result = {
        "output_kind": output_kind,
        "lag_samples_from_independent_impulse": lag_samples,
        "same_sample_without_delay_compensation": {
            "far_only_echo_to_total_output_power_ratio_db": echo_to_output_ratio_db(
                echo[a:b], base[a:b]
            ),
            "known_near_injection_increment": increment_metrics(
                base, injected, known_near, c, d, sample_unit="pcm_counts"
            ),
        },
        "fixed_lag_compensated": {
            "far_only_echo_to_total_output_power_ratio_db": echo_to_output_ratio_db(
                aligned_echo[a:b], aligned_base[a:b]
            ),
            "known_near_injection_increment": increment_metrics(
                aligned_base, aligned_injected, aligned_near, c, d,
                sample_unit="pcm_counts",
            ),
        },
    }
    return result


def _verify_binary(path: Path, expected_digest: str, label: str) -> tuple[Path, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"{label} SHA-256 differs from the documented local build")
    return resolved, digest


def _run_aec3(binary: Path, render: np.ndarray, microphone: np.ndarray,
              directory: Path, label: str) -> dict[str, np.ndarray]:
    mic = directory / f"{label}_mic.wav"
    far = directory / f"{label}_render.wav"
    final = directory / f"{label}_final.wav"
    linear = directory / f"{label}_linear.wav"
    order = directory / "call_order_rc.txt"
    if not order.exists():
        order.write_text("rc\n")
    write_wav(mic, microphone)
    write_wav(far, render)
    invocation = command(binary, mic, far, final, linear, order)
    completed = subprocess.run(invocation, cwd=directory, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"AEC3 {label} failed with code {completed.returncode}: "
            f"{completed.stderr[-1500:]}"
        )
    return {"linear": read_wav(linear, SAMPLES), "final": read_wav(final, SAMPLES)}


def _impulse_lag(output: np.ndarray, input_index: int) -> int:
    """Find exact PCM support onset in a separate no-reference pulse probe."""

    nonzero = np.flatnonzero(output)
    if nonzero.size == 0:
        raise ValueError("independent impulse probe produced silent output")
    first = int(nonzero[0])
    peak = int(np.argmax(np.abs(output.astype(np.int32))))
    if first != peak or not 0 <= first - input_index <= FRAME:
        raise ValueError("impulse response does not have an unambiguous short onset lag")
    return first - input_index


def run(speex_library: Path, audioproc: Path) -> dict:
    """Run all processors from fresh state on identical sample-indexed input."""

    speex_library, speex_digest = _verify_binary(
        speex_library, SPEEX_SHA256, "SpeexDSP library"
    )
    audioproc, aec3_digest = _verify_binary(audioproc, AEC3_SHA256, "AEC3 audioproc_f")
    components = known_components()
    render = components["render"]
    echo = components["echo"]
    near = components["near"]
    mixed = components["mixed"]

    freeze = np.zeros(SAMPLES, dtype=bool)
    freeze[NEAR_SCORE[0]:NEAR_SCORE[1]] = True
    nlms_base, _, _ = nlms(
        render.astype(float) / 32768.0, echo.astype(float) / 32768.0,
        32, step_size=0.4, freeze=freeze,
    )
    nlms_mixed, _, _ = nlms(
        render.astype(float) / 32768.0, mixed.astype(float) / 32768.0,
        32, step_size=0.4, freeze=freeze,
    )
    nlms_score = _score(
        nlms_base * 32768.0, nlms_mixed * 32768.0, near, echo,
        output_kind="teaching NLMS prior linear residual; known near-end oracle freeze",
        lag_samples=0,
    )

    speex_base, rate_base = speex_linear_aec(speex_library, render, echo)
    speex_mixed, rate_mixed = speex_linear_aec(speex_library, render, mixed)
    if rate_base != RATE or rate_mixed != RATE:
        raise RuntimeError("SpeexDSP sampling rate differs between runs")
    zero = np.zeros(SAMPLES, dtype="<i2")
    impulse = zero.copy()
    impulse_index = 3 * RATE + 37
    impulse[impulse_index] = 10000
    speex_impulse, impulse_rate = speex_linear_aec(speex_library, zero, impulse)
    if impulse_rate != RATE:
        raise RuntimeError("SpeexDSP impulse probe sampling rate differed")
    speex_lag = _impulse_lag(speex_impulse, impulse_index)
    speex_score = _score(
        speex_base, speex_mixed, near, echo,
        output_kind="SpeexDSP synchronous AUMDF core PCM output; no external RES preprocessor",
        lag_samples=speex_lag,
    )
    speex_score["pcm_output_sha256"] = {
        "base": _digest(speex_base), "near_injected": _digest(speex_mixed)
    }

    with tempfile.TemporaryDirectory(prefix="aec_same_input_") as temporary:
        directory = Path(temporary)
        aec3_base = _run_aec3(audioproc, render, echo, directory, "base")
        aec3_mixed = _run_aec3(audioproc, render, mixed, directory, "near")
        aec3_impulse = _run_aec3(audioproc, zero, impulse, directory, "impulse")
    aec3_scores = {}
    for tap in ("linear", "final"):
        lag = _impulse_lag(aec3_impulse[tap], impulse_index)
        score = _score(
            aec3_base[tap], aec3_mixed[tap], near, echo,
            output_kind=f"WebRTC AEC3 {tap} exported PCM output",
            lag_samples=lag,
        )
        score["pcm_output_sha256"] = {
            "base": _digest(aec3_base[tap]), "near_injected": _digest(aec3_mixed[tap])
        }
        aec3_scores[tap] = score

    return {
        "fixture": "one synthetic 16 kHz PCM16 reference; fixed 5-tap linear echo and independent known near-end injection; no room or speech recording",
        "seed": SEED,
        "samples": SAMPLES,
        "frame_samples": FRAME,
        "path_current_first": list(PATH),
        "echo_component": "echo is quantized from convolution of the quantized render; no background noise",
        "near_component": "independent seeded noise plus 503 Hz sine; exactly additive after PCM quantization",
        "far_only_score_interval_half_open": list(FAR_SCORE),
        "near_injection_score_interval_half_open": list(NEAR_SCORE),
        "input_pcm_sha256": {key: _digest(value) for key, value in components.items()},
        "binary_sha256": {"speexdsp": speex_digest, "aec3_audioproc_f": aec3_digest},
        "delay_probe": {
            "render": "all zero",
            "capture": "one 10000-count PCM impulse at sample 48037, otherwise zero",
            "lag_rule": "first nonzero output sample minus input impulse index; strongest sample must equal the first",
            "probe_state": "fresh state and the same rc call order, sample rate and APM controls",
        },
        "processing": {
            "nlms": nlms_score,
            "speex_core": speex_score,
            "aec3": aec3_scores,
        },
        "interpretation": (
            "The far-only ratio compares known quantized echo power with total processor "
            "output on identical samples. Only NLMS is a pure linear residual here; Speex "
            "core and AEC3 outputs include implementation-specific processing. The near "
            "increment subtracts two separate runs and includes state changes. It is not "
            "isolated near-end speech or double-talk ERLE. No delay or gain is fitted; "
            "the separately measured short impulse lag is compensated in one additional "
            "set of metrics. This does not fit delay or gain on the near-end score interval. "
            "Output taps, oracle control and quantization differ, so the rows are not a "
            "uniform algorithm ranking."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speex-library", type=Path, required=True)
    parser.add_argument("--audioproc", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.speex_library, args.audioproc), ensure_ascii=False,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
