"""R01: inspect a fixed real DEMAND excerpt without a clean-speech reference.

Input is signed PCM16, shape (samples, channels). Powers are *uncentered*
second moments in full-scale-squared units. No gain calibration, DC removal,
time alignment, or signal/noise decomposition is performed.
"""
from __future__ import annotations

import io
import wave
import numpy as np

SAMPLE_RATE = 16000
SAMPLES = 160000
CHANNELS = 16
FILENAMES = (
    "demand_nriver_16ch_10s.wav",
    "demand_nriver_ch01_10s.wav",
    "demand_nriver_mean02_10s.wav",
    "demand_nriver_mean16_10s.wav",
)


def validate_pcm(pcm: np.ndarray) -> np.ndarray:
    """Require a nonempty two-dimensional signed-16-bit PCM array."""
    x = np.asarray(pcm)
    if x.dtype.kind != "i" or x.dtype.itemsize != 2:
        raise ValueError("input must be signed PCM16; floats, bools and complex are invalid")
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("input must have nonempty (samples, channels) shape")
    return x


def pcm_wav(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode PCM without amplitude changes; channel order is preserved."""
    x = validate_pcm(pcm)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(x.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(x.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()


def read_pcm_wav(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode uncompressed PCM16 and reject truncated sample payloads."""
    with wave.open(io.BytesIO(blob), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
            raise ValueError("expected uncompressed PCM16")
        channels, frames, rate = stream.getnchannels(), stream.getnframes(), stream.getframerate()
        data = stream.readframes(frames)
    if channels < 1 or frames < 1 or len(data) != 2 * channels * frames:
        raise ValueError("empty or truncated WAV")
    return np.frombuffer(data, dtype="<i2").reshape(frames, channels).copy(), rate


def average_pcm(pcm: np.ndarray, channels: int) -> np.ndarray:
    """Equal-weight zero-delay average; nearest-even PCM quantization once."""
    x = validate_pcm(pcm)
    if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= x.shape[1]:
        raise ValueError("channels must select a nonempty input prefix")
    # Accumulate in float64, never int16. A convex mean cannot exceed PCM range.
    return np.rint(np.mean(x[:, :channels], axis=1, dtype=np.float64)).astype("<i2")[:, None]


def power_metrics(pcm: np.ndarray, channels: int) -> dict:
    """Compare measured mean power to a diagonal-only second-moment model.

    Actual power includes cross terms. The diagonal prediction retains each
    channel's actual second moment, so it does not assume equal microphone gain.
    Independence alone does not remove cross moments when channel means differ
    from zero; the data are deliberately not centered here.
    The difference still does not separate acoustic correlation from hardware.
    """
    x = validate_pcm(pcm)
    exported = average_pcm(x, channels)
    samples = x[:, :channels].astype(np.float64) / 32768.0
    second = samples.T @ samples / len(samples)
    diagonal = float(np.trace(second) / channels**2)
    exact = float(np.sum(second) / channels**2)
    actual = float(np.mean((exported[:, 0].astype(np.float64) / 32768.0)**2))
    if diagonal <= 0 or exact <= 0 or actual <= 0:
        raise ValueError("positive input and output power required for dB ratios")
    single = float(second[0, 0])
    if single <= 0:
        raise ValueError("channel 1 must have positive power")
    return {
        "channels": channels,
        "mean_channel_power": float(np.trace(second) / channels),
        "diagonal_only_power": diagonal,
        "cross_terms_power": exact - diagonal,
        "unquantized_mean_power": exact,
        "exported_mean_power": actual,
        "measured_minus_diagonal_db": float(10 * np.log10(actual / diagonal)),
        "mean_to_ch01_power_db": float(10 * np.log10(actual / single)),
        "max_quantization_error_pcm": float(np.max(np.abs(
            exported[:, 0].astype(float) - np.mean(x[:, :channels], axis=1, dtype=float)))),
    }


def experiment(pcm: np.ndarray) -> tuple[dict[str, bytes], dict]:
    """Create the four fixed R01 files and ten non-overlapping one-second results."""
    x = validate_pcm(pcm)
    if x.shape != (SAMPLES, CHANNELS):
        raise ValueError(f"R01 expects exactly {(SAMPLES, CHANNELS)}")
    exports = (x, x[:, :1], average_pcm(x, 2), average_pcm(x, 16))
    files = {name: pcm_wav(data) for name, data in zip(FILENAMES, exports)}
    metrics = {
        "exercise_id": "R01",
        "power_definition": "mean(x**2); PCM16 divided by 32768; DC retained",
        "full_excerpt": [power_metrics(x, n) for n in (2, 16)],
        "one_second_blocks": [
            {"start_sample": start, "stop_sample": start + SAMPLE_RATE,
             "results": [power_metrics(x[start:start + SAMPLE_RATE], n) for n in (2, 16)]}
            for start in range(0, SAMPLES, SAMPLE_RATE)
        ],
        "block_interpretation": "ten observed time blocks, not independent trials or confidence intervals",
    }
    return files, metrics
