"""Canonical chapter locations for the book's generated audio assets.

The main synthesis has one cross-chapter manifest in ch00, while each WAV lives
beside the chapter that teaches the corresponding experiment. Published site
URLs remain flat under ``site/audio/`` for existing readers.
"""

from pathlib import Path


MAIN_AUDIO_GROUP_CHAPTER = {
    "spatial": "ch01",
    "alignment_error": "ch02",
    "interpolation": "ch02",
    "dma_calibration": "ch03",
    "polarity": "ch03",
    "fractional_array": "ch04",
    "doa_ambiguity": "ch04",
    "gsc_gate": "ch05",
    "correlation": "ch05",
    "aec": "ch06",
    "aec_methods": "ch06",
    "aec_subband": "ch06",
    "aec_dropout": "ch06",
    "nonlinear": "ch06",
    "wpe": "ch07",
    "wpe_predictable": "ch07",
    "room_decay": "ch07",
    "separation": "ch08",
    "css_overlap": "ch08",
    "tracking": "ch09",
    "engineering": "ch10",
    "clock_drift": "ch10",
    "spectral_subtraction": "ch10",
    "agc_blocks": "ch10",
    "selection_tradeoff": "ch11",
    "conditioning": "appendix_a",
    "math_block": "appendix_a",
}


def main_audio_path(chapters_root: Path, group: str, filename: str) -> Path:
    """Return the sole chapter-owned location for one main-synthesis WAV."""
    if not filename.endswith(".wav") or Path(filename).name != filename:
        raise ValueError("audio filename must be a bare WAV name")
    return chapters_root / MAIN_AUDIO_GROUP_CHAPTER[group] / "audio" / filename


def main_audio_manifest_path(chapters_root: Path) -> Path:
    return chapters_root / "ch00" / "audio" / "MANIFEST.json"
