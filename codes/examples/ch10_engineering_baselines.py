"""Run the deterministic Chapter 10 engineering examples."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codes.array_tutorial.engineering import (  # noqa: E402
    HysteresisVAD,
    PeakProtectAGC,
    RingBuffer,
    estimate_sro_ppm,
    q15_dequantize,
    q15_dot,
    q15_quantize,
    resample_sro_to_reference,
    simulate_deadline_queue,
)


def main() -> None:
    times = np.arange(0.0, 61.0, 10.0)
    delays = 0.002 + 80e-6 * times
    ppm, initial_delay = estimate_sro_ppm(times, delays)
    source = np.sin(2.0 * np.pi * np.arange(1601) / 80.0)
    aligned = resample_sro_to_reference(source, ppm)

    vad = HysteresisVAD(on_threshold=0.04, off_threshold=0.01, hangover_frames=2)
    vad_states = [vad.update(np.full(160, amplitude)) for amplitude in (0.1, 0.3, 0.05, 0.0, 0.0, 0.0)]

    agc = PeakProtectAGC(target_peak=0.8, max_gain=4.0)
    protected, gain = agc.process(np.array([-1.5, 0.4, 1.2]))

    ring = RingBuffer(capacity=4)
    ring.write(np.arange(6.0))
    retained = ring.read(4)

    schedule = simulate_deadline_queue(
        np.array([6.0, 7.0, 18.0, 18.0, 4.0]),
        frame_period_ms=10.0,
        capacity_frames=2,
    )
    q15 = q15_quantize(np.array([-1.2, -0.5, 0.5, 1.2]))
    q15_mac = q15_dot(np.array([16384, 16384], dtype=np.int16), np.array([16384, 16384], dtype=np.int16))

    print(f"SRO={ppm:.1f} ppm, initial delay={initial_delay * 1e3:.1f} ms")
    print(f"SRO-aligned length: {aligned.size} samples")
    print(f"VAD states: {vad_states}")
    print(f"AGC gain={gain:.4f}, protected peak={np.max(np.abs(protected)):.4f}")
    print(f"Ring retained={retained.tolist()}, dropped={ring.dropped}")
    print(f"Schedule={schedule}")
    print(f"Q15={q15.tolist()}, decoded={q15_dequantize(q15).tolist()}")
    print(f"Q15 wide-accumulator dot={int(q15_mac)} ({float(q15_dequantize(q15_mac)):.4f})")


if __name__ == "__main__":
    main()
