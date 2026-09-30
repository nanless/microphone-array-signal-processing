"""Independent rational/trigonometric oracles and published-asset checks."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.chapters.ch02.core import stft_convolution as core
from codes.chapters.ch02.examples import generate_stft_convolution as assets


class FiniteWindowConvolutionTests(unittest.TestCase):
    def test_six_point_outputs_from_fraction_sample_sums(self):
        # No FFT, WOLA helper or tested function creates the expected values.
        F = Fraction
        x, h, g = list(map(F, [1, 2, 3, 4, 0, 0])), [F(1), F(0), F(1, 2)], [F(0), F(1, 2), F(1), F(1, 2)]
        full = [sum((h[q]*x[n-q] for q in range(3) if 0 <= n-q < 6), F(0)) for n in range(6)]
        den, circ, lin = [[F(0)]*6 for _ in range(3)]
        for start in (-2, 0, 2, 4):
            frame = [(x[start+q] if 0 <= start+q < 6 else F(0))*g[q] for q in range(4)]
            for q in range(4):
                n = start+q
                if 0 <= n < 6:
                    den[n] += g[q]**2
                    circ[n] += g[q]*sum((h[r]*frame[(q-r) % 4] for r in range(3)), F(0))
                    lin[n] += g[q]*sum((h[r]*frame[q-r] for r in range(3) if q-r >= 0), F(0))
        result = core.finite_window_example()
        for key, expected in (("full_linear_output", full), ("window_squared_sum", den),
                              ("framewise_fft4_output", [v/d for v, d in zip(circ, den)]),
                              ("framewise_fft8_cropped_output", [v/d for v, d in zip(lin, den)])):
            np.testing.assert_allclose(result[key], list(map(float, expected)), rtol=0, atol=2e-15)
        np.testing.assert_allclose(result["identity_roundtrip"], list(map(float, x)), atol=1e-15)
        self.assertEqual(result["window_start_samples"], [-2, 0, 2, 4])
        np.testing.assert_allclose(result["frames"][1]["windowed_frame"], [0, 1, 3, 2], atol=1e-15)
        np.testing.assert_allclose(result["frames"][1]["linear8_full_frame_tail"], [0, 1, 3, 2.5, 1.5, 1], atol=1e-15)

    def test_identity_zero_filter_and_error_inputs(self):
        x = np.array([1., -.5, 2., 0., .25])
        for control in (False, True):
            np.testing.assert_allclose(core.framewise_filter(x, [1.], n_fft=4, hop_length=2,
                                                            remove_circular_folding=control), x, atol=1e-15)
            np.testing.assert_array_equal(core.framewise_filter(x, [0.], n_fft=4, hop_length=2,
                                                               remove_circular_folding=control), np.zeros(5))
        for x_bad, h_bad in (([1j], [1]), ([1], [1j]), ([], [1]), ([1], []),
                             ([float("nan")], [1]), ([[1]], [1]), ([1], [1]*5)):
            with self.subTest(x=x_bad, h=h_bad), self.assertRaises(ValueError):
                core.framewise_filter(x_bad, h_bad, n_fft=4, hop_length=2)
        for length, nfft, hop in ((0, 4, 2), (6, True, 2), (6, 4, True), (6, 4, 5)):
            with self.assertRaises(ValueError):
                core.synthesis_support(length, nfft, hop)
        for bad in ([0]*32320, [1j]*32320, [1]*10):
            with self.assertRaises(ValueError):
                core.measure_waveform([1]*32320, bad)


class ConvolutionAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.scratch.name)/"assets"
        cls.manifest = assets.generate_assets(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_source_full_filter_and_mtf_using_time_sample_operations(self):
        # Source is independently evaluated with scalar sin and explicit burst gates.
        source = []
        for n in range(32320):
            envelope = 0.
            for start, stop in ((3200, 6400), (11200, 15200), (19200, 23200), (26400, 29600)):
                if start <= n < stop:
                    envelope = min(1., (n-start)/320., (stop-1-n)/320.)
            source.append(envelope*math.fsum(.06*math.sin(2*math.pi*f*n/16000)
                                            for f in (440, 997, 1733, 2819)))
        source = np.array(source)
        exact = source.copy()
        exact[320:] += .5*source[:-320]
        # Circular shift of each windowed frame is the independent inverse-DFT
        # identity for H=1+.5*exp(-j*2*pi*k*320/512); no FFT is used here.
        g = np.array([.5-.5*math.cos(2*math.pi*q/512) for q in range(512)])
        den, numerator = np.zeros(32320), np.zeros(32320)
        for start in range(-256, 32129, 128):
            frame = np.array([source[n] if 0 <= n < 32320 else 0. for n in range(start, start+512)])*g
            filtered = frame+.5*np.array([frame[(q-320) % 512] for q in range(512)])
            for q in range(512):
                n = start+q
                if 0 <= n < 32320:
                    numerator[n] += g[q]*filtered[q]
                    den[n] += g[q]**2
        mtf = numerator/den
        generated = core.build_convolution_audio()
        for name, expected in (("stft_roundtrip", source), ("full_convolution", exact), ("framewise_mtf", mtf)):
            np.testing.assert_allclose(generated["signals"][name], expected, rtol=0, atol=5e-14)
            payload = (self.directory/(name+".wav")).read_bytes()
            with wave.open(str(self.directory/(name+".wav")), "rb") as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth(), wav.getcomptype()),
                                 (16000, 1, 32320, 2, "NONE"))
                pcm = np.frombuffer(wav.readframes(32320), dtype="<i2").astype(float)/32768
            np.testing.assert_allclose(pcm, expected, rtol=0, atol=.5/32768+5e-14)
            row = self.manifest["samples"][name]["pcm_measurements"]["against_full_convolution"]
            self.assertAlmostEqual(row["energy"], math.fsum(float(v*v) for v in pcm), places=12)
            self.assertEqual(row["denominator_samples"], 32320)
            self.assertEqual(self.manifest["files"][name+".wav"]["sha256"], hashlib.sha256(payload).hexdigest())
        np.testing.assert_allclose(generated["synthesis_denominator"], den, atol=1e-15)
        self.assertGreater(abs(exact[29918]), 1e-6)  # retained late reflection sample
        self.assertEqual(source[29918], 0)
        self.assertEqual(self.manifest["common_export_gain"], 1.)
        self.assertLess(self.manifest["unmodified_roundtrip_against_input"]["max_abs_error"], 1e-14)

    def test_check_is_read_only_and_rejects_missing_extra_tampered(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        with patch.object(Path, "write_bytes", side_effect=AssertionError("check wrote bytes")), \
             patch.object(Path, "write_text", side_effect=AssertionError("check wrote text")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("check created directory")):
            assets.check_assets(self.directory)
            with self.assertRaises(ValueError):
                assets.check_assets(self.directory/"missing")
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        self.assertEqual(before, after)
        for mode in ("extra", "missing", "wav", "manifest"):
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/"case"
                shutil.copytree(self.directory, directory)
                if mode == "extra":
                    (directory/"unexpected.txt").write_text("extra")
                elif mode == "missing":
                    (directory/"framewise_mtf.wav").unlink()
                elif mode == "wav":
                    path = directory/"framewise_mtf.wav"
                    changed = bytearray(path.read_bytes()); changed[44+10000] ^= 1
                    path.write_bytes(changed)
                    # A forged matching file SHA does not make changed PCM valid.
                    manifest = json.loads((directory/"MANIFEST.json").read_text())
                    manifest["files"][path.name]["sha256"] = hashlib.sha256(changed).hexdigest()
                    (directory/"MANIFEST.json").write_text(json.dumps(manifest))
                else:
                    manifest = json.loads((directory/"MANIFEST.json").read_text())
                    manifest["source_sha256"][assets.SOURCE_PATHS[0]] = "0"*64
                    (directory/"MANIFEST.json").write_text(json.dumps(manifest))
                snapshot = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    assets.check_assets(directory)
                self.assertEqual(snapshot, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()})

    def test_published_current_set(self):
        manifest = assets.check_assets()
        self.assertEqual(set(manifest["files"]), {"stft_roundtrip.wav", "full_convolution.wav", "framewise_mtf.wav"})
        self.assertEqual(set(manifest["source_sha256"]), set(assets.SOURCE_PATHS))


if __name__ == "__main__":
    unittest.main()
