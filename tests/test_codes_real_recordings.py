"""Independent PCM, power and offline-provenance checks for real experiment R01."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.array_tutorial.real_recordings import (
    FILENAMES, average_pcm, experiment, pcm_wav, power_metrics, read_pcm_wav,
)
from codes.examples.prepare_real_recordings import (
    DEFAULT_OUTPUT, EXCERPT_PCM_SHA256, check, download_archive, verify_archive,
)


class RealRecordingTests(unittest.TestCase):
    def test_pcm_roundtrip_and_channel_order(self):
        source = np.array([[-32768, 32767], [7, -9]], dtype=np.int16)
        blob = pcm_wav(source)
        with wave.open(io.BytesIO(blob)) as stream:
            self.assertEqual(stream.getparams()[:4], (2, 2, 16000, 2))
            self.assertEqual(stream.readframes(2), b"\x00\x80\xff\x7f\x07\x00\xf7\xff")
        result, rate = read_pcm_wav(blob)
        np.testing.assert_array_equal(result, source)
        self.assertEqual(rate, 16000)

    def test_nearest_even_mean_and_no_integer_overflow(self):
        source = np.array([[0, 1], [1, 2], [-1, 0], [-2, -1], [32767, 32767]], dtype=np.int16)
        np.testing.assert_array_equal(average_pcm(source, 2)[:, 0], [0, 2, 0, -2, 32767])

    def test_hand_calculated_cross_terms(self):
        source = np.array([[-2, -2], [2, 2]], dtype=np.int16)
        result = power_metrics(source, 2)
        self.assertEqual(result["diagonal_only_power"], 2 / 32768**2)
        self.assertEqual(result["cross_terms_power"], 2 / 32768**2)
        self.assertEqual(result["exported_mean_power"], 4 / 32768**2)
        self.assertAlmostEqual(result["measured_minus_diagonal_db"], 3.010299956639812)
        self.assertEqual(result["mean_to_ch01_power_db"], 0)

    def test_dc_is_retained_not_covariance(self):
        source = np.full((4, 2), 3, dtype=np.int16)
        self.assertEqual(power_metrics(source, 2)["exported_mean_power"], 9 / 32768**2)

    def test_invalid_inputs(self):
        for source in (np.ones((2, 2)), np.ones((2, 2), dtype=complex),
                       np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=np.int32),
                       np.ones(3, dtype=np.int16), np.empty((0, 2), dtype=np.int16)):
            with self.assertRaises(ValueError):
                pcm_wav(source)
        x = np.ones((4, 2), dtype=np.int16)
        for channels in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                average_pcm(x, channels)
        for rate in (0, True, 16000.5):
            with self.assertRaises(ValueError):
                pcm_wav(x, rate)
        with self.assertRaises(ValueError):
            power_metrics(np.zeros((4, 2), dtype=np.int16), 2)
        with self.assertRaises(ValueError):
            power_metrics(np.array([[2, -2]], dtype=np.int16), 2)
        with self.assertRaises(ValueError):
            experiment(x)

    def test_truncated_wav(self):
        blob = pcm_wav(np.ones((10, 2), dtype=np.int16))
        with self.assertRaises(ValueError):
            read_pcm_wav(blob[:-2])

    def test_published_original_and_integer_arithmetic(self):
        with wave.open(str(DEFAULT_OUTPUT / FILENAMES[0])) as source:
            self.assertEqual(source.getparams()[:4], (16, 2, 16000, 160000))
            data = source.readframes(160000)
        self.assertEqual(hashlib.sha256(data).hexdigest(), EXCERPT_PCM_SHA256)
        x = np.frombuffer(data, dtype="<i2").reshape(-1, 16)
        self.assertGreater(int(x.min()), -32768)
        self.assertLess(int(x.max()), 32767)
        for name, count in zip(FILENAMES[1:], (1, 2, 16)):
            with wave.open(str(DEFAULT_OUTPUT / name)) as stream:
                actual = np.frombuffer(stream.readframes(160000), dtype="<i2")
            sums = x[:, :count].astype(np.int64).sum(axis=1)
            expected = np.rint(sums / count).astype(np.int16)
            np.testing.assert_array_equal(actual, expected)
            # Independent integer sum of squares, rather than covariance implementation.
            measured = sum(int(v)**2 for v in actual) / (len(actual) * 32768**2)
            if count > 1:
                metadata = json.loads((DEFAULT_OUTPUT / "MANIFEST.json").read_text())
                row = metadata["metrics"]["full_excerpt"][0 if count == 2 else 1]
                self.assertAlmostEqual(measured, row["exported_mean_power"], places=18)

    def test_offline_check_does_not_write_or_download(self):
        paths = list(DEFAULT_OUTPUT.iterdir())
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths if p.is_file()}
        with patch("urllib.request.urlopen", side_effect=AssertionError("network is forbidden")):
            self.assertEqual(check(DEFAULT_OUTPUT)["files"], 4)
        after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths if p.is_file()}
        self.assertEqual(before, after)

    def test_mismatched_cache_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.zip"
            path.write_bytes(b"not the pinned archive")
            with patch("urllib.request.urlopen", side_effect=AssertionError("network is forbidden")):
                with self.assertRaises(ValueError):
                    download_archive(path)
            self.assertEqual(path.read_bytes(), b"not the pinned archive")
            with self.assertRaises(ValueError):
                verify_archive(path)

    def test_corrupt_excerpt_and_metadata_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in FILENAMES:
                (output / name).write_bytes((DEFAULT_OUTPUT / name).read_bytes())
            manifest = (DEFAULT_OUTPUT / "MANIFEST.json").read_bytes()
            (output / "MANIFEST.json").write_bytes(manifest)
            broken = bytearray((output / FILENAMES[2]).read_bytes())
            broken[-1] ^= 1
            (output / FILENAMES[2]).write_bytes(broken)
            with self.assertRaises(ValueError):
                check(output)
            self.assertEqual((output / FILENAMES[2]).read_bytes(), broken)
            self.assertEqual((output / "MANIFEST.json").read_bytes(), manifest)
            (output / FILENAMES[2]).write_bytes((DEFAULT_OUTPUT / FILENAMES[2]).read_bytes())
            stale = json.loads(manifest)
            stale["processing"]["common_export_gain"] = 2
            stale_blob = json.dumps(stale).encode()
            (output / "MANIFEST.json").write_bytes(stale_blob)
            with self.assertRaises(ValueError):
                check(output)
            self.assertEqual((output / "MANIFEST.json").read_bytes(), stale_blob)


if __name__ == "__main__":
    unittest.main()
