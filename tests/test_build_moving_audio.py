"""移动声源媒体发布前必须核对三路文件与独立清单。"""

import shutil
import tempfile
from pathlib import Path
import unittest

from scripts import build_site


class MovingAudioStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "moving_audio"
        self.source.mkdir()
        for path in (build_site.ROOT / "codes/moving_audio").iterdir():
            if path.is_file():
                shutil.copy2(path, self.source / path.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_stages_three_waveforms_and_truth(self):
        destination = self.root / "published"
        names = build_site.stage_moving_audio(self.source, destination)
        self.assertEqual(names, set(build_site.MOVING_AUDIO_WAVS) | {"MANIFEST.json"})
        for name in names:
            self.assertEqual((destination / name).read_bytes(), (self.source / name).read_bytes())

    def test_rejects_changed_waveform(self):
        path = self.source / "moving_array.wav"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "摘要"):
            build_site.stage_moving_audio(self.source, self.root / "published")


class GssAudioStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "gss_audio"
        self.source.mkdir()
        for path in (build_site.ROOT / "codes/gss_audio").iterdir():
            if path.is_file():
                shutil.copy2(path, self.source / path.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_stages_waveforms_and_state(self):
        names = build_site.stage_gss_audio(self.source, self.root / "published")
        self.assertEqual(names, set(build_site.GSS_AUDIO_WAVS) | {"STATE.npz", "MANIFEST.json"})
        for name in names:
            self.assertEqual((self.root / "published" / name).read_bytes(),
                             (self.source / name).read_bytes())

    def test_rejects_changed_state(self):
        state = self.source / "STATE.npz"
        data = bytearray(state.read_bytes())
        data[-1] ^= 1
        state.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "摘要"):
            build_site.stage_gss_audio(self.source, self.root / "published")


if __name__ == "__main__":
    unittest.main()
