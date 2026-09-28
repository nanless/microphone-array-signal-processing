"""Independent media publication refuses stale or substituted tracking inputs."""
import hashlib
import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from scripts import build_site


class TrackingAudioStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        shutil.copytree(build_site.ROOT/'codes/chapters/ch09/tracking_audio',self.source)
        self.destination = self.root/'published'

    def test_exact_assets_and_bytes(self):
        names = build_site.stage_tracking_audio(self.source,self.destination)
        self.assertEqual(names,{'source.wav','array_noisy.wav','MANIFEST.json'})
        for name in names:
            self.assertEqual((self.source/name).read_bytes(),(self.destination/name).read_bytes())

    def test_missing_extra_directory_and_symlink_rejected(self):
        for failure in ('missing','extra','directory','symlink'):
            with self.subTest(failure=failure):
                case = self.root/failure
                shutil.copytree(self.source,case)
                if failure=='missing': (case/'source.wav').unlink()
                elif failure=='extra': (case/'extra').write_text('unexpected')
                elif failure=='directory': (case/'extra').mkdir()
                else:
                    (case/'source.wav').unlink()
                    (case/'source.wav').symlink_to(self.source/'source.wav')
                with self.assertRaises(ValueError):
                    build_site.stage_tracking_audio(case,self.destination)
                self.assertFalse(self.destination.exists())

    def test_changed_pcm_is_rejected(self):
        path=self.source/'array_noisy.wav'
        data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'摘要'):
            build_site.stage_tracking_audio(self.source,self.destination)
        self.assertFalse(self.destination.exists())

    def test_matching_hash_does_not_hide_wrong_pcm_format(self):
        path=self.source/'array_noisy.wav'
        with wave.open(str(path),'wb') as wav:
            wav.setparams((1,2,16000,32000,'NONE','not compressed'))
            wav.writeframes(bytes(64000))
        manifest_path=self.source/'MANIFEST.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['files'][path.name]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'PCM'):
            build_site.stage_tracking_audio(self.source,self.destination)
        self.assertFalse(self.destination.exists())
