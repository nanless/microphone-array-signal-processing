"""Hard-linked existing assets cannot route regeneration into an external inode."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from codes.chapters.ch09.examples import moving_source_audio as moving
from codes.chapters.ch09.examples import chapter09_tracking_audio as tracking


def snapshot(directory):
    return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()}


class TrackingAssetLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.cases=[]
        for name,module,original in (('moving',moving,moving.OUT),('tracking',tracking,tracking.OUTPUT)):
            directory=cls.root/name
            module.generate(directory)
            cls.cases.append((name,module,original,directory))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_real_generation_keeps_all_five_wav_bytes_and_source_sets(self):
        count=0
        for name,module,original,directory in self.cases:
            with self.subTest(family=name):
                for wav in directory.glob('*.wav'):
                    self.assertEqual(wav.read_bytes(),(original/wav.name).read_bytes());count+=1
                metadata=module.generate(directory,check=True)
                self.assertEqual(set(metadata['source_sha256']),set(module.SOURCE_PATHS))
                expected = {'codes/chapters/ch09/examples/moving_source_audio.py',
                    'codes/chapters/ch09/core/moving_source.py',
                    'codes/chapters/ch00/core/audio_samples.py',
                    'codes/chapters/ch02/core/conventions.py',
                    'codes/chapters/ch00/io_contracts.py'} if name == 'moving' else {
                    'codes/chapters/ch09/examples/chapter09_tracking_audio.py',
                    'codes/chapters/ch09/core/tracking_audio.py',
                    'codes/chapters/ch09/core/moving_source.py',
                    'codes/chapters/ch09/core/tracking.py',
                    'codes/chapters/ch04/core/doa.py',
                    'codes/chapters/ch04/core/covariance.py',
                    'codes/chapters/ch02/core/conventions.py',
                    'codes/chapters/ch00/core/audio_samples.py',
                    'codes/chapters/ch00/io_contracts.py'}
                self.assertEqual(set(module.SOURCE_PATHS), expected)
                self.assertEqual(len(module.SOURCE_PATHS), 5 if name == 'moving' else 9)
        self.assertEqual(count,5)

    def test_normal_checks_do_not_change_content_or_mtime(self):
        for name,module,original,directory in self.cases:
            with self.subTest(family=name):
                before=snapshot(directory);module.generate(directory,check=True)
                self.assertEqual(snapshot(directory),before)

    def test_wav_and_manifest_hardlinks_rejected_before_generation_or_check(self):
        for name,module,original,directory in self.cases:
            for member in ('source.wav','MANIFEST.json'):
                with self.subTest(family=name,member=member):
                    testdir=self.root/(name+'-'+member);shutil.copytree(directory,testdir)
                    external=self.root/(name+'-'+member+'-external')
                    external.write_bytes(b'KEEP_EXTERNAL_SENTINEL')
                    leaf=testdir/member;leaf.unlink();os.link(external,leaf)
                    before=snapshot(testdir);external_before=(external.read_bytes(),external.stat().st_mtime_ns)
                    self.assertEqual(leaf.stat().st_nlink,2)
                    for checking in (False,True):
                        with self.assertRaises(ValueError):module.generate(testdir,check=checking)
                        self.assertEqual(snapshot(testdir),before)
                        self.assertEqual((external.read_bytes(),external.stat().st_mtime_ns),external_before)


if __name__=='__main__':unittest.main()
