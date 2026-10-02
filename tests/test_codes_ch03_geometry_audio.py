"""Independent plane-wave, PCM readback and strictly read-only asset checks."""
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.chapters.ch03.core import geometry_audio as audio
from codes.chapters.ch03.examples import generate_geometry_audio as generator


class GeometryAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs,cls.manifest=generator.prepare_assets()

    def write_fixture(self,directory):
        directory=Path(directory)
        for name,blob in self.blobs.items():(directory/name).write_bytes(blob)
        (directory/'MANIFEST.json').write_text(json.dumps(self.manifest))
        return directory

    def test_three_files_formats_common_gain_source_set(self):
        self.assertEqual(set(self.blobs),{'geometry_reference.wav','geometry_u.wav','geometry_v.wav'})
        self.assertEqual(self.manifest['common_export_gain'],1)
        self.assertEqual(self.manifest['sample_rate_hz'],32000)
        self.assertEqual(len(self.manifest['source_sha256']),6)
        self.assertEqual(set(self.manifest['source_sha256']),set(generator.SOURCE_PATHS))
        for filename,blob in self.blobs.items():
            with wave.open(io.BytesIO(blob),'rb') as wav:
                self.assertEqual((wav.getframerate(),wav.getnframes(),wav.getsampwidth(),wav.getcomptype()),(32000,64000,2,'NONE'))
                self.assertEqual(wav.getnchannels(),1 if 'reference' in filename else 6)
            self.assertEqual(hashlib.sha256(blob).hexdigest(),self.manifest['files'][filename]['sha256'])

    def test_pcm_against_independent_continuous_triangle_projection(self):
        t=np.arange(64000)/32000;b=343/(math.sqrt(3)*.0375*8000);a=math.sqrt(1-b*b)
        for filename,blob in self.blobs.items():
            with wave.open(io.BytesIO(blob),'rb') as wav:
                channels=wav.getnchannels();actual=np.frombuffer(wav.readframes(64000),'<i2').reshape(64000,channels).T/32768
            for m in range(channels):
                if channels==1:shift=0
                else:
                    sign=1 if filename=='geometry_u.wav' else -1
                    # Explicit physical dot product, no geometry/delay helper.
                    shift=.0375*(math.cos(m*math.pi/3)*a+math.sin(m*math.pi/3)*sign*b)/343
                source_time=t-.002+shift
                envelope=np.minimum(np.clip((source_time-.1)/.02,0,1),np.clip((1.9-source_time)/.02,0,1))
                expected=.08*envelope*(np.sin(2*np.pi*4000*source_time)+np.sin(2*np.pi*8000*source_time))
                self.assertLessEqual(float(np.max(abs(actual[m]-expected))),.5/32768+1e-11)
                self.assertTrue(np.all(actual[m][:3000]==0))
                self.assertTrue(np.all(actual[m][62000:]==0))
            self.assertLess(float(np.max(abs(actual))),.15)

    def test_unknown_source_profile_from_six_sign_ratios(self):
        samples=self.manifest['samples']
        for name,other in [('direction_u','direction_v'),('direction_v','direction_u')]:
            floats=samples[name]['float_measurements'];pcm=samples[name]['pcm_measurements']
            self.assertEqual(pcm['scoring_interval_samples'],[4800,59200])
            self.assertEqual(pcm['samples_per_channel'],54400)
            for f,expected in zip(floats['tones'],[8/9,0]):
                np.testing.assert_allclose(f['amplitudes'],[.08]*6,atol=2e-13)
                profile=f['candidate_profiles'][other]
                self.assertAlmostEqual(profile['relative_residual'],expected,places=12)
                self.assertAlmostEqual(profile['observed_phasor_squared_sum'],6*.08**2,places=12)
            self.assertAlmostEqual(pcm['tones'][0]['candidate_profiles'][other]['relative_residual'],.888864418545076,places=12)
            self.assertAlmostEqual(pcm['tones'][1]['candidate_profiles'][other]['relative_residual'],3.9104611868e-9,delta=1e-15)
            self.assertLess(pcm['tones'][1]['candidate_profiles'][name]['relative_residual'],1e-8)

    def test_actual_published_assets_are_checked(self):
        manifest=generator.check_assets(generator.DEFAULT_OUTPUT)
        self.assertEqual(manifest,self.manifest)

    def test_check_reads_but_never_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=self.write_fixture(temporary)
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()}
            with patch.object(Path,'write_bytes',side_effect=AssertionError('check wrote bytes')),patch.object(Path,'write_text',side_effect=AssertionError('check wrote text')),patch.object(Path,'mkdir',side_effect=AssertionError('check mkdir')):
                generator.check_assets(directory)
            after={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()}
            self.assertEqual(before,after)

    def test_missing_extra_stale_source_and_fake_pcm_scores_fail(self):
        for kind in ('missing','extra','source','scores','wav','forged_sha'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                directory=self.write_fixture(temporary)
                manifest=json.loads((directory/'MANIFEST.json').read_text())
                if kind=='missing':(directory/'geometry_u.wav').unlink()
                elif kind=='extra':(directory/'unexpected.wav').write_bytes(b'extra')
                elif kind=='source':manifest['source_sha256'][generator.SOURCE_PATHS[0]]='0'*64
                elif kind=='scores':manifest['samples']['direction_u']['pcm_measurements']['tones'][0]['candidate_profiles']['direction_v']['relative_residual']=0
                else:
                    data=bytearray((directory/'geometry_u.wav').read_bytes());data[10000]^=64
                    (directory/'geometry_u.wav').write_bytes(data)
                    if kind=='forged_sha':manifest['files']['geometry_u.wav']['sha256']=hashlib.sha256(data).hexdigest()
                (directory/'MANIFEST.json').write_text(json.dumps(manifest))
                before={p.name:p.read_bytes() for p in directory.iterdir()}
                with self.assertRaises(ValueError):generator.check_assets(directory)
                self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir()})

    def test_scoring_rejects_invalid_reference_and_shapes(self):
        with self.assertRaises(ValueError):audio.measure_signal(np.zeros((6,64000)))
        with self.assertRaises(ValueError):audio.measure_signal(np.zeros((6,32000)))
        with self.assertRaises(ValueError):audio.measure_signal(np.zeros((6,64000),dtype=complex))


if __name__=='__main__':unittest.main()
