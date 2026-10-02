"""Independent real projections and actual PCM checks for known focusing."""
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
from codes.chapters.ch04.core import focus_audio as audio
from codes.chapters.ch04.examples import generate_focus_audio as generator


class FocusAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs, cls.manifest = generator.prepare_assets()

    def fixture(self, directory):
        root = Path(directory)
        for name, blob in self.blobs.items():
            (root/name).write_bytes(blob)
        (root/'MANIFEST.json').write_text(json.dumps(self.manifest))
        return root

    @staticmethod
    def pcm(blob):
        with wave.open(io.BytesIO(blob), 'rb') as w:
            assert (w.getframerate(), w.getnframes(), w.getsampwidth(), w.getcomptype()) == (16000,32024,2,'NONE')
            channels = w.getnchannels()
            return np.frombuffer(w.readframes(32024), '<i2').reshape(32024,channels).T/32768

    def test_fixed_file_source_and_common_gain_contract(self):
        self.assertEqual(set(self.blobs), {'focus_reference.wav','focus_delayed_source.wav',
                                          'focus_array.wav','focus_known_focused.wav'})
        self.assertEqual(set(self.manifest['source_sha256']), {
            'codes/chapters/ch04/core/focus_audio.py', 'codes/chapters/ch04/examples/generate_focus_audio.py',
            'codes/chapters/ch03/core/geometry.py', 'codes/chapters/ch02/core/conventions.py',
            'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py'})
        self.assertEqual(self.manifest['common_export_gain'], 1)
        for filename, blob in self.blobs.items():
            self.assertEqual(self.pcm(blob).shape[0], 4 if filename in ('focus_array.wav','focus_known_focused.wav') else 1)
            self.assertEqual(hashlib.sha256(blob).hexdigest(), self.manifest['files'][filename]['sha256'])

    def test_independent_continuous_source_and_complete_propagation_tail(self):
        n = np.arange(32024)
        def contribution(delay, frequency):
            k = n-delay
            valid = (k>=0)&(k<32000)
            env = np.zeros(len(n)); env[valid]=1
            first = valid&(k<320); last = valid&(k>=31680)
            env[first] = np.sin(np.pi*k[first]/638)**2
            env[last] = np.sin(np.pi*(31999-k[last])/638)**2
            return .08*env*np.cos(2*np.pi*frequency*k/16000)
        for filename, blob in self.blobs.items():
            actual=self.pcm(blob)
            for m in range(len(actual)):
                if filename=='focus_reference.wav': expected=sum(contribution(0,f) for f in (1000,3000))
                elif filename=='focus_delayed_source.wav': expected=sum(contribution(4,f) for f in (1000,3000))
                else:
                    m3 = [0,3,2,1][m] if filename=='focus_known_focused.wav' else m
                    expected=(contribution(12+4*m,1000)+contribution(16-4*m,1000)
                              +contribution(12+4*m3,3000)+contribution(16-4*m3,3000))
                self.assertLessEqual(float(np.max(abs(expected-actual[m]))), .5/32768+2e-12)
        reference=self.pcm(self.blobs['focus_reference.wav'])[0]
        delayed=self.pcm(self.blobs['focus_delayed_source.wav'])[0]
        np.testing.assert_array_equal(delayed[4:], reference[:-4])

    def test_ideal_rank_and_pcm_eigenvalues_from_independent_two_vector_gram(self):
        for name, expected in [('array',[0,0,0,8]),('known_focused',[0,0,4,4])]:
            row=self.manifest['samples'][name]
            np.testing.assert_allclose(row['float_measurements']['pooled']['eigenvalues'], expected, atol=3e-12)
            actual=self.pcm(self.blobs[row['file']])[:,2400:29600]
            n=np.arange(2400,29600); N=27200
            # Orthogonal real projections: each column norm squared is N/2.
            z=[]
            for f in (1000,3000):
                phase=2*np.pi*f*n/16000
                z.append((actual@np.cos(phase)-1j*(actual@np.sin(phase)))*2/(N*.08))
            # Nonzero eigenvalues of (vv^H+ww^H)/2 from its 2x2 Gram.
            a=sum(abs(x)**2 for x in z[0])/2; b=sum(abs(x)**2 for x in z[1])/2
            c=sum(x.conjugate()*y for x,y in zip(z[0],z[1]))/2
            gap=math.sqrt((a-b)**2+4*abs(c)**2)
            eig=[0,0,(a+b-gap)/2,(a+b+gap)/2]
            np.testing.assert_allclose(row['pcm_measurements']['pooled']['eigenvalues'], eig, atol=5e-11)
            self.assertEqual(row['pcm_measurements']['pooled']['rank'],1 if name=='array' else 2)
            self.assertEqual(row['pcm_measurements']['time_domain_sample_denominator'],108800)
            self.assertEqual(row['pcm_measurements']['pooled']['frequency_average_denominator'],2)
            self.assertLessEqual(row['quantization_max_abs_error'],.5/32768)

    def test_published_assets_checked_and_check_never_writes(self):
        self.assertEqual(generator.check_assets(generator.DEFAULT_OUTPUT),self.manifest)
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(temp)
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()}
            with patch.object(Path,'write_bytes',side_effect=AssertionError('write')),patch.object(Path,'write_text',side_effect=AssertionError('write')),patch.object(Path,'mkdir',side_effect=AssertionError('mkdir')):
                generator.check_assets(root)
            self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_missing_extra_source_score_pcm_and_forged_sha_fail_without_repair(self):
        for kind in ('missing','extra','source','score','pcm','forged_sha'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=self.fixture(temp); m=json.loads((root/'MANIFEST.json').read_text())
                if kind=='missing': (root/'focus_array.wav').unlink()
                elif kind=='extra': (root/'extra.wav').write_bytes(b'extra')
                elif kind=='source': m['source_sha256'][generator.SOURCE_PATHS[0]]='0'*64
                elif kind=='score': m['samples']['array']['pcm_measurements']['pooled']['rank']=2
                else:
                    data=bytearray((root/'focus_array.wav').read_bytes());data[10000]^=64
                    (root/'focus_array.wav').write_bytes(data)
                    if kind=='forged_sha':m['files']['focus_array.wav']['sha256']=hashlib.sha256(data).hexdigest()
                (root/'MANIFEST.json').write_text(json.dumps(m))
                before={p.name:p.read_bytes() for p in root.iterdir()}
                with self.assertRaises(ValueError): generator.check_assets(root)
                self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_shape_zero_energy_and_complex_input_rejected(self):
        for x in (np.zeros((4,32024)),np.zeros((3,32024)),np.zeros((4,32024),complex)):
            with self.assertRaises(ValueError): audio.measure_signal(x)


if __name__=='__main__': unittest.main()
