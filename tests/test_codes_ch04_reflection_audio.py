"""Independent finite-tone, quarter-cycle, PCM integer and filesystem controls."""
import copy
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
from codes.chapters.ch04.core import reflection_audio as audio
from codes.chapters.ch04.examples import generate_reflection_audio as generator


class ReflectionAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs, cls.manifest = generator.prepare_assets()

    def fixture(self, directory):
        root = Path(directory)
        for name, blob in self.blobs.items(): (root/name).write_bytes(blob)
        (root/'MANIFEST.json').write_text(json.dumps(self.manifest))
        return root

    def integer_pcm(self, blob):
        with wave.open(io.BytesIO(blob), 'rb') as reader:
            self.assertEqual((reader.getframerate(), reader.getnframes(), reader.getsampwidth(),
                              reader.getcomptype()), (16000, 32012, 2, 'NONE'))
            channels = reader.getnchannels()
            return np.frombuffer(reader.readframes(32012), '<i2').reshape(-1, channels).T.astype(np.int64)

    def test_numerically_equal_types_still_rejected(self):
        for key, value in audio.parameters().items():
            if type(value) not in (int, float): continue
            replacement = float(value) if type(value) is int else int(value)
            if replacement != value: continue
            bad = audio.parameters(); bad[key] = replacement
            with self.subTest(key=key), patch.object(generator, 'parameters', return_value=bad):
                with self.assertRaises(ValueError): generator.prepare_assets()
        for key in ('sample_rate_hz', 'source_samples'):
            bad = audio.parameters(); bad[key] = str(bad[key])
            with self.subTest(key=key), patch.object(generator, 'parameters', return_value=bad):
                with self.assertRaises(ValueError): generator.prepare_assets()

    def test_full_finite_source_fade_delays_and_complete_tail(self):
        n = np.arange(32012)
        def independently_delayed(delay):
            k = n-delay
            env = np.where((k>=0)&(k<32000), 1., 0.)
            first = (k>=0)&(k<320); last = (k>=31680)&(k<32000)
            env[first] = np.sin(np.pi*k[first]/638)**2
            env[last] = np.sin(np.pi*(31999-k[last])/638)**2
            # Exact four-point cos sequence; no production source implementation.
            return .1*env*np.array([1., 0., -1., 0.])[k%4]
        truth = {'reference': [independently_delayed(0)],
                 'direct': [independently_delayed(8)]*2,
                 'reflection': [independently_delayed(12), independently_delayed(11)],
                 'mixed': [independently_delayed(8)+independently_delayed(12),
                           independently_delayed(8)+independently_delayed(11)]}
        for name, row in self.manifest['samples'].items():
            actual = self.integer_pcm(self.blobs[row['file']])/32768
            self.assertEqual(actual.shape, (1 if name=='reference' else 2, 32012))
            self.assertLessEqual(float(np.max(abs(actual-truth[name]))), .5/32768+2e-12)
            self.assertEqual(hashlib.sha256(self.blobs[row['file']]).hexdigest(),
                             self.manifest['files'][row['file']]['sha256'])
        self.assertEqual(len(self.manifest['source_sha256']), 5)
        for path, sha in self.manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((generator.ROOT/path).read_bytes()).hexdigest(), sha)

    def test_independent_quarter_cycle_phasors_and_exact_pcm_integer_denominators(self):
        vectors = {'reference': np.array([1]), 'direct': np.array([1,1]),
                   'reflection': np.array([1,1j]), 'mixed': np.array([2,1+1j])}
        E = {'reference': [146046714400], 'direct': [146046714400]*2,
             'reflection': [146046714400]*2, 'mixed': [584186857600,292093428800]}
        for name, row in self.manifest['samples'].items():
            pcm = self.integer_pcm(self.blobs[row['file']])[:,2400:29600]
            # Four integer phases have exactly 6800 repetitions.
            z = (pcm[:,0::4].mean(axis=1)-pcm[:,2::4].mean(axis=1)
                 +1j*(pcm[:,3::4].mean(axis=1)-pcm[:,1::4].mean(axis=1)))/65536
            expected = 3277/32768*vectors[name]
            np.testing.assert_array_equal(z, expected)
            np.testing.assert_allclose(row['pcm_measurements']['phasor_real_imag'],
                                       np.stack((z.real,z.imag),axis=-1), atol=2e-13)
            np.testing.assert_allclose(row['float_measurements']['phasor_real_imag'],
                                       np.stack((.1*vectors[name].real,.1*vectors[name].imag),axis=-1), atol=2e-13)
            integers = row['pcm_integer_measurements']
            self.assertEqual([int(np.sum(channel**2)) for channel in pcm], E[name])
            self.assertEqual(integers['integer_squared_sum_E_per_channel'], E[name])
            self.assertEqual(integers['integer_denominator_D_per_channel'], 29205777612800)
            self.assertEqual(integers['integer_squared_sum_E_all_channels'], sum(E[name]))
            self.assertEqual(integers['integer_denominator_D_all_channels'], len(pcm)*29205777612800)
            self.assertEqual(integers['mean_square_all_channels'],sum(E[name])/(len(pcm)*29205777612800))
            self.assertEqual(integers['samples_per_channel'],27200)
        mixed = self.manifest['samples']['mixed']
        for domain in ('analytic','float_measurements','pcm_measurements'):
            self.assertAlmostEqual(mixed[domain]['amplitude_ratio'],1/math.sqrt(2),12)
            np.testing.assert_allclose(mixed[domain]['channel_1_over_channel_0_real_imag'],[.5,.5],atol=2e-12)
            self.assertAlmostEqual(mixed[domain]['phase_only_apparent_angle_deg'],14.477512185929925,10)
        virtual = self.manifest['virtual_covariance_control']
        np.testing.assert_allclose(virtual['eigenvalues'],[.1,6.1],atol=1e-14)
        self.assertEqual(self.manifest['parameters']['noise'],'none')
        self.assertIn('virtual',virtual['scope'])

    def test_fixed_true_parameters_fail_before_filesystem_mutation(self):
        def leaves(value,path=()):
            if isinstance(value,dict):
                for key,item in value.items(): yield from leaves(item,path+(key,))
            elif isinstance(value,list):
                for key,item in enumerate(value): yield from leaves(item,path+(key,))
            else: yield path,value
        for path,value in leaves(audio.parameters()):
            bad=copy.deepcopy(audio.parameters()); parent=bad
            for key in path[:-1]: parent=parent[key]
            parent[path[-1]]=True if type(value) in (int,float) else value+' wrong'
            with self.subTest(path=path),tempfile.TemporaryDirectory() as temp:
                target=Path(temp)/'absent'
                with patch.object(generator,'parameters',return_value=bad):
                    with self.assertRaises(ValueError): generator.generate_assets(target)
                self.assertFalse(target.exists())
        bad=audio.parameters();bad['extra']=1
        with patch.object(generator,'parameters',return_value=bad):
            with self.assertRaises(ValueError):generator.prepare_assets()

    def test_published_assets_bind_current_sources_and_readback(self):
        self.assertEqual(generator.check_assets(generator.DEFAULT_OUTPUT), self.manifest)

    def test_check_reads_actual_pcm_and_never_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(temp)
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()}
            with patch.object(Path,'write_bytes',side_effect=AssertionError('write')),patch.object(Path,'write_text',side_effect=AssertionError('write')),patch.object(Path,'mkdir',side_effect=AssertionError('mkdir')):
                generator.check_assets(root)
            self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_missing_extra_sources_scores_forged_pcm_and_json_rejected_without_repair(self):
        for kind in ('missing','extra','source','integer','float','pcm','forged_sha','bool','duplicate','nan'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=self.fixture(temp);m=copy.deepcopy(self.manifest)
                if kind=='missing': (root/'reflection_mixed.wav').unlink()
                elif kind=='extra':(root/'extra').write_bytes(b'extra')
                elif kind=='source':m['source_sha256'][generator.SOURCE_PATHS[0]]='0'*64
                elif kind=='integer':m['samples']['mixed']['pcm_integer_measurements']['integer_denominator_D_per_channel']+=1
                elif kind=='float':m['samples']['mixed']['float_measurements']['amplitude_ratio']=1.
                elif kind=='bool':m['parameters']['common_export_gain']=True
                elif kind in ('pcm','forged_sha'):
                    data=bytearray((root/'reflection_mixed.wav').read_bytes());data[20000]^=64
                    (root/'reflection_mixed.wav').write_bytes(data)
                    if kind=='forged_sha':m['files']['reflection_mixed.wav']['sha256']=hashlib.sha256(data).hexdigest()
                raw=json.dumps(m)
                if kind=='duplicate':raw=raw[:-1]+',"schema_version":1}'
                elif kind=='nan':raw=raw[:-1]+',"bad":NaN}'
                (root/'MANIFEST.json').write_text(raw)
                before={p.name:p.read_bytes() for p in root.iterdir()}
                with self.assertRaises(ValueError):generator.check_assets(root)
                self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_linked_members_and_parent_rejected_for_check_and_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);real=base/'real';real.mkdir();self.fixture(real)
            linked=base/'linked';linked.symlink_to(real,target_is_directory=True)
            for action in (generator.check_assets,generator.generate_assets):
                with self.assertRaises(ValueError):action(linked)
            wav=real/'reflection_mixed.wav';wav.unlink();wav.symlink_to(real/'reflection_direct.wav')
            for action in (generator.check_assets,generator.generate_assets):
                with self.assertRaises(ValueError):action(real)

    def test_invalid_measurement_shapes_energy_types_and_extremes(self):
        for x in (np.zeros((2,32012)),np.ones((3,32012)),np.ones((2,32012),complex),
                  np.full((2,32012),np.inf),np.full((2,32012),1e200),np.full((2,32012),1e-200)):
            with self.assertRaises(ValueError):audio.measure_signal(x)


if __name__=='__main__':unittest.main()
