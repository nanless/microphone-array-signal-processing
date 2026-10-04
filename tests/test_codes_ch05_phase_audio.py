"""Independent time/phase/PCM oracles and strict four-member asset contracts."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave
import numpy as np

from codes.chapters.ch05.core import phase_audio as core
from codes.chapters.ch05.examples import generate_phase_audio as generator


class PhaseAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals = core.generate_signals()
        cls.blobs, cls.manifest = generator.prepare_assets()

    def fixture(self, directory):
        output = Path(directory)
        for name, blob in self.blobs.items():
            (output/name).write_bytes(blob)
        (output/'MANIFEST.json').write_text(json.dumps(self.manifest, allow_nan=False))
        return output

    def test_independent_absolute_clock_envelope_and_conjugate_weight_sign(self):
        for n in (0, 1, 87, 319, 320, 2501, 29400, 31680, 31998, 31999):
            fade = math.sin(math.pi/2*min(n, 31999-n)/319)**2 if min(n, 31999-n) < 320 else 1.
            low, high = math.cos(2*math.pi*500*n/16000), math.cos(2*math.pi*1500*n/16000)
            high_sin = math.sin(2*math.pi*1500*n/16000)
            for name, expected in [('reference',low+high), ('flip',low-high), ('quadrature',low+high_sin)]:
                self.assertAlmostEqual(self.signals[name][0,n], .1*fade*expected, places=11)
        base = np.array([.5,.5])
        self.assertEqual(np.vdot(1j*base, np.ones(2)), -1j)
        self.assertGreater(self.signals['quadrature'][0,2401]-.1*math.cos(math.pi/16), 0)

    def test_exact_cycle_powers_error_and_delay_incompatibility(self):
        n = np.arange(2400,29600)
        # Independent absolute-time samples, not production's period lookup.
        reference = .1*(np.cos(2*np.pi*500*n/16000)+np.cos(2*np.pi*1500*n/16000))
        for name, high, error in [('reference', np.cos(2*np.pi*1500*n/16000),0),
                                  ('flip',-np.cos(2*np.pi*1500*n/16000),.02),
                                  ('quadrature',np.sin(2*np.pi*1500*n/16000),.01)]:
            y=.1*(np.cos(2*np.pi*500*n/16000)+high)
            self.assertAlmostEqual(float(y@y/27200), .01, places=13)
            self.assertAlmostEqual(float((y-reference)@(y-reference)/27200), error, places=13)
            result = self.manifest['samples'][name]['float_measurements']
            self.assertAlmostEqual(result['total_reference_mse'], error, places=14)
            self.assertEqual(core.analytic_measurements(name)['normalized_squared_reference_error'], error/.01)
        # Keeping the low tone requires tau=k/500. The high tone then has
        # three integer turns; neither sign flip nor quadrature can result.
        for k in (-17,-1,0,1,19):
            self.assertAlmostEqual(math.cos(2*math.pi*1500*k/500),1,places=12)

    def test_raw_pcm_integer_and_independent_phasor_projection(self):
        integer = {}
        for name, filename in core.FILE_NAMES.items():
            payload = self.blobs[filename]
            with wave.open(io.BytesIO(payload),'rb') as w:
                self.assertEqual((w.getframerate(),w.getnchannels(),w.getnframes(),w.getsampwidth(),w.getcomptype()),
                                 (16000,1,32000,2,'NONE'))
                integer[name] = np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.int64)
            self.assertEqual(hashlib.sha256(payload).hexdigest(),self.manifest['files'][filename]['sha256'])
        r=integer['reference'][2400:29600]
        energy=sum(int(v)**2 for v in r)
        denominator=27200*32768**2
        n=np.arange(2400,29600)
        ratios=[]
        for name in core.FILE_NAMES:
            y=integer[name][2400:29600]
            e=sum(int(v)**2 for v in y)
            residual=sum((int(a)-int(b))**2 for a,b in zip(y,r))
            report=self.manifest['samples'][name]['pcm_integer_measurements']
            self.assertEqual(report['integer_squared_sum_E_per_channel'],[e])
            self.assertEqual(report['integer_reference_squared_sum'],energy)
            self.assertEqual(report['integer_error_squared_sum_per_channel'],[residual])
            self.assertEqual(report['integer_denominator_D_per_channel'],denominator)
            self.assertEqual(report['mean_square_per_channel'],[e/denominator])
            for f in (500,1500):
                osc=np.exp(-2j*np.pi*f*n/16000)
                ratio=(y@osc)/(r@osc)
                expected={'reference':[1,1],'flip':[1,-1],'quadrature':[1,-1j]}[name][(f==1500)]
                self.assertLess(abs(ratio-expected),3e-4) # quantized ratios, not exact clean transfer
            ratios.append(e)
        # Equal energy is exact in the analytic model. Separately rounded
        # PCM combinations can differ; preserve their measured integers.
        self.assertLess((max(ratios)-min(ratios))/denominator,2e-6)

    def test_pcm_golden_sums_preserve_quantization_difference(self):
        # Golden integer values were independently summed from PCM16, rather
        # than replacing actual quantized powers by the common analytic .01.
        golden={'reference':(292026771800,0), 'flip':(292035584600,584062356400),
                'quadrature':(292051763500,292037208100)}
        for name,(energy,error) in golden.items():
            report=self.manifest['samples'][name]['pcm_integer_measurements']
            self.assertEqual(report['integer_squared_sum_E_per_channel'],[energy])
            self.assertEqual(report['integer_error_squared_sum_per_channel'],[error])
            self.assertEqual(report['integer_denominator_D_per_channel'],29205777612800)

    def test_parameter_guard_rejects_each_contract_field_before_generation(self):
        for key, value in generator.REQUIRED_PARAMETERS.items():
            p=copy.deepcopy(self.manifest['parameters'])
            p[key] = '__false_contract__'
            with self.subTest(key=key), mock.patch.object(generator,'parameters',return_value=p), mock.patch.object(generator,'generate_signals',side_effect=AssertionError('must guard first')):
                with self.assertRaisesRegex(ValueError,'true parameters'):
                    generator.prepare_assets()
        for key in ('sample_rate_hz','samples_per_channel','source_amplitude_per_frequency','common_export_gain','propagation_delay_samples'):
            p=copy.deepcopy(self.manifest['parameters']);p[key]=True
            with self.subTest(type=key), mock.patch.object(generator,'parameters',return_value=p):
                with self.assertRaises(ValueError): generator.prepare_assets()

    def test_check_is_strict_readonly_and_actual_pcm_scoring_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output=self.fixture(directory)
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
            with mock.patch.object(Path,'write_bytes',side_effect=AssertionError('check writes')), mock.patch.object(Path,'write_text',side_effect=AssertionError('check writes')), mock.patch.object(generator,'_integer_measurements',wraps=generator._integer_measurements) as score:
                self.assertEqual(generator.check_assets(output),self.manifest)
                self.assertGreaterEqual(score.call_count,6) # replay plus actual readback
            self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()})

    def test_bad_members_json_sha_and_forged_numerics_are_rejected_readonly(self):
        for case in ('extra','missing','linked_member','linked_parent','duplicate_json','nonfinite_json','bool_number','changed_pcm','forged_sha','source','score'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                base=Path(directory);output=base
                self.fixture(output)
                manifest=copy.deepcopy(self.manifest)
                path=output/core.FILE_NAMES['flip']
                if case=='extra': (output/'extra').write_bytes(b'x')
                elif case=='missing': path.unlink()
                elif case=='linked_member':
                    target=base/'saved';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
                elif case=='linked_parent':
                    link=base/'link';link.symlink_to(base,target_is_directory=True);output=link
                elif case=='duplicate_json': (output/'MANIFEST.json').write_text('{"x":1,"x":2}')
                elif case=='nonfinite_json': (output/'MANIFEST.json').write_text('{"x":NaN}')
                else:
                    if case=='bool_number': manifest['parameters']['sample_rate_hz']=True
                    elif case in ('changed_pcm','forged_sha'):
                        payload=bytearray(path.read_bytes());payload[-5]^=1;path.write_bytes(payload)
                        if case=='forged_sha': manifest['files'][path.name]['sha256']=hashlib.sha256(payload).hexdigest()
                    elif case=='source': manifest['source_sha256'][generator.SOURCE_PATHS[0]]='0'*64
                    else: manifest['samples']['flip']['pcm_integer_measurements']['integer_error_squared_sum_per_channel'][0]+=1
                    (output/'MANIFEST.json').write_text(json.dumps(manifest))
                with mock.patch.object(Path,'write_bytes',side_effect=AssertionError('check writes')),mock.patch.object(Path,'write_text',side_effect=AssertionError('check writes')):
                    with self.assertRaises(ValueError): generator.check_assets(output)

    def test_invalid_contract_cannot_create_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'not-created'
            p=copy.deepcopy(self.manifest['parameters']);p['noise']='white noise'
            with mock.patch.object(generator,'parameters',return_value=p):
                with self.assertRaises(ValueError): generator.generate_assets(output)
            self.assertFalse(output.exists())

    def test_generation_rejects_links_and_extra_members_before_preparation(self):
        for case in ('extra', 'linked_member', 'linked_parent'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                base=Path(directory);output=base/'assets';output.mkdir()
                if case=='extra': (output/'extra').write_bytes(b'protected')
                elif case=='linked_member':
                    target=base/'protected.wav';target.write_bytes(b'protected')
                    (output/core.FILE_NAMES['flip']).symlink_to(target)
                else:
                    link=base/'linked';link.symlink_to(output,target_is_directory=True);output=link
                before={p.name:p.read_bytes() for p in (base/'assets').iterdir()}
                with mock.patch.object(generator,'prepare_assets',side_effect=AssertionError('preflight first')):
                    with self.assertRaises(ValueError): generator.generate_assets(output)
                self.assertEqual(before,{p.name:p.read_bytes() for p in (base/'assets').iterdir()})

    def test_measurement_rejects_complex_zero_nonfinite_and_extreme_inputs(self):
        r=self.signals['reference']
        for x in (r.astype(complex),np.zeros((1,32000)),np.full((1,32000),1e200),np.full((1,32000),1e-200),np.full((1,32000),np.nan),np.zeros((2,32000))):
            with self.subTest(shape=x.shape),self.assertRaises(ValueError): core.measure_signal(x,r)


if __name__=='__main__': unittest.main()
