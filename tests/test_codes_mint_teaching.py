"""Independent rational, real-sum and signed-integer MINT fixture checks."""
import hashlib
import copy
import io
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
from fractions import Fraction as F
import numpy as np
from codes.chapters.ch07.core import mint_teaching as core
from codes.chapters.ch07.examples import mint_teaching_demo as demo


def read_integers(blob):
    with wave.open(io.BytesIO(blob)) as f:
        params=(f.getframerate(),f.getnchannels(),f.getnframes(),f.getsampwidth(),f.getcomptype())
        raw=f.readframes(f.getnframes())
    values=struct.unpack('<'+'h'*(len(raw)//2),raw)
    return params,[values[k::params[1]] for k in range(params[1])]


def encode_integers(channels, rate=16000):
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as f:
        f.setparams((len(channels),2,rate,0,'NONE','not compressed'))
        f.writeframes(struct.pack('<'+'h'*(len(channels)*len(channels[0])),
                                  *(v for row in zip(*channels) for v in row)))
    return buffer.getvalue()


def snapshot(directory):
    return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir() if p.is_file()}


class MintTeaching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory=Path(cls.temporary.name)/'assets'
        cls.manifest=demo.generate(cls.directory)  # real synthesis, encoding and actual read-check
        cls.experiment=core.run_experiment()

    def test_fraction_solution_and_equal_paths_regularized_case(self):
        for a,b,lam,expected in ((F(1,2),F(-1,2),0,[.5,.5]),
                                (F(1,2),F(49,100),0,[-49,50]),
                                (F(1,2),F(49,100),F(1,10000),[-16,17]),
                                (F(1,2),F(1,2),F(1,10000),[.5,.5])):
            np.testing.assert_array_equal(core.regularized_weights(a,b,lam),expected)
        # Differentiate the scalar constrained polynomial, independently.
        u1,u2=F(-16),F(17); a,b,lam=F(1,2),F(49,100),F(1,10000)
        self.assertEqual((a-b)*(a*u1+b*u2)+lam*(u1-u2),0)
        self.assertEqual((a*u1+b*u2)**2+lam*(u1*u1+u2*u2),F(817,5000))

    def test_weight_invalid_real_interfaces_and_nonunique_inverse(self):
        for value in (True,'0.5',.5+0j,[.5],float('inf'),float('nan')):
            for index in range(3):
                values=[.5,.49,.0001]; values[index]=value
                with self.subTest(value=value,index=index),self.assertRaises(ValueError):
                    core.regularized_weights(*values)
        for args in ((.5,.49,-1),(.5,.5,0)):
            with self.assertRaises(ValueError): core.regularized_weights(*args)

    def test_complete_tail_delayed_source_and_independent_real_cosines(self):
        c=self.experiment['components']; r=c['reference'][0]; delayed=c['delayed_source'][0]
        self.assertEqual(r.shape,(32512,)); np.testing.assert_array_equal(r[32000:],np.zeros(512))
        np.testing.assert_array_equal(delayed[:512],np.zeros(512))
        np.testing.assert_array_equal(delayed[512:],r[:32000])
        # Another route: scalar math trigonometry on an absolute sample clock.
        for n in (0,1,319,320,2400,29100,31679,31680,31998,31999):
            envelope=1.
            if n<320: envelope=math.sin(math.pi*n/(2*319))**2
            elif n>=31680: envelope=math.sin(math.pi*(31999-n)/(2*319))**2
            expected=.05*sum(math.cos(2*math.pi*f*n/16000) for f in (200,500,900,1200))*envelope
            self.assertAlmostEqual(r[n],expected,places=12)
        self.assertAlmostEqual(math.fsum(v*v for v in r[2400:29600])/27200,.005,places=14)
        self.assertGreater(sum(v*v for v in delayed[32000:]),0)
        np.testing.assert_array_equal(abs(c['noise_array']),np.full((2,32512),math.sqrt(5e-7)))

    def test_same_post_path_noise_and_fixed_one_tap_outputs(self):
        e=self.experiment; c=e['components']; noise=c['noise_array']
        for condition in ('well','near'):
            np.testing.assert_allclose(e['signals'][condition+'_array']-c[condition+'_clean_array'],noise,atol=3e-17)
        for key,w in (('well_exact',[.5,.5]),('near_exact',[-49,50]),('near_regularized',[-16,17])):
            condition='well' if key=='well_exact' else 'near'
            observed=e['signals'][condition+'_array']
            expected=np.array([w[0]*observed[0,n]+w[1]*observed[1,n] for n in range(32512)])
            np.testing.assert_allclose(e['signals'][key][0],expected,atol=3e-15)
            d=e['float_decomposition'][key]
            self.assertAlmostEqual(d['decomposed_total_mean_square'],e['float_measurements'][key]['total_reference_mse_per_channel'][0],places=15)
        self.assertNotEqual(e['finite_noise_statistics']['uncentered_second_moment'][0][1],0)
        self.assertNotEqual(e['float_decomposition']['near_regularized']['twice_clean_error_noise_cross_mean'],0)

    def test_actual_pcm_integer_oracle_channels_full_tail_quantization(self):
        m=self.manifest; decoded={}
        for key,filename in core.FILE_NAMES.items():
            blob=(self.directory/filename).read_bytes(); params,x=read_integers(blob)
            self.assertEqual(params,(16000,2 if key.endswith('_array') else 1,32512,2,'NONE'))
            self.assertEqual(m['files'][filename]['sha256'],hashlib.sha256(blob).hexdigest())
            decoded[key]=x
            floats=self.experiment['signals'][key]
            worst=max(abs(value/32768-floats[c,n]) for c,ch in enumerate(x) for n,value in enumerate(ch))
            self.assertEqual(worst,m['samples'][key]['quantization_max_abs_error'])
            self.assertLessEqual(worst,.5/32768)
            self.assertLess(max(abs(v) for ch in x for v in ch),32767)
        truth=decoded['reference'][0][2400:29600]; D=sum(v*v for v in truth)
        self.assertEqual(D,146024450180)
        output_expected={'well_exact':7223120,'near_exact':72001808458,'near_regularized':23945970352}
        for key,x in decoded.items():
            pm=m['samples'][key]['pcm_measurements']
            E=[sum((v-r)**2 for v,r in zip(ch[2400:29600],truth)) for ch in x]
            C=[sum(v*r for v,r in zip(ch[2400:29600],truth)) for ch in x]
            T=[sum(v*v for v in ch[32000:]) for ch in x]
            self.assertEqual(pm['integer_reference_squared_sum'],D)
            self.assertEqual(pm['integer_error_squared_sum_per_channel'],E)
            self.assertEqual(pm['integer_output_reference_cross_sum_per_channel'],C)
            self.assertEqual(pm['integer_tail_squared_sum_per_channel'],T)
            self.assertEqual(pm['total_reference_mse_per_channel'],[v/(27200*32768**2) for v in E])
            self.assertEqual(pm['relative_squared_reference_error_per_channel'],[v/D for v in E])
            self.assertEqual((pm['sample_denominator_per_channel'],pm['tail_samples_per_channel'],pm['pcm_decode_divisor']),(27200,512,32768))
            if key in output_expected: self.assertEqual(E,[output_expected[key]])

    def test_measurement_boundaries_not_silently_quantized(self):
        r=np.ones((1,32512))/32768
        for value in (True,'1',1+0j,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                core.measure_signal(np.full((1,32512),value),r)
        for invalid in (np.ones((1,32512)),np.full((1,32512),.1),np.full((1,32512),-1.1)):
            with self.assertRaises(ValueError): core.measure_signal(invalid,r,pcm=True)
        for pcm in (False,True):
            with self.assertRaises(ValueError): core.measure_signal(r,np.zeros_like(r),pcm=pcm)
        with self.assertRaises(ValueError): core.measure_signal(np.zeros((1,32511)),r)
        with self.assertRaises(ValueError): core.measure_signal(r,r,pcm=1)
        with self.assertRaises(ValueError): core.measure_signal(np.full_like(r,1e308),r)
        np.testing.assert_array_equal(core.measure_signal(-np.ones_like(r),r,pcm=True)['peak_per_channel'],[1])

    def test_read_check_is_readonly_and_no_replay_is_not_synthesis(self):
        before=snapshot(self.directory)
        with patch.object(demo,'prepare_assets',side_effect=AssertionError('must not generate during no-replay check')):
            self.assertEqual(demo.check_assets(self.directory,replay=False),self.manifest)
        self.assertEqual(demo.generate(self.directory,check=True),self.manifest)
        self.assertEqual(snapshot(self.directory),before)
        with self.assertRaises(ValueError): demo.generate(self.directory,check=1)
        with self.assertRaises(ValueError): demo.check_assets(self.directory,replay=1)

    def test_fixed_parameter_values_and_types_rejected_before_any_write(self):
        mutations = []
        for key in demo.REQUIRED_PARAMETERS:
            changed = copy.deepcopy(demo.REQUIRED_PARAMETERS)
            changed[key] = None
            mutations.append((key, changed))
        for key, value in (('seed', True), ('source_samples', 32000.0),
                           ('common_export_gain', True), ('amplitude_per_frequency', 0)):
            changed = copy.deepcopy(demo.REQUIRED_PARAMETERS)
            changed[key] = value
            mutations.append((key+' wrong type', changed))
        changed = copy.deepcopy(demo.REQUIRED_PARAMETERS)
        changed['paths']['first'][1] = .6
        mutations.append(('nested true path', changed))
        changed = copy.deepcopy(demo.REQUIRED_PARAMETERS)
        changed['designs']['near_exact']['direct_response'] = 1
        mutations.append(('nested float type', changed))
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary)/'not_created'
            before = snapshot(self.directory)
            for label, params in mutations:
                changed = {**self.experiment, 'parameters': params}
                for destination in (absent, self.directory):
                    with self.subTest(label=label, destination=destination), \
                            patch.object(demo, 'run_experiment', return_value=changed), \
                            patch.object(demo, 'pcm16_bytes') as encoder:
                        with self.assertRaises(ValueError):
                            demo.generate(destination)
                        encoder.assert_not_called()
                self.assertFalse(absent.exists())
                self.assertEqual(snapshot(self.directory), before)

    def test_runtime_declarations_limits_and_false_float_score_rejected_prewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary)/'not_created'
            params = copy.deepcopy(demo.REQUIRED_PARAMETERS)
            params['reflection_delay_samples'] = True
            with patch.object(demo, 'parameters', return_value=params), \
                    patch.object(demo, 'run_experiment') as synth, self.assertRaises(ValueError):
                demo.generate(absent)
            synth.assert_not_called()
            for change in ('limits', 'float_measurements'):
                altered = copy.deepcopy(self.experiment)
                if change == 'limits':
                    altered['limits'] = 'measured real room'
                else:
                    altered['float_measurements']['near_exact']['total_reference_mse_per_channel'] = [0.]
                with patch.object(demo, 'run_experiment', return_value=altered), \
                        patch.object(demo, 'pcm16_bytes') as encoder, self.assertRaises(ValueError):
                    demo.generate(absent)
                encoder.assert_not_called()
            self.assertFalse(absent.exists())

    def test_manifest_tampering_is_rejected_without_repair(self):
        mutations={
            'extra_source':lambda m:m['source_sha256'].update({'other.py':'0'*64}),
            'missing_source':lambda m:m['source_sha256'].pop(next(iter(m['source_sha256']))),
            'stale_source':lambda m:m['source_sha256'].update({next(iter(m['source_sha256'])):'0'*64}),
            'denominator':lambda m:m['samples']['near_exact']['pcm_measurements'].update(sample_denominator_per_channel=27199),
            'bool_integer':lambda m:m['samples']['near_exact']['pcm_measurements'].update(integer_reference_squared_sum=True),
            'nan':lambda m:m['samples']['near_exact']['pcm_measurements'].update(total_reference_mse_per_channel=[float('nan')]),
            'gain':lambda m:m.update(common_export_gain=True),
            'params':lambda m:m['parameters'].update(reflection_delay_samples=511),
            'float_component':lambda m:m['float_decomposition']['near_exact'].update(noise_mean_square=0),
            'bad_files_container':lambda m:m.update(files=None),
            'bad_sample_record':lambda m:m['samples'].update(reference=None),
        }
        for key,mutate in mutations.items():
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                p=Path(t)/'assets'; shutil.copytree(self.directory,p)
                m=json.loads((p/'MANIFEST.json').read_text()); mutate(m)
                (p/'MANIFEST.json').write_text(json.dumps(m)); before=snapshot(p)
                with self.assertRaises(ValueError): demo.generate(p,check=True)
                self.assertEqual(snapshot(p),before)

    def test_pcm_tampering_updated_sha_format_and_zero_reference(self):
        for case in ('byte','updated_sha','zero_reference','format'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                p=Path(t)/'assets'; shutil.copytree(self.directory,p)
                m=json.loads((p/'MANIFEST.json').read_text())
                name='mint_reference.wav' if case=='zero_reference' else 'mint_near_exact.wav'
                q=p/name
                if case in ('byte','updated_sha'):
                    blob=q.read_bytes(); q.write_bytes(blob[:-2]+b'\x01\x00')
                else: q.write_bytes(encode_integers([[0]*32512],8000 if case=='format' else 16000))
                if case!='byte':
                    m['files'][name]['sha256']=hashlib.sha256(q.read_bytes()).hexdigest()
                    (p/'MANIFEST.json').write_text(json.dumps(m))
                before=snapshot(p)
                # Actual PCM replay=False still re-scores and validates real format.
                with self.assertRaises(ValueError): demo.check_assets(p,replay=False)
                self.assertEqual(snapshot(p),before)

    def test_member_preflight_and_symlink_guard_leave_external_bytes_untouched(self):
        for case in ('missing','extra','directory','symlink','root_file','root_symlink'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                base=Path(t); p=base/'assets'; shutil.copytree(self.directory,p)
                external=base/'external'; external.write_bytes(b'private external fixture'); stat=external.stat().st_mtime_ns
                name=core.FILE_NAMES['well_exact']
                if case=='missing': (p/name).unlink()
                elif case=='extra': (p/'extra.wav').write_bytes(b'extra')
                elif case in ('directory','symlink'):
                    (p/name).unlink()
                    if case=='directory': (p/name).mkdir()
                    else: (p/name).symlink_to(external)
                elif case=='root_file': p=external
                elif case=='root_symlink':
                    alias=base/'alias'; alias.symlink_to(p,target_is_directory=True); p=alias
                with self.assertRaises(ValueError): demo.check_assets(p)
                if case!='missing':
                    with self.assertRaises(ValueError): demo.generate(p)
                self.assertEqual(external.read_bytes(),b'private external fixture')
                self.assertEqual(external.stat().st_mtime_ns,stat)


if __name__=='__main__': unittest.main()
