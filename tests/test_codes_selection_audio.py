"""Full real assets, independently summed PCM integers and signal equations."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave
import numpy as np
from codes.chapters.ch11.core.selection_audio import (
    build_fixture, analyze_fixture, analyze_pcm, analytic_results, STEMS,
)
from codes.chapters.ch11.examples.generate_selection_audio import (
    ROOT, MEMBERS, SOURCE_PATHS, expected_assets, generate, check_assets,
    check_main_selection_assets, validate_asset_directory,
)


def integers(data):
    with wave.open(io.BytesIO(data),'rb') as w:
        params=(w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())
        raw=w.readframes(w.getnframes())
    return params,struct.unpack('<'+'h'*(len(raw)//2),raw)


def wav_bytes(values, rate=16000):
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes(struct.pack('<'+'h'*len(values),*values))
    return stream.getvalue()


def snapshot(directory):
    return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in Path(directory).iterdir() if p.is_file()}


class SelectionAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=build_fixture()
        cls.buffers,cls.metadata=expected_assets()

    def temp_assets(self,root):
        dest=Path(root)/'audio';dest.mkdir()
        for name,data in self.buffers.items():(dest/name).write_bytes(data)
        return dest

    def test_analytic_values_from_real_trigonometric_sum(self):
        result=analytic_results(self.fixture)['candidates']
        for L in (3,9):
            h=1/sum(math.cos(2*math.pi*500*(k-(L-1)/2)/16000) for k in range(L))
            a=lambda f:h*sum(math.cos(2*math.pi*f*(k-(L-1)/2)/16000) for k in range(L))
            for scene in ('single','dual'):
                expected=(a(3500)**2 if scene=='single' else ((a(1500)-1)**2+a(3500)**2)/2)
                record=result[f'selection_{scene}_fir{L}.wav']
                self.assertAlmostEqual(record['aligned_total_nmse'],expected,14)
                self.assertAlmostEqual(record['signed_aligned_response']['3500'],a(3500),14)
        self.assertLess(result['selection_dual_fir9.wav']['signed_aligned_response']['3500'],0)

    def test_full_tail_and_sample_oracle(self):
        def tone(f,n):
            if not 0<=n<32000:return 0.
            t=n/16000
            return .2*min(1,t/.02,(2-t)/.02)*math.cos(2*math.pi*f*t)
        for scene in ('single','dual'):
            for L in (3,9):
                weight=1/sum(math.cos(2*math.pi*500*(k-(L-1)/2)/16000) for k in range(L))
                name=f'selection_{scene}_fir{L}'
                x=self.fixture['signals'][name]
                self.assertEqual(len(x),32008)
                for n in (0,1,7,318,319,1600,31999,32000,32001,32007):
                    expected=sum(weight*(tone(500,n-k)+tone(3500,n-k)+(tone(1500,n-k) if scene=='dual' else 0)) for k in range(L))
                    self.assertAlmostEqual(x[n],expected,12)
            self.assertNotEqual(self.fixture['signals'][f'selection_{scene}_fir9'][-1],0)
            np.testing.assert_array_equal(self.fixture['signals'][f'selection_{scene}_target'][32000:],np.zeros(8))
            np.testing.assert_array_equal(self.fixture['signals'][f'selection_{scene}_fir3'][32002:],np.zeros(6))

    def test_float_decomposition_and_common_scene_weights(self):
        records=analyze_fixture(self.fixture)
        for name,r in records['candidates'].items():
            self.assertAlmostEqual(r['target_distortion_power']+r['residual_noise_power']+r['cross_term_power'],r['total_error_power'],14)
        self.assertAlmostEqual(records['selection']['equal_cost_single_weight'],.4765184941037739,12)
        for q in (.25,.75):
            for L in (3,9):
                expected=q*records['candidates'][f'selection_single_fir{L}.wav']['aligned_total_nmse']+(1-q)*records['candidates'][f'selection_dual_fir{L}.wav']['aligned_total_nmse']
                self.assertEqual(records['selection']['at_q'][str(q)][f'fir{L}'],expected)
        self.assertLess(records['selection']['at_q']['0.25']['fir3'],records['selection']['at_q']['0.25']['fir9'])
        self.assertGreater(records['selection']['at_q']['0.75']['fir3'],records['selection']['at_q']['0.75']['fir9'])

    def test_pcm_integer_sse_and_denominator_independent(self):
        measured=self.metadata['pcm_analysis']['candidates']
        for scene,D in [('single',395812398600),('dual',791595378000)]:
            params,reference=integers(self.buffers[f'selection_{scene}_target.wav'])
            self.assertEqual(params,(16000,1,2,32008))
            denominator=sum(v*v for v in reference[1600:30400]);self.assertEqual(denominator,D)
            for L,E in [(3,87210066600 if scene=='single' else 91223542800),(9,152494200 if scene=='single' else 249713672400)]:
                name=f'selection_{scene}_fir{L}.wav';params,output=integers(self.buffers[name]);delay=(L-1)//2
                numerator=sum((a-b)**2 for a,b in zip(output[1600+delay:30400+delay],reference[1600:30400]))
                r=measured[name];self.assertEqual(numerator,E)
                self.assertEqual(r['integer_error_squared_sum'],E);self.assertEqual(r['integer_reference_squared_sum'],D)
                self.assertEqual(r['aligned_total_nmse'],float(Fraction(E,D)))
                self.assertEqual(r['scored_samples'],28800)
                # Standard-library Fourier projection, separate from NumPy measurement.
                segment=output[1600+delay:30400+delay]
                c=math.fsum(v*math.cos(2*math.pi*3500*(n+1600+delay)/16000) for n,v in enumerate(segment))
                s=math.fsum(v*math.sin(2*math.pi*3500*(n+1600+delay)/16000) for n,v in enumerate(segment))
                amplitude=2*math.hypot(c,s)/28800/32768
                self.assertAlmostEqual(r['fitted_amplitudes']['3500'],amplitude,12)
        for name,r in measured.items():
            if 'single' in name:self.assertIsNone(r['target_1500_retention'])

    def test_generate_and_check_are_full_real_and_check_readonly(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'new/deep/audio';generate(dest)
            before=snapshot(dest);m=check_assets(dest);self.assertEqual(snapshot(dest),before)
            self.assertEqual(set(m['files']),{stem+'.wav' for stem in STEMS})
            self.assertEqual(set(m['source_sha256']),set(SOURCE_PATHS))
            for path,h in m['source_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),h)
            self.assertEqual(m['common_export_gain'],.8)
            generate(dest,check=True);self.assertEqual(snapshot(dest),before)

    def test_bad_json_metadata_sources_and_actual_bytes_rejected_without_repair(self):
        for kind in ('duplicate','nan','overflow','bool_score','wrong_source','wrong_pcm','wrong_format','zero_reference','missing','extra'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                dest=self.temp_assets(d);p=dest/'MANIFEST.json';m=json.loads(p.read_bytes())
                if kind=='duplicate':p.write_text('{"schema_version":1,"schema_version":1}')
                elif kind=='nan':p.write_text('{"x":NaN}')
                elif kind=='overflow':p.write_text('{"x":1e999}')
                elif kind=='bool_score':m['pcm_analysis']['candidates']['selection_single_fir3.wav']['scored_samples']=True
                elif kind=='wrong_source':m['source_sha256'][SOURCE_PATHS[0]]='0'*64
                elif kind in ('wrong_pcm','wrong_format','zero_reference'):
                    name='selection_single_target.wav';_,x=integers((dest/name).read_bytes());x=list(x)
                    if kind=='zero_reference':x=[0]*32008
                    else:x[1800]+=1
                    data=wav_bytes(x,rate=8000 if kind=='wrong_format' else 16000);(dest/name).write_bytes(data)
                    m['files'][name]['sha256']=hashlib.sha256(data).hexdigest()
                elif kind=='missing':(dest/'selection_single_target.wav').unlink()
                elif kind=='extra':(dest/'extra').mkdir()
                if kind not in ('duplicate','nan','overflow'):p.write_text(json.dumps(m))
                before=snapshot(dest)
                with self.assertRaises(ValueError):check_assets(dest)
                self.assertEqual(snapshot(dest),before)

    def test_write_preflight_does_not_touch_symlink_or_extra_members(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);external=root/'external';external.mkdir();sentinel=external/'target';sentinel.write_bytes(b'outside')
            linked=root/'linked';linked.symlink_to(external,target_is_directory=True)
            for p in (linked/'new',linked/'../dest'):
                with self.assertRaises(ValueError):generate(p)
            self.assertFalse((external/'new').exists());self.assertFalse((root/'dest').exists())
            dest=self.temp_assets(d);(dest/'selection_single_target.wav').unlink();(dest/'selection_single_target.wav').symlink_to(sentinel)
            with self.assertRaises(ValueError):generate(dest)
            self.assertEqual(sentinel.read_bytes(),b'outside')
            for value in (1,'yes',None):
                with self.assertRaises(ValueError):validate_asset_directory(root/'absent',check=value)
            self.assertFalse((root/'absent').exists())

    def test_actual_main_files_and_updated_digest_tampering(self):
        actual=check_main_selection_assets()
        self.assertEqual(actual['integer_analysis']['selection_fir9']['integer_error_squared_sum'],249713672400)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);main=ROOT/'codes/chapters/ch00/audio/MANIFEST.json';m=json.loads(main.read_bytes())
            paths=list(m['generator_inputs'])+['codes/chapters/ch00/audio/MANIFEST.json']+[f'codes/chapters/ch11/audio/selection_{name}.wav' for name in ('clean','mixture','fir3','fir9')]
            for relative in paths:
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
            before={r:((root/r).read_bytes(),(root/r).stat().st_mtime_ns) for r in paths};check_main_selection_assets(root)
            self.assertEqual({r:((root/r).read_bytes(),(root/r).stat().st_mtime_ns) for r in paths},before)
            p=root/'codes/chapters/ch11/audio/selection_clean.wav';_,x=integers(p.read_bytes());x=list(x);x[1800]+=1;p.write_bytes(wav_bytes(x))
            for record in m['files']:
                if record['file']==p.name:record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'codes/chapters/ch00/audio/MANIFEST.json').write_text(json.dumps(m))
            before={r:((root/r).read_bytes(),(root/r).stat().st_mtime_ns) for r in paths}
            with self.assertRaises(ValueError):check_main_selection_assets(root)
            self.assertEqual({r:((root/r).read_bytes(),(root/r).stat().st_mtime_ns) for r in paths},before)

    def test_main_source_format_zero_and_missing_are_readonly_failures(self):
        for kind in ('source','format','zero','missing','duplicate','bool'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);m=json.loads((ROOT/'codes/chapters/ch00/audio/MANIFEST.json').read_bytes())
                paths=list(m['generator_inputs'])+['codes/chapters/ch00/audio/MANIFEST.json']+[f'codes/chapters/ch11/audio/selection_{name}.wav' for name in ('clean','mixture','fir3','fir9')]
                for relative in paths:
                    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
                p=root/'codes/chapters/ch11/audio/selection_clean.wav'
                if kind=='source':
                    path=next(iter(m['generator_inputs']));(root/path).write_bytes(b'altered source')
                    m['generator_inputs'][path]=hashlib.sha256((root/path).read_bytes()).hexdigest()
                elif kind=='missing':p.unlink()
                elif kind=='duplicate':m['files'].append(dict(m['files'][0]))
                elif kind=='bool':
                    next(r for r in m['files'] if r['file']==p.name)['channels']=True
                else:
                    _,values=integers(p.read_bytes());p.write_bytes(wav_bytes([0]*32000 if kind=='zero' else values,8000 if kind=='format' else 16000))
                    next(r for r in m['files'] if r['file']==p.name)['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
                (root/'codes/chapters/ch00/audio/MANIFEST.json').write_text(json.dumps(m))
                before={r:((root/r).read_bytes(),(root/r).stat().st_mtime_ns) for r in paths if (root/r).exists()}
                with self.assertRaises(ValueError):check_main_selection_assets(root)
                self.assertEqual({r:((root/r).read_bytes(),(root/r).stat().st_mtime_ns) for r in paths if (root/r).exists()},before)

if __name__=='__main__':unittest.main()
