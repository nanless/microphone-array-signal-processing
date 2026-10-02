"""Closed-form and actual integer oracles for known-weight audio."""
import copy
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
from codes.chapters.appendix_a.core.weighted_audio import build_fixture, analyze_fixture, analyze_pcm, STEMS
from codes.chapters.appendix_a.examples.generate_weighted_audio import expected_assets, check_assets, generate
from tests.test_codes_appendix_a_boundaries import snapshot


class WeightedAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.buffers,cls.metadata=expected_assets()

    def write_fixture(self,root):
        root.mkdir(parents=True,exist_ok=True)
        for name,data in self.buffers.items():(root/name).write_bytes(data)
        return root

    def test_analytic_float_components_from_independent_trigonometry(self):
        fixture=build_fixture();report=analyze_fixture(fixture)
        self.assertEqual(fixture['signals']['weighted_array'].shape,(2,32000))
        for n in (0,31999):self.assertEqual(fixture['signals']['weighted_target'][n],0)
        # Independently compute several samples using scalar math, not the
        # vector generator or its returned components as the expected value.
        for n in (1,201,1600,8001,30400,31998):
            env=min(1,n/640,(31999-n)/640)
            target=.2*math.cos(math.tau*700*n/16000)*env
            first=.03*math.cos(math.tau*3500*n/16000)*env
            second=.06*math.cos(math.tau*4000*n/16000)*env
            for name,w in [('weighted_ols',(.5,.5)),('weighted_gls',(.8,.2)),('weighted_reversed',(.2,.8))]:
                self.assertAlmostEqual(fixture['signals'][name][n],target+w[0]*first+w[1]*second,delta=3e-13)
        for name,expected in [('weighted_ols.wav',.0005625),('weighted_gls.wav',.00036),('weighted_reversed.wav',.00117)]:
            row=report['candidates'][name]
            self.assertAlmostEqual(row['total_error_power'],expected,delta=1e-16)
            self.assertEqual(row['target_distortion_power'],0)
            self.assertEqual(row['cross_term'],0)
            self.assertAlmostEqual(row['nmse'],expected/.02,delta=1e-14)

    def test_real_wave_integer_sums_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.write_fixture(Path(directory));before=snapshot(root);metadata=check_assets(root)
            self.assertEqual(before,snapshot(root))
            integers={}
            for stem in STEMS:
                with wave.open(str(root/(stem+'.wav'))) as reader:
                    self.assertEqual((reader.getframerate(),reader.getnframes(),reader.getsampwidth(),reader.getnchannels()),(16000,32000,2,2 if stem=='weighted_array' else 1))
                    ints=struct.unpack('<'+'h'*(32000*reader.getnchannels()),reader.readframes(32000))
                    integers[stem]=ints
            reference=integers['weighted_target'][1600:30400]
            D=sum(x*x for x in reference);self.assertEqual(D,618489148320)
            for stem,expected in [('weighted_ols',17394760800),('weighted_gls',11131529760),('weighted_reversed',36180064800)]:
                E=sum((x-y)**2 for x,y in zip(integers[stem][1600:30400],reference))
                self.assertEqual(E,expected)
                row=metadata['pcm_analysis']['candidates'][stem+'.wav']
                self.assertEqual(row['integer_error_squared_sum'],E)
                self.assertEqual(row['integer_reference_squared_sum'],D)
                self.assertEqual(row['scored_samples'],28800)
                self.assertEqual(row['nmse'],E/D)
            self.assertEqual(len(metadata['source_sha256']),4)
            self.assertEqual(metadata['common_export_gain'],1.)

    def test_cli_check_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.write_fixture(Path(directory));before=snapshot(root)
            result=subprocess.run([sys.executable,'-m','codes.chapters.appendix_a.examples.generate_weighted_audio','--output',str(root),'--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(before,snapshot(root))

    def test_all_check_failures_leave_bytes_and_mtime_unchanged(self):
        for kind in ('missing','extra','directory','pcm','source','score','bool','nan','infinity','overflow','duplicate'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=self.write_fixture(Path(directory));m=copy.deepcopy(self.metadata)
                if kind=='missing':(root/'weighted_ols.wav').unlink()
                elif kind=='extra':(root/'extra.wav').write_bytes(b'extra')
                elif kind=='directory':(root/'extra').mkdir()
                elif kind=='pcm':(root/'weighted_ols.wav').write_bytes(self.buffers['weighted_gls.wav'])
                elif kind=='source':m['source_sha256'][next(iter(m['source_sha256']))]='0'*64
                elif kind=='score':m['pcm_analysis']['candidates']['weighted_gls.wav']['integer_reference_squared_sum']+=1
                elif kind=='bool':m['files']['weighted_array.wav']['channels']=True
                elif kind=='nan':(root/'MANIFEST.json').write_text('{"x":NaN}')
                elif kind=='infinity':(root/'MANIFEST.json').write_text('{"x":Infinity}')
                elif kind=='overflow':(root/'MANIFEST.json').write_text('{"x":1e999}')
                else:(root/'MANIFEST.json').write_text('{"schema_version":1,"schema_version":1}')
                if kind in ('source','score','bool'):(root/'MANIFEST.json').write_text(json.dumps(m))
                before=snapshot(root)
                with self.assertRaises(ValueError):check_assets(root)
                self.assertEqual(before,snapshot(root))

    def test_zero_reference_even_valid_format_is_invalid(self):
        buffers=dict(self.buffers);blob=bytearray(buffers['weighted_target.wav']);blob[44:]=bytes(64000)
        buffers['weighted_target.wav']=bytes(blob);buffers.pop('MANIFEST.json')
        with self.assertRaisesRegex(ValueError,'reference power'):analyze_pcm(buffers)
        with tempfile.TemporaryDirectory() as directory:
            root=self.write_fixture(Path(directory));target=root/'weighted_target.wav'
            target.write_bytes(buffers['weighted_target.wav'])
            metadata=copy.deepcopy(self.metadata)
            metadata['files']['weighted_target.wav']['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
            (root/'MANIFEST.json').write_text(json.dumps(metadata))
            before=snapshot(root)
            with self.assertRaisesRegex(ValueError,'reference power'):check_assets(root)
            self.assertEqual(before,snapshot(root))

    def test_pcm_format_must_be_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.write_fixture(Path(directory));target=root/'weighted_array.wav'
            blob=bytearray(target.read_bytes());blob[24:28]=(8000).to_bytes(4,'little');target.write_bytes(blob)
            before=snapshot(root)
            with self.assertRaisesRegex(ValueError,'PCM format'):check_assets(root)
            self.assertEqual(before,snapshot(root))

    def test_paths_member_symlink_and_bad_check_types_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            top=Path(directory);outside=top/'outside';outside.mkdir();external=outside/'data';external.write_bytes(b'unchanged')
            root=self.write_fixture(top/'assets');target=root/'weighted_ols.wav';target.unlink();target.symlink_to(external)
            for check in (True,False):
                with self.assertRaises(ValueError):generate(root,check=check)
            self.assertEqual(external.read_bytes(),b'unchanged')
            linked=top/'linked';linked.symlink_to(outside,target_is_directory=True)
            for path in (linked/'new',linked/'..'/'new'):
                with self.assertRaises(ValueError):generate(path)
            self.assertFalse((top/'new').exists());self.assertFalse((outside/'new').exists())
            for value in ('yes',0,1,None):
                with self.assertRaises(ValueError):generate(top/'new',check=value)
