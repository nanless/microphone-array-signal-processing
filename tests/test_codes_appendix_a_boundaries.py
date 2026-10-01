"""Independent Fraction and published-byte oracles for Appendix A."""
import hashlib
import json
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from codes.chapters.appendix_a.core.math_foundations import fft_overlap_add, blockwise_circular_convolution
from codes.chapters.appendix_a.examples.check_main_math_audio import ROOT, check_main_math_assets
from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
from codes.chapters.ch00.core.audio_samples import math_block_case, prepare_exports


def main_fixture(directory):
    """Independent temporary publication, no writes to the repository."""
    directory=Path(directory)
    metadata=json.loads((ROOT/'codes/chapters/ch00/audio/MANIFEST.json').read_text())
    for name in INPUTS:
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    metadata['generator_inputs']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in INPUTS}
    files,groups=prepare_exports({'math_block':math_block_case()})
    metadata['groups']['math_block']=groups['math_block']
    replacements={}
    for name,(blob,info) in files.items():
        target=directory/'codes/chapters/appendix_a/audio'/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
        replacements[name]={'file':name,'chapter':'appendix_a','sha256':hashlib.sha256(blob).hexdigest(),**info}
    metadata['files']=[replacements.get(row['file'],row) for row in metadata['files']]
    target=directory/'codes/chapters/ch00/audio/MANIFEST.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(metadata,allow_nan=False))
    return directory


def snapshot(directory):
    return {str(p.relative_to(directory)):(p.read_bytes(),p.stat().st_mtime_ns)
            for p in Path(directory).rglob('*') if p.is_file()}


class AppendixABoundaries(unittest.TestCase):
    def test_fraction_representable_components_are_rejected_not_lost(self):
        cases=([1.,1e-200],[1e200],[1e200,1.],[1e-200])
        for x,h in ((cases[0],cases[1]),(cases[2],cases[3])):
            expected=[float(Fraction.from_float(a)*Fraction.from_float(h[0])) for a in x]
            self.assertNotEqual(expected[1],0)
            for function in (fft_overlap_add,blockwise_circular_convolution):
                with self.assertRaisesRegex(ValueError,'exponent span'):
                    function(x,h,2)
        tiny=Fraction.from_float(1e-162)
        self.assertEqual(float(4*tiny*tiny),np.nextafter(0.,1.))
        with self.assertRaisesRegex(ValueError,'below normal'):
            fft_overlap_add([1e-162]*4,[1e-162]*4,4)

    def test_normal_path_true_zero_tail_and_partial_blocks(self):
        for size in (1,2,3,7):
            np.testing.assert_allclose(fft_overlap_add([1,2,3],[1,.5],size),[1,2.5,4,1.5],atol=2e-15)
            np.testing.assert_array_equal(fft_overlap_add([0,0],[1,2],size),[0,0,0])
        # A normal exact cancellation is not an illegal positive underflow.
        np.testing.assert_array_equal(fft_overlap_add([1,1],[1,-1],2),[1,0,-1])
        np.testing.assert_allclose(fft_overlap_add(np.array([1,2],dtype=np.int64),[1.],2),[1,2])

    def test_numeric_types_and_intermediate_overflow(self):
        for function in (fft_overlap_add,blockwise_circular_convolution):
            for raw in ([True],['1'],np.array([1j],dtype=object),[10**400],[[1]]):
                with self.subTest(function=function.__name__,raw=repr(raw)):
                    with self.assertRaises(ValueError):function(raw,[1],2)
            with self.assertRaises(ValueError):function([1e308,-1e308],[1e-308],2)
            with self.assertRaises(ValueError):function([1],[1],10**400)

    def test_actual_main_integers_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=main_fixture(directory);before=snapshot(root);report=check_main_math_assets(root)
            self.assertEqual(snapshot(root),before)
            p=report['pcm_analysis'];self.assertEqual(p['first_dry_sample_500'],9830/32768)
            self.assertEqual(p['first_linear_echo_sample_620'],5898/32768)
            self.assertEqual(p['wrong_minus_linear_nonzero_count'],20)
            row=report['integer_analysis']
            self.assertEqual(row['integer_error_squared_sum'],20*5898**2)
            self.assertEqual(row['integer_reference_squared_sum'],10*(9830**2+5898**2))
            self.assertEqual(len(report['source_sha256']),19)

    def test_main_tamper_missing_format_json_and_source_are_not_repaired(self):
        for mutation in ('missing','bytes','format','zero','source','duplicate','nonfinite','overflow','bool'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=main_fixture(directory);target=root/'codes/chapters/appendix_a/audio/math_block_dry.wav'
                manifest=root/'codes/chapters/ch00/audio/MANIFEST.json'
                if mutation=='missing':target.unlink()
                elif mutation=='bytes':target.write_bytes(target.read_bytes()+b'bad')
                elif mutation=='format':
                    blob=bytearray(target.read_bytes());blob[24:28]=(8000).to_bytes(4,'little');target.write_bytes(blob)
                elif mutation=='zero':
                    blob=bytearray(target.read_bytes());blob[44:]=bytes(len(blob)-44);target.write_bytes(blob)
                elif mutation=='source':(root/INPUTS[0]).write_text('different source')
                elif mutation=='duplicate':manifest.write_text('{"schema_version":2,"schema_version":2}')
                elif mutation=='nonfinite':manifest.write_text('{"schema_version":NaN}')
                elif mutation=='overflow':manifest.write_text('{"schema_version":1e999}')
                else:
                    m=json.loads(manifest.read_text());m['schema_version']=True;manifest.write_text(json.dumps(m))
                before=snapshot(root)
                with self.assertRaises(ValueError):check_main_math_assets(root)
                self.assertEqual(before,snapshot(root))

    def test_main_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=main_fixture(Path(directory)/'repo');target=root/'codes/chapters/appendix_a/audio/math_block_dry.wav'
            outside=Path(directory)/'outside';outside.write_bytes(target.read_bytes());target.unlink();target.symlink_to(outside)
            before=outside.read_bytes()
            with self.assertRaises(ValueError):check_main_math_assets(root)
            self.assertEqual(outside.read_bytes(),before)
