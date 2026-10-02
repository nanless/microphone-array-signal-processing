"""Real generator IO integrations; model bytes are generated once per fixture."""
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
# These are fixed public entry points, not arbitrary filesystem discoveries.
CASES=(
 ('ch01.examples.generate_binaural_cues','binaural_audio','generate_assets'),
 ('ch02.examples.generate_stft_convolution','stft_audio','generate_assets'),
 ('ch03.examples.generate_geometry_audio','geometry_audio','main'),
 ('ch04.examples.generate_focus_audio','focus_audio','main'),
 ('ch05.examples.generate_derivative_audio','derivative_audio','main'),
 ('ch06.examples.generate_apa_audio','apa_audio','main'),
 ('ch07.examples.mint_teaching_demo','mint_audio','generate'),
 ('ch08.examples.mask_representation_demo','mask_audio','generate_assets'),
 ('ch08.examples.gss_teaching_demo','gss_audio','generate'),
 ('ch10.examples.generate_noise_mismatch','noise_audio','generate'),
 ('ch11.examples.generate_selection_audio','scenario_audio','generate'),
 ('appendix_a.examples.generate_weighted_audio','weighted_audio','generate'),
 ('appendix_b.examples.generate_response_audio','response_audio','generate'),
)


def snapshot(path):
    return {p.relative_to(path).as_posix():(p.read_bytes(),p.stat().st_mtime_ns)
            for p in path.rglob('*') if p.is_file() and not p.is_symlink()}


def invoke(module, api, directory, checking=False):
    if api=='main':
        with redirect_stdout(io.StringIO()):
            return module.main(['--output-dir',str(directory)]+(['--check'] if checking else []))
    if checking:
        if hasattr(module,'check_assets'):return module.check_assets(directory)
        return module.generate(directory,check=True)
    return getattr(module,api)(directory)


class AssetIOContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.fixtures=[]
        for name,folder,api in CASES:
            module=importlib.import_module('codes.chapters.'+name)
            out=cls.root/name.split('.')[0]/folder
            invoke(module,api,out)
            cls.fixtures.append((module,api,out,folder))
        cls.main=importlib.import_module('codes.chapters.ch00.examples.generate_audio_samples')
        cls.main_dir=cls.root/'main'
        cls.main.generate(cls.main_dir)
        cls.real=importlib.import_module('codes.chapters.ch02.examples.prepare_real_recordings')
        cls.real_dir=cls.root/'real'
        shutil.copytree(cls.real.DEFAULT_OUTPUT,cls.real_dir)
        # Replay the real pinned 16ch PCM, not a fake source or manually changed digest.
        pcm,_=cls.real.read_pcm_wav((cls.real_dir/cls.real.FILENAMES[0]).read_bytes())
        files,manifest=cls.real.artifacts(pcm)
        for name,data in files.items():(cls.real_dir/name).write_bytes(data)
        (cls.real_dir/'MANIFEST.json').write_text(json.dumps(manifest,allow_nan=False))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_real_generated_bytes_preserve_all_models(self):
        for module,api,out,folder in self.fixtures:
            with self.subTest(folder=folder):
                chapter=module.__name__.split('.')[2]
                official=ROOT/'codes/chapters'/chapter/folder
                for wav in out.glob('*.wav'):
                    self.assertEqual(wav.read_bytes(),(official/wav.name).read_bytes())
                if folder=='gss_audio':self.assertEqual((out/'STATE.npz').read_bytes(),(official/'STATE.npz').read_bytes())
                meta=json.loads((out/'MANIFEST.json').read_text())
                sources=meta.get('source_sha256',meta.get('generator_inputs'))
                self.assertIn('codes/chapters/ch00/io_contracts.py',sources)
                for name,digest in sources.items():self.assertEqual(digest,hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
        for wav in self.main_dir.glob('*/audio/*.wav'):
            self.assertEqual(wav.read_bytes(),(ROOT/'codes/chapters'/wav.relative_to(self.main_dir)).read_bytes())
        for name in self.real.FILENAMES:
            self.assertEqual((self.real_dir/name).read_bytes(),
                             (self.real.DEFAULT_OUTPUT/name).read_bytes())
        self.assertEqual(self.real.check(self.real_dir)['files'],4)

    def test_all_checks_are_read_only_and_reject_linked_parent(self):
        for module,api,out,folder in self.fixtures:
            with self.subTest(folder=folder):
                before=snapshot(out);invoke(module,api,out,True);self.assertEqual(snapshot(out),before)
                parent=self.root/('linked-'+folder);parent.symlink_to(out.parent,target_is_directory=True)
                try:
                    for checking in (False,True):
                        with self.assertRaises(ValueError):invoke(module,api,parent/out.name,checking)
                    self.assertEqual(snapshot(out),before)
                finally:parent.unlink()
        before=snapshot(self.main_dir)
        linked=self.root/'linked-main';linked.symlink_to(self.main_dir)
        for checking in (False,True):
            with self.assertRaises(ValueError):self.main.generate(linked,checking)
        self.main.generate(self.main_dir,True);self.assertEqual(snapshot(self.main_dir),before)

    def test_member_symlinks_and_extra_members_fail_before_writing(self):
        outside=self.root/'outside';outside.write_bytes(b'external sentinel')
        for module,api,out,folder in self.fixtures:
            with self.subTest(folder=folder):
                leaf=next(out.glob('*.wav'));original=leaf.read_bytes();leaf.unlink();leaf.symlink_to(outside)
                before=snapshot(out)
                try:
                    for checking in (False,True):
                        with self.assertRaises(ValueError):invoke(module,api,out,checking)
                    self.assertEqual(outside.read_bytes(),b'external sentinel');self.assertEqual(snapshot(out),before)
                finally:leaf.unlink();leaf.write_bytes(original)
                extra=out/'extra';extra.mkdir();before=snapshot(out)
                try:
                    for checking in (False,True):
                        with self.assertRaises(ValueError):invoke(module,api,out,checking)
                    self.assertEqual(snapshot(out),before)
                finally:extra.rmdir()

    def test_duplicate_json_and_boolean_numerical_values(self):
        for module,api,out,folder in self.fixtures:
            with self.subTest(folder=folder):
                leaf=out/'MANIFEST.json';original=leaf.read_bytes();meta=json.loads(original)
                key=next(iter(meta))
                bad=(b'{'+json.dumps(key).encode()+b':null,'+original.lstrip()[1:])
                leaf.write_bytes(bad);before=snapshot(out)
                try:
                    with self.assertRaises(ValueError):invoke(module,api,out,True)
                    self.assertEqual(snapshot(out),before)
                finally:leaf.write_bytes(original)
                if 'common_export_gain' in meta and meta['common_export_gain']==1:
                    meta['common_export_gain']=True;leaf.write_text(json.dumps(meta));before=snapshot(out)
                    try:
                        with self.assertRaises(ValueError):invoke(module,api,out,True)
                        self.assertEqual(snapshot(out),before)
                    finally:leaf.write_bytes(original)

    def test_main_late_member_error_does_not_modify_early_files(self):
        paths=sorted(self.main_dir.glob('*/audio/*.wav'));bad=paths[-1];original=bad.read_bytes()
        outside=self.root/'main-external';outside.write_bytes(b'leave me')
        bad.unlink();bad.symlink_to(outside);before=snapshot(self.main_dir)
        try:
            with self.assertRaises(ValueError):self.main.generate(self.main_dir)
            self.assertEqual(snapshot(self.main_dir),before)
            self.assertEqual(outside.read_bytes(),b'leave me')
        finally:bad.unlink();bad.write_bytes(original)
        leaf=self.main_dir/'ch00/audio/MANIFEST.json';original=leaf.read_bytes()
        meta=json.loads(original);meta['schema_version']=True;leaf.write_text(json.dumps(meta));before=snapshot(self.main_dir)
        try:
            with self.assertRaises(ValueError):self.main.generate(self.main_dir,True)
            self.assertEqual(snapshot(self.main_dir),before)
        finally:leaf.write_bytes(original)

    def test_demand_publication_requires_license_documents_and_preserves_them(self):
        before=snapshot(self.real_dir);self.real.check(self.real_dir);self.assertEqual(snapshot(self.real_dir),before)
        doc=self.real_dir/'LICENSE.txt';data=doc.read_bytes();doc.unlink()
        try:
            with self.assertRaises(ValueError):self.real.check(self.real_dir)
        finally:doc.write_bytes(data)
        alias=self.root/'linked-real';alias.symlink_to(self.real_dir.parent)
        with self.assertRaises(ValueError):self.real.check(alias/self.real_dir.name)
        # Path rejection occurs before opening the archive or writing any artifact.
        with patch.object(self.real,'load_excerpt',side_effect=AssertionError('too late')):
            with self.assertRaises(ValueError):self.real.prepare(alias/self.real_dir.name,self.real.DEFAULT_ARCHIVE)
        self.assertEqual(snapshot(self.real_dir).keys(),before.keys())


if __name__=='__main__':unittest.main()
