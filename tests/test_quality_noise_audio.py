"""Noise mismatch publication must bind real PCM, sources and visible controls."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_site, quality_check as quality
from codes.chapters.ch10.examples import generate_noise_mismatch as generator

ROOT = Path(__file__).resolve().parents[1]


class NoisePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'original'
            generator.generate(source)
            cls.assets = {p.name:p.read_bytes() for p in source.iterdir()}
        cls.original = json.loads(cls.assets['MANIFEST.json'])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'source'
        self.site = self.root/'site'
        self.published = self.site/'noise_audio'
        self.source.mkdir();self.published.mkdir(parents=True)
        (self.site/'research').mkdir()
        for path in self.original['source_sha256']:
            target = self.root/path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/path).read_bytes())
        for filename,data in self.assets.items():
            (self.source/filename).write_bytes(data)
            (self.published/filename).write_bytes(data)
        self.pages = [(self.site/'10_engineering-practice.html',''),
                      (self.site/'research/05_exercises_and_audio.html','../')]
        for page,prefix in self.pages:
            page.write_text(self.html(prefix))
        for name,value in [('ROOT',self.root),('SITE',self.site),('NOISE_AUDIO_ROOT',self.source)]:
            patcher=patch.object(quality,name,value);patcher.start();self.addCleanup(patcher.stop)

    def html(self,prefix):
        base=prefix+'noise_audio/'
        return ''.join(f'<audio controls preload="none" aria-label="{name}" src="{base}{name}"></audio>'
                       for name in sorted(self.original['files']))+f'<a href="{base}MANIFEST.json">Manifest</a>'

    def issues(self):
        errors=[];quality.check_noise_audio(errors);return errors

    def test_valid_assets_publish_without_changing_source(self):
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(),[])
        destination=self.root/'stage'
        self.assertEqual(build_site.stage_noise_audio(self.source,destination),set(self.assets))
        self.assertEqual({p.name:p.read_bytes() for p in destination.iterdir()},self.assets)
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_source_mismatch_and_false_pcm_denominator_fail_without_repair(self):
        for alter in (lambda d:d['source_sha256'].update({generator.SOURCE_PATHS[0]:'0'*64}),
                      lambda d:d['pcm_analysis']['score_windows']['before_step'].update(reference_squared_sum_pcm_integer=True),
                      lambda d:d['pcm_analysis']['score_windows']['after_step']['scores']['noise_fixed.wav'].update(nmse=0)):
            manifest=copy.deepcopy(self.original);alter(manifest)
            for folder in [self.source,self.published]:
                (folder/'MANIFEST.json').write_text(json.dumps(manifest))
            before={p.name:p.read_bytes() for p in self.source.iterdir()}
            self.assertTrue(self.issues())
            with self.assertRaises(ValueError):build_site.stage_noise_audio(self.source,self.root/'stage')
            self.assertFalse((self.root/'stage').exists())
            self.assertEqual(before,{p.name:p.read_bytes() for p in self.source.iterdir()})

    def test_bad_json_extra_directory_and_actual_publication_bytes_fail(self):
        for content in ('[]','{"x":1,"x":1}','{"x":NaN}','{"x":1e999}'):
            (self.source/'MANIFEST.json').write_text(content)
            with self.assertRaises(ValueError):build_site.stage_noise_audio(self.source,self.root/'stage')
            self.assertFalse((self.root/'stage').exists())
        (self.source/'MANIFEST.json').write_bytes(self.assets['MANIFEST.json'])
        extra=self.published/'extra';extra.mkdir();self.assertTrue(self.issues());extra.rmdir()
        target=self.published/'noise_fixed.wav';data=bytearray(target.read_bytes());data[20000]^=1;target.write_bytes(data)
        self.assertTrue(self.issues())

    def test_complete_parent_chain_and_lexical_escape_fail_before_writes(self):
        outside=self.root/'outside';outside.mkdir();(outside/'nested').mkdir()
        linked=self.root/'linked';linked.symlink_to(outside,target_is_directory=True)
        for destination in [linked/'stage',linked/'nested'/'stage',linked/'..'/'stage']:
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):build_site.stage_noise_audio(self.source,destination)
        self.assertEqual(list((outside/'nested').iterdir()),[])
        self.assertFalse((self.root/'stage').exists())
        source_link=self.root/'source_link';source_link.symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ValueError):build_site.stage_noise_audio(source_link,self.root/'stage')

    def test_visible_controls_and_manifest_required_on_both_pages(self):
        for page,prefix in self.pages:
            valid=self.html(prefix)
            for invalid in [valid.replace('<audio ','<audio hidden ',1),
                            valid.replace('controls ','',1),valid.replace('<audio ','<audio autoplay ',1),
                            '<details><summary>Audio</summary>'+valid+'</details>',
                            '<div inert>'+valid+'</div>',valid.replace('<a ','<a hidden ',1),
                            valid+valid[:valid.index('</audio>')+8]]:
                with self.subTest(page=page.name):
                    page.write_text(invalid);self.assertTrue(self.issues())
            page.write_text('<details open><summary>Audio</summary>'+valid+'</details>')
        self.assertEqual(self.issues(),[])

    def test_similar_remote_and_other_chapter_links_do_not_gain_noise_players(self):
        chapter=ROOT/'chapters/10_engineering-practice.md'
        valid='<a href="../codes/chapters/ch10/noise_audio/noise_fixed.wav">sample</a>'
        result=build_site.rewrite_site_links(valid,chapter)
        self.assertIn('src="noise_audio/noise_fixed.wav"',result)
        for href in ['https://example.com/noise_audio/noise_fixed.wav',
                     '../codes/chapters/ch09/noise_audio/noise_fixed.wav',
                     '../codes/chapters/ch10/noise_audio/unlisted.wav']:
            result=build_site.rewrite_site_links(f'<a href="{href}">sample</a>',chapter)
            self.assertNotIn('<audio ',result)


if __name__=='__main__':unittest.main()
