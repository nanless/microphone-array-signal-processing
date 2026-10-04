"""Independent PCM publication and finite rejection checks for chapter 8."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from scripts import build_site, quality_check as qc
ROOT=Path(__file__).resolve().parents[1]

class CSSPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.audio=self.root/'codes/chapters/ch08/css_audio'
        shutil.copytree(ROOT/'codes/chapters/ch08/css_audio',self.audio)
        self.report=self.root/'codes/chapters/ch08/reports/figure76_css_polarity_gain.json'
        self.report.parent.mkdir();shutil.copy2(ROOT/'codes/chapters/ch08/reports/figure76_css_polarity_gain.json',self.report)
        for name in json.loads(self.report.read_bytes())['source_sha256']:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        (self.root/'figures').mkdir();shutil.copy2(ROOT/'figures/fig76_css_polarity_gain.png',self.root/'figures/fig76_css_polarity_gain.png')
        self.site=self.root/'site';self.site.mkdir();build_site.stage_css_audio(self.audio,self.site/'css_audio')
        (self.site/'research').mkdir()
        for page,prefix in ((self.site/'08_speech-separation.html',''),(self.site/'research/05_exercises_and_audio.html','../')):
            page.write_text('\n'.join(f'<audio src="{prefix}css_audio/{n}" controls preload="none" aria-label="CSS"></audio>' for n in sorted(qc.CSS_AUDIO_WAVS))+f'<a href="{prefix}css_audio/MANIFEST.json">Manifest</a>')
    def check(self,function):
        errors=[]
        with patch.object(qc,'ROOT',self.root),patch.object(qc,'CSS_AUDIO_ROOT',self.audio),patch.object(qc,'SITE',self.site):function(errors)
        return errors
    def test_actual_independent_pcm_and_source_identity_pass(self):
        self.assertEqual(self.check(qc.check_css_audio),[])
        self.assertEqual(self.check(qc.check_css_figure),[])
    def test_changed_copy_is_rejected_and_not_repaired(self):
        path=self.site/'css_audio/css_naive.wav';raw=path.read_bytes();changed=raw[:-2]+b'\x01\x00';path.write_bytes(changed)
        self.assertTrue(self.check(qc.check_css_audio));self.assertEqual(path.read_bytes(),changed)
    def test_hidden_player_is_rejected(self):
        path=self.site/'08_speech-separation.html';path.write_text(path.read_text().replace('<audio','<audio hidden',1))
        self.assertTrue(self.check(qc.check_css_audio))
    def test_report_wrong_integer_type_is_rejected(self):
        original=json.loads(self.report.read_bytes())
        for location in ('figure','gain','error'):
            with self.subTest(location=location):
                report=copy.deepcopy(original)
                if location=='figure':report['figure']=76.0
                elif location=='gain':report['parameters']['common_export_gain']=True
                else:report['pcm_integer_measurements']['corrected']['primary']['integer_reference_error_squared_sum_per_channel'][0]=False
                self.report.write_text(json.dumps(report))
                self.assertIn('actual audio/report mismatch','\n'.join(self.check(qc.check_css_figure)))
    def test_unlisted_member_fails_before_destination_creation(self):
        (self.audio/'unexpected').write_text('preserve')
        target=self.root/'rejected'
        with self.assertRaises(ValueError):build_site.stage_css_audio(self.audio,target)
        self.assertFalse(target.exists())
    def test_two_page_depths_rewrite_all_player_urls(self):
        for source,prefix,target in ((build_site.SRC/'08_speech-separation.md','','../codes/chapters/ch08/css_audio'),(build_site.RESEARCH_ROOT/'05_exercises_and_audio.md','../','../../ch08/css_audio')):
            html=build_site.rewrite_site_links(f'<a href="{target}/css_naive.wav">Naive</a><a href="{target}/MANIFEST.json">Manifest</a>',source)
            self.assertIn(f'src="{prefix}css_audio/css_naive.wav"',html);self.assertIn(f'href="{prefix}css_audio/MANIFEST.json"',html)
    def test_both_site_digests_bind_every_css_asset(self):
        original_read = Path.read_bytes
        self.assertEqual(build_site.source_digest(), qc.site_source_digest())
        for digest in (build_site.source_digest, qc.site_source_digest):
            before = digest()
            for changed in sorted(build_site.CSS_AUDIO_ROOT.iterdir()):
                def read(path):
                    return original_read(path) + (b'changed' if path == changed else b'')
                with self.subTest(digest=digest.__qualname__, asset=changed.name):
                    with patch.object(Path, 'read_bytes', read):
                        self.assertNotEqual(before, digest())
    def test_research_topic_anchors_preserve_ids_and_scope_screen_clearance(self):
        source = build_site.RESEARCH_ROOT/'02_aec_wpe_separation.md'
        ids = ('bss', 'neural', 'tflocoformer', 'separation-upstream-audit', 'mffca')
        html, _ = build_site.render(source.read_text(), source)
        for anchor in ids:
            self.assertEqual(html.count(f'id="{anchor}"'), 1)
            self.assertIn(f'id="{anchor}" class="separation-topic-anchor"', html)
        fixture = '<a id="bss"></a>\n\n## Theme\n'
        elsewhere, _ = build_site.render(fixture, build_site.SRC/'fixture.md')
        self.assertNotIn('separation-topic-anchor', elsewhere)
        self.assertIn('@media screen{.main .separation-topic-anchor{scroll-margin-top:60px}}', build_site.CSS)
        from scripts import build_pdf
        self.assertNotIn('separation-topic-anchor', build_pdf.CSS)
    def test_new_and_original_exercise_ids_have_scroll_clearance(self):
        self.assertIn('.main a[id^="e08-"]{scroll-margin-top:60px}',build_site.CSS)
        source=build_site.SRC/'08_speech-separation.md';html,_=build_site.render(source.read_text(),source)
        for number in (7,29,30,31,32):self.assertEqual(html.count(f'id="e08-{number:02}"'),1)

if __name__=='__main__':unittest.main()
