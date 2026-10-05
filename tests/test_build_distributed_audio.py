"""Actual temporary publication and independent distributed-media failure controls."""
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_pdf, build_site, quality_check as quality


ROOT = Path(__file__).resolve().parents[1]
# Independent editorial contract: not copied from the builder's set.
WAVS = {
    'reference_node1.wav', 'reference_node2.wav', 'array_white.wav', 'array_correlated.wav',
    'local_node1.wav', 'central_white.wav', 'compressed_white.wav', 'central_correlated.wav',
    'compressed_correlated.wav', 'stale_correlated.wav', 'central_node2_correlated.wav',
    'remote_scalar_white.wav', 'transport_pcm16_white.wav', 'clock_misaligned_white.wav',
    'clock_linear_corrected_white.wav', 'packet_zerofill_white.wav', 'packet_local_fallback_white.wav',
}


class DistributedPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.site = Path(cls.temporary.name)/'site'
        with patch.object(build_site, 'OUT', cls.site), contextlib.redirect_stdout(io.StringIO()):
            build_site.main()

    def issues(self):
        errors = []
        with patch.object(quality, 'SITE', self.site):
            quality.check_distributed_audio(errors)
        return errors

    def test_actual_copies_visible_players_and_integer_score_contract(self):
        source = ROOT/'codes/chapters/ch13/distributed_audio'
        self.assertEqual({p.name for p in (self.site/'distributed_audio').iterdir()}, WAVS|{'MANIFEST.json'})
        for name in WAVS|{'MANIFEST.json'}:
            self.assertEqual((source/name).read_bytes(), (self.site/'distributed_audio'/name).read_bytes())
        self.assertEqual(self.issues(), [])
        scores = quality._distributed_integer_pcm(self.site/'distributed_audio')
        steady = scores['central_white']['windows']['steady']
        self.assertEqual((steady['integer_error_squared_sum_E'], steady['integer_reference_squared_sum_D']),
                         (32352145920, 210281178240))
        gap = scores['packet_zerofill_white']['windows']['packet']
        self.assertEqual((gap['samples'],gap['integer_error_squared_sum_E'],gap['integer_reference_squared_sum_D']),
                         (800,3983398320,5841143840))
        node2 = scores['central_node2_correlated']['windows']['steady']
        self.assertEqual(node2['integer_reference_squared_sum_D'],841117066920)
        self.assertNotEqual(node2['integer_reference_squared_sum_D'],4*steady['integer_reference_squared_sum_D'])

    def test_changed_actual_sample_fails_without_repair(self):
        path=self.site/'distributed_audio/central_white.wav'; original=path.read_bytes()
        try:
            changed=bytearray(original);changed[44+2*1600]^=1;path.write_bytes(changed)
            self.assertTrue(any('central_white.wav' in e for e in self.issues()))
            self.assertEqual(path.read_bytes(),bytes(changed))
        finally: path.write_bytes(original)

    def test_extra_and_linked_members_fail(self):
        folder=self.site/'distributed_audio'; extra=folder/'extra.txt'
        try:
            extra.write_text('extra');self.assertTrue(self.issues())
        finally: extra.unlink()
        member=folder/'central_white.wav'; original=member.read_bytes()
        try:
            member.unlink();member.symlink_to(ROOT/'codes/chapters/ch13/distributed_audio/central_white.wav')
            self.assertTrue(self.issues())
        finally: member.unlink();member.write_bytes(original)

    def test_hidden_player_and_closed_details_do_not_satisfy_media_contract(self):
        page=self.site/'13_distributed-enhancement.html';original=page.read_text()
        match=re.search(r'<audio\b[^>]*src="distributed_audio/central_white.wav"[^>]*>.*?</audio>',original,re.S)
        self.assertIsNotNone(match)
        for hidden in ('<div hidden>'+match[0]+'</div>', '<details>'+match[0]+'</details>'):
            try:
                page.write_text(original.replace(match[0],hidden,1))
                self.assertTrue(any('播放器' in e for e in self.issues()))
            finally: page.write_text(original)

    def test_published_player_properties_and_missing_manifest_are_detected(self):
        page=self.site/'research/05_exercises_and_audio.html';original=page.read_text()
        match=re.search(r'<audio\b[^>]*src="../distributed_audio/central_white.wav"[^>]*>',original)
        self.assertIsNotNone(match)
        for altered in (match[0].replace('preload="none"','preload="auto"'),match[0].replace('<audio','<audio autoplay')):
            try:
                page.write_text(original.replace(match[0],altered,1));self.assertTrue(self.issues())
            finally:page.write_text(original)
        try:
            page.write_text(original.replace('href="../distributed_audio/MANIFEST.json"','href="#main-content"'))
            self.assertTrue(any('清单' in e for e in self.issues()))
        finally:page.write_text(original)

    def test_routes_and_pdf_id_follow_stable_file_identity(self):
        chapter=ROOT/'chapters/13_distributed-enhancement.md'
        research=ROOT/'codes/chapters/ch00/research/05_exercises_and_audio.md'
        for source, url, expected in (
                (chapter,'../codes/chapters/ch13/distributed_audio/central_white.wav','distributed_audio/central_white.wav'),
                (research,'../../ch13/distributed_audio/central_white.wav','../distributed_audio/central_white.wav')):
            html=build_site.render('[控制]('+url+')',source)[0]
            self.assertIn('src="'+expected+'"',html)
        html=build_site.render('[清单](../codes/chapters/ch13/distributed_audio/MANIFEST.json?view=1#schema)',chapter)[0]
        self.assertIn('href="distributed_audio/MANIFEST.json?view=1#schema"',html)
        self.assertEqual(build_pdf.rewrite_repository_links(
            '<a href="13_distributed-enhancement.md#e13-24">GEVD</a>'
            '<a href="14_appendix-symbols-math.md#sec-1">附录A</a>'
            '<a href="15_appendix-guide.md#sec-13-6">练习</a>',ROOT/'chapters/00_overview.md'),
            '<a href="#ch-15-e13-24">GEVD</a><a href="#ch-12">附录A</a><a href="#ch-13-sec-13-6">练习</a>')
        self.assertIn('id="e13-24"',(self.site/'13_distributed-enhancement.html').read_text())

    def test_site_digest_binds_all_eighteen_distributed_members(self):
        original_read=Path.read_bytes
        for digest in (build_site.source_digest,quality.site_source_digest):
            before=digest()
            for changed in (ROOT/'codes/chapters/ch13/distributed_audio').iterdir():
                def read(path):
                    return original_read(path)+(b'changed' if path==changed else b'')
                with self.subTest(digest=digest.__name__,member=changed.name),patch.object(Path,'read_bytes',read):
                    self.assertNotEqual(before,digest())

    def test_staging_rejects_parent_links_and_extra_members_before_copy(self):
        import shutil
        source=ROOT/'codes/chapters/ch13/distributed_audio'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);copied=root/'source';shutil.copytree(source,copied)
            extra=copied/'unexpected.wav';extra.write_bytes(b'not a WAV')
            destination=root/'published'
            with self.assertRaises(ValueError):build_site.stage_distributed_audio(copied,destination)
            self.assertFalse(destination.exists())
            extra.unlink()
            link=root/'linked-parent';link.symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError):build_site.stage_distributed_audio(source,link/'published')
            self.assertFalse(destination.exists())


class DistributedFigureTests(unittest.TestCase):
    def report(self,number):
        name={70:'compression',71:'updates',72:'transport'}[number]
        return ROOT/f'codes/chapters/ch13/reports/figure{number}_distributed_{name}.json'

    def test_all_reports_pass_independent_arithmetic(self):
        for number in (70,71,72):
            quality._check_distributed_figure_report(self.report(number),number)

    def assert_rejected(self,number,mutate):
        data=json.loads(self.report(number).read_text());mutate(data)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'changed.json';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):quality._check_distributed_figure_report(path,number)

    def test_stale_effective_weight_and_shared_snapshot_corruption_fail(self):
        def stale(d):
            h=d['results']['trajectories']['round_robin']['history'][1]
            h['cached_outputs'][0]=h['outputs_at_last_solve'][0]
        self.assert_rejected(71,stale)
        def wrong_snapshot(d):
            h=d['results']['trajectories']['simultaneous']['history'][1]
            h['solutions'][1]['incoming_compressions'][0]=h['compressions_after'][0]
        self.assert_rejected(71,wrong_snapshot)

    def test_wrong_integer_reference_and_unsigned_cross_terms_fail(self):
        def wrong_D(d):d['results']['rows']['central_white']['pcm']['D']*=4
        self.assert_rejected(70,wrong_D)
        def wrong_gap(d):d['results']['rows']['packet_zerofill_white']['pcm']['packet']['samples']=28800
        self.assert_rejected(72,wrong_gap)
        def unsigned(d):
            row=d['results']['rows']['clock_misaligned_white']['float_components']
            row['target_noise_cross_power']=abs(row['target_noise_cross_power'])
        self.assert_rejected(72,unsigned)


if __name__=='__main__':unittest.main()
