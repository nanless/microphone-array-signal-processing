"""Actual PCM, read-only negative controls and scoped AEC publication layout."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from scripts import build_site, quality_check as qc

ROOT = Path(__file__).resolve().parents[1]


class ReferencePublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root/'codes/chapters/ch06/reference_audio'
        shutil.copytree(ROOT/'codes/chapters/ch06/reference_audio', self.audio)
        self.report = self.root/'codes/chapters/ch06/reports/figure74_reference_timing.json'
        self.report.parent.mkdir()
        shutil.copy2(ROOT/'codes/chapters/ch06/reports/figure74_reference_timing.json', self.report)
        for relative in json.loads(self.report.read_bytes())['source_sha256']:
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        (self.root/'figures').mkdir()
        shutil.copy2(ROOT/'figures/fig74_reference_timing.png', self.root/'figures/fig74_reference_timing.png')
        self.site = self.root/'site'
        self.site.mkdir()
        build_site.stage_reference_audio(self.audio, self.site/'reference_audio')
        (self.site/'research').mkdir()
        for page, prefix in ((self.site/'06_aec.html',''),
                             (self.site/'research/05_exercises_and_audio.html','../')):
            page.write_text('\n'.join(
                f'<audio src="{prefix}reference_audio/{name}" controls preload="none" aria-label="reference"></audio>'
                for name in sorted(qc.REFERENCE_AUDIO_WAVS))
                +f'<a href="{prefix}reference_audio/MANIFEST.json">Manifest</a>')

    def check(self, function):
        errors = []
        with patch.object(qc,'ROOT',self.root), patch.object(qc,'REFERENCE_AUDIO_ROOT',self.audio), patch.object(qc,'SITE',self.site):
            function(errors)
        return errors

    def test_actual_pcm_and_figure_pass_independent_publication_checks(self):
        self.assertEqual(self.check(qc.check_reference_audio), [])
        self.assertEqual(self.check(qc.check_reference_figure), [])

    def test_changed_copy_and_hidden_player_are_rejected_without_repair(self):
        path = self.site/'reference_audio/reference_echo.wav'
        original = path.read_bytes()
        changed = original[:-2]+b'\x01\x00'
        path.write_bytes(changed)
        self.assertTrue(self.check(qc.check_reference_audio))
        self.assertEqual(path.read_bytes(), changed)
        path.write_bytes(original)
        page = self.site/'06_aec.html'
        page.write_text(page.read_text().replace('<audio','<audio hidden',1))
        self.assertTrue(self.check(qc.check_reference_audio))

    def test_report_rejects_numeric_types_even_when_png_digest_matches(self):
        original = json.loads(self.report.read_bytes())
        with Image.open(self.root/'figures/fig74_reference_timing.png') as picture:
            info = dict(picture.info)
        for field in ('figure','gain','channels'):
            with self.subTest(field=field):
                report = copy.deepcopy(original)
                if field == 'figure': report['figure'] = 74.0
                elif field == 'gain': report['parameters']['common_export_gain'] = True
                else: report['samples']['early']['pcm_integer_measurements']['step']['channels'] = True
                self.report.write_text(json.dumps(report, allow_nan=False))
                metadata = {**info, 'NumericalReportDigest': hashlib.sha256(self.report.read_bytes()).hexdigest()}
                class Picture:
                    def __enter__(self):
                        self.info = metadata
                        return self
                    def __exit__(self, *args): return False
                with patch('PIL.Image.open', return_value=Picture()):
                    errors = self.check(qc.check_reference_figure)
                self.assertIn('actual audio/report mismatch', '\n'.join(errors))

    def test_tail_rms_rejects_changed_number_with_matching_report_digest(self):
        report = json.loads(self.report.read_bytes())
        report['plotted_tail_centered_rms'][0] *= 2
        self.report.write_text(json.dumps(report))
        self.assertIn('actual integer PCM', '\n'.join(self.check(qc.check_reference_figure)))

    def test_source_pcm_cannot_be_relabelled_by_updating_its_hash(self):
        path = self.audio/'reference_echo.wav'
        path.write_bytes(path.read_bytes()[:-2]+b'\x01\x00')
        manifest = json.loads((self.audio/'MANIFEST.json').read_bytes())
        manifest['files'][path.name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.audio/'MANIFEST.json').write_text(json.dumps(manifest))
        target = self.root/'rejected'
        with self.assertRaises(ValueError):
            build_site.stage_reference_audio(self.audio, target)
        self.assertFalse(target.exists())

    def test_stage_rejects_unlisted_files_and_linked_destinations(self):
        (self.audio/'unlisted').write_text('keep')
        with self.assertRaises(ValueError):
            build_site.stage_reference_audio(self.audio, self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())
        (self.audio/'unlisted').unlink()
        linked = self.root/'linked'
        linked.symlink_to(self.audio, target_is_directory=True)
        with self.assertRaises(ValueError):
            build_site.stage_reference_audio(self.audio, linked)

    def test_chapter_and_handbook_players_use_published_relative_paths(self):
        for current, prefix, target in (
                (build_site.SRC/'06_aec.md','','../codes/chapters/ch06/reference_audio'),
                (build_site.RESEARCH_ROOT/'05_exercises_and_audio.md','../','../../ch06/reference_audio')):
            html = build_site.rewrite_site_links(
                f'<a href="{target}/reference_echo.wav">Echo</a><a href="{target}/MANIFEST.json">Manifest</a>', current)
            self.assertIn(f'src="{prefix}reference_audio/reference_echo.wav"',html)
            self.assertIn(f'href="{prefix}reference_audio/MANIFEST.json"',html)

    def test_phone_column_budget_preserves_headers_values_and_other_chapters(self):
        table = ('| 算法 | 每样本主要计算 | 收敛受什么影响 | 双讲更新控制 | 适用条件 |\n'
                 '|---|---|---|---|---|\n| NLMS | 矢量内积 | 参考相关 | 冻结更新 | 已对齐参考 |')
        actual, _ = build_site.render(table,build_site.SRC/'06_aec.md')
        self.assertIn('--tutorial-table-width:48em', actual)
        self.assertIn('--tutorial-column-width:13em', actual)
        self.assertIn('冻结更新', actual)
        self.assertIn('scope="col"', actual)
        unrelated, _ = build_site.render(table,build_site.SRC/'05_beamforming.md')
        self.assertNotIn('tutorial-budget-table', unrelated)

    def test_real_markdown_headers_select_only_their_scoped_phone_budgets(self):
        cases = (
            (build_site.SRC/'06_aec.md', '| 处理输出 |', 54, '79.27 dB'),
            (build_site.SRC/'06_aec.md', '| 文件分量 |', 52, '4209136228'),
            (build_site.RESEARCH_ROOT/'05_exercises_and_audio.md', '| 文件 | 取点或输出 |', 60, '播放增益之前'),
            (build_site.RESEARCH_ROOT/'05_exercises_and_audio.md', '| 取点或输出 | 切换窗整数E |', 52, '8589947251'),
        )
        for source, header, budget, value in cases:
            with self.subTest(source=source.name, header=header):
                lines = source.read_text().splitlines()
                start = next(i for i, line in enumerate(lines) if line.startswith(header))
                stop = start
                while stop < len(lines) and lines[stop].startswith('|'):
                    stop += 1
                table = '\n'.join(lines[start:stop])
                actual, _ = build_site.render(table, source)
                self.assertIn(f'--tutorial-table-width:{budget}em', actual)
                self.assertIn(value, actual)
                self.assertIn('scope="col"', actual)
                unrelated, _ = build_site.render(table, build_site.SRC/'07_wpe-dereverberation.md')
                self.assertNotIn('tutorial-budget-table', unrelated)

    def test_chapter06_exercise_links_keep_ids_and_have_screen_scroll_clearance(self):
        self.assertIn('@media screen{.main a[id^="e06-"]{scroll-margin-top:60px}}', build_site.CSS)
        source = build_site.SRC/'06_aec.md'
        actual, _ = build_site.render(source.read_text(), source)
        for number in (40, 41, 42):
            self.assertEqual(actual.count(f'id="e06-{number}"'), 1)
            self.assertIn(f'E06-{number}：', actual)


if __name__ == '__main__':
    unittest.main()
