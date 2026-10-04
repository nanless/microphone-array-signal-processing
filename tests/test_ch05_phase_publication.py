"""Independent actual-PCM publication and strict numeric-type negative controls."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from scripts import quality_check as qc

ROOT = Path(__file__).resolve().parents[1]


class PhasePublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root/'codes/chapters/ch05/phase_audio'
        shutil.copytree(ROOT/'codes/chapters/ch05/phase_audio', self.audio)
        self.manifest = json.loads((self.audio/'MANIFEST.json').read_bytes())
        self.report_path = self.root/'codes/chapters/ch05/reports/figure73_phase_reference.json'
        self.report_path.parent.mkdir()
        shutil.copy2(ROOT/'codes/chapters/ch05/reports/figure73_phase_reference.json', self.report_path)
        report = json.loads(self.report_path.read_bytes())
        for relative in report['source_sha256']:
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        (self.root/'figures').mkdir()
        shutil.copy2(ROOT/'figures/fig73_phase_reference.png', self.root/'figures/fig73_phase_reference.png')
        self.site = self.root/'site'
        shutil.copytree(self.audio, self.site/'phase_audio')
        (self.site/'research').mkdir()
        for page, prefix in ((self.site/'05_beamforming.html',''),
                             (self.site/'research/05_exercises_and_audio.html','../')):
            page.write_text('\n'.join(f'<audio src="{prefix}phase_audio/{name}" controls preload="none" aria-label="phase"></audio>'
                                      for name in ('phase_reference.wav','phase_flip.wav','phase_quadrature.wav'))
                            +f'<a href="{prefix}phase_audio/MANIFEST.json">Manifest</a>')

    def check(self, function):
        errors = []
        with patch.object(qc,'ROOT',self.root), patch.object(qc,'PHASE_AUDIO_ROOT',self.audio), patch.object(qc,'SITE',self.site):
            function(errors)
        return errors

    def test_actual_audio_copies_and_independent_integer_scores_pass(self):
        self.assertEqual(self.check(qc.check_phase_audio), [])
        self.assertEqual(self.check(qc.check_phase_figure), [])

    def test_extra_file_and_changed_pcm_fail_without_rewriting(self):
        target = self.site/'phase_audio/phase_flip.wav'
        original = target.read_bytes()
        target.write_bytes(original[:-2]+b'\x01\x00')
        changed = target.read_bytes()
        self.assertTrue(self.check(qc.check_phase_audio))
        self.assertEqual(target.read_bytes(), changed)
        target.write_bytes(original)
        (self.site/'phase_audio/unlisted').write_bytes(b'not an asset')
        self.assertTrue(self.check(qc.check_phase_audio))

    def test_figure_rejects_equal_valued_wrong_types_with_matching_png_report_sha(self):
        original = json.loads(self.report_path.read_bytes())
        with Image.open(self.root/'figures/fig73_phase_reference.png') as picture:
            info = dict(picture.info)
        for field in ('figure','gain','channels'):
            with self.subTest(field=field):
                report = copy.deepcopy(original)
                if field == 'figure': report['figure'] = 73.0
                elif field == 'gain': report['parameters']['common_export_gain'] = True
                else: report['samples']['reference']['pcm_integer_measurements']['channels'] = True
                self.report_path.write_text(json.dumps(report, allow_nan=False))
                # Match the actual changed JSON SHA so this independently tests
                # the schema, rather than merely a stale-image checksum.
                metadata = {**info, 'NumericalReportDigest': hashlib.sha256(self.report_path.read_bytes()).hexdigest()}
                class Picture:
                    def __enter__(self):
                        self.info = metadata
                        return self
                    def __exit__(self, *args): return False
                with patch('PIL.Image.open', return_value=Picture()):
                    errors = self.check(qc.check_phase_figure)
                self.assertTrue(errors)
                self.assertIn('actual audio/report mismatch', '\n'.join(errors))

    def test_missing_and_hidden_players_fail(self):
        path = self.site/'05_beamforming.html'
        text = path.read_text()
        path.write_text(text.replace('<audio','<audio hidden',1))
        self.assertTrue(self.check(qc.check_phase_audio))
        path.write_text(text.replace('controls','autoplay',1))
        self.assertTrue(self.check(qc.check_phase_audio))


if __name__ == '__main__':
    unittest.main()
