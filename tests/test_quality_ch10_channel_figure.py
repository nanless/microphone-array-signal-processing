"""Isolated figure78 publication and preflight controls; no formal writes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from scripts import make_channel_figures as figure
from codes.chapters.ch10.examples import generate_channel_audio as generator


class ChannelFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.audio = cls.root/'audio'
        cls.manifest = generator.generate_assets(cls.audio)
        cls.png = cls.root/'fig78.png'
        cls.report_path = cls.root/'figure78.json'
        cls.before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in cls.audio.iterdir()}
        cls.report = figure.generate_figure(cls.png, cls.report_path, audio_directory=cls.audio)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_current_sources_actual_pcm_and_all_numerical_boundaries(self):
        report = json.loads(self.report_path.read_bytes())
        self.assertEqual(report['exercise_id'], 'E10-34')
        self.assertEqual(set(report['source_sha256']), set(generator.SOURCE_PATHS)|{'scripts/make_channel_figures.py'})
        self.assertEqual(report['source_sha256'], {name: hashlib.sha256((figure.ROOT/name).read_bytes()).hexdigest() for name in figure.SOURCE_PATHS})
        self.assertEqual(report['input_sha256'], {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.audio.iterdir()})
        self.assertEqual(report['samples'], self.manifest['samples'])
        self.assertEqual(report['analytic'], self.manifest['analytic'])
        p = report['plot_data']
        self.assertEqual(p['target_gain_analytic'], [1., 0., 1.])
        self.assertEqual(p['reference_NMSE_analytic'], [.5, 1.5, 1.5])
        self.assertEqual(p['effective_weights_after_prescribed_fault'], [[1., -.5, .5], [0., -.5, .5], [0., .5, .5]])
        self.assertEqual(p['reference_NMSE_pcm'], [73014637900/146027417700,
                        219018750300/146027417700, 219050854800/146027417700])
        self.assertEqual(p['waveform_interval_samples'], [2400, 2528])
        self.assertEqual(len(p['waveform_time_ms']), 128)
        self.assertEqual(p['waveform_time_ms'][0], 150)
        self.assertEqual(p['waveform_time_ms'][-1], 157.9375)
        self.assertGreater(p['projection_gain_pcm_diagnostic'][1], 0)
        self.assertLess(p['projection_gain_pcm_diagnostic'][1], .001)
        self.assertIn('no gain compensation', p['projection_scope'])
        self.assertIn('not per-frequency SCM', p['known_covariance_units'])
        self.assertEqual(self.before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.audio.iterdir()})

    def test_png_metadata_actual_file_identity(self):
        with Image.open(self.png) as image:
            self.assertGreater(image.width, 1000); self.assertGreater(image.height, 1000)
            self.assertEqual(image.info['SourceScript'], 'scripts/make_channel_figures.py')
            self.assertEqual(image.info['SourceScriptDigest'], self.report['script_sha256'])
            self.assertEqual(json.loads(image.info['GeneratorInputs']), self.report['source_sha256'])
            self.assertEqual(image.info['AudioManifestDigest'], self.report['input_sha256']['MANIFEST.json'])
            self.assertEqual(image.info['NumericalReportDigest'], hashlib.sha256(self.report_path.read_bytes()).hexdigest())

    def test_bad_destinations_are_rejected_before_inputs_and_no_report_is_touched(self):
        targets = [figure.ROOT/'figures/fig63_noise_mismatch.png',
                   figure.ROOT/'codes/chapters/ch10/core/channel_audio.py',
                   figure.ROOT/'codes/chapters/ch10/reports/figure63_noise_mismatch.json']
        for path in targets:
            with self.subTest(path=path), patch.object(figure, 'check_assets') as checked:
                with self.assertRaises(ValueError):
                    figure.generate_figure(path if path.suffix=='.png' else self.root/'not-written.png',
                                           path if path.suffix!='.png' else self.root/'not-written.json',
                                           audio_directory=self.audio)
                checked.assert_not_called()
                self.assertFalse((self.root/'not-written.png').exists())
                self.assertFalse((self.root/'not-written.json').exists())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); link = root/'link'; link.symlink_to(self.root, target_is_directory=True)
            with patch.object(figure, 'check_assets') as checked:
                with self.assertRaises(ValueError): figure.generate_figure(link/'bad.png', root/'out.json', audio_directory=self.audio)
                checked.assert_not_called()
            # Even an ordinary external path may not replace an input member.
            with patch.object(figure, 'check_assets') as checked:
                with self.assertRaises(ValueError): figure.generate_figure(root/'out.png', self.audio/'MANIFEST.json', audio_directory=self.audio)
                checked.assert_not_called()
            self.assertFalse((root/'out.png').exists()); self.assertFalse((root/'out.json').exists())

    def test_extra_input_refused_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'audio'; generator.generate_assets(source)
            (source/'extra.txt').write_text('keep')
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
            with self.assertRaises(ValueError): figure.generate_figure(root/'out.png', root/'out.json', audio_directory=source)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()})
            self.assertFalse((root/'out.png').exists()); self.assertFalse((root/'out.json').exists())


if __name__ == '__main__':
    unittest.main()
