"""New figure/report checks stay isolated from every published asset."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from codes.chapters.ch09.examples.chapter09_tracking_audio import generate
from scripts.make_tracking_figures import generate_figure, SOURCE_PATHS, ROOT


class TrackingLifecycleFigureTests(unittest.TestCase):
    def test_real_pcm_report_ages_provenance_and_new_figure_only(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);audio=base/'audio';generate(audio)
            saved={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in audio.iterdir()}
            image=base/'figure.png';report=base/'report.json'
            result=generate_figure(image,report,audio_directory=audio)
            self.assertEqual(saved,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in audio.iterdir()})
            self.assertEqual(json.loads(report.read_text()),result)
            self.assertEqual(result['input_sha256']['array_noisy.wav'],
                             hashlib.sha256((audio/'array_noisy.wav').read_bytes()).hexdigest())
            self.assertEqual(result['input_sha256']['source.wav'],
                             hashlib.sha256((audio/'source.wav').read_bytes()).hexdigest())
            self.assertEqual(set(result['source_sha256']),set(SOURCE_PATHS))
            for path,sha in result['source_sha256'].items():
                self.assertEqual(sha,hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
            p=result['plot_data']
            self.assertEqual(p['last_valid_measurement_ticks'][81],2*(81*160)+511)
            for k in range(82,106):
                state_age=(2*(160*k)+511)-(2*(160*81)+511)
                availability_age=2*(160*k+512)-(2*(160*81)+511)
                self.assertEqual(p['state_age_s'][k],state_age/32000)
                self.assertEqual(p['available_age_s'][k],availability_age/32000)
            self.assertEqual(result['lifecycle']['counts']['confirmed_publications'],187)
            self.assertEqual(p['available_time_s'][100],1.032)
            self.assertEqual(p['lifecycle_phase_code'][99],2)
            self.assertEqual(p['lifecycle_phase_code'][100],0)
            with Image.open(image) as png:
                self.assertEqual(png.info['SourceScript'],'scripts/make_tracking_figures.py')
                self.assertEqual(png.info['SourceScriptDigest'],result['script_sha256'])
                self.assertEqual(png.info['AudioManifestDigest'],result['input_sha256']['MANIFEST.json'])
                self.assertGreater(png.width,1000)
                self.assertGreater(png.height,1200)

    def test_disallowed_report_and_linked_parents_rejected_before_input_read(self):
        protected=ROOT/'codes/chapters/ch09/reports/figure22_tracking.json'
        old=(protected.read_bytes(),protected.stat().st_mtime_ns)
        with tempfile.TemporaryDirectory() as name:
            base=Path(name)
            with patch('scripts.make_tracking_figures.run_demo',side_effect=AssertionError('must not read')):
                with self.assertRaises(ValueError):generate_figure(base/'a.png',protected)
                with self.assertRaises(ValueError):generate_figure(ROOT/'figures/fig22_tracking.png',base/'a.json')
                target=base/'target';target.mkdir();link=base/'linked';link.symlink_to(target)
                with self.assertRaises(ValueError):generate_figure(link/'a.png',base/'a.json')
                self.assertEqual(list(target.iterdir()),[])
        self.assertEqual(old,(protected.read_bytes(),protected.stat().st_mtime_ns))
