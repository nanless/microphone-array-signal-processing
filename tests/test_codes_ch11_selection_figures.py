"""Independent scalar model oracle and isolated Figure79 publication paths."""
import cmath
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
from matplotlib.figure import Figure
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selection_figure_source', ROOT/'scripts/make_selection_figures.py')
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


class SelectionFigureTests(unittest.TestCase):
    def test_actual_artist_fonts_at_print_and_web_embedding_widths(self):
        # Inspect the prepared artists, rather than echoing a source fontsize
        # constant. Use the exported tight PNG width, including its padding.
        # Math subscripts have their usual smaller glyphs; this checks the
        # nominal text sizes, as the publication requirement specifies.
        font_sizes = []
        savefig = Figure.savefig

        def capture_fonts(fig, *args, **kwargs):
            fig.set_dpi(kwargs['dpi'])
            fig.canvas.draw()
            font_sizes.extend(artist.get_fontsize() for artist in fig.findobj(Text)
                              if artist.get_visible() and artist.get_text().strip())
            renderer = fig.canvas.get_renderer()
            wng_axes = fig.axes[2]
            legend_box = wng_axes.get_legend().get_window_extent(renderer)
            for annotation in wng_axes.texts:
                self.assertFalse(legend_box.overlaps(annotation.get_window_extent(renderer)),
                                 annotation.get_text())
            evidence_axes = fig.axes[3]
            table = evidence_axes.tables[0]
            for cell in table.get_celld().values():
                cell_box = cell.get_window_extent(renderer)
                text_box = cell.get_text().get_window_extent(renderer)
                self.assertGreaterEqual(text_box.x0, cell_box.x0)
                self.assertLessEqual(text_box.x1, cell_box.x1)
                self.assertGreaterEqual(text_box.y0, cell_box.y0)
                self.assertLessEqual(text_box.y1, cell_box.y1)
            table_box = table.get_window_extent(renderer)
            explanatory_boxes = [text.get_window_extent(renderer) for text in evidence_axes.texts]
            for box in explanatory_boxes:
                self.assertFalse(table_box.overlaps(box))
            self.assertFalse(explanatory_boxes[0].overlaps(explanatory_boxes[1]))
            return savefig(fig, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary)/'figure.png'
            with mock.patch.object(Figure, 'savefig', capture_fonts):
                figure.generate_figure(image, Path(temporary)/'report.json')
            self.assertTrue(font_sizes)
            with Image.open(image) as png:
                width_inches = png.width/150
            minimum_pt = min(font_sizes)
            print_size_pt = minimum_pt*(165/25.4)/width_inches
            web_size_css_px = minimum_pt/72*860/width_inches
            self.assertGreaterEqual(print_size_pt, 7)
            self.assertGreaterEqual(web_size_css_px, 11)

    def test_external_figure_report_and_independent_scalar_statistics(self):
        # No production solver is called for this oracle. R=bb^H+qI has the
        # closed scalar Sherman-Morrison inverse; DI uses a separate sinc Gamma.
        positions = [-.04, 0., .04]
        k = 2*math.pi*1000/343
        b = [cmath.exp(1j*k*x*math.sin(math.radians(20))) for x in positions]
        actual = [cmath.exp(1j*k*x*math.sin(math.radians(5))) for x in positions]
        gamma = [[math.sin(k*abs(x-y))/(k*abs(x-y)) if x != y else 1.
                  for y in positions] for x in positions]
        expected = []
        for alpha in (None, 0., .1, 1.):
            if alpha is None:
                weight = [1/3]*3
            else:
                q = .01+alpha*1.01
                s = sum(z.conjugate() for z in b)
                u = [(1-z*s/(q+3))/q for z in b]
                denominator = sum(u)
                weight = [z/denominator for z in u]
            norm = sum(abs(z)**2 for z in weight)
            h = sum(w.conjugate()*a for w, a in zip(weight, actual))
            noise = abs(sum(w.conjugate()*a for w, a in zip(weight, b)))**2+.01*norm
            diffuse = sum(weight[i].conjugate()*gamma[i][j]*weight[j]
                          for i in range(3) for j in range(3)).real
            expected.append((noise, abs(h-1), -10*math.log10(norm),
                             -10*math.log10(diffuse), noise+abs(h-1)**2))
        with tempfile.TemporaryDirectory() as temporary:
            image, report_path = Path(temporary)/'figure.png', Path(temporary)/'report.json'
            report = figure.generate_figure(image, report_path)
            saved = json.loads(report_path.read_text())
            self.assertEqual(saved, report)
            self.assertEqual(report['exercise_id'], 'E11-27')
            self.assertEqual(len(report['source_sha256']), 8)
            for path, digest in report['source_sha256'].items():
                self.assertEqual(digest, hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
            for row, values in zip(report['numerical_case']['candidates'], expected):
                for key, value in zip(('actual_noise_power', 'response_amplitude_error', 'WNG_dB', 'DI_dB', 'actual_NMSE'), values):
                    self.assertAlmostEqual(row[key], value, places=11)
            self.assertEqual(report['plot_data']['acoustic_verdicts'], ['pass', 'fail', 'fail', 'pass'])
            self.assertEqual(report['plot_data']['device_verdicts'], ['undetermined', 'fail', 'fail', 'undetermined'])
            with Image.open(image) as png:
                self.assertGreater(png.width, 1200)
                self.assertGreater(png.height, 800)
                self.assertEqual(png.info['NumericalReportDigest'], hashlib.sha256(report_path.read_bytes()).hexdigest())
                self.assertEqual(json.loads(png.info['GeneratorInputs']), report['source_sha256'])

    def test_invalid_output_paths_rejected_before_model_and_preserve_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root/'sentinel.png'; sentinel.write_bytes(b'keep')
            linked = root/'linked'; linked.symlink_to(root, target_is_directory=True)
            directory = root/'directory.png'; directory.mkdir()
            bad_images = [linked/'new.png', linked/'../new.png', directory,
                          ROOT/'figures/fig46_selection_pareto.png', ROOT/'source.png', root/'wrong.txt']
            with mock.patch.object(figure, 'beamformer_selection_case') as model:
                for path in bad_images:
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        figure.generate_figure(path, root/'result.json')
                for path in [linked/'new.json', root/'wrong.txt', ROOT/'codes/chapters/ch11/reports/meeting_kernel_contracts.json']:
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        figure.generate_figure(sentinel, path)
                model.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'keep')
            self.assertFalse((root/'result.json').exists())
            self.assertFalse((root/'new.png').exists())

    def test_invalid_numerical_result_rejected_before_either_output(self):
        case = figure.beamformer_selection_case()
        case['candidates'][0]['actual_noise_power'] = float('nan')
        with tempfile.TemporaryDirectory() as temporary:
            image, report = Path(temporary)/'figure.png', Path(temporary)/'report.json'
            image.write_bytes(b'old image'); report.write_bytes(b'old report')
            with mock.patch.object(figure, 'beamformer_selection_case', return_value=case):
                with self.assertRaises(ValueError):
                    figure.generate_figure(image, report)
            self.assertEqual(image.read_bytes(), b'old image')
            self.assertEqual(report.read_bytes(), b'old report')


if __name__ == '__main__':
    unittest.main()
