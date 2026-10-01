"""Equal-DRR five-WAV publication with independent PCM and spectral expectations."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_site, quality_check as quality
from codes.chapters.appendix_b.examples import generate_response_audio as generator

ROOT = Path(__file__).resolve().parents[1]


class ResponsePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets, cls.manifest = generator.expected_assets()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'codes/chapters/appendix_b/response_audio'
        self.site = self.root / 'site'
        self.published = self.site / 'response_audio'
        self.source.mkdir(parents=True)
        self.published.mkdir(parents=True)
        (self.site / 'research').mkdir()
        for name, data in self.assets.items():
            (self.source / name).write_bytes(data)
            (self.published / name).write_bytes(data)
        for path in self.manifest['source_sha256']:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / path).read_bytes())
        script = self.root / 'scripts/make_figures.py'
        script.parent.mkdir()
        script.write_text('# independent fixture\n')
        self.pages = [(self.site / '13_appendix-guide.html', ''),
                      (self.site / 'research/05_exercises_and_audio.html', '../')]
        for page, prefix in self.pages:
            page.write_text(self.html(prefix))
        for name, value in [('ROOT', self.root), ('SITE', self.site),
                            ('RESPONSE_AUDIO_ROOT', self.source)]:
            patcher = patch.object(quality, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def html(self, prefix):
        base = prefix + 'response_audio/'
        return ''.join(f'<audio controls preload="none" aria-label="{name}" src="{base}{name}"></audio>'
                       for name in sorted(self.manifest['files'])) + f'<a href="{base}MANIFEST.json">Manifest</a>'

    def issues(self):
        errors = []
        quality.check_response_audio(errors)
        return errors

    def test_actual_integer_scores_and_staging_are_readonly(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(), [])
        # Recorded by an independent proposed-fixture calculation before
        # the teaching generator existed; not derived by the tested helper.
        rows = quality._response_integer_pcm(self.published)
        for method, numerator in [('full_a', 209299622400), ('full_b', 99939578400)]:
            row = rows['response_' + method + '.wav']
            self.assertEqual(row['integer_error_squared_sum'], numerator)
            self.assertEqual(row['integer_reference_squared_sum'], 309262788000)
            self.assertEqual(row['scored_samples'], 28800)
            self.assertEqual(row['nmse'], numerator / 309262788000)
        stage = self.root / 'stage'
        self.assertEqual(build_site.stage_response_audio(self.source, stage), set(self.assets))
        self.assertEqual({p.name: p.read_bytes() for p in stage.iterdir()}, self.assets)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                                  for p in self.source.iterdir()})

    def test_false_integer_gain_source_and_scores_cannot_publish(self):
        for alter in [lambda m: m.update(common_export_gain=.8),
                      lambda m: m['source_sha256'].update({generator.SOURCE_PATHS[0]: '0' * 64}),
                      lambda m: m['pcm_analysis']['candidates']['response_full_a.wav'].update(integer_reference_squared_sum=True),
                      lambda m: m['pcm_analysis']['candidates']['response_full_b.wav'].update(nmse=0.)]:
            modified = copy.deepcopy(self.manifest)
            alter(modified)
            for folder in (self.source, self.published):
                (folder / 'MANIFEST.json').write_text(json.dumps(modified))
            before = {p.name: p.read_bytes() for p in self.source.iterdir()}
            self.assertTrue(self.issues())
            with self.assertRaises(ValueError):
                build_site.stage_response_audio(self.source, self.root / 'stage')
            self.assertFalse((self.root / 'stage').exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def test_real_byte_change_extra_members_and_json_fail(self):
        for text in ['[]', '{"x":1,"x":1}', '{"x":NaN}', '{"x":1e999}']:
            (self.source / 'MANIFEST.json').write_text(text)
            with self.assertRaises(ValueError):
                build_site.stage_response_audio(self.source, self.root / 'stage')
        (self.source / 'MANIFEST.json').write_bytes(self.assets['MANIFEST.json'])
        extra = self.published / 'unexpected'
        extra.mkdir()
        self.assertTrue(self.issues())
        extra.rmdir()
        target = self.published / 'response_full_b.wav'
        data = bytearray(target.read_bytes())
        data[10000] ^= 1
        target.write_bytes(data)
        self.assertTrue(self.issues())

    def test_linked_parents_and_cancelled_traversal_fail_before_writes(self):
        outside = self.root / 'outside'
        outside.mkdir()
        linked = self.root / 'linked'
        linked.symlink_to(outside, target_is_directory=True)
        for target in [linked / 'stage', linked / '..' / 'stage']:
            with self.assertRaises(ValueError):
                build_site.stage_response_audio(self.source, target)
        self.assertEqual(list(outside.iterdir()), [])
        alias = self.root / 'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            build_site.stage_response_audio(alias, self.root / 'stage')

    def test_all_visible_controls_and_manifest_required(self):
        for page, prefix in self.pages:
            valid = self.html(prefix)
            for invalid in [valid.replace('<audio ', '<audio hidden ', 1),
                            valid.replace('controls ', '', 1),
                            valid.replace('<audio ', '<audio autoplay ', 1),
                            '<div inert>' + valid + '</div>',
                            valid.replace('<a ', '<a hidden ', 1),
                            '<details><summary>Audio</summary>' + valid + '</details>']:
                page.write_text(invalid)
                self.assertTrue(self.issues())
            page.write_text('<details open><summary>Audio</summary>' + valid + '</details>')
        self.assertEqual(self.issues(), [])

    def test_only_owned_links_get_players(self):
        source = ROOT / 'chapters/13_appendix-guide.md'
        valid = '../codes/chapters/appendix_b/response_audio/response_full_b.wav'
        self.assertIn('src="response_audio/response_full_b.wav"',
                      build_site.rewrite_site_links(f'<a href="{valid}">sample</a>', source))
        for href in ['https://example.com/response_audio/response_full_b.wav',
                     '../codes/chapters/ch11/response_audio/response_full_b.wav',
                     '../codes/chapters/appendix_b/response_audio/unlisted.wav']:
            self.assertNotIn('<audio ', build_site.rewrite_site_links(f'<a href="{href}">sample</a>', source))

    def report(self):
        import math
        gains = {'a': [(2+math.sqrt(2))/4, .5], 'b': [(2-math.sqrt(2))/4, .5]}
        rows = {}
        for name, numerator in [('full_a', 209299622400), ('full_b', 99939578400)]:
            rows['response_'+name+'.wav'] = {
                'integer_error_squared_sum': numerator,
                'integer_reference_squared_sum': 309262788000, 'scored_samples': 28800,
                'mse': numerator/(28800*32768**2), 'nmse': numerator/309262788000,
            }
        return {
            'schema_version': 1,
            'script_sha256': hashlib.sha256((self.root/'scripts/make_figures.py').read_bytes()).hexdigest(),
            'audio_manifest_sha256': hashlib.sha256((self.source/'MANIFEST.json').read_bytes()).hexdigest(),
            'scope': 'known short FIRs and equal-power tones; no real room, blind estimation or listening test',
            'score_window': [1600, 30400], 'frequencies_hz': [2000, 4000],
            'rir_drr_db': 10*math.log10(2), 'reflection_squared_gain': gains,
            'analytic_output_reflection_nmse': {name: sum(values)/2 for name, values in gains.items()},
            'integer_pcm': rows,
        }

    def test_report_cannot_replace_spectral_weights_with_rir_energy_ratio(self):
        path = self.root/'report.json'
        valid = self.report(); path.write_text(json.dumps(valid))
        quality._check_equal_drr_report(path)
        for alter in [lambda r: r.update(schema_version=True),
                      lambda r: r.update(script_sha256='0'*64),
                      lambda r: r.update(score_window=[0,32002]),
                      lambda r: r.update(frequencies_hz=[0,8000]),
                      lambda r: r['reflection_squared_gain'].update(a=[.5,.5]),
                      lambda r: r['analytic_output_reflection_nmse'].update(a=.5),
                      lambda r: r['integer_pcm']['response_full_a.wav'].update(integer_error_squared_sum=209299622401)]:
            invalid=copy.deepcopy(valid);alter(invalid);path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):quality._check_equal_drr_report(path)


if __name__ == '__main__':
    unittest.main()
