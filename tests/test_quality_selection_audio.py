"""Real eight-file selection publication, independent PCM and negative fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import build_site, quality_check as quality
from codes.chapters.ch11.examples import generate_selection_audio as generator

ROOT = Path(__file__).resolve().parents[1]


class SelectionPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets, cls.manifest = generator.expected_assets()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'codes/chapters/ch11/scenario_audio'
        self.site = self.root/'site'
        self.published = self.site/'scenario_audio'
        self.source.mkdir(parents=True); self.published.mkdir(parents=True)
        (self.site/'research').mkdir()
        for name,data in self.assets.items():
            (self.source/name).write_bytes(data); (self.published/name).write_bytes(data)
        for path in self.manifest['source_sha256']:
            target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/path).read_bytes())
        script=self.root/'scripts/make_figures.py';script.parent.mkdir();script.write_text('# fixture\n')
        self.pages=[(self.site/'11_selection-guide.html',''),
                    (self.site/'research/05_exercises_and_audio.html','../')]
        for page,prefix in self.pages:page.write_text(self.html(prefix))
        for name,value in [('ROOT',self.root),('SITE',self.site),('SCENARIO_AUDIO_ROOT',self.source)]:
            patcher=patch.object(quality,name,value);patcher.start();self.addCleanup(patcher.stop)

    def html(self,prefix):
        base=prefix+'scenario_audio/'
        return ''.join(f'<audio controls preload="none" aria-label="{name}" src="{base}{name}"></audio>'
                       for name in sorted(self.manifest['files']))+f'<a href="{base}MANIFEST.json">Manifest</a>'

    def issues(self):
        errors=[];quality.check_scenario_audio(errors);return errors

    def test_real_assets_and_staging_are_readonly(self):
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(),[])
        destination=self.root/'stage'
        self.assertEqual(build_site.stage_scenario_audio(self.source,destination),set(self.assets))
        self.assertEqual({p.name:p.read_bytes() for p in destination.iterdir()},self.assets)
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_wrong_source_integer_denominator_and_gain_cannot_publish(self):
        for alter in [lambda m:m['source_sha256'].update({generator.SOURCE_PATHS[0]:'0'*64}),
                      lambda m:m['pcm_analysis']['candidates']['selection_single_fir3.wav'].update(integer_reference_squared_sum=True),
                      lambda m:m.update(common_export_gain=1.),
                      lambda m:m['pcm_analysis']['candidates']['selection_dual_fir9.wav'].update(aligned_total_nmse=0.)]:
            modified=copy.deepcopy(self.manifest);alter(modified)
            for folder in (self.source,self.published):(folder/'MANIFEST.json').write_text(json.dumps(modified))
            before={p.name:p.read_bytes() for p in self.source.iterdir()}
            self.assertTrue(self.issues())
            with self.assertRaises(ValueError):build_site.stage_scenario_audio(self.source,self.root/'stage')
            self.assertFalse((self.root/'stage').exists())
            self.assertEqual(before,{p.name:p.read_bytes() for p in self.source.iterdir()})

    def test_strict_json_members_and_actual_pcm_changes_fail(self):
        for text in ['[]','{"x":1,"x":1}','{"x":NaN}','{"x":1e999}']:
            (self.source/'MANIFEST.json').write_text(text)
            with self.assertRaises(ValueError):build_site.stage_scenario_audio(self.source,self.root/'stage')
        (self.source/'MANIFEST.json').write_bytes(self.assets['MANIFEST.json'])
        extra=self.published/'extra';extra.mkdir();self.assertTrue(self.issues());extra.rmdir()
        target=self.published/'selection_single_fir9.wav'
        data=bytearray(target.read_bytes());data[10000]^=1;target.write_bytes(data)
        self.assertTrue(self.issues())

    def test_parent_links_and_lexical_escape_fail_before_writes(self):
        outside=self.root/'outside';outside.mkdir();(outside/'nested').mkdir()
        linked=self.root/'linked';linked.symlink_to(outside,target_is_directory=True)
        for target in [linked/'stage',linked/'nested'/'stage',linked/'..'/'stage']:
            with self.assertRaises(ValueError):build_site.stage_scenario_audio(self.source,target)
        self.assertEqual(list((outside/'nested').iterdir()),[])
        self.assertFalse((self.root/'stage').exists())
        alias=self.root/'source_link';alias.symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ValueError):build_site.stage_scenario_audio(alias,self.root/'stage')

    def test_all_eight_visible_keyboard_controls_and_manifest_required(self):
        for page,prefix in self.pages:
            valid=self.html(prefix)
            for invalid in [valid.replace('<audio ','<audio hidden ',1),valid.replace('controls ','',1),
                            valid.replace('<audio ','<audio autoplay ',1),'<div inert>'+valid+'</div>',
                            '<details><summary>Audio</summary>'+valid+'</details>',valid.replace('<a ','<a hidden ',1),
                            valid+valid[:valid.index('</audio>')+8]]:
                page.write_text(invalid);self.assertTrue(self.issues())
            page.write_text('<details open><summary>Audio</summary>'+valid+'</details>')
        self.assertEqual(self.issues(),[])

    def test_only_owned_scenario_links_get_players(self):
        chapter=ROOT/'chapters/11_selection-guide.md'
        valid='<a href="../codes/chapters/ch11/scenario_audio/selection_single_fir3.wav">sample</a>'
        self.assertIn('src="scenario_audio/selection_single_fir3.wav"',build_site.rewrite_site_links(valid,chapter))
        for href in ['https://example.com/scenario_audio/selection_single_fir3.wav',
                     '../codes/chapters/ch10/scenario_audio/selection_single_fir3.wav',
                     '../codes/chapters/ch11/scenario_audio/unlisted.wav']:
            self.assertNotIn('<audio ',build_site.rewrite_site_links(f'<a href="{href}">sample</a>',chapter))

    def report(self):
        # Fixed independent wave/struct sums recorded in the readonly review;
        # no expected-value helper from quality or the figure is called.
        energy={'single':395812398600,'dual':791595378000}
        errors={'single':{'fir3':87210066600,'fir9':152494200},
                'dual':{'fir3':91223542800,'fir9':249713672400}}
        integers={scene:{kind:{'source_window':[1600,30400],
                  'output_window':[1600+delay,30400+delay],'scored_samples':28800,
                  'integer_error_squared_sum':errors[scene][kind],
                  'integer_reference_squared_sum':energy[scene]}
                  for kind,delay in [('fir3',1),('fir9',4)]} for scene in energy}
        pcm={scene:{kind:count/energy[scene] for kind,count in errors[scene].items()} for scene in energy}
        analytic={'single':{'fir3':.22034251695016538,'fir9':.0003838134799433653},
                  'dual':{'fir3':.11525515368901426,'fir9':.3154807375197196}}
        def decision(values):
            endpoints={k:[q*values['single'][k]+(1-q)*values['dual'][k] for q in [.25,.75]] for k in ['fir3','fir9']}
            # Solve the two linear functions using endpoint intercept/slopes.
            intercept=values['dual']['fir3']-values['dual']['fir9']
            slope=(values['single']['fir3']-values['dual']['fir3'])-(values['single']['fir9']-values['dual']['fir9'])
            return {'q_single_weight':[.25,.75],'endpoint_costs':endpoints,
                    'interval_worst':{k:max(v) for k,v in endpoints.items()},
                    'scene_worst':{k:max(values['single'][k],values['dual'][k]) for k in endpoints},
                    'crossing_q_single':-intercept/slope}
        return {'schema_version':1,'script_sha256':hashlib.sha256((self.root/'scripts/make_figures.py').read_bytes()).hexdigest(),
                'audio_manifest_sha256':hashlib.sha256((self.source/'MANIFEST.json').read_bytes()).hexdigest(),
                'scope':'mathematical tones; scenario-weighted NMSE, not pooled reference energy or mean dB',
                'integer_pcm':integers,'pcm_nmse':pcm,'analytic_nmse':analytic,
                'pcm_decision':decision(pcm),'analytic_decision':decision(analytic)}

    def test_figure_report_independently_bound_to_actual_pcm(self):
        report=self.report()
        path=self.root/'report.json';path.write_text(json.dumps(report))
        before=path.read_bytes(),path.stat().st_mtime_ns
        quality._check_selection_scenarios_report(path)
        self.assertEqual(before,(path.read_bytes(),path.stat().st_mtime_ns))
        for alter in [lambda r:r.update(schema_version=True),lambda r:r.update(script_sha256='0'*64),
                      lambda r:r['pcm_decision'].update(crossing_q_single=0.5),
                      lambda r:r['analytic_nmse']['single'].update(fir9=.2),
                      lambda r:r['integer_pcm']['dual']['fir3'].update(scored_samples=False),
                      lambda r:r['pcm_decision']['interval_worst'].update(fir3=r['pcm_decision']['scene_worst']['fir3']),
                      lambda r:r['pcm_nmse']['dual'].update(fir9=True)]:
            changed=copy.deepcopy(report);alter(changed);path.write_text(json.dumps(changed))
            before=path.read_bytes(),path.stat().st_mtime_ns
            with self.assertRaises(ValueError):quality._check_selection_scenarios_report(path)
            self.assertEqual(before,(path.read_bytes(),path.stat().st_mtime_ns))


if __name__=='__main__':unittest.main()
