"""Independent publication arithmetic and isolated negative controls."""
import copy
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import quality_check as q, build_site
from codes.chapters.ch10.examples.generate_channel_audio import generate_assets


class ChannelPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.directory=Path(cls.temp.name)/'audio'
        cls.manifest=generate_assets(cls.directory)
        cls.pcm=q._check_channel_pcm(cls.directory,cls.manifest)
        cls.report=q._read_audio_manifest(q.ROOT/'codes/chapters/ch10/reports/figure78_channel_failure.json')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_known_integer_scores_and_readonly_repetition(self):
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        pcm=q._check_channel_pcm(self.directory,self.manifest)
        q._check_channel_figure_report(self.report,self.manifest,pcm,self.directory)
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.directory.iterdir()})
        reference=pcm['reference'][0][2400:29600]
        self.assertEqual(sum(x*x for x in reference),146027417700)
        for role,energy in [('healthy_output',73014637900),('stale_output',219018750300),
                            ('recomputed_output',219050854800)]:
            self.assertEqual(sum((x-y)**2 for x,y in zip(pcm[role][0][2400:29600],reference)),energy)

    def test_altered_target_covariance_denominator_and_pcm_values_fail(self):
        mutations=[lambda m:m['model']['selected_covariance'][0].__setitem__(1,0),
            lambda m:m['model']['selected_weights'].__setitem__(0,-.5),
            lambda m:m['analytic']['stale_output'].__setitem__('target_gain',1.),
            lambda m:m['samples']['reference']['pcm_integer_measurements'].__setitem__('integer_denominator_D_per_channel',27200),
            lambda m:m['samples']['recomputed_output']['pcm_integer_measurements']['integer_reference_error_squared_sum_per_channel'].__setitem__(0,219018750300),
            lambda m:m['samples']['stale_output']['pcm_measurements']['target_frequency_phasor_imag_per_channel'].__setitem__(0,-.1),
            lambda m:m['parameters'].__setitem__('known_failed_channel',False),
            lambda m:m.__setitem__('common_export_gain',True)]
        for mutation in mutations:
            data=copy.deepcopy(self.manifest);mutation(data)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                q._check_channel_pcm(self.directory,data)

    def test_plot_drift_gain_compensation_or_wrong_window_fails(self):
        mutations=[lambda r:r['plot_data']['target_gain_analytic'].__setitem__(1,1.),
            lambda r:r['plot_data']['reference_NMSE_pcm'].__setitem__(1,1.5),
            lambda r:r['plot_data']['waveform_interval_samples'].__setitem__(0,2401),
            lambda r:r['plot_data']['waveform_pcm']['reference'].__setitem__(0,1/32768),
            lambda r:r['plot_data'].__setitem__('projection_scope','fitted gain compensation'),
            lambda r:r['input_sha256'].__setitem__('channel_reference.wav','0'*64),
            lambda r:r.__setitem__('script_sha256','0'*64)]
        for mutation in mutations:
            report=copy.deepcopy(self.report);mutation(report)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                q._check_channel_figure_report(report,self.manifest,self.pcm,self.directory)

    def test_stage_preflight_preserves_unrelated_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/'published';destination.mkdir()
            keep=destination/'keep.txt';keep.write_bytes(b'keep')
            before=(keep.read_bytes(),keep.stat().st_mtime_ns)
            with self.assertRaises(ValueError):build_site.stage_channel_audio(self.directory,destination)
            self.assertEqual(before,(keep.read_bytes(),keep.stat().st_mtime_ns))
            self.assertEqual({p.name for p in destination.iterdir()},{'keep.txt'})

    def test_media_links_exact_scope_and_source_digest(self):
        source=q.ROOT/'chapters/10_engineering-practice.md'
        html,_=build_site.render('[参考](../codes/chapters/ch10/channel_audio/channel_reference.wav)\n\n[清单](../codes/chapters/ch10/channel_audio/MANIFEST.json)',source)
        self.assertIn('src="channel_audio/channel_reference.wav"',html)
        self.assertIn('href="channel_audio/MANIFEST.json"',html)
        digest=build_site.source_digest()
        from scripts import build_pdf
        for function in (build_site.source_digest,build_pdf.source_digest,q.site_source_digest,q.source_digest):
            original=Path.read_bytes
            def changed(path,*args,**kwargs):
                data=original(path,*args,**kwargs)
                return data+b'\n# test source drift\n' if path.name=='make_channel_figures.py' else data
            initial=function()
            with patch.object(Path,'read_bytes',changed):self.assertNotEqual(initial,function())

    def test_new_phone_table_budgets_do_not_override_existing_research_budgets(self):
        path=q.ROOT/'scripts/build_site.py'
        tree=ast.parse(path.read_text())
        assignment=next(node for node in tree.body if isinstance(node,ast.Assign)
                        and any(isinstance(target,ast.Name) and target.id=='NARROW_TABLE_POLICIES'
                                for target in node.targets))
        keys=[ast.dump(key,include_attributes=False) for key in assignment.value.keys]
        self.assertEqual(len(keys),len(set(keys)))
        # The Chapter06 tail table and Chapter10 new PCM table coexist on this page.
        source=q.RESEARCH_ROOT/'05_exercises_and_audio.md'
        for headers,budget in [(['取点或输出','切换窗整数E','稳定窗整数E','尾窗整数E'],52),
                               (['输出','PCM误差整数能量E','实际PCM MSE','实际PCM NMSE'],54)]:
            markdown='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*4)+' |\n| A | 1 | 2 | 3 |\n'
            html,_=build_site.render(markdown,source)
            self.assertIn('tutorial-budget-table',html)
            self.assertIn('--tutorial-table-width:'+str(budget)+'em',html)


if __name__=='__main__':unittest.main()
