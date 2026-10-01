"""Real PCM fixtures check publication failures actually reach the caller."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_site, quality_check as quality

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'mint_reference.wav', 'mint_well_array.wav', 'mint_near_array.wav',
         'mint_well_exact.wav', 'mint_near_exact.wav', 'mint_near_regularized.wav'}


class MINTPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_root = ROOT/'codes/chapters/ch07/mint_audio'
        cls.assets = {name: (asset_root/name).read_bytes() for name in NAMES | {'MANIFEST.json'}}
        cls.original = json.loads(cls.assets['MANIFEST.json'])
        cls.sources = {name: (ROOT/name).read_bytes() for name in cls.original['source_sha256']}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'codes/chapters/ch07/mint_audio'
        self.site = self.root/'site'
        self.published = self.site/'mint_audio'
        self.source.mkdir(parents=True); self.published.mkdir(parents=True)
        (self.site/'research').mkdir()
        for name, blob in self.sources.items():
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        for name, blob in self.assets.items():
            (self.source/name).write_bytes(blob); (self.published/name).write_bytes(blob)
        self.manifest = copy.deepcopy(self.original)
        self.pages = ((self.site/'07_wpe-dereverberation.html', ''),
                      (self.site/'research/05_exercises_and_audio.html', '../'))
        for page, prefix in self.pages:
            page.write_text(self.html(prefix))
        for name, value in (('ROOT', self.root), ('SITE', self.site), ('MINT_AUDIO_ROOT', self.source)):
            patcher = patch.object(quality, name, value); patcher.start(); self.addCleanup(patcher.stop)

    @staticmethod
    def html(prefix):
        return '\n'.join(f'<audio controls preload="none" aria-label="sample {name}" src="{prefix}mint_audio/{name}"></audio>'
                         for name in sorted(NAMES)) + f'<a href="{prefix}mint_audio/MANIFEST.json">Manifest</a>'

    def save_manifest(self):
        text = json.dumps(self.manifest, ensure_ascii=False)
        for folder in (self.source, self.published):
            (folder/'MANIFEST.json').write_text(text)

    def issues(self):
        problems = []
        quality.check_mint_audio(problems)
        return problems

    def assert_rejected(self):
        # This explicit outer-list assertion prevents a shadowed errors list
        # from turning a detected PCM failure into a false publication pass.
        self.assertTrue(self.issues())
        with self.assertRaises(ValueError):
            build_site.stage_mint_audio(self.source, self.root/'staged')

    def test_valid_pcm_sources_players_and_read_only_source_pass(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        self.assertEqual(self.issues(), [])
        self.assertEqual(build_site.stage_mint_audio(self.source, self.root/'staged'), NAMES | {'MANIFEST.json'})
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_direct_quality_entry_resolves_chapter_replay(self):
        script = ROOT/'scripts/quality_check.py'
        code = ('import runpy,sys; from pathlib import Path; '
                'sys.path[0]=str(Path(sys.argv[1]).parent); '
                'ns=runpy.run_path(sys.argv[1]); fn=ns["check_mint_audio"]; '
                'root=Path(sys.argv[2]); fn.__globals__.update(ROOT=root,SITE=root/"site",'
                'MINT_AUDIO_ROOT=root/"codes/chapters/ch07/mint_audio"); '
                'problems=[]; fn(problems); assert not problems, problems')
        result = subprocess.run([sys.executable, '-c', code, str(script), str(self.root)],
                                cwd=script.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_float_boolean_is_not_a_score(self):
        self.manifest['samples']['near_exact']['float_measurements']['total_reference_mse_per_channel'] = [True]
        self.save_manifest(); self.assert_rejected()

    def test_finite_looking_json_exponent_overflow_is_rejected(self):
        text = self.assets['MANIFEST.json'].decode()
        old = str(self.original['float_decomposition']['near_exact']['noise_mean_square'])
        self.assertIn(old, text)
        for folder in (self.source, self.published):
            (folder/'MANIFEST.json').write_text(text.replace(old, '1e999'))
        self.assert_rejected()

    def test_false_float_component_is_rejected(self):
        self.manifest['float_decomposition']['near_exact']['noise_mean_square'] = 0
        self.save_manifest(); self.assert_rejected()

    def test_pcm_boolean_denominator_reaches_error_caller(self):
        self.manifest['samples']['near_exact']['pcm_measurements']['integer_reference_squared_sum'] = True
        self.save_manifest(); self.assert_rejected()

    def test_false_pcm_measurement_reaches_error_caller(self):
        self.manifest['samples']['near_exact']['pcm_measurements']['total_reference_mse_per_channel'] = [0.0]
        self.save_manifest(); self.assert_rejected()

    def test_missing_member_rejects_source_and_publication(self):
        (self.source/'mint_near_exact.wav').unlink(); self.assert_rejected()

    def test_symlink_member_is_rejected(self):
        member = self.source/'mint_near_exact.wav'; member.unlink()
        member.symlink_to(self.published/'mint_near_exact.wav'); self.assert_rejected()

    def test_stale_source_digest_is_rejected(self):
        key = next(iter(self.manifest['source_sha256']))
        self.manifest['source_sha256'][key] = '0'*64
        self.save_manifest(); self.assert_rejected()

    def test_changed_published_byte_is_rejected(self):
        path = self.published/'mint_near_exact.wav'; blob = path.read_bytes()
        path.write_bytes(blob[:-1]+bytes([blob[-1]^1])); self.assertTrue(self.issues())

    def test_missing_duplicate_hidden_or_inert_players_are_rejected(self):
        page, prefix = self.pages[0]
        valid = self.html(prefix)
        variants = (valid.replace('<audio ', '<audio hidden ', 1),
                    '<template>'+valid+'</template>',
                    '<noscript>'+valid+'</noscript>',
                    '<section inert>'+valid+'</section>',
                    '<div style="display: none">'+valid+'</div>',
                    valid+valid,
                    valid.replace('controls', '', 1),
                    valid.replace('preload="none"', 'autoplay', 1),
                    valid.replace('aria-label=', 'data-label=', 1))
        for html in variants:
            with self.subTest(html=html[:90]):
                page.write_text(html); self.assertTrue(self.issues())

    def test_unrelated_media_and_hidden_unrelated_content_do_not_fail(self):
        page, prefix = self.pages[0]
        page.write_text(self.html(prefix)+'<section hidden><audio src="other.wav"></audio></section>'
                        '<template><audio src="other.wav"></audio></template>'
                        '<audio controls src="unrelated.wav"></audio>')
        self.assertEqual(self.issues(), [])


if __name__ == '__main__':
    unittest.main()
