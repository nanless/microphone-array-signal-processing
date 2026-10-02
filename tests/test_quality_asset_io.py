"""Literal publication fixtures; no builder or generator supplies expectations."""
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import quality_check as qc


# Family dimensions and source identities are independent publication contracts.
FAMILIES = {
    'binaural': ('BINAURAL_AUDIO_ROOT', 'binaural_audio', '01_problem-definition.html', 16000, 32008,
                 {'reference.wav': 2, 'itd_only.wav': 2, 'ild_only.wav': 2, 'consistent.wav': 2, 'conflicting.wav': 2},
                 ('codes/chapters/ch01/core/binaural_cues.py', 'codes/chapters/ch01/examples/generate_binaural_cues.py', 'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py')),
    'stft': ('STFT_AUDIO_ROOT', 'stft_audio', '02_basics-signal-model.html', 16000, 32320,
             {'stft_roundtrip.wav': 1, 'full_convolution.wav': 1, 'framewise_mtf.wav': 1},
             ('codes/chapters/ch02/core/stft_convolution.py', 'codes/chapters/ch02/examples/generate_stft_convolution.py', 'codes/chapters/ch02/core/spectral.py', 'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py')),
    'geometry': ('GEOMETRY_AUDIO_ROOT', 'geometry_audio', '03_array-geometry.html', 32000, 64000,
                 {'geometry_reference.wav': 1, 'geometry_u.wav': 6, 'geometry_v.wav': 6},
                 ('codes/chapters/ch03/core/geometry_audio.py', 'codes/chapters/ch03/examples/generate_geometry_audio.py', 'codes/chapters/ch03/core/geometry.py', 'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py')),
    'focus': ('FOCUS_AUDIO_ROOT', 'focus_audio', '04_doa-estimation.html', 16000, 32024,
              {'focus_reference.wav': 1, 'focus_delayed_source.wav': 1, 'focus_array.wav': 4, 'focus_known_focused.wav': 4},
              ('codes/chapters/ch04/core/focus_audio.py', 'codes/chapters/ch04/examples/generate_focus_audio.py', 'codes/chapters/ch03/core/geometry.py', 'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py')),
    'derivative': ('DERIVATIVE_AUDIO_ROOT', 'derivative_audio', '05_beamforming.html', 16000, 32002,
                   {'derivative_reference.wav': 1, 'derivative_single.wav': 1, 'derivative_array.wav': 3, 'derivative_constrained.wav': 1},
                   ('codes/chapters/ch05/core/derivative_audio.py', 'codes/chapters/ch05/examples/generate_derivative_audio.py', 'codes/chapters/ch05/core/beamforming.py', 'codes/chapters/ch04/core/covariance.py', 'codes/chapters/ch03/core/geometry.py', 'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py', 'codes/chapters/ch00/io_contracts.py')),
}


class QualityAssetIOTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, family):
        attr, directory, chapter, fs, count, channels, sources = FAMILIES[family]
        root = self.root / family
        source, site = root / 'source', root / 'site'
        source.mkdir(parents=True)
        (site / 'research').mkdir(parents=True)
        manifest = {'sample_rate_hz': fs, 'samples_per_channel': count,
                    'common_export_gain': 1, 'files': {}, 'source_sha256': {}}
        for path in sources:
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'literal identity fixture\n')
            manifest['source_sha256'][path] = hashlib.sha256(file.read_bytes()).hexdigest()
        for name, ch in channels.items():
            data = io.BytesIO()
            with wave.open(data, 'wb') as wav:
                wav.setnchannels(ch); wav.setsampwidth(2); wav.setframerate(fs)
                wav.writeframes(b'\x01\x00' * ch * count)
            blob = data.getvalue()
            (source / name).write_bytes(blob)
            manifest['files'][name] = {'sample_rate_hz': fs, 'channels': ch,
                                      'samples_per_channel': count,
                                      'sha256': hashlib.sha256(blob).hexdigest()}
        (source / 'MANIFEST.json').write_text(json.dumps(manifest))
        shutil.copytree(source, site / directory)
        for page, prefix in ((site / chapter, ''), (site / 'research/05_exercises_and_audio.html', '../')):
            page.write_text('\n'.join(f'<audio src="{prefix}{directory}/{name}" controls preload="none" aria-label="cue"></audio>' for name in channels)
                            + f'<a href="{prefix}{directory}/MANIFEST.json">Manifest</a>')
        return root, source, site, manifest

    def check(self, family, root, source, site):
        attr = FAMILIES[family][0]
        errors = []
        with patch.object(qc, 'ROOT', root), patch.object(qc, attr, source), patch.object(qc, 'SITE', site):
            getattr(qc, 'check_' + family + '_audio')(errors)
        return errors

    def rewrite(self, family, source, site, text):
        (source / 'MANIFEST.json').write_text(text)
        (site / FAMILIES[family][1] / 'MANIFEST.json').write_text(text)

    def test_five_legacy_structural_wav_contracts_accept_literal_fixtures(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                self.assertEqual(self.check(family, *self.fixture(family)[:3]), [])

    def test_bools_and_float_wave_dimensions_rejected_in_each_legacy_family(self):
        for family in FAMILIES:
            root, source, site, original = self.fixture(family)
            for location, value in [('gain', True), ('rate', 16000.), ('channels', True), ('count', float(FAMILIES[family][4]))]:
                with self.subTest(family=family, location=location):
                    m = copy.deepcopy(original)
                    if location == 'gain': m['common_export_gain'] = value
                    elif location == 'rate': m['sample_rate_hz'] = value
                    else: m['files'][next(iter(m['files']))]['channels' if location == 'channels' else 'samples_per_channel'] = value
                    self.rewrite(family, source, site, json.dumps(m))
                    self.assertTrue(self.check(family, root, source, site))

    def test_duplicate_gain_and_nested_keys_are_not_last_value_wins(self):
        root, source, site, manifest = self.fixture('binaural')
        ordinary = json.dumps(manifest)
        for text in (ordinary.replace('"common_export_gain": 1', '"common_export_gain": 0, "common_export_gain": 1'),
                     ordinary.replace('"channels": 2', '"channels": 1, "channels": 2', 1)):
            with self.subTest(text=text[:70]):
                self.rewrite('binaural', source, site, text)
                self.assertIn('duplicate JSON key', '\n'.join(self.check('binaural', root, source, site)))

    def test_nonfinite_json_rejected_even_in_unchecked_annotation(self):
        root, source, site, manifest = self.fixture('binaural')
        for token in ('NaN', 'Infinity', '-Infinity', '1e400'):
            with self.subTest(token=token):
                text = json.dumps(manifest)[:-1] + ', "annotation": ' + token + '}'
                self.rewrite('binaural', source, site, text)
                self.assertTrue(self.check('binaural', root, source, site))

    def test_source_and_published_lexical_parent_links_are_rejected(self):
        for family in FAMILIES:
            root, source, site, _ = self.fixture(family)
            alias = root / 'alias'; alias.symlink_to(root, target_is_directory=True)
            for linked_source, linked_site in ((alias / 'source', site), (source, alias / 'site')):
                with self.subTest(family=family, source=linked_source):
                    self.assertIn('symbolic link', '\n'.join(self.check(family, root, linked_source, linked_site)))

    def test_real_audio_parent_link_is_rejected_before_metadata_or_model_reads(self):
        real = self.root / 'real'; real.mkdir()
        site = self.root / 'site'; site.mkdir()
        alias = self.root / 'alias'; alias.symlink_to(self.root, target_is_directory=True)
        for source, published in ((alias / 'real', site), (real, alias / 'site')):
            # Build the exact ordinary inventory so the linked ancestor is the
            # independently chosen failure, not a missing model fixture.
            for folder in (real, site / 'real_audio'):
                folder.mkdir(exist_ok=True)
                for name in qc.EXPECTED_REAL_AUDIO_FILES: (folder / name).write_bytes(b'fixture')
            errors = []
            with patch.object(qc, 'REAL_AUDIO_ROOT', source), patch.object(qc, 'SITE', published):
                qc.check_real_audio(errors)
            self.assertIn('symbolic link', '\n'.join(errors))

    def test_extra_directory_and_hardlinked_member_are_rejected(self):
        root, source, site, _ = self.fixture('binaural')
        for folder in (source, site / 'binaural_audio'):
            extra = folder / 'extra'; extra.mkdir()
            self.assertTrue(self.check('binaural', root, source, site)); extra.rmdir()
            name = folder / 'reference.wav'; other = root / 'hardlink'
            os.link(name, other)
            self.assertIn('singly linked', '\n'.join(self.check('binaural', root, source, site)))
            other.unlink()

    def test_hidden_template_inert_and_collapsed_players_cannot_satisfy_inventory(self):
        root, source, site, _ = self.fixture('binaural')
        page = site / '01_problem-definition.html'; original = page.read_text()
        for start, end in [('<template>', '</template>'), ('<div hidden>', '</div>'),
                           ('<div inert>', '</div>'), ('<div aria-hidden="true">', '</div>'),
                           ('<div style="display:none">', '</div>'), ('<noscript>', '</noscript>'),
                           ('<details><summary>show</summary>', '</details>')]:
            with self.subTest(start=start):
                page.write_text(start + original + end)
                self.assertIn('可见播放器', '\n'.join(self.check('binaural', root, source, site)))
        page.write_text('<details open><summary>show</summary>' + original + '</details>')
        self.assertEqual(self.check('binaural', root, source, site), [])

    def test_visible_manifest_link_and_nonblank_labels_required(self):
        root, source, site, _ = self.fixture('binaural')
        page = site / '01_problem-definition.html'; original = page.read_text()
        page.write_text(original.replace('aria-label="cue"', 'aria-label=" "', 1))
        self.assertTrue(self.check('binaural', root, source, site))
        page.write_text(original.replace('<a href=', '<a hidden href='))
        self.assertIn('可见独立资产链接', '\n'.join(self.check('binaural', root, source, site)))

    def test_near_named_other_family_media_do_not_satisfy_or_poison_selected_inventory(self):
        root, source, site, _ = self.fixture('binaural')
        page = site / '01_problem-definition.html'; original = page.read_text()
        page.write_text(original + '<audio src="binaural_audio_extra/reference.wav"></audio>')
        self.assertEqual(self.check('binaural', root, source, site), [])
        page.write_text(original.replace('binaural_audio/reference.wav', 'binaural_audio_extra/reference.wav'))
        self.assertTrue(self.check('binaural', root, source, site))

    def test_metadata_policy_is_finite_and_preserves_boolean_controls_and_namespaces(self):
        valid = {'sample_rate_hz': 16000, 'channels': 1, 'common_export_gain': 1.,
                 'samples': {'x': {'rms': .1, 'zero_output': True}},
                 'frames': {'observation_valid': [True, False]}, 'count_label': 'three'}
        self.assertIs(qc._validate_audio_metadata(valid), valid)
        for key, value in [('schema_version', True), ('samples', 1.), ('duration_s', True),
                           ('rms', [.1, False]), ('peak', '1'), ('common_gain', False), ('common_export_gain', [1])]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError): qc._validate_audio_metadata({key: value})


    def test_only_channel_rms_accepts_an_array_of_the_declared_length(self):
        valid = {'files': [{'channels': 2, 'rms': [.1, .2], 'peak': .2}]}
        self.assertIs(qc._validate_audio_metadata(valid, per_channel_rms=True), valid)
        for rms in ([], [.1], [.1, False], [[.1], .2], .1):
            with self.subTest(rms=rms):
                with self.assertRaises(ValueError):
                    qc._validate_audio_metadata({'files': [{'channels': 2, 'rms': rms}]},
                                                per_channel_rms=True)
        with self.assertRaises(ValueError):
            qc._validate_audio_metadata({'peak': [.1]})
        with self.assertRaises(ValueError):
            qc._validate_audio_metadata(valid)


if __name__ == '__main__':
    unittest.main()
