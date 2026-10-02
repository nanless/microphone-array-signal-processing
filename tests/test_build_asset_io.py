"""Independent structural WAV fixtures test publication IO, not acoustic scores."""
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import build_site


class AssetStageIOTest(unittest.TestCase):
    FAMILIES = (
        ('real', 'RealAudioPublishingTests'),
        ('binaural', 'BinauralPublicationTest'),
        ('stft', 'STFTPublicationTest'),
        ('geometry', 'GeometryPublicationTest'),
        ('focus', 'FocusPublicationTest'),
        ('derivative', 'DerivativePublicationTest'),
    )

    def fixture(self, family, class_name):
        module = importlib.import_module('tests.test_build_' + family + '_audio')
        fixture = getattr(module, class_name)()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture, getattr(build_site, 'stage_' + family + '_audio')

    def test_independent_valid_sets_are_still_published(self):
        for family, class_name in self.FAMILIES:
            with self.subTest(family=family):
                fixture, stage = self.fixture(family, class_name)
                target = fixture.root / 'published'
                names = stage(fixture.source, target)
                self.assertEqual(names, {p.name for p in fixture.source.iterdir()})
                for path in fixture.source.glob('*.wav'):
                    self.assertEqual(path.read_bytes(), (target / path.name).read_bytes())

    def test_linked_source_parent_is_rejected_without_copying(self):
        for family, class_name in self.FAMILIES:
            with self.subTest(family=family):
                fixture, stage = self.fixture(family, class_name)
                alias = fixture.root / 'alias'
                alias.symlink_to(fixture.source.parent, target_is_directory=True)
                target = fixture.root / 'published'
                with self.assertRaises(ValueError):
                    stage(alias / fixture.source.name, target)
                self.assertFalse(target.exists())

    def test_linked_target_parent_preserves_external_sentinel(self):
        for family, class_name in self.FAMILIES:
            with self.subTest(family=family):
                fixture, stage = self.fixture(family, class_name)
                external = fixture.root / 'external'
                external.mkdir()
                sentinel = external / 'sentinel'
                sentinel.write_bytes(b'KEEP')
                before = sentinel.stat().st_mtime_ns
                alias = fixture.root / 'alias'
                alias.symlink_to(external, target_is_directory=True)
                with self.assertRaises(ValueError):
                    stage(fixture.source, alias / 'published')
                self.assertEqual({p.name for p in external.iterdir()}, {'sentinel'})
                self.assertEqual(sentinel.read_bytes(), b'KEEP')
                self.assertEqual(sentinel.stat().st_mtime_ns, before)

    def test_parent_traversal_is_rejected_before_creating_target(self):
        for family, class_name in self.FAMILIES:
            with self.subTest(family=family):
                fixture, stage = self.fixture(family, class_name)
                with self.assertRaises(ValueError):
                    stage(fixture.source, fixture.root / 'unused' / '..' / 'published')
                self.assertFalse((fixture.root / 'published').exists())
                self.assertFalse((fixture.root / 'unused').exists())

    def test_hardlinked_and_extra_source_members_are_rejected(self):
        for family, class_name in self.FAMILIES:
            for invalid in ('hardlink', 'extra_directory', 'fifo'):
                with self.subTest(family=family, invalid=invalid):
                    fixture, stage = self.fixture(family, class_name)
                    if invalid == 'hardlink':
                        os.link(next(fixture.source.glob('*.wav')), fixture.root / 'other-link')
                    elif invalid == 'extra_directory':
                        (fixture.source / 'extra').mkdir()
                    else:
                        os.mkfifo(fixture.source / 'special')
                    with self.assertRaises(ValueError):
                        stage(fixture.source, fixture.root / 'published')
                    self.assertFalse((fixture.root / 'published').exists())

    def test_strict_manifest_duplicate_nonfinite_and_numeric_types(self):
        for family, class_name in self.FAMILIES:
            for invalid in ('duplicate', 'nan', 'overflow', 'bool_gain', 'bool_count', 'peak_array', 'rms_shape'):
                with self.subTest(family=family, invalid=invalid):
                    fixture, stage = self.fixture(family, class_name)
                    path = fixture.source / 'MANIFEST.json'
                    original = path.read_text().rstrip()
                    if invalid == 'duplicate':
                        text = original[:-1] + ', "common_export_gain":0, "common_export_gain":1}'
                    elif invalid == 'nan':
                        text = original[:-1] + ', "unused_number":NaN}'
                    elif invalid == 'overflow':
                        text = original[:-1] + ', "unused_number":1e999}'
                    else:
                        data = json.loads(original)
                        if invalid in ('peak_array', 'rms_shape'):
                            records = data['files']
                            record = next(iter(records.values())) if isinstance(records, dict) else records[0]
                            record['peak' if invalid == 'peak_array' else 'rms'] = []
                        else:
                            data['common_export_gain' if invalid == 'bool_gain' else 'schema_version'] = True
                        text = json.dumps(data)
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        stage(fixture.source, fixture.root / 'published')
                    self.assertFalse((fixture.root / 'published').exists())

    def test_existing_target_preserved_before_any_copy(self):
        for family, class_name in self.FAMILIES:
            with self.subTest(family=family):
                fixture, stage = self.fixture(family, class_name)
                target = fixture.root / 'published'
                target.mkdir()
                (target / 'sentinel').write_bytes(b'KEEP')
                with self.assertRaises(ValueError):
                    stage(fixture.source, target)
                self.assertEqual({p.name for p in target.iterdir()}, {'sentinel'})
                self.assertEqual((target / 'sentinel').read_bytes(), b'KEEP')

    def test_main_mapping_rejects_linked_manifest_or_wav_parent(self):
        fixture, _stage = self.fixture('stft', 'STFTPublicationTest')
        root = fixture.root / 'chapters'
        manifest = root / 'ch00/audio/MANIFEST.json'
        wave = root / 'ch01/audio/example.wav'
        manifest.parent.mkdir(parents=True)
        wave.parent.mkdir(parents=True)
        wave.write_bytes((fixture.source / 'full_convolution.wav').read_bytes())
        manifest.write_text(json.dumps({'files': [{'file': 'example.wav',
                                                   'group': 'spatial', 'chapter': 'ch01'}]}))
        with patch.object(build_site, 'CODE_CHAPTERS', root):
            self.assertEqual(build_site.main_audio_sources(), {wave.resolve(): 'example.wav'})
            for directory in (manifest.parent, wave.parent):
                preserved = directory.with_name(directory.name + '-preserved')
                directory.rename(preserved)
                directory.symlink_to(preserved, target_is_directory=True)
                with self.assertRaises(ValueError):
                    build_site.main_audio_sources()
                directory.unlink()
                preserved.rename(directory)


if __name__ == '__main__':
    unittest.main()
