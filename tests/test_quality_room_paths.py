"""Room staging must reject unsafe/existing targets before any file operation.

Use the committed, checked 21-member source; no PRA installation, simulation,
network or source repair occurs. Every output is isolated in /private/tmp.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_site
from codes.chapters.appendix_b.examples import check_room_assets as room_assets


class RoomStagingPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='masp-room-path-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def snapshot(self, directory):
        return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in directory.iterdir()}

    def test_actual_ordinary_staging_copies_all_members_without_changing_source(self):
        room_assets.check_assets(room_assets.ROOM)
        before = self.snapshot(room_assets.ROOM)
        destination = self.root/'stage'
        members = build_site.stage_room_audio(room_assets.ROOM, destination)
        self.assertEqual(members, room_assets.MEMBERS)
        self.assertEqual({p.name for p in destination.iterdir()}, room_assets.MEMBERS)
        for name in room_assets.MEMBERS:
            self.assertEqual((destination/name).read_bytes(), (room_assets.ROOM/name).read_bytes())
        self.assertEqual(self.snapshot(room_assets.ROOM), before)

    def test_linked_parent_cannot_publish_into_external_directory(self):
        outside = self.root/'outside'
        outside.mkdir()
        (outside/'old.txt').write_bytes(b'external original bytes')
        before = self.snapshot(outside)
        (self.root/'linked').symlink_to(outside, target_is_directory=True)
        with patch.object(room_assets, 'check_assets') as source_check:
            with self.assertRaises(ValueError):
                build_site.stage_room_audio(room_assets.ROOM, self.root/'linked'/'stage')
            source_check.assert_not_called()
        self.assertEqual(self.snapshot(outside), before)
        self.assertFalse((outside/'stage').exists())

    def test_cancelled_traversal_and_plain_traversal_are_rejected(self):
        outside = self.root/'outside'
        outside.mkdir()
        (outside/'old.txt').write_bytes(b'unchanged')
        (self.root/'linked').symlink_to(outside, target_is_directory=True)
        ordinary = self.root/'ordinary'
        ordinary.mkdir()
        before = self.snapshot(outside)
        for path in (self.root/'linked'/'..'/'stage', ordinary/'..'/'stage'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                build_site.stage_room_audio(room_assets.ROOM, path)
        self.assertEqual(self.snapshot(outside), before)
        self.assertFalse((self.root/'stage').exists())

    def test_existing_empty_or_populated_directory_is_not_changed(self):
        for name, populated in (('empty', False), ('used', True)):
            target = self.root/name
            target.mkdir()
            if populated:
                (target/'old.txt').write_bytes(b'existing destination data')
            before = self.snapshot(target)
            with self.subTest(name=name), patch.object(room_assets, 'check_assets') as source_check:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    build_site.stage_room_audio(room_assets.ROOM, target)
                source_check.assert_not_called()
            self.assertEqual(self.snapshot(target), before)

    def test_existing_file_or_linked_target_is_not_replaced(self):
        target = self.root/'file'
        target.write_bytes(b'existing ordinary file')
        directory = self.root/'directory'
        directory.mkdir()
        (directory/'old.txt').write_bytes(b'existing linked directory')
        alias = self.root/'alias'
        alias.symlink_to(directory, target_is_directory=True)
        before = self.snapshot(directory)
        for path in (target, alias):
            with self.subTest(path=path), self.assertRaises(ValueError):
                build_site.stage_room_audio(room_assets.ROOM, path)
        self.assertEqual(target.read_bytes(), b'existing ordinary file')
        self.assertEqual(self.snapshot(directory), before)
        self.assertTrue(alias.is_symlink())

    def test_near_named_ordinary_parent_is_allowed(self):
        # A normal directory named "linked" is not a symbolic link. Guards
        # should inspect path topology, not reject a suspicious spelling.
        parent = self.root/'linked'
        parent.mkdir()
        members = build_site.stage_room_audio(room_assets.ROOM, parent/'stage')
        self.assertEqual(members, room_assets.MEMBERS)
        self.assertEqual(len(list((parent/'stage').iterdir())), 21)


if __name__ == '__main__':
    unittest.main()
