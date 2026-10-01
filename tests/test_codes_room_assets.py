"""Actual optional PRA generation; strict ordinary paths and no-write audits."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from codes.chapters.appendix_b.examples import room_srp_exercise as generator
from codes.chapters.appendix_b.examples.check_room_assets import check_assets


class ActualRoomAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pyroomacoustics as pra
            import scipy
        except ImportError:
            raise unittest.SkipTest('actual PRA/SciPy asset generation is optional')
        if pra.__version__ != '0.10.0':
            raise unittest.SkipTest('PRA 0.10.0 is required')
        cls.temp = tempfile.TemporaryDirectory(prefix='masp-room-test-')
        cls.source = Path(cls.temp.name)/'audio'
        report = generator.run_experiment()
        generator.export_audio(report, cls.source)
        generator.plot_results(report, cls.source/'ROOM_RESULTS.png')
        generator.write_results(report, cls.source/'ROOM_RESULTS.png', cls.source, cls.source/'RESULTS.json')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy(self, directory):
        destination = directory/'room'
        shutil.copytree(self.source, destination)
        return destination

    def bind_manifest_for_adversarial_test(self, directory, document):
        # The test deliberately gives the corrupted manifest its own correct
        # digest: checksum consistency alone must not establish validity.
        path = directory/'MANIFEST.json'
        path.write_text(json.dumps(document))
        report = json.loads((directory/'RESULTS.json').read_bytes())
        report['assets']['audio_manifest']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (directory/'RESULTS.json').write_text(json.dumps(report))

    def test_actual_21_members_and_both_read_only_modes(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        audit = check_assets(self.source)
        self.assertEqual(audit['validation']['actual_pcm_files'], 18)
        self.assertFalse(audit['validation']['rir_and_output_replayed'])
        self.assertTrue(check_assets(self.source, replay=True)['validation']['rir_and_output_replayed'])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()})

    def test_missing_extra_directory_symlink_and_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); directory = self.copy(root)
            extra = directory/'extra'; extra.mkdir()
            with self.assertRaises(ValueError): check_assets(directory)
            extra.rmdir()
            member = directory/'fixed_near_left_source.wav'; original = member.read_bytes(); member.unlink()
            with self.assertRaises(ValueError): check_assets(directory)
            outside = root/'outside'; outside.write_bytes(original); member.symlink_to(outside)
            with self.assertRaises(ValueError): check_assets(directory)
            self.assertEqual(outside.read_bytes(), original)
            with self.assertRaises(ValueError): check_assets(directory/'..'/'room')
            with self.assertRaises(ValueError): check_assets(self.source, replay=1)

    def test_source_metadata_types_and_actual_wav_digest_reject(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(Path(td))
            manifest = json.loads((directory/'MANIFEST.json').read_bytes())
            manifest['source_sha256'][generator.SOURCE_PATHS[0]] = '0'*64
            self.bind_manifest_for_adversarial_test(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'source closure'): check_assets(directory)
            manifest['source_sha256'] = generator.source_hashes()
            manifest['source_and_microphone_clocks_aligned'] = 1
            self.bind_manifest_for_adversarial_test(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'type'): check_assets(directory)
            manifest['source_and_microphone_clocks_aligned'] = True
            self.bind_manifest_for_adversarial_test(directory, manifest)
            path = directory/'fixed_far_left_full.wav'; data = bytearray(path.read_bytes()); data[-2] ^= 1; path.write_bytes(data)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'digest'): check_assets(directory)
            self.assertEqual(path.read_bytes(), before)

    def test_strict_json_and_valid_format_zero_excitation_reject(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(Path(td))
            path = directory/'MANIFEST.json'; original = path.read_bytes()
            for invalid in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError): check_assets(directory)
            path.write_bytes(original)
            member = directory/'fixed_near_left_source.wav'; data = bytearray(member.read_bytes()); data[44:] = b'\0'*(len(data)-44); member.write_bytes(data)
            manifest = json.loads(original)
            next(r for r in manifest['files'] if r['file'] == member.name)['sha256'] = hashlib.sha256(data).hexdigest()
            self.bind_manifest_for_adversarial_test(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'excitation replay'): check_assets(directory)

    def test_malformed_row_shape_and_huge_numeric_fields_raise_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(Path(td))
            path = directory/'RESULTS.json'
            original = json.loads(path.read_bytes())
            bad = json.loads(path.read_bytes()); bad['results'][0]['drr_db_median'] = 10**400
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): check_assets(directory)
            bad = json.loads(json.dumps(original)); bad['results'][0] = []
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): check_assets(directory)

    def test_valid_hex_but_wrong_pra_identity_cannot_be_self_bound(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(Path(td))
            manifest = json.loads((directory/'MANIFEST.json').read_bytes())
            report = json.loads((directory/'RESULTS.json').read_bytes())
            manifest['environment']['pyroomacoustics_source_sha256']['room.py'] = '0'*64
            report['environment']['pyroomacoustics_source_sha256']['room.py'] = '0'*64
            (directory/'RESULTS.json').write_text(json.dumps(report))
            self.bind_manifest_for_adversarial_test(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'fixed PRA Python source identities'):
                check_assets(directory)


if __name__ == '__main__':
    unittest.main()
