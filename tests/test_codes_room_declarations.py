"""Published room declarations: default checking needs no installed PRA."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from codes.chapters.appendix_b.examples import check_room_assets as checker


class PublishedRoomDeclarationTests(unittest.TestCase):
    def copy(self, root):
        return Path(shutil.copytree(checker.ROOM, Path(root)/'room'))

    def test_correct_published_assets_are_read_only_without_rir_replay(self):
        before = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in checker.ROOM.iterdir()}
        with patch.object(checker.room, 'run_experiment', side_effect=AssertionError('PRA replay')), \
                patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir')), \
                patch.object(Path, 'write_bytes', side_effect=AssertionError('write')):
            result = checker.check_assets()
        self.assertEqual(result['validation']['library_global_delay_samples'], 40)
        self.assertEqual(result['validation']['fractional_delay_filter_length_samples'], 81)
        self.assertFalse(result['validation']['rir_and_output_replayed'])
        self.assertEqual(before, {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                                  for p in checker.ROOM.iterdir()})

    def test_fixed_doa_each_field_and_type_reject_without_writes(self):
        controls = [('n_fft', 256), ('n_fft', 512.), ('hop_length', True), ('hop_length', 64),
                    ('frequency_band_hz', [500., 1000.]), ('frequency_band_hz', [300, 2000]),
                    ('azimuth_grid_deg', [-80., 80, 1]), ('azimuth_grid_deg', [-90, 90, 1]),
                    ('method', 'near-field'), ('extra', 'unknown')]
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(td)
            path = directory/'RESULTS.json'; original = json.loads(path.read_bytes())
            for key, value in controls:
                with self.subTest(key=key, value=value):
                    result = json.loads(json.dumps(original)); result['doa'][key] = value
                    path.write_text(json.dumps(result)); before = path.read_bytes()
                    with patch.object(Path, 'write_bytes', side_effect=AssertionError('write')), \
                            patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir')):
                        with self.assertRaisesRegex(ValueError, 'DOA'): checker.check_assets(directory)
                    self.assertEqual(before, path.read_bytes())
            result = json.loads(json.dumps(original)); result['doa'].pop('hop_length')
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError): checker.check_assets(directory)

    def test_filter_length_fixed_typed_and_40_padding_relationship(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(td); path = directory/'RESULTS.json'
            original = json.loads(path.read_bytes())
            for value in (1, 80, 83, 81., True, None):
                with self.subTest(value=value):
                    result = json.loads(json.dumps(original)); result['fractional_delay_filter_length_samples'] = value
                    path.write_text(json.dumps(result))
                    with self.assertRaisesRegex(ValueError, 'filter length'): checker.check_assets(directory)

    def test_resource_order_boundaries_remain_parameterized(self):
        # This tests a declaration/resource boundary, not rerun acoustics at a
        # different image order. Default validation explicitly is not PRA replay.
        with tempfile.TemporaryDirectory() as td:
            directory = self.copy(td)
            manifest_path = directory/'MANIFEST.json'; result_path = directory/'RESULTS.json'
            manifest = json.loads(manifest_path.read_bytes()); result = json.loads(result_path.read_bytes())
            for order in (2, 80, 1, 81, True, 2.):
                with self.subTest(order=order):
                    manifest['max_order'] = order; result['actual_max_order'] = order
                    manifest_path.write_text(json.dumps(manifest))
                    result['assets']['audio_manifest']['sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                    result_path.write_text(json.dumps(result))
                    if type(order) is int and 2 <= order <= 80:
                        checker.check_assets(directory)
                    else:
                        with self.assertRaises(ValueError): checker.check_assets(directory)


if __name__ == '__main__':
    unittest.main()
