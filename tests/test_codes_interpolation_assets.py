"""Current sources and actual four-WAV interpolation, never repaired."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from codes.chapters.appendix_b.examples.check_main_interpolation import ROOT, check_main_interpolation_assets
from codes.chapters.ch00.examples.generate_audio_samples import INPUTS


class MainInterpolationAssetsTests(unittest.TestCase):
    def copy(self, root):
        for relative in INPUTS:
            target = root/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/relative, target)
        manifest = root/'codes/chapters/ch00/audio/MANIFEST.json'; manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'codes/chapters/ch00/audio/MANIFEST.json', manifest)
        directory = root/'codes/chapters/ch02/audio'; directory.mkdir(parents=True)
        for path in (ROOT/'codes/chapters/ch02/audio').glob('interpolation_*.wav'):
            shutil.copyfile(path, directory/path.name)
        return manifest, directory

    def test_actual_read_and_check_leave_every_byte_and_mtime(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.copy(root)
            before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            _, actual = check_main_interpolation_assets(root)
            self.assertEqual(len(actual), 4)
            self.assertEqual(before, {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})

    def test_source_mismatch_zero_reference_and_bool_format_metadata_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest, directory = self.copy(root)
            source = root/INPUTS[0]; original = source.read_bytes(); source.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'source digest'): check_main_interpolation_assets(root)
            source.write_bytes(original)
            record = json.loads(manifest.read_bytes())
            row = next(r for r in record['files'] if r['file'] == 'interpolation_ideal_half.wav')
            path = directory/row['file']; blob = bytearray(path.read_bytes()); blob[44:] = b'\0'*(len(blob)-44); path.write_bytes(blob)
            row['sha256'] = hashlib.sha256(blob).hexdigest(); manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError): check_main_interpolation_assets(root)
            before = path.read_bytes()
            self.assertEqual(path.read_bytes(), before)
            row['channels'] = True; manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError): check_main_interpolation_assets(root)

    def test_duplicate_json_and_symlink_reject_without_following(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest, directory = self.copy(root)
            original = manifest.read_bytes()
            manifest.write_bytes(b'{"schema_version":2,"schema_version":2}')
            with self.assertRaisesRegex(ValueError, 'duplicate'): check_main_interpolation_assets(root)
            manifest.write_bytes(original)
            member = directory/'interpolation_linear_half.wav'; data = member.read_bytes(); member.unlink()
            outside = root/'outside.wav'; outside.write_bytes(data); member.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'ordinary'): check_main_interpolation_assets(root)
            self.assertEqual(outside.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
