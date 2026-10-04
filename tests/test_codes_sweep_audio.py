"""Source-bound, read-only sweep assets and independent PCM/error checks."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from codes.chapters.ch02.examples import generate_sweep_audio as generator


class SweepAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'audio'
        generator.generate_assets(self.directory)

    def snapshot(self):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.directory.iterdir()}

    def test_actual_pcm_headers_integer_power_complete_tail_and_read_only_replay(self):
        before = self.snapshot()
        manifest = generator.check_assets(self.directory)
        self.assertEqual(before, self.snapshot())
        expected_energies = {'sweep_source.wav': 682551316575,
                             'sweep_complete.wav': 479665609386,
                             'sweep_noisy.wav': 479941314935,
                             'sweep_cut.wav': 479595214158}
        for name, count in generator.WAV_LENGTHS.items():
            with wave.open(str(self.directory / name), 'rb') as stream:
                self.assertEqual((stream.getframerate(), stream.getnchannels(),
                                  stream.getsampwidth(), stream.getnframes()), (16000, 1, 2, count))
                pcm = [v[0] for v in struct.iter_unpack('<h', stream.readframes(count))]
            energy = sum(v*v for v in pcm)
            self.assertEqual(energy, expected_energies[name])
            record = manifest['files'][name]
            self.assertEqual(record['pcm_integer_measurements']['squared_sum_E'], energy)
            self.assertEqual(record['pcm_integer_measurements']['integer_denominator_D'], count*32768**2)
            self.assertEqual(record['sha256'], hashlib.sha256((self.directory/name).read_bytes()).hexdigest())
            if name == 'sweep_complete.wav':
                self.assertGreater(sum(v*v for v in pcm[32000:]), 0)
            if name == 'sweep_cut.wav':
                self.assertEqual(pcm[32000:], [0]*160)

    def test_pcm_inverse_retains_all_taps_and_separates_quantization_and_regularization(self):
        manifest = generator.check_assets(self.directory)
        self.assertEqual(manifest['scoring']['full_interval'], [0, 65536])
        self.assertEqual(manifest['scoring']['true_support_interval'], [0, 161])
        self.assertEqual(manifest['parameters']['rho_values'], [0., 1e-8, 1e-6, 1e-4])
        # Independent full complex FFT expression, including both negative and
        # positive frequencies; do not call the tested inverse or score helper.
        decoded = {}
        for name in generator.WAV_LENGTHS:
            with wave.open(str(self.directory/name), 'rb') as stream:
                decoded[name] = np.array([v[0] for v in struct.iter_unpack('<h', stream.readframes(stream.getnframes()))])/32768
        U = np.fft.fft(decoded['sweep_source.wav'], n=65536)
        h = np.zeros(65536)
        h[[0, 160]] = [.8, .25]
        denominator = .8**2 + .25**2
        rows = manifest['pcm_measurements']['rows']
        self.assertEqual(len(rows), 12)
        for row in rows:
            Y = np.fft.fft(decoded['sweep_' + row['case'] + '.wav'], n=65536)
            epsilon = row['rho'] * float(np.max(abs(U)**2))
            H = Y/U if epsilon == 0 else U.conj()*Y/(abs(U)**2+epsilon)
            estimated = np.fft.ifft(H).real
            error = estimated-h
            self.assertAlmostEqual(float(error@error)/denominator /
                                   row['full_ir_relative_error_energy'], 1., delta=2e-6)
            self.assertAlmostEqual(row['full_ir_relative_error_energy'],
                                   row['true_support_relative_error_energy'] + row['outside_support_relative_energy'],
                                   delta=1e-8)
        self.assertLess(manifest['float_measurements']['rows'][0]['full_ir_relative_error_energy'], 1e-20)
        self.assertGreater(rows[0]['full_ir_relative_error_energy'], .4)
        self.assertLess(rows[0]['true_support_relative_error_energy'], .005)

    def test_changed_case_configuration_is_rejected_before_asset_writes(self):
        for key, changed in (('end_frequency_hz', 5000.), ('noise_seed', 7),
                             ('common_export_gain', True)):
            with self.subTest(key=key):
                case = generator.build_sweep_cases()
                case['parameters'][key] = changed
                before = self.snapshot()
                with patch.object(generator, 'build_sweep_cases', return_value=case):
                    with self.assertRaises(ValueError):
                        generator.generate_assets(self.directory)
                self.assertEqual(before, self.snapshot())

    def test_modified_wav_is_rejected_without_repair(self):
        target = self.directory/'sweep_cut.wav'
        target.write_bytes(target.read_bytes()[:-2]+b'\x01\x00')
        before = self.snapshot()
        with self.assertRaises(ValueError):
            generator.check_assets(self.directory)
        self.assertEqual(before, self.snapshot())

    def test_stale_sources_scores_and_metadata_types_are_rejected_without_repair(self):
        original = (self.directory/'MANIFEST.json').read_text()
        for change in ('source', 'score', 'type'):
            with self.subTest(change=change):
                manifest = json.loads(original)
                if change == 'source':
                    manifest['source_sha256'][generator.SOURCE_PATHS[0]] = '0'*64
                elif change == 'score':
                    manifest['pcm_measurements']['rows'][0]['full_ir_relative_error_energy'] = 0.
                else:
                    manifest['schema_version'] = True
                (self.directory/'MANIFEST.json').write_text(json.dumps(manifest))
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    generator.check_assets(self.directory)
                self.assertEqual(before, self.snapshot())

    def test_nonstandard_or_duplicate_json_and_extra_member_fail_read_only(self):
        original = (self.directory/'MANIFEST.json').read_text()
        for text in ('{"schema_version":NaN}', '{"schema_version":1,"schema_version":1}'):
            (self.directory/'MANIFEST.json').write_text(text)
            before = self.snapshot()
            with self.assertRaises(ValueError):
                generator.check_assets(self.directory)
            self.assertEqual(before, self.snapshot())
        (self.directory/'MANIFEST.json').write_text(original)
        (self.directory/'extra.txt').write_text('preserve this')
        before = self.snapshot()
        for operation in (generator.check_assets, generator.generate_assets):
            with self.assertRaises(ValueError):
                operation(self.directory)
            self.assertEqual(before, self.snapshot())

    def test_parent_leaf_links_and_directory_members_are_rejected_before_writes(self):
        linked = Path(self.temp.name)/'linked'
        linked.symlink_to(self.directory, target_is_directory=True)
        before = self.snapshot()
        for operation in (generator.generate_assets, generator.check_assets):
            with self.assertRaises(ValueError):
                operation(linked)
        self.assertEqual(before, self.snapshot())
        target = self.directory/'sweep_cut.wav'
        target.unlink()
        target.symlink_to(self.directory/'sweep_complete.wav')
        with self.assertRaises(ValueError):
            generator.generate_assets(self.directory)
        target.unlink()
        target.mkdir()
        with self.assertRaises(ValueError):
            generator.generate_assets(self.directory)


if __name__ == '__main__':
    unittest.main()
