"""Offline source identities, output boundaries, and bounded original controls."""
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from codes.chapters.ch15.examples import audit_upstream_distributed_contracts as audit
from codes.chapters.ch00.io_contracts import strict_json_loads, write_json_report
from codes.chapters.ch00.upstream.fetch_upstreams import run_git


class DistributedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve() / 'repo'
        self.repo.mkdir()
        run_git(['init', '-q'], cwd=self.repo)
        run_git(['config', 'user.name', 'Offline fixture'], cwd=self.repo)
        run_git(['config', 'user.email', 'fixture@example.invalid'], cwd=self.repo)
        run_git(['remote', 'add', 'origin', audit.PADER_ORIGIN], cwd=self.repo)
        self.file = self.repo / 'LICENSE'
        self.file.write_text('ordinary identity fixture\n')
        run_git(['add', 'LICENSE'], cwd=self.repo)
        run_git(['commit', '-qm', 'offline fixture'], cwd=self.repo)
        self.revision = run_git(['rev-parse', 'HEAD'], cwd=self.repo)
        self.files = {'LICENSE': hashlib.sha256(self.file.read_bytes()).hexdigest()}

    def verify(self, **overrides):
        options = {'revision': self.revision, 'origin': audit.PADER_ORIGIN, 'files': self.files}
        options.update(overrides)
        return audit.verify_checkout(self.repo, **options)

    def test_real_sha_and_blob_agree(self):
        row = self.verify()['files']['LICENSE']
        self.assertEqual(row['sha256'], self.files['LICENSE'])
        self.assertEqual(row['head_blob'], row['actual_blob'])

    def test_wrong_origin_revision_and_digest(self):
        with self.assertRaisesRegex(ValueError, 'origin mismatch'):
            self.verify(origin='https://wrong.invalid/source.git')
        with self.assertRaisesRegex(ValueError, 'HEAD mismatch'):
            self.verify(revision='0' * 40)
        with self.assertRaisesRegex(ValueError, '40-character'):
            self.verify(revision=self.revision[:12])
        with self.assertRaisesRegex(ValueError, 'SHA/blob mismatch'):
            self.verify(files={'LICENSE': '0' * 64})

    def test_dirty_and_untracked_source(self):
        extra = self.repo / 'extra.txt'
        extra.write_text('extra')
        with self.assertRaisesRegex(ValueError, 'completely clean'):
            self.verify()
        extra.unlink()
        self.file.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'completely clean'):
            self.verify()

    def test_linked_or_escaping_source(self):
        for name in ('../LICENSE', str(self.file)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'relative checkout'):
                self.verify(files={name: self.files['LICENSE']})
        link = self.repo.parent / 'alias'
        link.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            audit.verify_checkout(link, revision=self.revision, origin=audit.PADER_ORIGIN, files=self.files)
        os.link(self.file, self.repo.parent / 'license-copy')
        with self.assertRaisesRegex(ValueError, 'singly linked'):
            self.verify()


class DistributedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()

    def test_protected_source_cache_and_historical_reports(self):
        for path in (audit.CACHE / 'paderwasn/report.json', audit.LOCK, Path(audit.__file__),
                     audit.ROOT / 'reviews/old.json',
                     audit.ROOT / 'codes/chapters/ch14/reports/upstream_imaging_contracts.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.report_target(path)
        self.assertEqual(audit.report_target(audit.CURRENT_REPORT), audit.CURRENT_REPORT)

    def test_symlink_hardlink_directory_and_parent_traversal(self):
        ordinary = self.directory / 'ordinary.json'
        ordinary.write_text('{}')
        link = self.directory / 'link.json'
        link.symlink_to(ordinary)
        hardlink = self.directory / 'hard.json'
        os.link(ordinary, hardlink)
        for path in (link, hardlink, self.directory, self.directory / '..' / 'escape.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.report_target(path)

    def test_preflight_before_original_calls(self):
        with patch.object(audit, 'run_audit') as run:
            with self.assertRaises(ValueError):
                audit.main(['--report', str(audit.CACHE / 'paderwasn/bad.json')])
            run.assert_not_called()

    def test_stdout_has_no_write(self):
        value = {'status': 'audit_completed_with_original_differences_and_unobserved_peaks'}
        stream = io.StringIO()
        with patch.object(audit, 'run_audit', return_value=value), \
                patch.object(audit, 'write_json_report') as writer, redirect_stdout(stream):
            self.assertEqual(audit.main([]), 0)
        writer.assert_not_called()
        self.assertEqual(strict_json_loads(stream.getvalue()), value)

    def test_explicit_current_external_output_preserves_status(self):
        value = {'status': 'audit_completed_with_original_differences_and_unobserved_peaks',
                 'matlab_executed': False}
        target = self.directory / 'current.json'
        with patch.object(audit, 'run_audit', return_value=value), redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(['--report', str(target)]), 0)
        self.assertEqual(strict_json_loads(target.read_bytes()), value)
        with self.assertRaises(ValueError):
            write_json_report(self.directory / 'bad.json', {'value': float('inf')})
        self.assertFalse((self.directory / 'bad.json').exists())

    def test_extraction_preserves_body_and_omits_module_import(self):
        source = self.directory / 'fixture.py'
        source.write_text('raise RuntimeError("module must not execute")\n'
                          '@missing_decorator\ndef original(x):\n    return x + 2\n')
        records, namespace = [], {}
        audit.extract_original(source, ['original'], namespace, records)
        self.assertEqual(namespace['original'](3), 5)
        self.assertTrue(records[0]['mathematical_body_unchanged'])
        self.assertEqual(records[0]['removed_decorators'], ['missing_decorator'])
        self.assertNotEqual(records[0]['original_definition_ast_sha256'],
                            records[0]['executed_definition_ast_sha256'])

    def test_locked_license_and_selection_cannot_be_relabelled(self):
        projects = [
            {'id': 'danse-wola', 'revision': audit.WOLA_REVISION, 'url': audit.WOLA_ORIGIN,
             'license': 'LicenseRef-Bertrand-DANSE-3-Conditions', 'entrypoints': list(audit.WOLA_FILES)},
            {'id': 'paderwasn', 'revision': audit.PADER_REVISION, 'url': audit.PADER_ORIGIN,
             'license': 'MIT', 'entrypoints': list(audit.PADER_FILES)},
        ]
        lock = self.directory / 'sources.json'
        write_json_report(lock, {'projects': projects})
        # The identity stub isolates the lock/selection boundary; it is never
        # presented as an original-source execution or a production report.
        selection = {'status': 'source_selection_mismatch', 'source_selection_verified': False}
        with patch.object(audit, 'LOCK', lock), patch.object(audit, 'verify_checkout') as verify, \
                patch.object(audit, 'inspect_project', return_value=selection):
            verify.return_value = {'required_source_identity_verified': True}
            result = audit.verify_sources(self.directory / 'cache')
            self.assertFalse(result['projects']['paderwasn']['acquisition_scope']['source_selection_verified'])
            projects[0]['license'] = 'BSD-3-Clause'
            write_json_report(lock, {'projects': projects})
            with self.assertRaisesRegex(ValueError, 'file-header license'):
                audit.verify_sources(self.directory / 'cache')
            projects[0]['license'] = 'LicenseRef-Bertrand-DANSE-3-Conditions'
            projects[1]['entrypoints'].remove('paderwasn/synchronization/utils.py')
            write_json_report(lock, {'projects': projects})
            with self.assertRaisesRegex(ValueError, 'entrypoints'):
                audit.verify_sources(self.directory / 'cache')


class DistributedIndependentControls(unittest.TestCase):
    def test_math_and_scheduling_are_not_original_matlab(self):
        rows = {r['name']: r for r in audit.independent_matlab_controls()}
        np.testing.assert_array_equal(rows['negative_scm_algebraic_max_then_abs']['reconstructed_scm'],
                                      [[0, 0], [0, 1]])
        self.assertFalse(rows['symmetric_hann_cola']['unit_cola'])
        self.assertGreater(rows['symmetric_hann_cola']['max_unit_error'], .003)
        time = rows['external_target_two_frame_effect']
        self.assertEqual(time['first_affected_broadcast_frame'], 2)
        self.assertEqual([r['broadcast_weight'] for r in time['timeline']], [1., 1., 2., 2.5])
        self.assertFalse(rows['many_noise_frames_do_not_imply_spd']['inverse_exists'])
        self.assertEqual(rows['unprocessed_terminal_support']['zero_based_processed_starts'], [0, 4, 8, 12])
        for row in rows.values():
            self.assertIn('not', row['execution'])

    def test_incorrect_original_control_fails_instead_of_recording_success(self):
        fake = SimpleNamespace(coarse_sync=lambda sig, ref, n: (sig, ref, 999))
        with self.assertRaises(ValueError):
            audit.original_cases(fake)
        with self.assertRaisesRegex(ValueError, 'recorded contract'):
            audit.comparison('bad numerical control', [5.], [0.], tolerance=1e-4)


@unittest.skipUnless((audit.CACHE / 'paderwasn/paderwasn/synchronization/sync.py').is_file()
                     and (audit.CACHE / 'danse-wola/WOLA_DANSE1.m').is_file(),
                     'optional fixed sources absent; no network or substitute original code')
class DistributedOriginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.run_audit()
        cls.rows = {r['name']: r for r in cls.report['cases']}

    def test_identity_pre_post_and_selection_are_distinct(self):
        self.assertEqual(self.report['source_identity_before'], self.report['source_identity_after'])
        for identifier, revision, files in (('paderwasn', audit.PADER_REVISION, audit.PADER_FILES),
                                             ('danse-wola', audit.WOLA_REVISION, audit.WOLA_FILES)):
            identity = self.report['source_identity_before']['projects'][identifier]
            self.assertEqual(identity['head'], revision)
            self.assertTrue(identity['required_source_identity_verified'])
            self.assertIn('acquisition_scope', identity)
            for path, expected in files.items():
                self.assertEqual(identity['files'][path]['sha256'], expected)

    def test_short_input_and_no_observation_are_preserved(self):
        short = self.rows['coarse_short_input_offset_baseline']
        self.assertEqual(short['actual'], [-4])
        self.assertEqual(short['independent_expected'], [0])
        self.assertFalse(short['matched_independent_expected'])
        self.assertEqual(short['retained_lengths'], [0, 0])
        silence = self.rows['coarse_silence_no_observable_peak']
        self.assertEqual(silence['actual_offset'], -7)
        self.assertFalse(silence['peak_observed'])
        gcc = self.rows['gcc_zero_spectrum_no_observation']
        self.assertIsNone(gcc['independent_identifiable_lag'])
        self.assertFalse(gcc['peak_observed'])

    def test_periodic_lag_precision_is_only_helper_tolerance(self):
        for lag in (0., 3., -5., 2.25, 31.75, 32., 33.):
            row = self.rows['gcc_phase_lag_' + str(lag)]
            self.assertLessEqual(abs(row['periodic_lag_error']), 1e-4)
        self.assertLess(self.rows['gcc_phase_lag_31.75']['actual_lag'], -32.)

    def test_scope_does_not_claim_unexecuted_algorithms(self):
        scope = self.report['execution_scope']
        for field in ('matlab_executed', 'octave_executed', 'dwacd_executed', 'online_wacd_executed',
                      'sro_resampler_executed', 'gevd_danse_executed', 'network_transport_executed'):
            self.assertFalse(scope[field])
        self.assertEqual(len(self.report['original_definitions']), 3)
        self.assertEqual(self.report['counts']['unobserved_peaks_with_numeric_output'], 2)


if __name__ == '__main__':
    unittest.main()
