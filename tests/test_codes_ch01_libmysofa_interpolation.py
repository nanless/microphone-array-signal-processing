"""Offline provenance, independent interpolation expectations and report IO."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from codes.chapters.ch00.io_contracts import strict_json_loads, write_json_report
from codes.chapters.ch00.upstream.fetch_upstreams import run_git
from codes.chapters.ch01.examples import audit_libmysofa_interpolation as audit

# Written from exact arithmetic, not produced by the original C implementation.
CONTROL_STDOUT = '''global_exact 2 4 8 16
global_midpoint 3 5 4 8
global_zero_midpoint 3 5 0 0
per_direction_same_midpoint 3 5 8 16
per_direction_vary_midpoint 3 5 10 18
'''


class InterpolationExpectationsTest(unittest.TestCase):
    def test_common_delay_failure_is_not_mathematical_success(self):
        rows = audit.evaluate(CONTROL_STDOUT)
        self.assertEqual(len(rows), 5)
        common = rows[1]
        self.assertEqual(common['independent_expected_ir_left_right'], [(2+4)/2, (4+6)/2])
        self.assertEqual(common['independent_expected_delays_raw'], [8., 16.])
        self.assertEqual(common['observed_delays_raw'], [4., 8.])
        self.assertEqual(common['delay_error_from_mathematical_expectation'], [-4., -8.])
        self.assertFalse(common['mathematical_invariants_passed'])
        self.assertTrue(common['expected_upstream_behavior_observed'])
        self.assertEqual([r['name'] for r in rows if not r['mathematical_invariants_passed']], ['global_midpoint'])
        self.assertEqual(rows[-1]['independent_expected_delays_raw'], [(8+12)/2, (16+20)/2])
        self.assertTrue(all(r['ir_interpolation_invariant_passed'] for r in rows))

    def test_changed_output_is_detected_separately_from_expected_upstream_failure(self):
        corrected = audit.evaluate(CONTROL_STDOUT.replace('global_midpoint 3 5 4 8', 'global_midpoint 3 5 8 16'))[1]
        self.assertTrue(corrected['mathematical_invariants_passed'])
        self.assertFalse(corrected['expected_upstream_behavior_observed'])
        wrong_ir = audit.evaluate(CONTROL_STDOUT.replace('global_exact 2 4 8 16', 'global_exact 20 4 8 16'))[0]
        self.assertFalse(wrong_ir['ir_interpolation_invariant_passed'])
        self.assertFalse(wrong_ir['expected_upstream_behavior_observed'])

    def test_missing_reordered_nonfinite_and_extra_fields_are_rejected(self):
        for stdout in ('', CONTROL_STDOUT.split('\n', 1)[1],
                       CONTROL_STDOUT.replace('global_exact', 'wrong'),
                       CONTROL_STDOUT.replace('global_exact 2', 'global_exact nan'),
                       CONTROL_STDOUT.replace('global_exact 2', 'global_exact inf'),
                       CONTROL_STDOUT + 'extra 1 2 3 4\n',
                       CONTROL_STDOUT.replace('global_exact 2', 'global_exact 2 7')):
            with self.subTest(stdout=stdout[:40]), self.assertRaises(ValueError):
                audit.evaluate(stdout)


class InterpolationSourceIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkout = Path(self.temporary.name) / 'source'
        self.checkout.mkdir()
        self.origin = 'https://example.invalid/official.git'
        self.payload = b'unchanged source and original notice\n'
        (self.checkout / 'source.c').write_bytes(self.payload)
        run_git(['init', '-q'], cwd=self.checkout)
        run_git(['remote', 'add', 'origin', self.origin], cwd=self.checkout)
        run_git(['add', 'source.c'], cwd=self.checkout)
        run_git(['-c', 'user.name=Offline Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'offline fixture'], cwd=self.checkout)
        self.revision = run_git(['rev-parse', 'HEAD'], cwd=self.checkout)
        self.files = {'source.c': hashlib.sha256(self.payload).hexdigest()}

    def verify(self, **overrides):
        args = {'revision': self.revision, 'origin': self.origin, 'files': self.files}
        args.update(overrides)
        return audit.verify_checkout(self.checkout, **args)

    def test_official_origin_head_sha_and_blob_are_independently_recorded(self):
        # An outer Git operation must not redirect the independent checkout.
        with mock.patch.dict(os.environ, {'GIT_DIR': '/nonexistent/outer-git',
                                         'GIT_WORK_TREE': '/nonexistent/outer-tree'}):
            evidence = self.verify()
        self.assertTrue(evidence['required_source_identity_verified'])
        member = evidence['files']['source.c']
        self.assertEqual(member['bytes'], len(self.payload))
        self.assertEqual(member['sha256'], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(member['head_blob'], member['actual_blob'])
        self.assertEqual((self.checkout / 'source.c').read_bytes(), self.payload)

    def test_wrong_origin_head_sha_and_dirty_checkout_are_rejected(self):
        for changes in ({'origin': 'https://example.invalid/other.git'},
                        {'revision': '0' * 40}, {'revision': self.revision[:7]},
                        {'files': {'source.c': '0' * 64}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.verify(**changes)
        (self.checkout / 'untracked.txt').write_text('preserve me')
        with self.assertRaisesRegex(ValueError, 'clean'):
            self.verify()
        self.assertEqual((self.checkout / 'untracked.txt').read_text(), 'preserve me')
        (self.checkout / 'untracked.txt').unlink()
        (self.checkout / 'source.c').write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'clean'):
            self.verify()
        self.assertEqual((self.checkout / 'source.c').read_bytes(), b'edited')

    def test_unsafe_source_member_paths_are_rejected(self):
        for relative in ('../escape', '/absolute', ''):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.verify(files={relative: '0' * 64})


class InterpolationReportTest(unittest.TestCase):
    def test_only_new_current_report_can_be_written_inside_repository(self):
        self.assertEqual(audit.report_target(audit.CURRENT_REPORT), audit.CURRENT_REPORT)
        for forbidden in (audit.LOCK, audit.STATUS, audit.CACHE / 'libmysofa/LICENSE',
                          audit.ROOT / 'codes/chapters/ch01/reports/libmysofa_loudness.json',
                          Path(audit.__file__), Path(__file__).resolve(),
                          audit.ROOT / 'codes/chapters/ch01/reports/unapproved.json'):
            with self.subTest(path=forbidden), self.assertRaises(ValueError):
                audit.report_target(forbidden)

    def test_external_ordinary_report_is_atomic_strict_json_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = audit.report_target(root / 'nested/result.json')
            data = {'cases': audit.evaluate(CONTROL_STDOUT), 'all_mathematical_invariants_passed': False}
            write_json_report(target, data)
            self.assertEqual(strict_json_loads(target.read_bytes()), data)
            self.assertEqual(list(target.parent.iterdir()), [target])
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                write_json_report(target, {'invalid': float('nan')})
            self.assertEqual(target.read_bytes(), before)
            link = root / 'linked.json'
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                audit.report_target(link)
            directory_link = root / 'directory-link'
            directory_link.symlink_to(target.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                audit.report_target(directory_link / 'new.json')
            hardlink = root / 'hardlink.json'
            os.link(target, hardlink)
            with self.assertRaises(ValueError):
                audit.report_target(hardlink)

    def test_unsafe_destination_is_rejected_before_original_execution(self):
        with mock.patch('sys.argv', ['audit', '--report', str(audit.LOCK)]), \
                mock.patch.object(audit, 'run_audit') as run:
            with self.assertRaises(ValueError):
                audit.main()
            run.assert_not_called()

    def test_default_stdout_does_not_publish_or_upgrade_math_failure(self):
        report = {'expected_upstream_behavior_observed': True,
                  'all_mathematical_invariants_passed': False,
                  'cases': audit.evaluate(CONTROL_STDOUT)}
        output = io.StringIO()
        with mock.patch('sys.argv', ['audit']), mock.patch.object(audit, 'run_audit', return_value=report), \
                mock.patch.object(audit, 'write_json_report') as write, contextlib.redirect_stdout(output):
            audit.main()
        write.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), report)


if __name__ == '__main__':
    unittest.main()
