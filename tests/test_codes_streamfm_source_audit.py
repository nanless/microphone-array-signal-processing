"""Offline fixtures; tests never require the optional Stream.FM download."""
import tempfile
import unittest
from pathlib import Path
from codes.examples.streamfm_source_audit import audit_file, inspect_source

FIXTURE = '''
class CausalResnetBlockBigGANpp:
    def init_state(self):
        return (1, 2)
    def forward_step(self, x, *, state):
        a, b = state
        return x, (a, b)
class CausalConv2d:
    def __init__(self):
        self.depthwise_separable = False
        self.pointwise_conv = None
    def forward_step(self, x, *, state):
        if self.depthwise_separable:
            x = self.pointwise_conv(x)
        return x, state
'''


class StreamFMSourceAudit(unittest.TestCase):
    def test_consistent_contract(self):
        result = inspect_source(FIXTURE)
        self.assertFalse(result['resnet_direct_contract_mismatch'])
        self.assertEqual(result['conv_selected_reads_without_local_assignment'], [])

    def test_mismatch_and_missing_attributes(self):
        bad = FIXTURE.replace('a, b = state', 'a, b, c = state').replace(
            'self.depthwise_separable = False', 'other.depthwise_separable = False').replace(
            'self.pointwise_conv = None', 'other.pointwise_conv = None')
        result = inspect_source(bad)
        self.assertTrue(result['resnet_direct_contract_mismatch'])
        self.assertEqual(result['conv_selected_reads_without_local_assignment'], ['depthwise_separable', 'pointwise_conv'])

    def test_similar_comment_or_other_object_is_not_evidence(self):
        text = FIXTURE + '\n# self.depthwise_separable is undefined; a,b,c = state\n'
        result = inspect_source(text)
        self.assertFalse(result['resnet_direct_contract_mismatch'])
        self.assertEqual(result['conv_selected_reads_without_local_assignment'], [])

    def test_wrong_source_digest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.py'
            path.write_text(FIXTURE)
            with self.assertRaisesRegex(ValueError, 'differs'):
                audit_file(path)


if __name__ == '__main__':
    unittest.main()
