"""Static-audit fixtures do not require upstream sources or PyTorch."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from codes.examples.tflocoformer_source_audit import audit, inspect_separator, REVISION

FIXTURE = '''
class Other:
    Norm = {"rmsgrouporm": object}
class TFLocoformerSeparator:
    def __init__(self, norm_type="rmsgrouporm"):
        self.conv = nn.Conv2d(2, 8, 3)
    def forward(self, input):
        if input.ndim == 4:
            assert input.shape[1] == 1
            batch0 = input.transpose(1, 2)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        return batch
class LocoformerBlock:
    def __init__(self):
        Norm = {"layernorm": nn.LayerNorm, "rmsgroupnorm": RMSGroupNorm}
        assert norm_type in Norm
'''


class SourceAuditTests(TestCase):
    def test_mismatch_and_explicit_dimensions(self):
        actual = inspect_separator(FIXTURE)
        self.assertTrue(actual["default_norm"]["mismatch"])
        cases = actual["four_dimensional_path"]["cases"]
        self.assertFalse(cases[0]["passes_axis1_assertion"])
        self.assertEqual(cases[1]["real_imag_conv_input"], [2, 10, 1, 17])
        self.assertFalse(cases[1]["matches_conv_channels"])
        self.assertTrue(cases[2]["matches_conv_channels"])

    def test_valid_default_and_irrelevant_names(self):
        # An unrelated class has the misspelled key; only the consumer matters.
        fixed = FIXTURE.replace('norm_type="rmsgrouporm"', 'norm_type="rmsgroupnorm"')
        self.assertFalse(inspect_separator(fixed)["default_norm"]["mismatch"])

    def test_changed_path_is_not_reported_as_same_defect(self):
        for source in (FIXTURE.replace("input.shape[1]", "input.shape[2]"),
                       FIXTURE.replace("input.transpose(1, 2)", "input")):
            with self.assertRaises(ValueError):
                inspect_separator(source)

    @patch("codes.examples.tflocoformer_source_audit.subprocess.check_output", return_value="different")
    def test_changed_revision_is_rejected(self, _):
        with self.assertRaisesRegex(ValueError, "revision"):
            audit(Path("unused"))

    @patch("codes.examples.tflocoformer_source_audit.subprocess.check_output", return_value=REVISION)
    def test_changed_bytes_are_rejected(self, _):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "standalone").mkdir()
            (root / "standalone/tflocoformer_separator.py").write_text(FIXTURE)
            with self.assertRaisesRegex(ValueError, "digest"):
                audit(root)
