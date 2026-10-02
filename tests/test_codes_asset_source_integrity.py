"""A changed generator must invalidate published GSS, moving and tracking assets."""

import copy
import unittest
from unittest.mock import patch

from scripts import quality_check


class AssetSourceIntegrityTest(unittest.TestCase):
    def test_generator_digest_mismatch_is_rejected(self):
        original_read = quality_check._read_audio_manifest
        for asset_dir, field, check in (
            ("gss_audio", "generator_inputs", quality_check.check_gss_audio),
            ("moving_audio", "source_sha256", quality_check.check_moving_audio),
            ("tracking_audio", "source_sha256", quality_check.check_tracking_audio),
        ):
            with self.subTest(asset_dir=asset_dir):
                owner = {"gss_audio": "ch08", "moving_audio": "ch09",
                         "tracking_audio": "ch09"}[asset_dir]
                manifest = quality_check.ROOT / "codes" / "chapters" / owner / asset_dir / "MANIFEST.json"
                recorded = original_read(manifest)
                source = next(iter(recorded[field]))
                def read_manifest(path, *args, **kwargs):
                    # Preserve the real ordinary-path and strict-JSON read;
                    # inject only a false provenance claim at its consumer.
                    value = original_read(path, *args, **kwargs)
                    if path == manifest:
                        value = copy.deepcopy(value)
                        value[field][source] = "0" * 64
                    return value

                errors = []
                with patch.object(quality_check, "_read_audio_manifest",
                                  side_effect=read_manifest) as reader:
                    check(errors)
                reader.assert_called_with(manifest)
                self.assertTrue(any("生成源码已变化" in message and source in message
                                    for message in errors), errors)


if __name__ == "__main__":
    unittest.main()
