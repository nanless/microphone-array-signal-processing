"""A changed generator must invalidate published GSS and moving-source assets."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import quality_check


class AssetSourceIntegrityTest(unittest.TestCase):
    def test_generator_digest_mismatch_is_rejected(self):
        original_read = Path.read_text
        for asset_dir, field, check in (
            ("gss_audio", "generator_inputs", quality_check.check_gss_audio),
            ("moving_audio", "source_sha256", quality_check.check_moving_audio),
        ):
            with self.subTest(asset_dir=asset_dir):
                manifest = quality_check.ROOT / "codes" / asset_dir / "MANIFEST.json"
                recorded = json.loads(original_read(manifest, encoding="utf-8"))
                source = next(iter(recorded[field]))
                recorded[field][source] = "0" * 64

                def read_text(path, *args, **kwargs):
                    if path == manifest:
                        return json.dumps(recorded)
                    return original_read(path, *args, **kwargs)

                errors = []
                with patch.object(Path, "read_text", read_text):
                    check(errors)
                self.assertTrue(any("生成源码已变化" in message and source in message
                                    for message in errors), errors)


if __name__ == "__main__":
    unittest.main()
