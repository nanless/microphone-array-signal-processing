import importlib.metadata
import hashlib
from pathlib import Path
import unittest

from codes.examples.compare_online_wpe_reference import (
    compare_locked_upstream,
    fixed_index_hand_calculation,
    state_lifetime_experiment,
)


class TestOnlineWPEReference(unittest.TestCase):
    def test_adapted_reference_retains_upstream_notice(self):
        root = Path(__file__).resolve().parents[1]
        notice = root / "codes/licenses/nara_wpe_MIT.txt"
        self.assertEqual(hashlib.sha256(notice.read_bytes()).hexdigest(),
                         "eef25788ae3423a6d6188f3fc0d23c083d8e7f9ab18a1d1e495df5879f2edaf0")
        script = (root / "codes/examples/compare_online_wpe_reference.py").read_text()
        self.assertIn("../licenses/nara_wpe_MIT.txt", script)
        self.assertIn("Copyright (c) 2018 Communications Engineering Group, Paderborn University", script)

    def test_fixed_index_hand_calculation(self):
        result = fixed_index_hand_calculation()
        self.assertEqual(
            result["offline_build_y_tilde"],
            {"regressor_index": 2, "regressor": 30.0, "residual": 10.0},
        )
        self.assertEqual(
            result["stateless_online_wpe_step"],
            {"regressor_index": 1, "regressor": 20.0, "residual": 20.0},
        )
        self.assertEqual(
            result["stateful_OnlineWPE_step_frame"],
            {"regressor_index": 0, "regressor": 10.0, "residual": 30.0},
        )

    def test_state_is_continuous_across_outer_chunks_but_not_reset(self):
        result = state_lifetime_experiment()
        self.assertEqual(result["continuous_vs_same_state_chunks_max_abs"], 0.0)
        self.assertEqual(result["first_changed_zero_based_frame_after_reset"], 24)
        self.assertGreater(result["continuous_vs_reset_max_abs"], 1.0)

    def test_locked_upstream_when_available(self):
        try:
            version = importlib.metadata.version("nara-wpe")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("optional nara-wpe is not installed")
        if version != "0.0.11":
            self.skipTest(f"optional nara-wpe 0.0.11 not installed (found {version})")
        result = compare_locked_upstream()
        self.assertEqual(result["status"], "checked")
        self.assertLess(result["adapted_numpy_vs_upstream_max_abs"], 1e-12)


if __name__ == "__main__":
    unittest.main()
