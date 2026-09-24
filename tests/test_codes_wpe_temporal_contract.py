"""Independent time-index and state checks for the WPE interface experiment."""

import importlib.metadata
import math
import unittest

import numpy as np

from codes.examples.wpe_temporal_contract import (
    CHUNK_PARTITIONS,
    FRAME_COUNT,
    PERTURBED_FRAME,
    PERTURBATION,
    fixed_frames,
    process_chunks,
    report,
    temporal_summary,
)
from codes.examples.compare_online_wpe_reference import NumpyOnlineWPE011


def new_state():
    return NumpyOnlineWPE011(
        taps=2, delay=2, alpha=0.95, frequency_bins=2, channels=1
    )


class TestWPETemporalContract(unittest.TestCase):
    def test_input_change_is_exactly_at_declared_frame(self):
        original, changed = fixed_frames()
        self.assertEqual(original.shape, (48, 2, 1))
        np.testing.assert_array_equal(original[:PERTURBED_FRAME], changed[:PERTURBED_FRAME])
        np.testing.assert_array_equal(original[PERTURBED_FRAME + 1:], changed[PERTURBED_FRAME + 1:])
        np.testing.assert_array_equal(
            changed[PERTURBED_FRAME] - original[PERTURBED_FRAME],
            np.full((2, 1), PERTURBATION),
        )

    def test_causal_prefix_and_same_frame_difference_have_independent_expected_values(self):
        original, changed = fixed_frames()
        out_original = process_chunks(new_state, original, (FRAME_COUNT,))
        out_changed = process_chunks(new_state, changed, (FRAME_COUNT,))
        # Identical input prefixes must build identical pre-update state.
        np.testing.assert_array_equal(
            out_original[:PERTURBED_FRAME], out_changed[:PERTURBED_FRAME]
        )
        # At the changed frame the old filter and regressor are identical;
        # only the current observation differs, by the known perturbation.
        np.testing.assert_allclose(
            out_changed[PERTURBED_FRAME] - out_original[PERTURBED_FRAME],
            np.full((2, 1), PERTURBATION), rtol=0, atol=2e-15,
        )
        self.assertAlmostEqual(
            float(np.max(np.abs(out_changed[PERTURBED_FRAME] - out_original[PERTURBED_FRAME]))),
            math.sqrt(29.0), places=14,
        )

    def test_outer_chunking_is_bitwise_invariant(self):
        frames, _ = fixed_frames()
        whole = process_chunks(new_state, frames, CHUNK_PARTITIONS[0])
        for partition in CHUNK_PARTITIONS[1:]:
            with self.subTest(partition=partition):
                np.testing.assert_array_equal(whole, process_chunks(new_state, frames, partition))

    def test_rejects_invalid_partitions(self):
        frames, _ = fixed_frames()
        for bad in ((), (47,), (49,), (0, 48), (True, 47), (-1, 49)):
            with self.subTest(partition=bad):
                with self.assertRaises(ValueError):
                    process_chunks(new_state, frames, bad)

    def test_summary_uses_distinct_online_and_offline_claims(self):
        summary = temporal_summary(new_state)
        self.assertEqual(summary["earlier_online_output_max_abs_difference"], 0.0)
        self.assertEqual(summary["changed_input_samples"], 2)
        self.assertTrue(all(row["bitwise_equal_to_whole"] for row in summary["outer_chunk_checks"]))

    def test_locked_upstream_when_installed(self):
        try:
            version = importlib.metadata.version("nara-wpe")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("optional nara-wpe 0.0.11 is not installed")
        if version != "0.0.11":
            self.skipTest(f"optional nara-wpe 0.0.11 not installed (found {version})")
        result = report()["locked_upstream"]
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["online"]["earlier_online_output_max_abs_difference"], 0.0)
        self.assertTrue(all(row["bitwise_equal_to_whole"] for row in result["online"]["outer_chunk_checks"]))
        self.assertGreater(result["offline_control"]["earlier_output_max_abs_difference"], 0.01)


if __name__ == "__main__":
    unittest.main()
