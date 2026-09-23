"""Independent hand values and boundaries for the teaching PBFDAF baseline."""

import unittest

import numpy as np

from codes.array_tutorial.aec_partitioned import PartitionedFDAFState
from codes.examples.aec_partitioned_demo import run_demo


class TestPartitionedFDAFState(unittest.TestCase):
    def test_two_blocks_match_hand_update_and_projection(self):
        # N=2, P=2, mu=1/2, delta=1, x=[1,0,0,0], d=[1,0,2,0].
        # Block 0: X0=E0=[1,-1,1,-1], bin power+delta=2,
        # so the first partition becomes [1/4,0,0,0] in time.
        state = PartitionedFDAFState(4, 2, step_size=.5, epsilon=1.)
        e0, y0 = state.process([1., 0.], [1., 0.])
        np.testing.assert_allclose(y0, [0., 0.], atol=1e-14)
        np.testing.assert_allclose(e0, [1., 0.], atol=1e-14)
        np.testing.assert_allclose(state.partition_time_coefficients,
                                   [[.25, 0., 0., 0.], [0., 0., 0., 0.]], atol=1e-14)

        # Block 1: X1=[1,1,1,1], old X0=[1,-1,1,-1], E1=[2,-2,2,-2].
        # Bin power+delta=3. The p=0 candidate has illegal lag-2 tap 1/3;
        # its projection removes that tap. The p=1 legal first tap is 1/3.
        e1, y1 = state.process([0., 0.], [2., 0.])
        np.testing.assert_allclose(y1, [0., 0.], atol=1e-14)
        np.testing.assert_allclose(e1, [2., 0.], atol=1e-14)
        np.testing.assert_allclose(state.last_candidate_partition_time_coefficients,
                                   [[.25, 0., 1 / 3, 0.], [1 / 3, 0., 0., 0.]],
                                   atol=1e-14)
        np.testing.assert_allclose(state.partition_time_coefficients,
                                   [[.25, 0., 0., 0.], [1 / 3, 0., 0., 0.]],
                                   atol=1e-14)
        np.testing.assert_allclose(state.fir_weights, [.25, 0., 1 / 3, 0.], atol=1e-14)

    def test_frozen_known_fir_matches_independent_direct_convolution(self):
        reference = np.array([1., -2., 3., .5, 0., 1., -1., 2.])
        path = np.array([.4, -.2, .1, .05, -.03])  # L=5, N=2, P=3.
        state = PartitionedFDAFState(5, 2, step_size=0., initial_weights=path)
        residual, estimate = state.process(reference, np.zeros_like(reference))
        expected = np.convolve(reference, path, mode="full")[:len(reference)]
        np.testing.assert_allclose(estimate, expected, atol=1e-14)
        np.testing.assert_allclose(residual, -expected, atol=1e-14)
        np.testing.assert_allclose(state.fir_weights, path, atol=1e-14)
        np.testing.assert_allclose(state.partition_time_coefficients[:, 2:], 0., atol=1e-14)
        self.assertEqual(state.partitions, 3)
        self.assertEqual(state.fft_length, 4)

    def test_last_partial_partition_stays_inside_requested_fir_length(self):
        state = PartitionedFDAFState(3, 2, step_size=.5, epsilon=1.)
        state.process([1., 0., 0., 0.], [1., 0., 2., 0.])
        self.assertEqual(state.fir_weights.shape, (3,))
        np.testing.assert_allclose(state.partition_time_coefficients[1, 1:], 0., atol=1e-14)

    def test_unconstrained_variant_keeps_circular_candidate(self):
        state = PartitionedFDAFState(4, 2, step_size=.5, epsilon=1.,
                                     constrain_gradient=False)
        state.process([1., 0., 0., 0.], [1., 0., 2., 0.])
        np.testing.assert_allclose(state.partition_time_coefficients,
                                   [[.25, 0., 1 / 3, 0.], [1 / 3, 0., 0., 0.]],
                                   atol=1e-14)
        with self.assertRaisesRegex(ValueError, "not a length-L FIR"):
            _ = state.fir_weights

    def test_whole_blocks_across_calls_match_one_call_and_reset(self):
        rng = np.random.default_rng(20260923)
        reference = rng.normal(size=20)
        path = np.array([.6, -.15, .05, .01, .02])
        microphone = np.convolve(reference, path, mode="full")[:len(reference)]
        microphone[6:10] += rng.normal(size=4) * .2
        freeze = np.array([False, False, False, True, True,
                           False, False, False, False, False])
        kwargs = dict(filter_length=5, block_length=2, step_size=.3, epsilon=.01)
        whole = PartitionedFDAFState(**kwargs)
        expected_e, expected_y = whole.process(reference, microphone, freeze=freeze)
        chunks = PartitionedFDAFState(**kwargs)
        outputs = []
        predictions = []
        for lo, hi in ((0, 4), (4, 4), (4, 12), (12, 20)):
            e, y = chunks.process(reference[lo:hi], microphone[lo:hi],
                                  freeze=freeze[lo // 2:hi // 2])
            outputs.append(e)
            predictions.append(y)
        np.testing.assert_array_equal(np.concatenate(outputs), expected_e)
        np.testing.assert_array_equal(np.concatenate(predictions), expected_y)
        np.testing.assert_array_equal(chunks.partition_time_coefficients,
                                      whole.partition_time_coefficients)
        np.testing.assert_array_equal(chunks.reference_overlap, reference[-2:])

        chunks.reset()
        np.testing.assert_array_equal(chunks.fir_weights, np.zeros(5))
        np.testing.assert_array_equal(chunks.reference_overlap, np.zeros(2))
        self.assertIsNone(chunks.last_candidate_partition_time_coefficients)
        rerun_e, rerun_y = chunks.process(reference, microphone, freeze=freeze)
        np.testing.assert_array_equal(rerun_e, expected_e)
        np.testing.assert_array_equal(rerun_y, expected_y)

    def test_freeze_keeps_weights_but_advances_reference_history(self):
        state = PartitionedFDAFState(4, 2, step_size=.5, epsilon=1.)
        e0, y0 = state.process([1., 0.], [1., 0.], freeze=[True])
        np.testing.assert_array_equal(e0, [1., 0.])
        np.testing.assert_array_equal(y0, [0., 0.])
        np.testing.assert_array_equal(state.fir_weights, np.zeros(4))
        np.testing.assert_array_equal(state.reference_overlap, [1., 0.])
        self.assertIsNone(state.last_candidate_partition_time_coefficients)
        # The current block is zero, but the delayed p=1 reference is nonzero.
        state.process([0., 0.], [2., 0.])
        # The current overlap-save FFT still contains the previous two
        # samples, so its power also enters the denominator: 1+1+1=3.
        np.testing.assert_allclose(state.fir_weights, [0., 0., 1 / 3, 0.], atol=1e-14)

    def test_zero_reference_does_not_update_or_divide_by_zero(self):
        state = PartitionedFDAFState(3, 2, epsilon=1e-8)
        e, y = state.process(np.zeros(6), [1., -1., 2., -2., 3., -3.])
        np.testing.assert_array_equal(y, np.zeros(6))
        np.testing.assert_array_equal(e, [1., -1., 2., -2., 3., -3.])
        np.testing.assert_array_equal(state.fir_weights, np.zeros(3))

    def test_invalid_inputs_and_overflow_leave_state_unchanged(self):
        state = PartitionedFDAFState(3, 2, initial_weights=[.5, -.2, .1])
        original = state.partition_time_coefficients.copy()
        for reference, microphone, freeze in (
            ([1., 0., 0.], [1., 0., 0.], None),
            ([1., 0.], [1.], None),
            ([1. + 0j, 0.], [1., 0.], None),
            ([1., 0.], [1. + 0j, 0.], None),
            ([float("nan"), 0.], [1., 0.], None),
            ([1., 0.], [1., 0.], [1]),
            ([1., 0.], [1., 0.], [True, False]),
        ):
            with self.subTest(reference=reference, freeze=freeze):
                with self.assertRaises(ValueError):
                    state.process(reference, microphone, freeze=freeze)
                np.testing.assert_array_equal(state.partition_time_coefficients, original)
                np.testing.assert_array_equal(state.reference_overlap, [0., 0.])
        # The first block would update, but the second block's squared FFT
        # magnitude cannot be represented. No part of the call is committed.
        with self.assertRaises(ValueError):
            state.process([1., 0., 1e308, 0.], [1., 0., 1., 0.])
        np.testing.assert_array_equal(state.partition_time_coefficients, original)
        np.testing.assert_array_equal(state.reference_overlap, [0., 0.])

    def test_constructor_validation_and_returned_copies(self):
        for args, kwargs in (
            ((0, 2), {}), ((2, False), {}), ((2, 2), {"epsilon": 0.}),
            ((2, 2), {"step_size": 2.}),
            ((2, 2), {"constrain_gradient": 1}),
            ((2, 2), {"initial_weights": [1.]}),
            ((2, 2), {"initial_weights": [1. + 0j, 0.]}),
            ((1, 2), {"step_size": 0., "initial_weights": [1e308]}),
        ):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                PartitionedFDAFState(*args, **kwargs)
        state = PartitionedFDAFState(2, 2)
        copy = state.partition_time_coefficients
        copy[:] = 9.
        np.testing.assert_array_equal(state.partition_time_coefficients, np.zeros((1, 4)))


class TestPartitionedDemo(unittest.TestCase):
    def test_demo_contains_hand_oracle_and_honest_scope(self):
        result = run_demo()
        self.assertEqual(result["block_length"], 2)
        self.assertEqual(result["partitions"], 2)
        self.assertEqual(result["residual"], [1., 0., 2., 0.])
        self.assertAlmostEqual(result["candidate_second_block"][0][2], 1 / 3)
        self.assertIn("not Speex", result["scope"])


if __name__ == "__main__":
    unittest.main()
