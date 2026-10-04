"""Independent E10-34 answers and channel-order/constraint boundaries."""
import unittest
from unittest.mock import patch
import numpy as np

from codes.chapters.ch10.core.channel_selection import (
    select_channel_observations, select_mvdr_channels, teaching_channel_selection,
)
from codes.chapters.ch05.core.beamforming import apply_beamformer, mvdr_weights


class ChannelSelectionTests(unittest.TestCase):
    def setUp(self):
        # Exact hand input, not reconstructed from the fixture under test.
        self.r = np.array([[1., 1., 0.], [1., 2., 1.], [0., 1., 2.]])
        self.a = np.ones(3)

    def test_old_zeroed_constraint_and_recomputed_answer(self):
        x = teaching_channel_selection()
        np.testing.assert_allclose(x['controls']['healthy']['weights'], [1, -.5, .5], atol=2e-15)
        np.testing.assert_allclose(x['controls']['stale']['weights'], [-.5, .5], atol=2e-15)
        np.testing.assert_allclose(x['controls']['recomputed']['weights'], [.5, .5], atol=2e-15)
        for name, response, noise, damage, total in (
                ('healthy', 1, .5, 0, .5), ('stale', 0, .5, 1, 1.5),
                ('recomputed', 1, 1.5, 0, 1.5)):
            row = x['controls'][name]
            np.testing.assert_allclose(row['target_response_real_imag'], [response, 0], atol=2e-15)
            self.assertAlmostEqual(row['normalized_noise_power'], noise)
            self.assertAlmostEqual(row['normalized_target_damage'], damage)
            self.assertAlmostEqual(row['normalized_total_error'], total)

    def test_selection_keeps_both_axes_and_physical_order(self):
        x = select_mvdr_channels(self.r, self.a, [2, 0])
        np.testing.assert_array_equal(x['selection_matrix'], [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(x['covariance'], [[2, 0], [0, 1]])
        np.testing.assert_allclose(x['weights'], [1/3, 2/3], atol=1e-15)
        values = np.array([[10, 11], [20, 21], [30, 31]])
        picked = select_channel_observations(values, [2, 0])
        np.testing.assert_array_equal(picked, [[30, 31], [10, 11]])
        picked[0, 0] = 99
        self.assertEqual(values[2, 0], 30)
        self.assertFalse(np.shares_memory(x['covariance'], self.r))

    def test_complex_phase_and_frequency_stack(self):
        r = np.array([[2, 1j], [-1j, 2]])
        a = np.array([1, 1j])
        model = select_mvdr_channels(np.stack([r, 10*r]), np.stack([a, 2j*a]), [1, 0])
        np.testing.assert_allclose(model['weights'][0], [.5j, .5], atol=1e-15)
        np.testing.assert_allclose(model['weights'][1], [-.25, .25j], atol=1e-15)
        # Two prescribed target amplitudes, distinct frequency weights.
        source = np.array([[2, 3], [4, 5]])
        data = np.stack([a[:, None]*source[0], (2j*a)[:, None]*source[1]], axis=1)
        output = apply_beamformer(select_channel_observations(data, [1, 0]), model['weights'])
        np.testing.assert_allclose(output, source, atol=2e-15)

    def test_full_identity_and_existing_solver_is_the_only_solver(self):
        with patch('codes.chapters.ch10.core.channel_selection.mvdr_weights', wraps=mvdr_weights) as solver:
            x = select_mvdr_channels(self.r, self.a, [0, 1, 2])
            self.assertEqual(solver.call_count, 1)
        np.testing.assert_array_equal(x['selection_matrix'], np.eye(3))
        np.testing.assert_array_equal(x['covariance'], self.r)
        np.testing.assert_allclose(x['weights'], [1, -.5, .5], atol=1e-15)

    def test_one_channel_and_full_rank_deficiency(self):
        # A zero noise variance in a discarded channel does not destroy the
        # selected scalar positive variance; no full-array inverse is needed.
        x = select_mvdr_channels(np.diag([0., 4.]), np.array([1, 2j]), [1])
        np.testing.assert_allclose(x['weights'], [.5j], atol=1e-15)
        self.assertAlmostEqual(np.vdot(x['weights'], x['steering']).real, 1)
        with self.assertRaises(np.linalg.LinAlgError):
            select_mvdr_channels(np.ones((2, 2)), np.ones(2), [0, 1])
        loaded = select_mvdr_channels(np.ones((2, 2)), np.ones(2), [0, 1], relative_diagonal_loading=.1)
        np.testing.assert_allclose(loaded['weights'], [.5, .5], atol=1e-15)

    def test_covariance_scale_does_not_change_response(self):
        for scale in (1e-200, 1., 1e200):
            x = select_mvdr_channels(scale*self.r, self.a, [1, 2])
            np.testing.assert_allclose(x['weights'], [.5, .5], rtol=0, atol=2e-15)
            np.testing.assert_allclose(x['covariance']/scale, [[2, 1], [1, 2]], atol=1e-15)

    def test_non_psd_omitted_channel_is_not_hidden(self):
        for r in (np.diag([-1., 1.]), np.array([[1, 2], [2, 1]])):
            with self.assertRaises(np.linalg.LinAlgError):
                select_mvdr_channels(r, np.ones(2), [1], relative_diagonal_loading=10.)
        with self.assertRaises(ValueError):
            select_mvdr_channels(np.array([[1, 1j], [1j, 1]]), np.ones(2), [1])

    def test_invalid_model_and_selection(self):
        for indices in ([], [0, 0], [-1], [3], [True, 1], [0., 1.], [[0]], '01', {0, 1}):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                select_mvdr_channels(self.r, self.a, indices)
        for r, a in ((np.eye(2), self.a), (np.ones((1, 2, 2)), np.ones(2)),
                     (np.eye(2)*np.nan, np.ones(2)), (np.eye(2), [1, np.inf]),
                     (np.eye(2), ['1', '1']), (np.zeros((0, 0)), [])):
            with self.subTest(r=r, a=a), self.assertRaises(ValueError):
                select_mvdr_channels(r, a, [0])
        for r, a in ((np.eye(2), np.zeros(2)), (np.zeros((2, 2)), np.ones(2))):
            with self.assertRaises(np.linalg.LinAlgError):
                select_mvdr_channels(r, a, [0])
        for values in (np.array(1), np.zeros((0, 2)), np.ones((2, 0)),
                       [[1, np.nan]], ['text'], [True, False]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                select_channel_observations(values, [0])


if __name__ == '__main__':
    unittest.main()
