"""Independent publication controls for broadcast-coordinate statistics."""
import copy
import unittest
from unittest.mock import patch

from scripts import quality_check as quality
from codes.chapters.ch15.core.distributed import broadcast_statistics_control


class DistributedStatisticsPublicationTests(unittest.TestCase):
    def test_fixed_physical_cost_and_extreme_residual_controls(self):
        errors = []
        quality.check_distributed_numerical_controls(errors)
        self.assertEqual(errors, [])

    def test_old_cross_vector_cannot_be_reused_with_new_scm(self):
        result = broadcast_statistics_control()
        result['converted_mixture_statistics']['cross_covariance'] = [1., 1.]
        with self.assertRaisesRegex(ValueError, 'cross'):
            quality._verify_distributed_statistics_control(result)

    def test_solved_mixed_coordinates_are_not_the_current_operator(self):
        result = broadcast_statistics_control()
        case = result['cases']['unconverted_mixture']
        case['effective_weights'] = case['receiver_weights'].copy()
        with self.assertRaisesRegex(ValueError, 'effective weights'):
            quality._verify_distributed_statistics_control(result)

    def test_zero_or_optimal_cost_cannot_replace_observed_bad_current_cost(self):
        for fabricated in (0., 1/3, True):
            result = copy.deepcopy(broadcast_statistics_control())
            result['cases']['unconverted_mixture']['physical_components']['total_mse'] = fabricated
            with self.subTest(cost=fabricated), self.assertRaisesRegex(ValueError, 'total_mse'):
                quality._verify_distributed_statistics_control(result)

    def test_false_online_estimator_scope_is_a_publication_error(self):
        result = broadcast_statistics_control()
        result['scope'] = 'executed blind TI-DANSE network'
        with patch('codes.chapters.ch15.core.distributed.broadcast_statistics_control', return_value=result):
            errors = []
            quality.check_distributed_numerical_controls(errors)
        self.assertEqual(len(errors), 1)
        self.assertIn('scope', errors[0])


if __name__ == '__main__':
    unittest.main()
