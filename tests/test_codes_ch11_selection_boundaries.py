"""Explicit token/time and exact decision oracles, independent of helpers."""
from fractions import Fraction
import math
import unittest
import numpy as np
from codes.chapters.ch11.core.selection import (
    upper_limit_verdict, interval_dominates, zero_event_poisson_upper_bound,
    token_edit_distance, small_slot_word_errors, tiny_time_constrained_edit_distance,
)

class SelectionBoundariesTests(unittest.TestCase):
    def test_explicit_tokens_not_character_scoring(self):
        self.assertEqual(token_edit_distance(['北京'],['北海']),1)
        self.assertEqual(small_slot_word_errors([['北京']],[0],[['北海']])['cp_errors'],1)
        for args in [('a',[0],[['a']]),([['a']],[0],['ab']),([['a']],[0],b'a'),(None,[0],[['a']])]:
            with self.subTest(args=args),self.assertRaises(ValueError):small_slot_word_errors(*args)
        for left,right in [('abc',['abc']),(['a'],'a'),(['a'],[3]),(None,[])]:
            with self.assertRaises(ValueError):token_edit_distance(left,right)
        self.assertEqual(token_edit_distance([],['word']),1)
        self.assertEqual(small_slot_word_errors([['a']],[0],[[]])['orc_errors'],1)
        self.assertEqual(token_edit_distance(np.array(['a']),('a',)),0)

    def test_interval_shapes_and_exact_order(self):
        for intervals,limits in [(1,[2]),([1],[2]),(['12'],[2]),([[1,2]],2),([],[]),([[[1],2]],[3])]:
            with self.subTest(intervals=intervals),self.assertRaises(ValueError):upper_limit_verdict(intervals,limits)
        big=10**400
        self.assertEqual(upper_limit_verdict([[big,big+1]],[big]),'undetermined')
        self.assertTrue(interval_dominates([[80,90],[.10,.11]],[[90,100],[.12,.13]]))
        self.assertFalse(interval_dominates([[80,100],[.10,.12]],[[90,110],[.11,.13]]))
        self.assertFalse(interval_dominates([[1,1]],[[1,1]]))
        self.assertFalse(interval_dominates([None],[[1,2]]))
        self.assertTrue(interval_dominates([[big,big]],[[big+1,big+2]]))
        for a,b in [([None],[[2,1]]),([],[]),([[1,2]],[]),([True],[[1,2]])]:
            with self.assertRaises(ValueError):interval_dominates(a,b)

    def test_poisson_probability_and_units(self):
        rate=zero_event_poisson_upper_bound(2)
        self.assertAlmostEqual(rate,1.4978661367769954,14)
        self.assertAlmostEqual(math.exp(-2*rate),.05,15)
        self.assertAlmostEqual(zero_event_poisson_upper_bound(4),rate/2,15)
        self.assertGreater(zero_event_poisson_upper_bound(10**300),0)
        for t,c in [(0,.95),(-1,.95),(True,.95),(2,True),(2,1),(2,0),(math.inf,.95),(10**400,.95),(5e-324,.95),(1e308,5e-324)]:
            with self.subTest(t=t,c=c),self.assertRaises(ValueError):zero_event_poisson_upper_bound(t,c)

    def test_time_diagonal_requires_positive_overlap(self):
        f=tiny_time_constrained_edit_distance
        self.assertEqual(f(['a'],['a'],[[0,1]],[[1,2]]),2)
        self.assertEqual(f(['a'],['b'],[[0,1]],[[.5,1.5]]),1)
        self.assertEqual(f(['a'],['a'],[[0,1]],[[1,2]],collar=.01),0)
        self.assertEqual(f(['a','b'],['a','b'],[[0,1],[2,3]],[[0,1],[5,6]]),2)
        # Preserve a tiny positive overlap/collar despite a huge origin.
        big=10**400
        self.assertEqual(f(['a'],['a'],[[big,big+1]],[[big+1,big+2]],collar=5e-324),0)
        self.assertEqual(f([],['a'],[],[[0,1]]),1)
        self.assertEqual(f(['a'],['a'],[[-2,-1]],[[-2,-1]]),0)
        for args in [('a',['a'],[[0,1]],[[0,1]],0),(['a'],['a'],[[0,0]],[[0,1]],0),(['a'],['a'],[],[[0,1]],0),(['a'],['a'],[[0,1]],[[0,1]],True),(['a'],['a'],[[2,1]],[[0,1]],0),(['a'],['a'],[[0,1]],[[0,1]],-1)]:
            with self.subTest(args=args),self.assertRaises(ValueError):f(*args)

if __name__=='__main__':unittest.main()
