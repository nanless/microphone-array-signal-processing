"""Independent finite expectations for E30/E31 and CSS application gains."""
from fractions import Fraction as F
import unittest
import numpy as np
from codes.chapters.ch08.core.consistency_teaching import (
    run_experiment, mixture_projection, rectangular_consistency)
from codes.chapters.ch08.core.overiva_teaching import background_constraint, run_experiment as overiva_control
from codes.chapters.ch08.core.css import overlap_application_gain


class ConstraintTeachingTests(unittest.TestCase):
    def test_shared_samples_hand_calculation(self):
        # Independent sample-space calculation: two frame descriptions of the
        # same sample are averaged, with no STFT/FFT implementation in oracle.
        r = run_experiment()
        first = [F(3, 4)]*4
        second = [F(1)]*4
        expected_a = first[:2]+[(first[i+2]+second[i])/2 for i in range(2)]+second[2:]
        # P(raw) has shared samples 1/2 in both frames, then independent
        # mixture corrections use .75 in frame 0 and .25 in frame 1.
        expected_frames_b = [[F(3,4),F(3,4),F(7,8),F(7,8)],
                             [F(5,8),F(5,8),F(1),F(1)]]
        expected_b = expected_frames_b[0][:2]+[
            (expected_frames_b[0][i+2]+expected_frames_b[1][i])/2 for i in range(2)
        ]+expected_frames_b[1][2:]
        np.testing.assert_array_equal(r['mix_then_consistency']['waveforms'][0], list(map(float,expected_a)))
        np.testing.assert_array_equal(r['consistency_then_mix']['waveforms'][0], list(map(float,expected_b)))
        self.assertEqual(r['order_max_waveform_difference'], 1/8)
        self.assertEqual(r['consistency_then_mix']['consistency_max_spectrum_error'], 1/4)
        for order in ('mix_then_consistency','consistency_then_mix'):
            self.assertEqual(r[order]['sum_spectrum_max_error'], 0.)
        self.assertEqual(r['mix_then_consistency']['consistency_max_spectrum_error'], 0.)

    def test_equal_weights_and_zero_raw_are_distinct_controls(self):
        r = run_experiment()
        self.assertEqual(r['equal_weight_order_max_spectrum_difference'],0.)
        z = np.zeros((2,3,2))
        x, v = r['mixture_spectrum'], r['positive_variances']
        a = rectangular_consistency(mixture_projection(z,x,v))
        b = mixture_projection(rectangular_consistency(z),x,v)
        # Their frames differ but their synthesized shared sample averages
        # agree; this is why E30 needs the nonzero starting spectrum.
        fa, fb = np.fft.irfft(a,n=4,axis=1), np.fft.irfft(b,n=4,axis=1)
        for frames_a,frames_b in zip(fa.transpose(0,2,1),fb.transpose(0,2,1)):
            wa=np.r_[frames_a[0,:2],(frames_a[0,2:]+frames_a[1,:2])/2,frames_a[1,2:]]
            wb=np.r_[frames_b[0,:2],(frames_b[0,2:]+frames_b[1,:2])/2,frames_b[1,2:]]
            np.testing.assert_array_equal(wa,wb)

    def test_projection_input_contract(self):
        y=np.zeros((2,3,2)); x=np.ones((3,2)); v=np.ones_like(y)
        for bad in (np.zeros_like(v), -v, v.astype(bool),np.full_like(v,np.inf)):
            with self.assertRaises(ValueError): mixture_projection(y,x,bad)
        with self.assertRaises(ValueError): mixture_projection(y,x,np.ones((2,1,2)))
        y=y.astype(complex); y[0,0,0]=1j
        with self.assertRaises(ValueError): rectangular_consistency(y)

    def test_background_constraint_rational_control(self):
        r=background_constraint([[1,.5]],[[2,1],[1,2]])
        np.testing.assert_allclose(r['J'],[[float(F(4,5))]],atol=1e-15)
        np.testing.assert_allclose(r['target_second_moment'],[[float(F(7,2))]])
        np.testing.assert_allclose(r['background_second_moment'],[[float(F(42,25))]])
        np.testing.assert_allclose(r['target_background_cross_second_moment'],0,atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(r['extended_demixing']).real,-7/5)

    def test_pca_loses_weak_target_even_with_loading(self):
        r=overiva_control()
        np.testing.assert_array_equal(r['weak_target_after_projection'],[0,0])
        # Derive the retained space from eigenvectors rather than trusting the
        # example's stated rows, for both unloaded and loaded matrices.
        c=np.diag([9,4,1]); weak=np.array([0,0,1])
        for delta in (0,1,100):
            values,vectors=np.linalg.eigh(c+delta*np.eye(3))
            retained=vectors[:,np.argsort(values)[-2:]].T
            np.testing.assert_array_equal(retained@weak,[0,0])

    def test_complex_target_row_uses_conjugate_cross_moment(self):
        # A real-only fixture would not distinguish H from ordinary transpose.
        w=np.array([[1,1j]])
        c=np.array([[2,1j],[-1j,3]])
        r=background_constraint(w,c)
        np.testing.assert_allclose(r['J'],[[-4j/3]])
        u=r['background_rows']
        np.testing.assert_allclose(w@c@u.conj().T,0,atol=1e-15)
        self.assertGreater(abs((w@c@u.T)[0,0]),1)

    def test_background_coordinate_chart_rejects_singular_top_block(self):
        with self.assertRaises(np.linalg.LinAlgError): background_constraint([[0,1]],np.eye(2))
        with self.assertRaises(np.linalg.LinAlgError): background_constraint([[1,.5]],np.ones((2,2)))
        with self.assertRaises(ValueError): background_constraint([[1,1]],[[2,1j],[1j,2]])
        with self.assertRaises(ValueError): background_constraint(np.eye(2),np.eye(2))
        with self.assertRaises(ValueError): background_constraint([[2,1]],np.eye(2)*1e308)

    def test_css_gain_applies_to_current_without_centering(self):
        previous=np.array([[1,2,3],[2,4,6]])
        current=-2*previous
        np.testing.assert_array_equal(overlap_application_gain(previous,current),[-.5,-.5])
        # Different DC offset: original-sample LS is 20/29, unlike a centered
        # slope of 1. The helper must retain the objective in equation 8-32.
        np.testing.assert_allclose(overlap_application_gain(previous,previous+1),[20/29,68/83])

    def test_css_gain_numeric_boundary(self):
        p=np.array([[1,-1],[2,-2]],dtype=float)
        np.testing.assert_array_equal(overlap_application_gain(p*1e200,-2*p*1e200),[-.5,-.5])
        huge=np.full((2,2),1e308)
        np.testing.assert_allclose(overlap_application_gain(huge,huge*[1,.5]),[1.2,1.2])
        for current in (np.zeros_like(p),p*1e-10,p.astype(complex),np.full_like(p,np.nan)):
            with self.assertRaises(ValueError): overlap_application_gain(p,current)
        with self.assertRaises(ValueError): overlap_application_gain(p,p,minimum_rms=-1)
        np.testing.assert_array_equal(overlap_application_gain(np.zeros_like(p),p),[0,0])


if __name__=='__main__':
    unittest.main()
