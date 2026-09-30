"""Independent population matrices for the two Chapter 4 focusing figures."""
import math
import unittest
import numpy as np
from scripts.make_figures import focus_noise_example, coherent_frequency_example

class FocusFigureTests(unittest.TestCase):
    def test_noise_axis_amplification_is_not_signal_gain(self):
        total, noise, powers, noise_powers, whitened = focus_noise_example()
        np.testing.assert_array_equal(total-noise, np.ones((2, 2)))
        np.testing.assert_array_equal(total, [[4,-1],[-1,4]])
        np.testing.assert_array_equal(noise, [[3,-2],[-2,3]])
        np.testing.assert_allclose(powers, [3,5], atol=1e-14)
        np.testing.assert_allclose(noise_powers, [1,5], atol=1e-14)
        np.testing.assert_allclose(whitened, [3,1], atol=1e-14)
        # Target [1,1] has unchanged signal energy 2. Orthogonal noise alone wins.
        self.assertAlmostEqual(powers[0]-noise_powers[0], 2)
        self.assertAlmostEqual(powers[1]-noise_powers[1], 0)

    def test_two_frequency_rank_comes_from_changed_source_ratio(self):
        spectra = coherent_frequency_example()
        for values in spectra[:3]:
            np.testing.assert_allclose(values,[0,0,0,8],atol=4e-15,rtol=0)
        np.testing.assert_allclose(spectra[3],[0,0,4,4],atol=4e-15,rtol=0)
        # Hand matrices: unpooled low phasor is (1-j)*[1,1,-1,-1].
        v=np.array([1,1,-1,-1]); w=np.array([1,-1,-1,1])
        raw=2*np.outer(v,v); pooled=np.outer(v,v)+np.outer(w,w)
        np.testing.assert_allclose(np.linalg.eigvalsh(raw),spectra[2],atol=4e-15,rtol=0)
        np.testing.assert_allclose(np.linalg.eigvalsh(pooled),spectra[3],atol=4e-15,rtol=0)
        self.assertEqual(int(v@w),0)

if __name__=='__main__':unittest.main()
