"""Actual lighting direction must remain distinguishable after imperfect editing."""
import unittest
import numpy as np
from PIL import Image
from src.mini_pilot.lighting_reward_v2 import analytic_exposure
from src.mini_pilot.lighting_reward_v3 import mean_lighting_terms


class MeanLightingTest(unittest.TestCase):
    def setUp(self):
        self.source=Image.fromarray(np.tile(np.array([140,90,60],dtype=np.uint8),(32,40,1)))
        self.box=(0,0,40,32)

    def test_identity_zero_target_positive_darken_negative(self):
        a=mean_lighting_terms(self.source,self.source,self.box)
        b=mean_lighting_terms(self.source,analytic_exposure(self.source,self.box,1.03),self.box)
        c=mean_lighting_terms(self.source,analytic_exposure(self.source,self.box,.97),self.box)
        self.assertEqual(a["brightness_progress"],0.)
        self.assertGreater(b["brightness_progress"],.3)
        self.assertLess(c["brightness_progress"],0.)

    def test_wrong_exposures_retain_a_directional_ranking(self):
        darker=mean_lighting_terms(self.source,analytic_exposure(self.source,self.box,.8),self.box)
        less_dark=mean_lighting_terms(self.source,analytic_exposure(self.source,self.box,.9),self.box)
        self.assertGreater(less_dark["brightness_progress"],darker["brightness_progress"])
        self.assertGreater(darker["brightness_progress"],-1.)

    def test_spatial_rearrangement_cannot_change_the_mean_property(self):
        array=np.asarray(self.source).copy();array[:16]//=2
        source=Image.fromarray(array);permuted=Image.fromarray(array[::-1].copy())
        self.assertAlmostEqual(mean_lighting_terms(source,permuted,self.box)["brightness_progress"],0.)
        # This intentionally leaves structure validation to separate metrics.


if __name__=="__main__":unittest.main()
