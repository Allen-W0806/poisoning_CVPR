"""Falsifiable direction, color and padding tests for the task-specific reward."""
import unittest

import numpy as np
from PIL import Image

from src.mini_pilot.lighting_reward_v2 import analytic_exposure, lighting_terms


class LightingTermsTest(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (48, 32), (255, 255, 255))
        self.box = (8, 4, 40, 28)
        self.source.paste(Image.new("RGB", (32, 24), (150, 90, 60)), self.box[:2])

    def test_identity_zero_brighten_positive_darken_negative(self):
        noop = lighting_terms(self.source, self.source, self.box)
        brighter = lighting_terms(self.source, analytic_exposure(self.source, self.box, 1.03), self.box)
        darker = lighting_terms(self.source, analytic_exposure(self.source, self.box, .97), self.box)
        self.assertEqual(noop["brightness_progress"], 0.)
        self.assertEqual(noop["color_cost"], 0.)
        self.assertGreater(brighter["brightness_progress"], .7)
        self.assertLess(darker["brightness_progress"], 0.)
        self.assertLess(brighter["color_cost"], .3)

    def test_border_changes_cannot_earn_brightness_progress(self):
        modified = Image.new("RGB", self.source.size, "black")
        modified.paste(self.source.crop(self.box), self.box[:2])
        self.assertEqual(lighting_terms(self.source, modified, self.box),
                         lighting_terms(self.source, self.source, self.box))

    def test_channel_swap_penalized(self):
        rgb = np.asarray(self.source).copy()
        swapped = Image.fromarray(rgb[:, :, ::-1])
        self.assertGreater(lighting_terms(self.source, swapped, self.box)["color_cost"], 1.)


if __name__ == "__main__":
    unittest.main()
