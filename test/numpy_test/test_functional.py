import unittest
import numpy as np

import spike_data_augmentation.functional as F

import utils


class TestFunctionalAPI(unittest.TestCase):
    def setUp(self):
        self.random_xytp = utils.create_random_input_xytp()

    def testFlipLR(self):
        original_x = self.random_xytp[0][0, 0].copy()

        events, images = F.flip_lr_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            flip_probability=1.0,
        )

        same_pixel = np.isclose(
            self.random_xytp[2][0] - self.random_xytp[0][0, 0], original_x
        )

        self.assertTrue(
            same_pixel,
            "When flipping left and right x must map to the opposite pixel, i.e. x' = sensor width - x",
        )

    def testFlipPolarity(self):
        original_polarities = self.random_xytp[0][:, 3].copy()

        events = F.flip_polarity_numpy(
            self.random_xytp[0], flip_probability=1, ordering=self.random_xytp[3]
        )

        self.assertTrue(
            np.array_equal(original_polarities * -1, events[:, 3]),
            "When flipping polarity with probability 1, all event polarities must flip",
        )

        self.random_xytp[0][:, 3] = original_polarities.copy()

        events = F.flip_polarity_numpy(
            self.random_xytp[0], flip_probability=0, ordering=self.random_xytp[3]
        )

        self.assertTrue(
            np.array_equal(original_polarities, events[:, 3]),
            "When flipping polarity with probability 0, no event polarities must flip",
        )
