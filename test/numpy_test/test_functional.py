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

    def testFlipUD(self):
        original_y = self.random_xytp[0][0, 1].copy()

        events, images = F.flip_ud_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            flip_probability=1.0,
        )

        print(self.random_xytp[2], self.random_xytp[0][0, 1])
        print(events[0, 0])

        same_pixel = np.isclose(
            self.random_xytp[2][1] - self.random_xytp[0][0, 1], original_y
        )

        self.assertTrue(
            same_pixel,
            "When flipping up and down y must map to the opposite pixel, i.e. y' = sensor width - y",
        )

    def testSpatialJitter(self):
        original_events = self.random_xytp[0].copy()

        events = F.spatial_jitter_numpy(
            self.random_xytp[0],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            variance_x=2,
            variance_y=2,
            sigma_x_y=0,
        )

        self.assertTrue(len(events) == len(original_events))
        self.assertTrue((events[:, 2] == original_events[:, 2]).all())
        self.assertTrue((events[:, 3] == original_events[:, 3]).all())
        self.assertFalse((events[:, 0] == original_events[:, 0]).all())
        self.assertFalse((events[:, 1] == original_events[:, 1]).all())
