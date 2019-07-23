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

    def testEventDropout(self):
        original = self.random_xytp[0]
        drop_probability = 0.2

        events = F.drop_event_numpy(original, drop_probability=drop_probability)

        self.assertTrue(
            np.isclose(events.shape[0], (1 - drop_probability) * original.shape[0]),
            "Event dropout should result in drop_probability*len(original) events dropped out.",
        )

        self.assertTrue(
            np.isclose(np.sum((events[:, 2] - np.sort(events[:, 2])) ** 2), 0),
            "Event dropout should maintain temporal order.",
        )

        events = F.drop_event_numpy(
            original, drop_probability=drop_probability, random_drop_probability=True
        )

        self.assertTrue(
            events.shape[0] >= (1 - drop_probability) * original.shape[0],
            "Event dropout with random drop probability should result in less than drop_probability*len(original) events dropped out.",
        )
