import unittest
import numpy as np
import spike_data_augmentation.transforms as transforms
import utils


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.random_xytp = utils.create_random_input_xytp()
        self.original_events = self.random_xytp[0].copy()
        self.original_images = self.random_xytp[1].copy()

    def testTimeJitter(self):
        events = self.random_xytp[0].copy()
        images = None
        variance = 0.1
        transform = transforms.Compose([transforms.TimeJitter(variance=variance)])
        transform(
            events=events, sensor_size=self.random_xytp[2], ordering=self.random_xytp[3]
        )

        self.assertTrue(len(events) == len(self.original_events))
        self.assertTrue((events[:, 0] == self.original_events[:, 0]).all())
        self.assertTrue((events[:, 1] == self.original_events[:, 1]).all())
        self.assertFalse((events[:, 2] == self.original_events[:, 2]).all())
        self.assertTrue((events[:, 3] == self.original_events[:, 3]).all())
        self.assertTrue(
            np.isclose(
                events[:, 2].all(), self.original_events[:, 2].all(), atol=variance
            )
        )

    def testTimeReversalSpatialJitter(self):
        events = self.random_xytp[0].copy()
        images = self.random_xytp[1].copy()
        flip_probability = 1
        multi_image = self.random_xytp[4]
        variance_x = 1
        variance_y = 1
        sigma_x_y = 0
        transform = transforms.Compose(
            [
                transforms.TimeReversal(flip_probability=flip_probability),
                transforms.SpatialJitter(
                    variance_x=variance_x, variance_y=variance_y, sigma_x_y=sigma_x_y
                ),
            ]
        )
        events, images = transform(
            events=events,
            images=images,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=multi_image,
        )

        self.assertTrue(len(events) == len(self.original_events))
        self.assertTrue(
            np.isclose(
                events[:, 0].all(), self.original_events[:, 0].all(), atol=variance_x
            )
        )
        self.assertFalse((events[:, 0] == self.original_events[:, 0]).all())
        self.assertTrue(
            np.isclose(
                events[:, 1].all(), self.original_events[:, 1].all(), atol=variance_y
            )
        )
        self.assertFalse((events[:, 1] == self.original_events[:, 1]).all())
        self.assertTrue((events[:, 3] == self.original_events[:, 3] * (-1)).all())
        self.assertTrue(
            (
                events[:, 2]
                == np.max(self.original_events[:, 2]) - self.original_events[:, 2]
            ).all()
        )
        self.assertTrue((images[::-1] == self.original_images).all())
