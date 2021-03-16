import unittest
import numpy as np
import tonic.transforms as transforms
import utils


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.ordering = "xytp"
        self.random_xytp = utils.create_random_input_with_ordering(self.ordering)
        self.original_events = self.random_xytp[0].copy()
        self.original_images = self.random_xytp[1].copy()

    def testTimeJitter(self):
        events = self.random_xytp[0].copy()
        images = None
        variance = max(self.random_xytp[0][:, 2]) / 10
        transform = transforms.Compose(
            [transforms.TimeJitter(variance=variance, clip_negative=False)]
        )
        transform(
            events=events, sensor_size=self.random_xytp[2], ordering=self.ordering
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

    def testToTimesurfaces(self):
        events = self.random_xytp[0].copy()
        surf_dims = (7, 7)
        transform = transforms.Compose(
            [
                transforms.ToTimesurface(
                    surface_dimensions=surf_dims, tau=5e3, decay="lin"
                )
            ]
        )
        surfaces = transform(
            events=events, sensor_size=self.random_xytp[2], ordering=self.ordering
        )
        self.assertEqual(surfaces.shape[0], len(self.original_events))
        self.assertEqual(surfaces.shape[1], 2)
        self.assertEqual(surfaces.shape[2:], surf_dims)

    def testTimeReversalSpatialJitter(self):
        events = self.random_xytp[0].copy()
        images = self.random_xytp[1].copy()
        flip_probability = 1
        multi_image = self.random_xytp[3]
        variance_x = 1
        variance_y = 1
        sigma_x_y = 0
        transform = transforms.Compose(
            [
                transforms.TimeReversal(flip_probability=flip_probability),
                transforms.SpatialJitter(
                    variance_x=variance_x,
                    variance_y=variance_y,
                    sigma_x_y=sigma_x_y,
                    clip_outliers=False,
                ),
            ]
        )
        events, images = transform(
            events=events,
            images=images,
            sensor_size=self.random_xytp[2],
            ordering=self.ordering,
            multi_image=multi_image,
        )

        self.assertTrue(
            len(events) == len(self.original_events),
            "Number of events should be the same.",
        )
        spatial_var_x = np.isclose(
            events[:, 0].all(), self.original_events[:, 0].all(), atol=variance_x
        )
        self.assertTrue(
            spatial_var_x, "Spatial jitter should be within chosen variance x."
        )
        self.assertFalse(
            (events[:, 0] == self.original_events[:, 0]).all(),
            "X coordinates should be different.",
        )
        spatial_var_y = np.isclose(
            events[:, 1].all(), self.original_events[:, 1].all(), atol=variance_y
        )
        self.assertTrue(
            spatial_var_y, "Spatial jitter should be within chosen variance y."
        )
        self.assertFalse(
            (events[:, 1] == self.original_events[:, 1]).all(),
            "Y coordinates should be different.",
        )
        self.assertTrue(
            (events[:, 3] == self.original_events[:, 3] * (-1)).all(),
            "Polarities should be flipped.",
        )
        time_reversal = (
            events[:, 2]
            == np.max(self.original_events[:, 2]) - self.original_events[:, 2]
        ).all()
        self.assertTrue(
            time_reversal,
            "Condition of time reversal t_i' = max(t) - t_i has to be fullfilled",
        )
        self.assertTrue(
            (images[::-1] == self.original_images).all(),
            "Images should be in reversed order.",
        )

    def testDropoutFlipUD(self):
        events = self.random_xytp[0].copy()
        images = self.random_xytp[1].copy()
        multi_image = self.random_xytp[3]
        flip_probability = 1
        drop_probability = 0.5

        transform = transforms.Compose(
            [
                transforms.DropEvents(drop_probability=drop_probability),
                transforms.FlipUD(flip_probability=flip_probability),
            ]
        )

        events, images = transform(
            events=events,
            images=images,
            sensor_size=self.random_xytp[2],
            ordering=self.ordering,
            multi_image=multi_image,
        )

        drop_events = np.isclose(
            events.shape[0], (1 - drop_probability) * self.original_events.shape[0]
        )
        self.assertTrue(
            drop_events,
            "Event dropout should result in drop_probability*len(original) events"
            " dropped out.",
        )

        temporal_order = np.isclose(
            np.sum((events[:, 2] - np.sort(events[:, 2])) ** 2), 0
        )
        self.assertTrue(temporal_order, "Temporal order should be maintained.")

        first_dropped_index = np.where(events[0, 2] == self.original_events[:, 2])[0][0]
        flipped_events = (
            self.random_xytp[2][1] - 1 - self.original_events[first_dropped_index, 1]
            == events[0, 1]
        )
        self.assertTrue(
            flipped_events,
            "When flipping up and down y must map to the opposite pixel, i.e. y' ="
            " sensor width - y",
        )

    def testTimeSkewFlipPolarityFlipLR(self):
        events = self.random_xytp[0].copy()
        images = self.random_xytp[1].copy()
        multi_image = self.random_xytp[3]
        coefficient = 1.5
        offset = 0
        flip_probability_pol = 1
        flip_probability_lr = 1

        transform = transforms.Compose(
            [
                transforms.TimeSkew(coefficient=coefficient, offset=offset),
                transforms.FlipPolarity(flip_probability=flip_probability_pol),
                transforms.FlipLR(flip_probability=flip_probability_lr),
            ]
        )

        events, images = transform(
            events=events,
            images=images,
            sensor_size=self.random_xytp[2],
            ordering=self.ordering,
            multi_image=multi_image,
        )

        self.assertTrue(len(events) == len(self.original_events))
        self.assertTrue((events[:, 2] >= self.original_events[:, 2]).all())
        self.assertTrue(np.min(events[:, 2]) >= 0)

        self.assertTrue(
            (events[:, 3] == self.original_events[:, 3] * (-1)).all(),
            "Polarities should be flipped.",
        )

        same_pixel = np.isclose(
            (self.random_xytp[2][0] - 1) - events[0, 0], self.original_events[0, 0]
        )
        self.assertTrue(
            same_pixel,
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x",
        )
