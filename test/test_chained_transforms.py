import pytest
import numpy as np
import tonic.transforms as transforms
import utils


class TestChainedTransforms:
    def testTimeReversalSpatialJitter(self):
        ordering = "xytp"
        (
            orig_events,
            original_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        flip_probability = 1
        variance_x = 1
        variance_y = 1
        sigma_x_y = 0
        transform = transforms.Compose(
            [
                transforms.RandomTimeReversal(ordering=ordering, flip_probability=flip_probability),
                transforms.SpatialJitter(
                    ordering=ordering, 
                    sensor_size=sensor_size,
                    variance_x=variance_x,
                    variance_y=variance_y,
                    sigma_x_y=sigma_x_y,
                    clip_outliers=False,
                ),
            ]
        )
        events = transform(
            events=orig_events.copy(),
        )

        assert len(events) == len(orig_events), "Number of events should be the same."
        spatial_var_x = np.isclose(
            events[:, 0].all(), orig_events[:, 0].all(), atol=variance_x
        )
        assert spatial_var_x, "Spatial jitter should be within chosen variance x."

        assert (
            events[:, 0] != orig_events[:, 0]
        ).all(), "X coordinates should be different."
        spatial_var_y = np.isclose(
            events[:, 1].all(), orig_events[:, 1].all(), atol=variance_y
        )
        assert spatial_var_y, "Spatial jitter should be within chosen variance y."
        assert (
            events[:, 1] != orig_events[:, 1]
        ).all(), "Y coordinates should be different."
        assert (
            events[:, 3] == orig_events[:, 3] * (-1)
        ).all(), "Polarities should be flipped."
        time_reversal = (
            events[:, 2] == np.max(orig_events[:, 2]) - orig_events[:, 2]
        ).all()
        assert (
            time_reversal
        ), "Condition of time reversal t_i' = max(t) - t_i has to be fullfilled"

    def testDropoutFlipUD(self):
        ordering = "xytp"
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        flip_probability = 1
        drop_probability = 0.5

        transform = transforms.Compose(
            [
                transforms.DropEvent(drop_probability=drop_probability),
                transforms.RandomFlipUD(ordering=ordering, sensor_size=sensor_size, flip_probability=flip_probability),
            ]
        )

        events = transform(
            events=orig_events.copy(),
        )

        drop_events = np.isclose(
            events.shape[0], (1 - drop_probability) * orig_events.shape[0]
        )
        assert drop_events, (
            "Event dropout should result in drop_probability*len(original) events"
            " dropped out."
        )

        temporal_order = np.isclose(
            np.sum((events[:, 2] - np.sort(events[:, 2])) ** 2), 0
        )
        assert temporal_order, "Temporal order should be maintained."

        first_dropped_index = np.where(events[0, 2] == orig_events[:, 2])[0][0]
        flipped_events = (
            sensor_size[1] - 1 - orig_events[first_dropped_index, 1] == events[0, 1]
        )
        assert flipped_events, (
            "When flipping up and down y must map to the opposite pixel, i.e. y' ="
            " sensor width - y"
        )

    def testTimeSkewFlipPolarityFlipLR(self):
        ordering = "xytp"
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        coefficient = 1.5
        offset = 0
        flip_probability_pol = 1
        flip_probability_lr = 1

        transform = transforms.Compose(
            [
                transforms.TimeSkew(ordering=ordering, coefficient=coefficient, offset=offset),
                transforms.RandomFlipPolarity(ordering=ordering, flip_probability=flip_probability_pol),
                transforms.RandomFlipLR(ordering=ordering, sensor_size=sensor_size, flip_probability=flip_probability_lr),
            ]
        )

        events = transform(
            events=orig_events.copy(),
        )

        assert len(events) == len(orig_events)
        assert (events[:, 2] >= orig_events[:, 2]).all()
        assert np.min(events[:, 2]) >= 0

        assert (
            events[:, 3] == orig_events[:, 3] * (-1)
        ).all(), "Polarities should be flipped."

        same_pixel = np.isclose((sensor_size[0] - 1) - events[0, 0], orig_events[0, 0])
        assert same_pixel, (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )
