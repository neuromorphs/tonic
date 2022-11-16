import numpy as np
import pytest
from utils import create_random_input

import tonic.transforms as transforms


def testTimeReversalSpatialJitter():
    orig_events, sensor_size = create_random_input()

    flip_probability = 1
    var_x = 3
    var_y = 3
    sigma_xy = 0
    transform = transforms.Compose(
        [
            transforms.RandomTimeReversal(p=flip_probability),
            transforms.SpatialJitter(
                sensor_size=sensor_size,
                var_x=var_x,
                var_y=var_y,
                sigma_xy=sigma_xy,
                clip_outliers=False,
            ),
        ]
    )
    events = transform(orig_events)

    assert "RandomTimeReversal" in str(transform)
    assert "SpatialJitter" in str(transform)

    assert len(events) == len(orig_events), "Number of events should be the same."
    spatial_var_x = np.isclose(events["x"].all(), orig_events["x"].all(), atol=var_x)
    assert spatial_var_x, "Spatial jitter should be within chosen variance x."
    assert (events["x"] != orig_events["x"]).any(), "X coordinates should be different."
    spatial_var_y = np.isclose(events["y"].all(), orig_events["y"].all(), atol=var_y)
    assert spatial_var_y, "Spatial jitter should be within chosen variance y."
    assert (events["y"] != orig_events["y"]).any(), "Y coordinates should be different."
    assert np.array_equal(orig_events["p"], np.invert(events["p"][::-1].astype(bool)))
    assert np.array_equal(
        orig_events["t"], np.max(orig_events["t"]) - events["t"][::-1]
    )
    assert events is not orig_events


def testDropoutFlipUD():
    orig_events, sensor_size = create_random_input()

    flip_probability = 1
    drop_probability = 0.5

    transform = transforms.Compose(
        [
            transforms.DropEvent(p=drop_probability),
            transforms.RandomFlipUD(sensor_size=sensor_size, p=flip_probability),
        ]
    )

    events = transform(orig_events)

    drop_events = np.isclose(
        events.shape[0], (1 - drop_probability) * orig_events.shape[0]
    )
    assert drop_events, (
        "Event dropout should result in drop_probability*len(original) events"
        " dropped out."
    )

    temporal_order = np.isclose(np.sum((events["t"] - np.sort(events["t"])) ** 2), 0)
    assert temporal_order, "Temporal order should be maintained."

    first_dropped_index = np.where(events["t"][0] == orig_events["t"])[0][0]
    flipped_events = (
        sensor_size[1] - 1 - orig_events["y"][first_dropped_index] == events["y"][0]
    )
    assert flipped_events, (
        "When flipping up and down y must map to the opposite pixel, i.e. y' ="
        " sensor width - y"
    )
    assert events is not orig_events


def testTimeSkewFlipPolarityFlipLR():
    orig_events, sensor_size = create_random_input()

    coefficient = 1.5
    offset = 0
    flip_probability_pol = 1
    flip_probability_lr = 1

    transform = transforms.Compose(
        [
            transforms.TimeSkew(coefficient=coefficient, offset=offset),
            transforms.RandomFlipPolarity(p=flip_probability_pol),
            transforms.RandomFlipLR(sensor_size=sensor_size, p=flip_probability_lr),
        ]
    )

    events = transform(orig_events)

    assert len(events) == len(orig_events)
    assert (events["t"] >= orig_events["t"]).all()
    assert np.min(events["t"]) >= 0

    assert (
        events["p"] == np.invert(orig_events["p"].astype(bool)).astype(int)
    ).all(), "Polarities should be flipped."

    same_pixel = np.isclose((sensor_size[0] - 1) - events["x"][0], orig_events["x"][0])
    assert same_pixel, (
        "When flipping left and right x must map to the opposite pixel, i.e. x' ="
        " sensor width - x"
    )
    assert events is not orig_events
