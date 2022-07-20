import pytest
import numpy as np
import tonic.transforms as transforms
import itertools
from utils import create_random_input


@pytest.mark.parametrize("min, max", itertools.product((0, 1000), (None, 5000)))
def test_crop_time(min, max):
    orig_events, sensor_size = create_random_input()

    transform = transforms.CropTime(min=min, max=max)

    events = transform(orig_events)

    assert events is not orig_events
    if min is not None:
        assert not events["t"][0] < min
    if max is not None:
        assert not events["t"][-1] > max


@pytest.mark.parametrize("filter_time", [10000, 5000])
def test_transform_denoise(filter_time):
    orig_events, sensor_size = create_random_input()

    transform = transforms.Denoise(filter_time=filter_time)

    events = transform(orig_events)

    assert len(events) > 0, "Not all events should be filtered"
    assert len(events) < len(
        orig_events
    ), "Result should be fewer events than original event stream"
    assert np.isin(events, orig_events).all(), (
        "Denoising should not add additional events that were not present in"
        " original event stream"
    )
    assert events is not orig_events


@pytest.mark.parametrize(
    "p",
    [
        0.2,
        0.5,
    ],
)
def test_transform_drop_events(p):
    orig_events, sensor_size = create_random_input()

    transform = transforms.DropEvent(p=p)

    events = transform(orig_events)

    assert np.isclose(events.shape[0], (1 - p) * orig_events.shape[0]), (
        "Event dropout should result in p*len(original) events" " dropped out."
    )
    assert np.isclose(
        np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
    ), "Event dropout should maintain temporal order."
    assert events is not orig_events


@pytest.mark.parametrize(
    "duration_ratio",
    [(0.1), (0.2), (0.3), (0.4), (0.5), (0.6), (0.7), (0.8), (0.9)],
)
def test_transform_drop_events_by_time(duration_ratio):
    orig_events, sensor_size = create_random_input()

    transform = transforms.DropEventByTime(duration_ratio=duration_ratio)

    events = transform(orig_events)

    assert len(events) < len(orig_events)

    t_start = orig_events["t"].min()
    t_end = orig_events["t"].max()

    # checks that there is no events during a period of the defined duration ratio
    duration = (t_end - t_start) * duration_ratio

    diffs = np.diff(events["t"])

    assert np.any(
        diffs >= duration
    ), f"There should be no events during {duration} in the obtained sequence."


@pytest.mark.parametrize(
    "area_ratio",
    [(0.1), (0.2), (0.3), (0.4), (0.5), (0.6), (0.7), (0.8), (0.9)],
)
def test_transform_drop_events_by_area(area_ratio):
    orig_events, sensor_size = create_random_input()

    transform = transforms.DropEventByArea(sensor_size, area_ratio)

    events = transform(orig_events)

    assert len(events) < len(orig_events)  # events were dropped by the transform

    # checks that an area of the right dimension contains no events in the resulting sequence
    cut_w = int(area_ratio * sensor_size[0])
    cut_h = int(area_ratio * sensor_size[1])

    to_im = transforms.ToImage(sensor_size)
    frame = to_im(events)
    orig_frame = to_im(orig_events)
    cmp = frame - orig_frame
    dropped_events = len(orig_events) - len(events)

    # goal: find the area that contains the same number of dropped events
    dropped_area_found = False
    for bbx1 in range(0, (sensor_size[0] - cut_w)):
        bbx2 = bbx1 + cut_w
        for bby1 in range(0, (sensor_size[1] - cut_h)):
            bby2 = bby1 + cut_h

            if abs(np.sum(cmp[:, bby1:bby2, bbx1:bbx2])) == dropped_events:
                dropped_area_found = True
                break

    assert (
        dropped_area_found is True
    ), f"There should be an area with {dropped_events} events dropped in the obtained sequence."


def test_transform_decimation():
    n = 10

    orig_events, sensor_size = create_random_input(sensor_size=(1, 1, 2), n_events=1000)
    transform = transforms.Decimation(n=n)

    events = transform(orig_events)

    assert len(events) == 100


@pytest.mark.parametrize(
    "coordinates, hot_pixel_frequency",
    [(((9, 11), (10, 12), (11, 13)), None), (None, 10000)],
)
def test_transform_drop_pixel(coordinates, hot_pixel_frequency):
    orig_events, sensor_size = create_random_input(sensor_size=(20, 20, 2))
    orig_events = np.concatenate((orig_events, np.ones(10000, dtype=orig_events.dtype)))
    orig_events = orig_events[np.argsort(orig_events["t"])]

    transform = transforms.DropPixel(
        coordinates=coordinates, hot_pixel_frequency=hot_pixel_frequency
    )

    events = transform(orig_events)

    assert len(events) < len(orig_events)

    if coordinates:
        for x, y in coordinates:
            assert not np.logical_and(events["x"] == x, events["y"] == y).sum()

    if hot_pixel_frequency:
        assert not np.logical_and(events["x"] == 1, events["y"] == 1).sum()
    assert events is not orig_events


@pytest.mark.parametrize(
    "coordinates, hot_pixel_frequency",
    [(((9, 11), (10, 12), (11, 13)), None), (None, 5000)],
)
def test_transform_drop_pixel_raster(coordinates, hot_pixel_frequency):
    raster_test = np.random.randint(0, 100, (50, 2, 100, 200))
    frame_test = np.random.randint(0, 100, (2, 100, 200))
    transform = transforms.DropPixel(
        coordinates=coordinates, hot_pixel_frequency=hot_pixel_frequency
    )
    raster = transform(raster_test)
    frame = transform(frame_test)

    if coordinates:
        for x, y in coordinates:
            assert raster[:, :, x, y].sum() == 0
            assert frame[:, x, y].sum() == 0
    if hot_pixel_frequency:
        merged_polarity_raster = raster.sum(0).sum(0)
        merged_polarity_frame = frame.sum(0)
        assert not merged_polarity_frame[merged_polarity_frame > 5000].sum().sum()
        assert not merged_polarity_raster[merged_polarity_raster > 5000].sum().sum()


@pytest.mark.parametrize("time_factor, spatial_factor", [(1, 0.25), (1e-3, 1)])
def test_transform_downsample(time_factor, spatial_factor):
    orig_events, sensor_size = create_random_input()

    transform = transforms.Downsample(
        time_factor=time_factor, spatial_factor=spatial_factor
    )

    events = transform(orig_events)

    assert np.array_equal(
        (orig_events["t"] * time_factor).astype(orig_events["t"].dtype), events["t"]
    )
    assert np.array_equal(np.floor(orig_events["x"] * spatial_factor), events["x"])
    assert np.array_equal(np.floor(orig_events["y"] * spatial_factor), events["y"])
    assert events is not orig_events


@pytest.mark.parametrize("target_size", [(50, 50), (10, 5)])
def test_transform_random_crop(target_size):
    orig_events, sensor_size = create_random_input()

    transform = transforms.RandomCrop(sensor_size=sensor_size, target_size=target_size)
    events = transform(orig_events)

    assert np.all(events["x"]) < target_size[0] and np.all(
        events["y"] < target_size[1]
    ), "Cropping needs to map the events into the new space."
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_lr(p):
    orig_events, sensor_size = create_random_input()

    transform = transforms.RandomFlipLR(sensor_size=sensor_size, p=p)

    events = transform(orig_events)

    if p == 1:
        assert ((sensor_size[0] - 1) - orig_events["x"] == events["x"]).all(), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )
    else:
        assert np.array_equal(orig_events, events)
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_polarity(p):
    orig_events, sensor_size = create_random_input()

    transform = transforms.RandomFlipPolarity(p=p)

    events = transform(orig_events)

    if p == 1:
        assert np.array_equal(np.invert(orig_events["p"].astype(bool)), events["p"]), (
            "When flipping polarity with probability 1, all event polarities must"
            " flip"
        )
    else:
        assert np.array_equal(orig_events["p"], events["p"]), (
            "When flipping polarity with probability 0, no event polarities must"
            " flip"
        )
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_polarity_bools(p):
    orig_events, sensor_size = create_random_input(
        dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", bool)])
    )

    transform = transforms.RandomFlipPolarity(p=p)

    events = transform(orig_events)

    if p == 1:
        assert np.array_equal(np.invert(orig_events["p"].astype(bool)), events["p"]), (
            "When flipping polarity with probability 1, all event polarities must"
            " flip"
        )
    else:
        assert np.array_equal(orig_events["p"], events["p"]), (
            "When flipping polarity with probability 0, no event polarities must"
            " flip"
        )
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_ud(p):
    orig_events, sensor_size = create_random_input()

    transform = transforms.RandomFlipUD(sensor_size=sensor_size, p=p)

    events = transform(orig_events)

    if p == 1:
        assert np.array_equal((sensor_size[1] - 1) - orig_events["y"], events["y"]), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )
    else:
        assert np.array_equal(orig_events, events)
    assert events is not orig_events


def test_transform_merge_polarities():
    orig_events, sensor_size = create_random_input()
    transform = transforms.MergePolarities()
    events = transform(orig_events)
    assert len(np.unique(orig_events["p"])) == 2
    assert len(np.unique(events["p"])) == 1
    assert events is not orig_events


def test_transform_numpy_array():
    orig_events, sensor_size = create_random_input()
    transform = transforms.NumpyAsType(int)
    events = transform(orig_events)
    assert events.dtype == int
    assert events is not orig_events


def test_transform_numpy_array_unstructured():
    orig_events, sensor_size = create_random_input()
    transform = transforms.NumpyAsType(int)
    events = transform(orig_events)
    assert events.dtype == int
    assert events is not orig_events


@pytest.mark.parametrize("delta", [10000, 5000])
def test_transform_refractory_period(delta):
    orig_events, sensor_size = create_random_input()

    transform = transforms.RefractoryPeriod(delta=delta)

    events = transform(orig_events)

    assert len(events) > 0, "Not all events should be filtered"
    assert len(events) < len(
        orig_events
    ), "Result should be fewer events than original event stream"
    assert np.isin(
        events, orig_events
    ).all(), "Added additional events that were not present in original event stream"
    assert events.dtype == events.dtype
    assert events is not orig_events


@pytest.mark.parametrize(
    "variance, clip_outliers", [(30, False), (100, True), (3.5, True), (0.8, False)]
)
def test_transform_spatial_jitter(variance, clip_outliers):
    orig_events, sensor_size = create_random_input()

    transform = transforms.SpatialJitter(
        sensor_size=sensor_size,
        var_x=variance,
        var_y=variance,
        sigma_xy=0,
        clip_outliers=clip_outliers,
    )

    events = transform(orig_events)

    if not clip_outliers:
        assert len(events) == len(orig_events)
        assert (events["t"] == orig_events["t"]).all()
        assert (events["p"] == orig_events["p"]).all()
        assert (events["x"] != orig_events["x"]).any()
        assert (events["y"] != orig_events["y"]).any()
        assert np.isclose(events["x"].all(), orig_events["x"].all(), atol=2 * variance)
        assert np.isclose(events["y"].all(), orig_events["y"].all(), atol=2 * variance)

        assert (
            events["x"] - orig_events["x"]
            == (events["x"] - orig_events["x"]).astype(int)
        ).all()

        assert (
            events["y"] - orig_events["y"]
            == (events["y"] - orig_events["y"]).astype(int)
        ).all()

    else:
        assert len(events) < len(orig_events)
    assert events is not orig_events


@pytest.mark.parametrize(
    "std, clip_negative, sort_timestamps",
    [(10, True, True), (50, False, False), (0, True, False)],
)
def test_transform_time_jitter(std, clip_negative, sort_timestamps):
    orig_events, sensor_size = create_random_input()

    transform = transforms.TimeJitter(
        std=std, clip_negative=clip_negative, sort_timestamps=sort_timestamps
    )

    events = transform(orig_events)

    if clip_negative:
        assert (events["t"] >= 0).all()
    else:
        assert len(events) == len(orig_events)
    if sort_timestamps:
        np.testing.assert_array_equal(events["t"], np.sort(events["t"]))
    if not sort_timestamps and not clip_negative:
        np.testing.assert_array_equal(events["x"], orig_events["x"])
        np.testing.assert_array_equal(events["y"], orig_events["y"])
        np.testing.assert_array_equal(events["p"], orig_events["p"])
        assert (
            events["t"] - orig_events["t"]
            == (events["t"] - orig_events["t"]).astype(int)
        ).all()
    assert events is not orig_events


@pytest.mark.parametrize("p", [1, 0])
def test_transform_time_reversal(p):
    orig_events, sensor_size = create_random_input()

    original_t = orig_events["t"][0]
    max_t = np.max(orig_events["t"])

    transform = transforms.RandomTimeReversal(p=p)
    events = transform(orig_events)

    if p == 1:
        assert np.array_equal(orig_events["t"], max_t - events["t"][::-1])
        assert np.array_equal(
            orig_events["p"],
            np.invert(events["p"][::-1].astype(bool)),
        )
    elif p == 0:
        assert np.array_equal(orig_events, events)
    assert events is not orig_events


@pytest.mark.parametrize("coefficient, offset", [(3.1, 100), (0.3, 0), (2.7, 10)])
def test_transform_time_skew(coefficient, offset):
    orig_events, sensor_size = create_random_input()

    transform = transforms.TimeSkew(coefficient=coefficient, offset=offset)

    events = transform(orig_events)

    assert len(events) == len(orig_events)
    assert np.min(events["t"]) >= offset
    assert (events["t"] == (events["t"]).astype(int)).all()
    assert all((orig_events["t"] * coefficient + offset).astype(int) == events["t"])
    assert events is not orig_events


@pytest.mark.parametrize("n", [100, 0])
def test_transform_uniform_noise(n):
    orig_events, sensor_size = create_random_input()

    transform = transforms.UniformNoise(sensor_size=sensor_size, n=n)

    events = transform(orig_events)

    assert len(events) == len(orig_events) + n
    assert np.isin(orig_events, events).all()
    assert np.isclose(
        np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
    ), "Event noise should maintain temporal order."
    assert events is not orig_events


def test_transform_time_alignment():
    orig_events, sensor_size = create_random_input()

    transform = transforms.TimeAlignment()

    events = transform(orig_events)

    assert np.min(events["t"]) == 0
    assert events is not orig_events
