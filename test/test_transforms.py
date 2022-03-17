import pytest
import numpy as np
import tonic.transforms as transforms
from utils import create_random_input


class TestTransforms:
    @pytest.mark.parametrize("filter_time", [10000, 5000])
    def test_transform_denoise(self, filter_time):
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

    @pytest.mark.parametrize("p, random_p", [(0.2, False), (0.5, True)])
    def test_transform_drop_events(self, p, random_p):
        orig_events, sensor_size = create_random_input()

        transform = transforms.DropEvent(p=p, random_p=random_p)

        events = transform(orig_events)

        if random_p:
            assert events.shape[0] >= (1 - p) * orig_events.shape[0], (
                "Event dropout with random drop probability should result in less than "
                " p*len(original) events dropped out."
            )
        else:
            assert np.isclose(events.shape[0], (1 - p) * orig_events.shape[0]), (
                "Event dropout should result in p*len(original) events" " dropped out."
            )
        assert np.isclose(
            np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
        ), "Event dropout should maintain temporal order."
        assert events is not orig_events

    @pytest.mark.parametrize(
        "coordinates, hot_pixel_frequency",
        [(((9, 11), (10, 12), (11, 13)), None), (None, 10000)],
    )
    def test_transform_drop_pixel(self, coordinates, hot_pixel_frequency):
        orig_events, sensor_size = create_random_input(sensor_size=(20, 20, 2))
        orig_events = np.concatenate(
            (orig_events, np.ones(10000, dtype=orig_events.dtype))
        )
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
    def test_transform_drop_pixel_raster(self, coordinates, hot_pixel_frequency):
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
    def test_transform_downsample(self, time_factor, spatial_factor):
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
    def test_transform_random_crop(self, target_size):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RandomCrop(
            sensor_size=sensor_size, target_size=target_size
        )
        events = transform(orig_events)

        assert np.all(events["x"]) < target_size[0] and np.all(
            events["y"] < target_size[1]
        ), "Cropping needs to map the events into the new space."
        assert events is not orig_events

    @pytest.mark.parametrize("p", [1.0, 1.0])
    def test_transform_flip_lr(self, p):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RandomFlipLR(sensor_size=sensor_size, p=p)

        events = transform(orig_events)

        assert ((sensor_size[0] - 1) - orig_events["x"] == events["x"]).all(), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )
        assert events is not orig_events

    @pytest.mark.parametrize("p", [1.0, 0])
    def test_transform_flip_polarity(self, p):
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
    def test_transform_flip_polarity_bools(self, p):
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

    @pytest.mark.parametrize("p", [1.0, 1.0])
    def test_transform_flip_ud(self, p):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RandomFlipUD(sensor_size=sensor_size, p=p)

        events = transform(orig_events)

        assert np.array_equal((sensor_size[1] - 1) - orig_events["y"], events["y"]), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )
        assert events is not orig_events

    def test_transform_merge_polarities(self):
        orig_events, sensor_size = create_random_input()
        transform = transforms.MergePolarities()
        events = transform(orig_events)
        assert len(np.unique(orig_events["p"])) == 2
        assert len(np.unique(events["p"])) == 1
        assert events is not orig_events

    def test_transform_numpy_array(self):
        orig_events, sensor_size = create_random_input()
        transform = transforms.NumpyAsType(int)
        events = transform(orig_events)
        assert events.dtype == int
        assert events is not orig_events

    def test_transform_numpy_array_unstructured(self):
        orig_events, sensor_size = create_random_input()
        transform = transforms.NumpyAsType(int)
        events = transform(orig_events)
        assert events.dtype == int
        assert events is not orig_events

    @pytest.mark.parametrize("refractory_period", [10000, 5000])
    def test_transform_refractory_period(self, refractory_period):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RefractoryPeriod(refractory_period=refractory_period)

        events = transform(orig_events)

        assert len(events) > 0, "Not all events should be filtered"
        assert len(events) < len(
            orig_events
        ), "Result should be fewer events than original event stream"
        assert np.isin(
            events, orig_events
        ).all(), (
            "Added additional events that were not present in original event stream"
        )
        assert events.dtype == events.dtype
        assert events is not orig_events

    @pytest.mark.parametrize(
        "variance, clip_outliers", [(30, False), (100, True), (3.5, True), (0.8, False)]
    )
    def test_transform_spatial_jitter(self, variance, clip_outliers):
        orig_events, sensor_size = create_random_input()

        transform = transforms.SpatialJitter(
            sensor_size=sensor_size,
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
            clip_outliers=clip_outliers,
        )

        events = transform(orig_events)

        if not clip_outliers:
            assert len(events) == len(orig_events)
            assert (events["t"] == orig_events["t"]).all()
            assert (events["p"] == orig_events["p"]).all()
            assert (events["x"] != orig_events["x"]).any()
            assert (events["y"] != orig_events["y"]).any()
            assert np.isclose(
                events["x"].all(), orig_events["x"].all(), atol=2 * variance
            )
            assert np.isclose(
                events["y"].all(), orig_events["y"].all(), atol=2 * variance
            )

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
    def test_transform_time_jitter(self, std, clip_negative, sort_timestamps):
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

    @pytest.mark.parametrize("p", [1000, 50])
    def test_transform_time_reversal(self, p):
        orig_events, sensor_size = create_random_input()

        original_t = orig_events["t"][0]
        original_p = orig_events["p"][0]

        max_t = np.max(orig_events["t"])

        transform = transforms.RandomTimeReversal(p=p)

        events = transform(orig_events)

        same_time = np.isclose(max_t - original_t, events["t"][0])
        same_polarity = np.isclose(events["p"][0], -1.0 * original_p)

        assert same_time, "When flipping time must map t_i' = max(t) - t_i"
        assert same_polarity, "When flipping time polarity should be flipped"
        assert events is not orig_events

    @pytest.mark.parametrize("coefficient, offset", [(3.1, 100), (0.3, 0), (2.7, 10)])
    def test_transform_time_skew(self, coefficient, offset):
        orig_events, sensor_size = create_random_input()

        transform = transforms.TimeSkew(coefficient=coefficient, offset=offset)

        events = transform(orig_events)

        assert len(events) == len(orig_events)
        assert np.min(events["t"]) >= offset
        assert (events["t"] == (events["t"]).astype(int)).all()
        assert all((orig_events["t"] * coefficient + offset).astype(int) == events["t"])
        assert events is not orig_events

    @pytest.mark.parametrize("n_noise_events", [100, 0])
    def test_transform_uniform_noise(self, n_noise_events):
        orig_events, sensor_size = create_random_input()

        transform = transforms.UniformNoise(
            sensor_size=sensor_size, n_noise_events=n_noise_events
        )

        events = transform(orig_events)

        assert len(events) == len(orig_events) + n_noise_events
        assert np.isin(orig_events, events).all()
        assert np.isclose(
            np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
        ), "Event noise should maintain temporal order."
        assert events is not orig_events

    def test_transform_time_alignment(self):
        orig_events, sensor_size = create_random_input()

        transform = transforms.TimeAlignment()

        events = transform(orig_events)

        assert np.min(events["t"]) == 0
        assert events is not orig_events
