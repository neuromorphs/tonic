import pytest
import numpy as np
import tonic.transforms as transforms
from utils import create_random_input

dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])


class TestTransforms:
    @pytest.mark.parametrize("target_size", [(50, 50), (10, 5)])
    def test_transform_random_crop(self, target_size):
        orig_events, orig_sensor_size = create_random_input()

        transform = transforms.RandomCrop(target_size=target_size)
        events, sensor_size = transform((orig_events.copy(), orig_sensor_size))

        assert np.all(events["x"]) < target_size[0] and np.all(
            events["y"] < target_size[1]
        ), "Cropping needs to map the events into the new space."
        assert (
            sensor_size == target_size
        ), "Sensor size needs to match the target cropping size"

    @pytest.mark.parametrize("filter_time", [(10000), (5000)])
    def test_transform_denoise(self, filter_time):
        orig_events, sensor_size = create_random_input()

        transform = transforms.Denoise(filter_time=filter_time)

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        assert len(events) > 0, "Not all events should be filtered"
        assert len(events) < len(
            orig_events
        ), "Result should be fewer events than original event stream"
        assert np.isin(events, orig_events).all(), (
            "Denoising should not add additional events that were not present in"
            " original event stream"
        )

    @pytest.mark.parametrize(
        "drop_probability, random_drop_probability", [(0.2, False), (0.5, True)],
    )
    def test_transform_drop_events(self, drop_probability, random_drop_probability):
        orig_events, sensor_size = create_random_input()

        transform = transforms.DropEvent(
            drop_probability=drop_probability,
            random_drop_probability=random_drop_probability,
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        if random_drop_probability:
            assert events.shape[0] >= (1 - drop_probability) * orig_events.shape[0], (
                "Event dropout with random drop probability should result in less than "
                " drop_probability*len(original) events dropped out."
            )
        else:
            assert np.isclose(
                events.shape[0], (1 - drop_probability) * orig_events.shape[0]
            ), (
                "Event dropout should result in drop_probability*len(original) events"
                " dropped out."
            )
        assert np.isclose(
            np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
        ), "Event dropout should maintain temporal order."

    @pytest.mark.parametrize("time_factor, spatial_factor", [(1, 0.25), (1e-3, 1)])
    def test_transform_downsample(self, time_factor, spatial_factor):
        orig_events, sensor_size = create_random_input()

        transform = transforms.Downsample(
            time_factor=time_factor,
            spatial_factor=spatial_factor,
            sensor_size=sensor_size,
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        assert np.array_equal((orig_events["t"] * time_factor).astype(orig_events["t"].dtype), events["t"])
        assert np.array_equal(np.floor(orig_events["x"] * spatial_factor), events["x"])
        assert np.array_equal(np.floor(orig_events["y"] * spatial_factor), events["y"])

    @pytest.mark.parametrize("flip_probability", [(1.0), (1.0)])
    def test_transform_flip_lr(self, flip_probability):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RandomFlipLR(
            flip_probability=flip_probability, sensor_size=sensor_size
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        assert ((sensor_size[0] - 1) - orig_events["x"] == events["x"]).all(), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )

    @pytest.mark.parametrize("flip_probability", [(1.0), (0)])
    def test_transform_flip_polarity(self, flip_probability):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RandomFlipPolarity(flip_probability=flip_probability)

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        if flip_probability == 1:
            assert np.array_equal(orig_events["p"] * -1, events["p"]), (
                "When flipping polarity with probability 1, all event polarities must"
                " flip"
            )
        else:
            assert np.array_equal(orig_events["p"], events["p"]), (
                "When flipping polarity with probability 0, no event polarities must"
                " flip"
            )

    @pytest.mark.parametrize("flip_probability", [(1.0), (1.0)])
    def test_transform_flip_ud(self, flip_probability):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RandomFlipUD(
            flip_probability=flip_probability, sensor_size=sensor_size
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        assert np.array_equal((sensor_size[1] - 1) - orig_events["y"], events["y"]), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )

    @pytest.mark.parametrize("refractory_period", [(1000), (50)])
    def test_transform_refractory_period(self, refractory_period):
        orig_events, sensor_size = create_random_input()

        transform = transforms.RefractoryPeriod(refractory_period=refractory_period,)

        events, sensor_size = transform((orig_events.copy(), sensor_size))

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

    @pytest.mark.parametrize(
        "variance, integer_jitter, clip_outliers",
        [
            (30, True, False),
            (100, True, True),
            (3.5, False, True),
            (0.8, False, False),
        ],
    )
    def test_transform_spatial_jitter(self, variance, integer_jitter, clip_outliers):
        orig_events, sensor_size = create_random_input()

        transform = transforms.SpatialJitter(
            sensor_size=sensor_size,
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
            integer_jitter=integer_jitter,
            clip_outliers=clip_outliers,
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        if not clip_outliers:
            assert len(events) == len(orig_events)
            assert (events["t"] == orig_events["t"]).all()
            assert (events["p"] == orig_events["p"]).all()
            assert (events["x"] != orig_events["x"]).any()
            assert (events["y"] != orig_events["y"]).any()
            assert np.isclose(
                events["x"].all(), orig_events["x"].all(), atol=2 * variance,
            )
            assert np.isclose(
                events["y"].all(), orig_events["y"].all(), atol=2 * variance,
            )

            if integer_jitter:
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

    @pytest.mark.parametrize(
        "std, integer_jitter, clip_negative, sort_timestamps",
        [(10, False, True, True), (50, True, False, False), (0, True, True, False),],
    )
    def test_transform_time_jitter(
        self, std, integer_jitter, clip_negative, sort_timestamps
    ):
        orig_events, sensor_size = create_random_input()

        # we do this to ensure integer timestamps before testing for int jittering
        if integer_jitter:
            orig_events["t"] = orig_events["t"].round()

        transform = transforms.TimeJitter(
            std=std,
            integer_jitter=integer_jitter,
            clip_negative=clip_negative,
            sort_timestamps=sort_timestamps,
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

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
            if integer_jitter:
                assert (
                    events["t"] - orig_events["t"]
                    == (events["t"] - orig_events["t"]).astype(int)
                ).all()

    @pytest.mark.parametrize("flip_probability", [(1000), (50)])
    def test_transform_time_reversal(self, flip_probability):
        orig_events, sensor_size = create_random_input()

        original_t = orig_events["t"][0]
        original_p = orig_events["p"][0]

        max_t = np.max(orig_events["t"])

        transform = transforms.RandomTimeReversal(flip_probability=flip_probability,)

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        same_time = np.isclose(max_t - original_t, events["t"][0])
        same_polarity = np.isclose(events["p"][0], -1.0 * original_p)

        assert same_time, "When flipping time must map t_i' = max(t) - t_i"
        assert same_polarity, "When flipping time polarity should be flipped"
        assert events.dtype == events.dtype

    @pytest.mark.parametrize(
        "coefficient, offset",
        [(3.1, 100), (0.7, 0), (2.7, 10)],
    )
    def test_transform_time_skew(self, coefficient, offset):
        orig_events, sensor_size = create_random_input()

        transform = transforms.TimeSkew(
            coefficient=coefficient, offset=offset, integer_time=integer_time,
        )

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        assert len(events) == len(orig_events)
        assert np.min(events["t"]) >= offset
        if integer_time:
            assert (events["t"] == (events["t"]).astype(int)).all()

        else:
            if coefficient > 1:
                assert (events["t"] - offset > orig_events["t"]).all()

            if coefficient < 1:
                assert (events["t"] - offset < orig_events["t"]).all()

            assert (events["t"] != (events["t"]).astype(int)).any()

    @pytest.mark.parametrize("n_noise_events", [(100), (0)])
    def test_transform_uniform_noise(self, n_noise_events):
        orig_events, sensor_size = create_random_input()

        transform = transforms.UniformNoise(n_noise_events=n_noise_events,)

        events, sensor_size = transform((orig_events.copy(), sensor_size))

        assert len(events) == len(orig_events) + n_noise_events
        assert np.isin(orig_events, events).all()
        assert np.isclose(
            np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
        ), "Event noise should maintain temporal order."

    def test_transform_time_alignment(self):
        orig_events, sensor_size = create_random_input()

        transform = transforms.TimeAlignment()

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            images=orig_images.copy(),
        )

        assert np.min(events["t"]) == 0
