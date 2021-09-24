import pytest
import numpy as np
import tonic.transforms as transforms
from utils import create_random_input

dtype = np.dtype([('x', int), ('y', int), ('t', int), ('p', int)])

class TestTransforms:
    @pytest.mark.parametrize(
        "target_size", [(50, 50), (10, 5)]
    )
    def test_transform_random_crop(self, target_size):
        print(target_size)
        orig_events, orig_sensor_size = create_random_input()

        transform = transforms.RandomCrop(target_size=target_size)
        events, sensor_size = transform((orig_events.copy(), orig_sensor_size))

        assert np.all(events['x']) < target_size[0] and np.all(
            events['y'] < target_size[1]
        ), "Cropping needs to map the events into the new space."
        assert sensor_size == target_size, "Sensor size needs to match the target cropping size"


    @pytest.mark.parametrize("ordering, filter_time", [("xytp", 1000), ("typx", 500)])
    def test_transform_denoise(self, ordering, filter_time):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        transform = transforms.Denoise(filter_time=filter_time, ordering=ordering)

        events = transform(events=orig_events.copy(),)

        assert len(events) > 0, "Not all events should be filtered"
        assert len(events) < len(
            orig_events
        ), "Result should be fewer events than original event stream"
        assert np.isin(events, orig_events).all(), (
            "Denoising should not add additional events that were not present in"
            " original event stream"
        )

    @pytest.mark.parametrize(
        "ordering, drop_probability, random_drop_probability",
        [("xytp", 0.2, False), ("typx", 0.5, True)],
    )
    def test_transform_drop_events(
        self, ordering, drop_probability, random_drop_probability
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.DropEvent(
            drop_probability=drop_probability,
            random_drop_probability=random_drop_probability,
        )

        events = transform(events=orig_events.copy())

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
            np.sum((events['t'] - np.sort(events['t'])) ** 2), 0
        ), "Event dropout should maintain temporal order."

        
    @pytest.mark.parametrize(
        "ordering, time_factor, spatial_factor", [("xytp", 1, 0.25), ("typx", 1e-3, 1)]
    )
    def test_transform_downsample(self, ordering, time_factor, spatial_factor):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.Downsample(ordering=ordering,
            time_factor=time_factor, spatial_factor=spatial_factor, sensor_size=sensor_size
        )

        events = transform(
            events=orig_events.copy()
        )

        assert np.array_equal(orig_events['t'] * time_factor, events['t'])
        assert np.array_equal(
            np.floor(orig_events['x'] * spatial_factor), events['x']
        )
        assert np.array_equal(
            np.floor(orig_events['y'] * spatial_factor), events['y']
        )

    @pytest.mark.parametrize(
        "ordering, flip_probability", [("xytp", 1.0), ("typx", 1.0)]
    )
    def test_transform_flip_lr(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.RandomFlipLR(flip_probability=flip_probability, ordering=ordering, sensor_size=sensor_size)

        events = transform(
            events=orig_events.copy(),
        )

        assert (
            (sensor_size[0] - 1) - orig_events['x'] == events['x']
        ).all(), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )

    @pytest.mark.parametrize("ordering, flip_probability", [("xytp", 1.0), ("typx", 0)])
    def test_transform_flip_polarity(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)
        transform = transforms.RandomFlipPolarity(ordering=ordering, flip_probability=flip_probability)

        events = transform(
            events=orig_events.copy(),
        )

        if flip_probability == 1:
            assert np.array_equal(orig_events['p'] * -1, events['p']), (
                "When flipping polarity with probability 1, all event polarities must"
                " flip"
            )
        else:
            assert np.array_equal(orig_events['p'], events['p']), (
                "When flipping polarity with probability 0, no event polarities must"
                " flip"
            )

    @pytest.mark.parametrize(
        "ordering, flip_probability", [("xytp", 1.0), ("typx", 1.0)]
    )
    def test_transform_flip_ud(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)
        transform = transforms.RandomFlipUD(flip_probability=flip_probability, ordering=ordering, sensor_size=sensor_size)

        events = transform(
            events=orig_events.copy(),
        )
        assert np.array_equal(
            (sensor_size[1] - 1) - orig_events['y'], events['y']
        ), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )

    @pytest.mark.parametrize(
        "ordering, refractory_period", [("xytp", 1000), ("typx", 50)]
    )
    def test_transform_refractory_period(self, ordering, refractory_period):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        transform = transforms.RefractoryPeriod(refractory_period=refractory_period, ordering=ordering)

        events = transform(
            events=orig_events.copy(),
        )

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
        "ordering, variance, integer_jitter, clip_outliers",
        [
            ("xytp", 30, True, False),
            ("typx", 100, True, True),
            ("typx", 3.5, False, True),
            ("typx", 0.8, False, False),
        ],
    )
    def test_transform_spatial_jitter(
        self, ordering, variance, integer_jitter, clip_outliers
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)
        transform = transforms.SpatialJitter(
            ordering=ordering,
            sensor_size=sensor_size,
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
            integer_jitter=integer_jitter,
            clip_outliers=clip_outliers,
        )

        events = transform(
            events=orig_events.copy(),
        )

        if not clip_outliers:
            assert len(events) == len(orig_events)
            assert (events['t'] == orig_events['t']).all()
            assert (events['p'] == orig_events['p']).all()
            assert (events['x'] != orig_events['x']).any()
            assert (events['y'] != orig_events['y']).any()
            assert np.isclose(
                events['x'].all(),
                orig_events['x'].all(),
                atol=2 * variance,
            )
            assert np.isclose(
                events['y'].all(),
                orig_events['y'].all(),
                atol=2 * variance,
            )

            if integer_jitter:
                assert (
                    events['x'] - orig_events['x']
                    == (events['x'] - orig_events['x']).astype(int)
                ).all()

                assert (
                    events['y'] - orig_events['y']
                    == (events['y'] - orig_events['y']).astype(int)
                ).all()

        else:
            assert len(events) < len(orig_events)

    @pytest.mark.parametrize(
        "ordering, std, integer_jitter, clip_negative, sort_timestamps",
        [
            ("xytp", 10, False, True, True),
            ("typx", 50, True, False, False),
            ("pxty", 0, True, True, False),
        ],
    )
    def test_transform_time_jitter(
        self, ordering, std, integer_jitter, clip_negative, sort_timestamps
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        # we do this to ensure integer timestamps before testing for int jittering
        if integer_jitter:
            orig_events['t'] = orig_events['t'].round()

        transform = transforms.TimeJitter(
            ordering=ordering,
            std=std,
            integer_jitter=integer_jitter,
            clip_negative=clip_negative,
            sort_timestamps=sort_timestamps,
        )

        events = transform(
            events=orig_events.copy()
        )

        if clip_negative:
            assert (events['t'] >= 0).all()
        else:
            assert len(events) == len(orig_events)
        if sort_timestamps:
            np.testing.assert_array_equal(
                events['t'], np.sort(events['t'])
            )
        if not sort_timestamps and not clip_negative:
            np.testing.assert_array_equal(events['x'], orig_events['x'])
            np.testing.assert_array_equal(events['y'], orig_events['y'])
            np.testing.assert_array_equal(events['p'], orig_events['p'])
            if integer_jitter:
                assert (
                    events['t'] - orig_events['t']
                    == (events['t'] - orig_events['t']).astype(int)
                ).all()

    @pytest.mark.parametrize(
        "ordering, flip_probability", [("xytp", 1000), ("typx", 50)]
    )
    def test_transform_time_reversal(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        original_t = orig_events[0, t_index]
        original_p = orig_events[0, p_index]

        max_t = np.max(orig_events['t'])

        transform = transforms.RandomTimeReversal(flip_probability=flip_probability, ordering=ordering)

        events = transform(
            events=orig_events.copy()
        )

        same_time = np.isclose(max_t - original_t, events[0, t_index])
        same_polarity = np.isclose(events[0, p_index], -1.0 * original_p)

        assert same_time, "When flipping time must map t_i' = max(t) - t_i"
        assert same_polarity, "When flipping time polarity should be flipped"
        assert events.dtype == events.dtype

    @pytest.mark.parametrize(
        "ordering, offset, coefficient, integer_time",
        [("xytp", 100, 3.1, True), ("typx", 0, 0.7, False), ("ptyx", 10, 2.7, False)],
    )
    def test_transform_time_skew(self, ordering, offset, coefficient, integer_time):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.TimeSkew(
            ordering=ordering, coefficient=coefficient, offset=offset, integer_time=integer_time,
        )

        events = transform(
            events=orig_events.copy()
        )

        assert len(events) == len(orig_events)
        assert np.min(events['t']) >= offset
        if integer_time:
            assert (events['t'] == (events['t']).astype(int)).all()

        else:
            if coefficient > 1:
                assert (events['t'] - offset > orig_events['t']).all()

            if coefficient < 1:
                assert (events['t'] - offset < orig_events['t']).all()

            assert (events['t'] != (events['t']).astype(int)).any()

    @pytest.mark.parametrize("ordering, n_noise_events", [("xytp", 100), ("typx", 0)])
    def test_transform_uniform_noise(self, ordering, n_noise_events):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.UniformNoise(
            n_noise_events=n_noise_events, ordering=ordering,
        )

        events = transform(
            events=orig_events.copy()
        )

        assert len(events) == len(orig_events)+n_noise_events
        assert np.isin(orig_events, events).all()
        assert np.isclose(
            np.sum((events['t'] - np.sort(events['t'])) ** 2), 0
        ), "Event noise should maintain temporal order."

    @pytest.mark.parametrize("ordering",[("xytp"), ("typx")])
    def test_transform_time_alignment(self, ordering):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.TimeAlignment()

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            images=orig_images.copy(),
            multi_image=is_multi_image,
        )

        assert np.min(events['t']) == 0
