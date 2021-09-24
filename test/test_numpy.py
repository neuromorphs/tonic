import pytest
import numpy as np
import tonic.functional as F
import utils


class TestFunctionalNumpy:
    @pytest.mark.parametrize(
        "ordering, target_size", [("xytp", (50, 50)), ("typx", (10, 5))]
    )
    def testCrop(self, ordering, target_size):
        (
            events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events =  F.crop_numpy(
            events,
            
            sensor_size=sensor_size,
            ordering=ordering,
            
            target_size=target_size,
        )
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        assert np.all(events['x']) < target_size[0] and np.all(
            events['y'] < target_size[1]
        ), "Cropping needs to map the events into the new space"


    @pytest.mark.parametrize(
        "ordering, drop_probability, random_drop_probability",
        [("xytp", 0.2, False), ("typx", 0.5, True)],
    )
    def testDropEvents(self, ordering, drop_probability, random_drop_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events = F.drop_event_numpy(
            orig_events.copy(),
            drop_probability=drop_probability,
            random_drop_probability=random_drop_probability,
        )
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

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


    @pytest.mark.parametrize("ordering, filter_time", [("xytp", 1000), ("typx", 500)])
    def testDenoise(self, ordering, filter_time):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.denoise_numpy(
            orig_events,
            ordering=ordering,
            filter_time=filter_time,
        )

        assert len(events) > 0, "Not all events should be filtered"
        assert len(events) < len(
            orig_events
        ), "Result should be fewer events than original event stream"
        assert np.isin(events, orig_events).all(), (
            "Denoising should not add additional events that were not present in"
            " original event stream"
        )

    @pytest.mark.parametrize("ordering", ["xytp", "typx"])
    def testMixEvents(self, ordering):
        (
            stream1,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        (
            stream2,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        events = (stream1, stream2)

        mixed_events_no_offset, _ = F.mix_ev_streams_numpy(
            events,
            offsets=None,
            check_conflicts=False,
            sensor_size=sensor_size,
            ordering=ordering,
        )

        mixed_events_random_offset, _ = F.mix_ev_streams_numpy(
            events,
            offsets="Random",
            check_conflicts=False,
            sensor_size=sensor_size,
            ordering=ordering,
        )

        mixed_events_defined_offset, _ = F.mix_ev_streams_numpy(
            events,
            offsets=(0, 100),
            check_conflicts=False,
            sensor_size=sensor_size,
            ordering=ordering,
        )

        mixed_events_conflict, num_conflicts = F.mix_ev_streams_numpy(
            (stream1, stream1),
            offsets=None,
            check_conflicts=True,
            sensor_size=sensor_size,
            ordering=ordering,
        )

        no_offset_monotonic = np.all(
            mixed_events_no_offset[1:, t_index] >= mixed_events_no_offset[:-1, t_index],
            axis=0,
        )
        random_offset_monotonic = np.all(
            mixed_events_random_offset[1:, t_index]
            >= mixed_events_random_offset[:-1, t_index],
            axis=0,
        )
        defined_offset_monotonic = np.all(
            mixed_events_defined_offset[1:, t_index]
            >= mixed_events_defined_offset[:-1, t_index],
            axis=0,
        )
        conflict_offset_monotonic = np.all(
            mixed_events_conflict[1:, t_index] >= mixed_events_conflict[:-1, t_index],
            axis=0,
        )
        all_colisions_detected = len(stream1) == num_conflicts

        assert (
            all_colisions_detected
        ), "Missed some event colisions, may cause processing problems."
        assert no_offset_monotonic, "Result was not monotonic."
        assert random_offset_monotonic, "Result was not monotonic."
        assert defined_offset_monotonic, "Result was not monotonic."
        assert conflict_offset_monotonic, "Result was not monotonic."

    @pytest.mark.parametrize(
        "ordering, refractory_period", [("xytp", 1000), ("typx", 50)]
    )
    def testRefractoryPeriod(self, ordering, refractory_period):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.refractory_period_numpy(
            events=orig_events,
            ordering=ordering,
            refractory_period=refractory_period,
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
    def testSpatialJitter(self, ordering, variance, integer_jitter, clip_outliers):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.spatial_jitter_numpy(
            orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
            integer_jitter=integer_jitter,
            clip_outliers=clip_outliers,
        )
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

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
    def testTimeJitter(
        self, ordering, std, integer_jitter, clip_negative, sort_timestamps
    ):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        # we do this to ensure integer timestamps before testing for int jittering
        if integer_jitter:
            orig_events['t'] = orig_events['t'].round()
        events = F.time_jitter_numpy(
            orig_events.copy(),
            ordering=ordering,
            std=std,
            integer_jitter=integer_jitter,
            clip_negative=clip_negative,
            sort_timestamps=sort_timestamps,
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
        "ordering, offset, coefficient, integer_time",
        [("xytp", 100, 3.1, True), ("typx", 0, 0.7, False), ("ptyx", 10, 2.7, False)],
    )
    def testTimeSkew(self, ordering, offset, coefficient, integer_time):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        events = F.time_skew_numpy(
            orig_events.copy(),
            ordering=ordering,
            coefficient=coefficient,
            offset=offset,
            integer_time=integer_time,
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

    @pytest.mark.parametrize(
        "ordering, time_window, spike_count, n_time_bins, n_event_bins, overlap,"
        " include_incomplete, merge_polarities",
        [
            ("xytp", 2000, None, None, None, 0, False, True),
            ("txyp", 2000, None, None, None, 200, True, False),
            ("xytp", 1000, None, None, None, 100, True, True),
            ("txyp", None, 2000, None, None, 0, False, True),
            ("xytp", None, 2000, None, None, 200, True, False),
            ("txyp", None, 2000, None, None, 100, True, True),
            ("xytp", None, None, 5, None, 0, False, True),
            ("xytp", None, None, 5, None, 0.1, False, False),
            ("xytp", None, None, 5, None, 0.25, True, False),
            ("xytp", None, None, None, 5, 0, True, False),
            ("xytp", None, None, None, 5, 0.1, False, True),
            ("xytp", None, None, None, 5, 0.25, False, False),
        ],
    )
    def testToFrame(
        self,
        ordering,
        time_window,
        spike_count,
        n_time_bins,
        n_event_bins,
        overlap,
        include_incomplete,
        merge_polarities,
    ):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        frames = F.to_frame_numpy(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            time_window=time_window,
            spike_count=spike_count,
            n_time_bins=n_time_bins,
            n_event_bins=n_event_bins,
            overlap=overlap,
            include_incomplete=include_incomplete,
            merge_polarities=merge_polarities,
        )

        if time_window is not None:
            stride = time_window - overlap
            times = orig_events['t']
            if include_incomplete:
                assert frames.shape[0] == int(
                    np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1
                )
            else:
                assert frames.shape[0] == int(
                    np.floor(((times[-1] - times[0]) - time_window) / stride) + 1
                )

        if spike_count is not None:
            stride = spike_count - overlap
            n_events = orig_events.shape[0]
            if include_incomplete:
                assert frames.shape[0] == int(
                    np.ceil((n_events - spike_count) / stride) + 1
                )
            else:
                assert frames.shape[0] == int(
                    np.floor((n_events - spike_count) / stride) + 1
                )

        if n_time_bins is not None:
            assert frames.shape[0] == n_time_bins

        if n_event_bins is not None:
            assert frames.shape[0] == n_event_bins

        if merge_polarities:
            assert frames.shape[1] == 1

    @pytest.mark.parametrize(
        "ordering, surface_dimensions, tau, merge_polarities",
        [("xytp", (15, 15), 100, True), ("typx", (3, 3), 10, False), ("txyp", None, 1e4, False)],
    )
    def testToTimesurface(self, ordering, surface_dimensions, tau, merge_polarities):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        surfaces = F.to_timesurface_numpy(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            surface_dimensions=surface_dimensions,
            tau=tau,
            merge_polarities=merge_polarities,
        )
        assert surfaces.shape[0] == len(orig_events)
        assert surfaces.shape[1] == 1 if merge_polarities else 2
        if surface_dimensions:
            assert surfaces.shape[2:] == surface_dimensions
        else:
            assert surfaces.shape[2:] == sensor_size

    @pytest.mark.parametrize("ordering", ["xytp", "typx"])
    def testToAveragedTimesurface(self, ordering):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        cell_size = 10
        surface_size = 5
        temporal_window = 100
        tau = 100
        merge_polarities = True

        surfaces = F.to_averaged_timesurface(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            cell_size=cell_size,
            surface_size=surface_size,
            temporal_window=temporal_window,
            tau=tau,
            merge_polarities=merge_polarities,
        )
        assert surfaces.shape[0] == len(orig_events)
        assert surfaces.shape[1] == 1
        assert surfaces.shape[2] == surface_size

    @pytest.mark.parametrize("ordering, n_time_bins", [("xytp", 10), ("typx", 1)])
    def testToVoxelGrid(self, ordering, n_time_bins):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        volumes = F.to_voxel_grid_numpy(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            n_time_bins=n_time_bins,
        )
        assert volumes.shape == (n_time_bins, *sensor_size[::-1])
