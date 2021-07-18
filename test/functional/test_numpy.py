import unittest
from parameterized import parameterized
import numpy as np
import tonic.functional as F
import utils
import math


class TestFunctionalNumpy(unittest.TestCase):
    def findXytpPermutation(self, ordering):
        x_index = ordering.find("x")
        y_index = ordering.find("y")
        t_index = ordering.find("t")
        p_index = ordering.find("p")
        return x_index, y_index, t_index, p_index

    @parameterized.expand([("xytp", (50, 50)), ("typx", (10, 5))])
    def testCrop(self, ordering, target_size):
        (
            events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events, images = F.crop_numpy(
            events,
            images=images,
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
            target_size=target_size,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        self.assertTrue(
            np.all(events[:, x_index]) < target_size[0]
            and np.all(events[:, y_index] < target_size[1]),
            "Cropping needs to map the events into the new space",
        )

        self.assertTrue(
            images.shape[2] == target_size[0] and images.shape[1] == target_size[1],
            "Cropping needs to map the images into the new space",
        )

    @parameterized.expand([("xytp", 0.2, False), ("typx", 0.5, True)])
    def testDropEvents(self, ordering, drop_probability, random_drop_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events = F.drop_events_numpy(
            orig_events.copy(),
            drop_probability=drop_probability,
            random_drop_probability=random_drop_probability,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        if random_drop_probability:
            self.assertTrue(
                events.shape[0] >= (1 - drop_probability) * orig_events.shape[0],
                "Event dropout with random drop probability should result in less than"
                " drop_probability*len(original) events dropped out.",
            )
        else:
            self.assertTrue(
                np.isclose(
                    events.shape[0], (1 - drop_probability) * orig_events.shape[0]
                ),
                "Event dropout should result in drop_probability*len(original) events"
                " dropped out.",
            )
        self.assertTrue(
            np.isclose(
                np.sum((events[:, t_index] - np.sort(events[:, t_index])) ** 2), 0
            ),
            "Event dropout should maintain temporal order.",
        )

    @parameterized.expand([("xytp", 1.0), ("typx", 1.0)])
    def testFlipLR(self, ordering, flip_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events, images = F.flip_lr_numpy(
            orig_events.copy(),
            images=images,
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
            flip_probability=flip_probability,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)
        self.assertTrue(
            (
                (sensor_size[0] - 1) - orig_events[:, x_index] == events[:, x_index]
            ).all(),
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x",
        )

    @parameterized.expand([("xytp", 1.0), ("typx", 0)])
    def testFlipPolarity(self, ordering, flip_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.flip_polarity_numpy(
            orig_events.copy(), ordering=ordering, flip_probability=flip_probability
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)
        if flip_probability == 1:
            self.assertTrue(
                np.array_equal(orig_events[:, p_index] * -1, events[:, p_index]),
                "When flipping polarity with probability 1, all event polarities must"
                " flip",
            )
        else:
            self.assertTrue(
                np.array_equal(orig_events[:, p_index], events[:, p_index]),
                "When flipping polarity with probability 0, no event polarities must"
                " flip",
            )

    @parameterized.expand([("xytp", 1.0), ("typx", 1.0)])
    def testFlipUD(self, ordering, flip_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events, images = F.flip_ud_numpy(
            orig_events.copy(),
            images=images,
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
            flip_probability=flip_probability,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)
        self.assertTrue(
            (
                (sensor_size[1] - 1) - orig_events[:, y_index] == events[:, y_index]
            ).all(),
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x",
        )

    @parameterized.expand([("xytp", 1000), ("typx", 500)])
    def testDenoise(self, ordering, filter_time):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.denoise_numpy(
            orig_events,
            sensor_size=sensor_size,
            ordering=ordering,
            filter_time=filter_time,
        )

        self.assertTrue(len(events) > 0, "Not all events should be filtered")
        self.assertTrue(
            len(events) < len(orig_events),
            "Result should be fewer events than original event stream",
        )
        self.assertTrue(
            np.isin(events, orig_events).all(),
            "Denoising should not add additional events that were not present in"
            " original event stream",
        )

    @parameterized.expand(["xytp", "typx"])
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
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

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

        self.assertTrue(
            all_colisions_detected,
            "Missed some event colisions, may cause processing problems.",
        )
        self.assertTrue(no_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(random_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(defined_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(conflict_offset_monotonic, "Result was not monotonic.")

    @parameterized.expand([("xytp", 1000), ("typx", 50)])
    def testRefractoryPeriod(self, ordering, refractory_period):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.refractory_period_numpy(
            events=orig_events,
            sensor_size=sensor_size,
            ordering=ordering,
            refractory_period=refractory_period,
        )

        self.assertTrue(len(events) > 0, "Not all events should be filtered")
        self.assertTrue(
            len(events) < len(orig_events),
            "Result should be fewer events than original event stream",
        )
        self.assertTrue(
            np.isin(events, orig_events).all(),
            "Added additional events that were not present in original event stream",
        )
        self.assertTrue(events.dtype == events.dtype)

    @parameterized.expand(
        [
            ("xytp", 30, True, False),
            ("typx", 100, True, True),
            ("typx", 3.5, False, True),
            ("typx", 0.8, False, False),
        ]
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
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        if not clip_outliers:
            self.assertTrue(len(events) == len(orig_events))
            self.assertTrue((events[:, t_index] == orig_events[:, t_index]).all())
            self.assertTrue((events[:, p_index] == orig_events[:, p_index]).all())
            self.assertTrue((events[:, x_index] != orig_events[:, x_index]).any())
            self.assertTrue((events[:, y_index] != orig_events[:, y_index]).any())
            self.assertTrue(
                np.isclose(
                    events[:, x_index].all(),
                    orig_events[:, x_index].all(),
                    atol=2 * variance,
                )
            )
            self.assertTrue(
                np.isclose(
                    events[:, y_index].all(),
                    orig_events[:, y_index].all(),
                    atol=2 * variance,
                )
            )
            if integer_jitter:
                self.assertTrue(
                    (
                        events[:, x_index] - orig_events[:, x_index]
                        == (events[:, x_index] - orig_events[:, x_index]).astype(int)
                    ).all()
                )
                self.assertTrue(
                    (
                        events[:, y_index] - orig_events[:, y_index]
                        == (events[:, y_index] - orig_events[:, y_index]).astype(int)
                    ).all()
                )
        else:
            self.assertTrue(len(events) < len(orig_events))

    @parameterized.expand(["xytp"])
    def testStTransform(self, ordering):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        spatial_transform = np.array(((1, 0, 10), (0, 1, 10), (0, 0, 1)))
        temporal_transform = np.array((2, 0))
        events = F.st_transform(
            orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            roll=False,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        self.assertTrue(
            np.all(events[:, x_index]) < sensor_size[0]
            and np.all(events[:, y_index] < sensor_size[1]),
            "Transformation does not map beyond sensor size",
        )

    @parameterized.expand(
        [
            ("xytp", 10, False, True, True),
            ("typx", 50, True, False, False),
            ("pxty", 0, True, True, False),
        ]
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
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        # we do this to ensure integer timestamps before testing for int jittering
        if integer_jitter:
            orig_events[:, t_index] = orig_events[:, t_index].round()
        events = F.time_jitter_numpy(
            orig_events.copy(),
            ordering=ordering,
            std=std,
            integer_jitter=integer_jitter,
            clip_negative=clip_negative,
            sort_timestamps=sort_timestamps,
        )
        if clip_negative:
            self.assertTrue((events[:, t_index] >= 0).all())
        else:
            self.assertEqual(len(events), len(orig_events))
        if sort_timestamps:
            np.testing.assert_array_equal(
                events[:, t_index], np.sort(events[:, t_index])
            )
        if not sort_timestamps and not clip_negative:
            np.testing.assert_array_equal(events[:, x_index], orig_events[:, x_index])
            np.testing.assert_array_equal(events[:, y_index], orig_events[:, y_index])
            np.testing.assert_array_equal(events[:, p_index], orig_events[:, p_index])
            if integer_jitter:
                self.assertTrue(
                    (
                        events[:, t_index] - orig_events[:, t_index]
                        == (events[:, t_index] - orig_events[:, t_index]).astype(int)
                    ).all()
                )

    @parameterized.expand([("xytp", 1000), ("typx", 50)])
    def testTimeReversal(self, ordering, flip_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        original_t = orig_events[0, t_index]
        original_p = orig_events[0, p_index]

        max_t = np.max(orig_events[:, t_index])

        events, images = F.time_reversal_numpy(
            orig_events,
            images=images,
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
            flip_probability=flip_probability,
        )
        same_time = np.isclose(max_t - original_t, events[0, t_index])
        same_polarity = np.isclose(events[0, p_index], -1.0 * original_p)

        self.assertTrue(same_time, "When flipping time must map t_i' = max(t) - t_i")
        self.assertTrue(same_polarity, "When flipping time polarity should be flipped")
        self.assertTrue(events.dtype == events.dtype)

    @parameterized.expand(
        [("xytp", 100, 3.1, True), ("typx", 0, 0.7, False), ("ptyx", 10, 2.7, False)]
    )
    def testTimeSkew(self, ordering, offset, coefficient, integer_time):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        events = F.time_skew_numpy(
            orig_events.copy(),
            ordering=ordering,
            coefficient=coefficient,
            offset=offset,
            integer_time=integer_time,
        )
        self.assertTrue(len(events) == len(orig_events))
        self.assertTrue(np.min(events[:, t_index]) >= offset)
        if integer_time:
            self.assertTrue(
                (events[:, t_index] == (events[:, t_index]).astype(int)).all()
            )
        else:
            if coefficient > 1:
                self.assertTrue(
                    (events[:, t_index] - offset > orig_events[:, t_index]).all()
                )
            if coefficient < 1:
                self.assertTrue(
                    (events[:, t_index] - offset < orig_events[:, t_index]).all()
                )
            self.assertTrue(
                (events[:, t_index] != (events[:, t_index]).astype(int)).any()
            )

    @parameterized.expand(
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
        ]
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
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

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
            times = orig_events[:, t_index]
            if include_incomplete:
                self.assertEqual(
                    frames.shape[0],
                    int(np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1),
                )
            else:
                self.assertEqual(
                    frames.shape[0],
                    int(np.floor(((times[-1] - times[0]) - time_window) / stride) + 1),
                )

        if spike_count is not None:
            stride = spike_count - overlap
            n_events = orig_events.shape[0]
            if include_incomplete:
                self.assertEqual(
                    frames.shape[0], int(np.ceil((n_events - spike_count) / stride) + 1)
                )
            else:
                self.assertEqual(
                    frames.shape[0],
                    int(np.floor((n_events - spike_count) / stride) + 1),
                )

        if n_time_bins is not None:
            self.assertEqual(frames.shape[0], n_time_bins)

        if n_event_bins is not None:
            self.assertEqual(frames.shape[0], n_event_bins)

        if merge_polarities:
            self.assertEqual(frames.shape[1], 1)

    @parameterized.expand([("xytp", (15, 15), 100, True), ("typx", (3, 3), 10, False)])
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
        self.assertEqual(surfaces.shape[0], len(orig_events))
        self.assertEqual(surfaces.shape[1], 1 if merge_polarities else 2)
        self.assertEqual(surfaces.shape[2:], surface_dimensions)

    @parameterized.expand(["xytp", "typx"])
    def testToAveragedTimesurfaceXytp(self, ordering):
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
        self.assertEqual(surfaces.shape[0], len(orig_events))
        self.assertEqual(surfaces.shape[1], 1)
        self.assertEqual(surfaces.shape[2], surface_size)

    @parameterized.expand(["xytp", "typx"])
    def testUniformNoiseXytp(self, ordering):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        noisy_events = F.uniform_noise_numpy(
            orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            scaling_factor_to_micro_sec=1000000,
            noise_density=1e-8,
        )

        self.assertTrue(len(noisy_events) > len(orig_events))
        self.assertTrue(np.isin(orig_events, noisy_events).all())

    @parameterized.expand([("xytp", 10), ("typx", 1)])
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
        self.assertEqual(volumes.shape, (n_time_bins, *sensor_size[::-1]))
