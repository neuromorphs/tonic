import unittest
from parameterized import parameterized
import numpy as np
import tonic.functional as F
import utils
import ipdb
import math


class TestFunctionalAPI(unittest.TestCase):
    def findXytpPermutation(self, ordering):
        x_index = ordering.find("x")
        y_index = ordering.find("y")
        t_index = ordering.find("t")
        p_index = ordering.find("p")
        return x_index, y_index, t_index, p_index

    @parameterized.expand(
        [("xytp", (50, 50)), ("typx", (10, 5)),]
    )
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

    @parameterized.expand(
        [("xytp", 0.2, False), ("typx", 0.5, True),]
    )
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
                    events.shape[0], (1 - drop_probability) * orig_events.shape[0],
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

    @parameterized.expand(
        [("xytp", 1.0), ("typx", 1.0),]
    )
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

    @parameterized.expand(
        [("xytp", 1.0), ("typx", 0),]
    )
    def testFlipPolarity(self, ordering, flip_probability):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        events = F.flip_polarity_numpy(
            orig_events.copy(), ordering=ordering, flip_probability=flip_probability,
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

    @parameterized.expand(
        [("xytp", 1.0), ("typx", 1.0),]
    )
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

    @parameterized.expand(
        [("xytp", 1000), ("typx", 500),]
    )
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

    @parameterized.expand(
        ["xytp", "typx",]
    )
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

    @parameterized.expand(
        [("xytp", 1000), ("typx", 50),]
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
        [("xytp", 30), ("typx", 10),]
    )
    def testSpatialJitter(self, ordering, variance):
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
            integer_coordinates=False,
            clip_outliers=False,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        self.assertTrue(len(events) == len(orig_events))
        self.assertTrue((events[:, t_index] == orig_events[:, t_index]).all())
        self.assertTrue((events[:, p_index] == orig_events[:, p_index]).all())
        self.assertFalse((events[:, x_index] == orig_events[:, x_index]).all())
        self.assertFalse((events[:, y_index] == orig_events[:, y_index]).all())
        self.assertTrue(
            np.isclose(
                events[:, x_index].all(), orig_events[:, x_index].all(), atol=variance
            )
        )
        self.assertTrue(
            np.isclose(
                events[:, y_index].all(), orig_events[:, y_index].all(), atol=variance
            )
        )

    @parameterized.expand(
        ["xytp",]
    )
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
        [("xytp", 10), ("typx", 50),]
    )
    def testTimeJitter(self, ordering, variance):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        events = F.time_jitter_numpy(
            orig_events.copy(),
            ordering=ordering,
            variance=variance,
            integer_timestamps=False,
            clip_negative=False,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        self.assertTrue(len(events) == len(orig_events))
        self.assertTrue((events[:, x_index] == orig_events[:, x_index]).all())
        self.assertTrue((events[:, y_index] == orig_events[:, y_index]).all())
        self.assertFalse((events[:, t_index] == orig_events[:, t_index]).all())
        self.assertTrue((events[:, p_index] == orig_events[:, p_index]).all())

    @parameterized.expand(
        [("xytp", 1000), ("typx", 50),]
    )
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
        [("xytp", 100, 3.1), ("typx", 0, 0.7),]
    )
    def testTimeSkew(self, ordering, offset, coefficient):
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
        )
        self.assertTrue(len(events) == len(orig_events))
        self.assertTrue(np.min(events[:, t_index]) >= offset)
        if coefficient > 1:
            self.assertTrue(
                (events[:, t_index] - offset > orig_events[:, t_index]).all()
            )
        elif coefficient < 1:
            self.assertTrue(
                (events[:, t_index] - offset < orig_events[:, t_index]).all()
            )

    @parameterized.expand(
        [("xytp", 1000), ("typx", 500),]
    )
    def testToRatecodedFrame(self, ordering, frame_time):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        frames = F.to_ratecoded_frame_numpy(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            frame_time=frame_time,
        )
        self.assertEqual(
            frames.shape,
            (
                math.ceil(orig_events[-1, t_index] / frame_time),
                sensor_size[1],
                sensor_size[0],
            ),
        )

    @parameterized.expand(
        [("xytp", (5, 5), 100), ("typx", (3, 3), 10),]
    )
    def testToTimesurface(self, ordering, surface_dimensions, tau):
        (
            orig_events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        merge_polarities = True

        surfaces = F.to_timesurface_numpy(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            surface_dimensions=surface_dimensions,
            tau=tau,
            merge_polarities=merge_polarities,
        )
        self.assertEqual(surfaces.shape[0], len(orig_events))
        self.assertEqual(surfaces.shape[1], 1)
        self.assertEqual(surfaces.shape[2:], surface_dimensions)

    @parameterized.expand(
        ["xytp", "typx",]
    )
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

    @parameterized.expand(
        ["xytp", "typx",]
    )
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
