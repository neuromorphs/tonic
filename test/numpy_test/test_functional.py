import unittest
import numpy as np
import spike_data_augmentation.functional as F
import utils


class TestFunctionalAPI(unittest.TestCase):
    def setUp(self):
        self.random_xytp = utils.create_random_input_with_ordering("xytp")
        self.random_txyp = utils.create_random_input_with_ordering("txyp")

    def testFlipLRxytp(self):
        original_x = self.random_xytp[0][0, 0].copy()

        events, images = F.flip_lr_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            flip_probability=1.0,
        )

        same_pixel = np.isclose(
            self.random_xytp[2][0] - self.random_xytp[0][0, 0], original_x
        )

        self.assertTrue(
            same_pixel,
            "When flipping left and right x must map to the opposite pixel, i.e. x' = sensor width - x",
        )

    def testFlipLRtxyp(self):
        original_x = self.random_txyp[0][0, 1].copy()

        events, images = F.flip_lr_numpy(
            self.random_txyp[0],
            images=self.random_txyp[1],
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
            multi_image=self.random_txyp[4],
            flip_probability=1.0,
        )

        same_pixel = np.isclose(
            self.random_txyp[2][0] - self.random_txyp[0][0, 1], original_x
        )

        self.assertTrue(
            same_pixel,
            "When flipping left and right x must map to the opposite pixel, i.e. x' = sensor width - x",
        )

    def testFlipPolarity(self):
        original_polarities = self.random_xytp[0][:, 3].copy()

        events = F.flip_polarity_numpy(
            self.random_xytp[0], flip_probability=1, ordering=self.random_xytp[3]
        )

        self.assertTrue(
            np.array_equal(original_polarities * -1, events[:, 3]),
            "When flipping polarity with probability 1, all event polarities must flip",
        )

        self.random_xytp[0][:, 3] = original_polarities.copy()

        events = F.flip_polarity_numpy(
            self.random_xytp[0], flip_probability=0, ordering=self.random_xytp[3]
        )

        self.assertTrue(
            np.array_equal(original_polarities, events[:, 3]),
            "When flipping polarity with probability 0, no event polarities must flip",

    def testFlipUDxytp(self):
        original_y = self.random_xytp[0][0, 1].copy()

        events, images = F.flip_ud_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            flip_probability=1.0,
        )

        same_pixel = np.isclose(
            self.random_xytp[2][1] - self.random_xytp[0][0, 1], original_y
        )

        self.assertTrue(
            same_pixel,
            "When flipping up and down y must map to the opposite pixel, i.e. y' = sensor width - y",
        )

    def testFlipUDtxyp(self):
        original_y = self.random_txyp[0][0, 2].copy()

        events, images = F.flip_ud_numpy(
            self.random_txyp[0],
            images=self.random_txyp[1],
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
            multi_image=self.random_txyp[4],
            flip_probability=1.0,
        )

        same_pixel = np.isclose(
            self.random_txyp[2][1] - self.random_txyp[0][0, 2], original_y
        )

        self.assertTrue(
            same_pixel,
            "When flipping up and down y must map to the opposite pixel, i.e. y' = sensor width - y",
        )

    def testEventDropoutXytp(self):
        original = self.random_xytp[0]
        drop_probability = 0.2

        events = F.drop_event_numpy(original, drop_probability=drop_probability)

        self.assertTrue(
            np.isclose(events.shape[0], (1 - drop_probability) * original.shape[0]),
            "Event dropout should result in drop_probability*len(original) events dropped out.",
        )

        self.assertTrue(
            np.isclose(np.sum((events[:, 2] - np.sort(events[:, 2])) ** 2), 0),
            "Event dropout should maintain temporal order.",
        )

        events = F.drop_event_numpy(
            original, drop_probability=drop_probability, random_drop_probability=True
        )

        self.assertTrue(
            events.shape[0] >= (1 - drop_probability) * original.shape[0],
            "Event dropout with random drop probability should result in less than drop_probability*len(original) events dropped out.",
        )

    def testEventDropoutTxyp(self):
        original = self.random_txyp[0]
        drop_probability = 0.2

        events = F.drop_event_numpy(original, drop_probability=drop_probability)

        self.assertTrue(
            np.isclose(events.shape[0], (1 - drop_probability) * original.shape[0]),
            "Event dropout should result in drop_probability*len(original) events dropped out.",
        )

        self.assertTrue(
            np.isclose(np.sum((events[:, 0] - np.sort(events[:, 0])) ** 2), 0),
            "Event dropout should maintain temporal order.",
        )

        events = F.drop_event_numpy(
            original, drop_probability=drop_probability, random_drop_probability=True
        )

        self.assertTrue(
            events.shape[0] >= (1 - drop_probability) * original.shape[0],
            "Event dropout with random drop probability should result in less than drop_probability*len(original) events dropped out.",
        )

    def testSpatialJitterXytp(self):
        original_events = self.random_xytp[0].copy()
        variance = 3

        events = F.spatial_jitter_numpy(
            self.random_xytp[0],
            ordering=self.random_xytp[3],
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
        )

        self.assertTrue(len(events) == len(original_events))
        self.assertTrue((events[:, 2] == original_events[:, 2]).all())
        self.assertTrue((events[:, 3] == original_events[:, 3]).all())
        self.assertFalse((events[:, 0] == original_events[:, 0]).all())
        self.assertFalse((events[:, 1] == original_events[:, 1]).all())
        self.assertTrue(
            np.isclose(events[:, 0].all(), original_events[:, 0].all(), atol=variance)
        )
        self.assertTrue(
            np.isclose(events[:, 1].all(), original_events[:, 1].all(), atol=variance)
        )

    def testSpatialJitterTxyp(self):
        original_events = self.random_txyp[0].copy()
        variance = 2

        events = F.spatial_jitter_numpy(
            self.random_txyp[0],
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
        )

        self.assertTrue(len(events) == len(original_events))
        self.assertTrue((events[:, 0] == original_events[:, 0]).all())
        self.assertTrue((events[:, 3] == original_events[:, 3]).all())
        self.assertFalse((events[:, 1] == original_events[:, 1]).all())
        self.assertFalse((events[:, 2] == original_events[:, 2]).all())
        self.assertTrue(
            np.isclose(events[:, 1].all(), original_events[:, 1].all(), atol=variance)
        )
        self.assertTrue(
            np.isclose(events[:, 2].all(), original_events[:, 2].all(), atol=variance)
        )
          
    def testTimeJitter(self):
        original_events = self.random_xytp[0].copy()
        variance = 0.1
        events = F.time_jitter_numpy(
            self.random_xytp[0], ordering=self.random_xytp[3], variance=variance
        )

        self.assertTrue(len(events) == len(original_events))
        self.assertTrue((events[:, 0] == original_events[:, 0]).all())
        self.assertTrue((events[:, 1] == original_events[:, 1]).all())
        self.assertFalse((events[:, 2] == original_events[:, 2]).all())
        self.assertTrue((events[:, 3] == original_events[:, 3]).all())

    def testMixEvXytp(self):
        stream_1 = utils.create_random_input_with_ordering("xytp")
        stream_2 = utils.create_random_input_with_ordering("xytp")

        events = (stream_1[0], stream_2[0])

        mixed_events_no_offset, _ = F.mix_ev_streams(
            events,
            offsets=None,
            check_conflicts=False,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
        )

        mixed_events_random_offset, _ = F.mix_ev_streams(
            events,
            offsets="Random",
            check_conflicts=False,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
        )

        mixed_events_defined_offset, _ = F.mix_ev_streams(
            events,
            offsets=(0, 100),
            check_conflicts=False,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
        )

        mixed_events_conflict, num_conflicts = F.mix_ev_streams(
            (stream_1[0], stream_1[0]),
            offsets=None,
            check_conflicts=True,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
        )

        no_offset_monotonic = np.all(
            mixed_events_no_offset[1:, 2] >= mixed_events_no_offset[:-1, 2], axis=0
        )
        random_offset_monotonic = np.all(
            mixed_events_random_offset[1:, 2] >= mixed_events_random_offset[:-1, 2],
            axis=0,
        )
        defined_offset_monotonic = np.all(
            mixed_events_defined_offset[1:, 2] >= mixed_events_defined_offset[:-1, 2],
            axis=0,
        )
        conflict_offset_monotonic = np.all(
            mixed_events_conflict[1:, 2] >= mixed_events_conflict[:-1, 2], axis=0
        )
        all_colisions_detected = len(stream_1[0]) == num_conflicts

        self.assertTrue(
            all_colisions_detected,
            "Missed some event colisions, may cause processing problems.",
        )
        self.assertTrue(no_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(random_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(defined_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(conflict_offset_monotonic, "Result was not monotonic.")

    def testMixEvTxyp(self):
        stream_1 = utils.create_random_input_with_ordering("txyp")
        stream_2 = utils.create_random_input_with_ordering("txyp")
        events = (stream_1[0], stream_2[0])

        mixed_events_no_offset, _ = F.mix_ev_streams(
            events,
            offsets=None,
            check_conflicts=False,
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
        )

        mixed_events_random_offset, _ = F.mix_ev_streams(
            events,
            offsets="Random",
            check_conflicts=False,
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
        )

        mixed_events_defined_offset, _ = F.mix_ev_streams(
            events,
            offsets=(0, 100),
            check_conflicts=False,
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
        )

        mixed_events_conflict, num_conflicts = F.mix_ev_streams(
            (stream_1[0], stream_1[0]),
            offsets=None,
            check_conflicts=True,
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
        )

        no_offset_monotonic = np.all(
            mixed_events_no_offset[1:, 0] >= mixed_events_no_offset[:-1, 0], axis=0
        )
        random_offset_monotonic = np.all(
            mixed_events_random_offset[1:, 0] >= mixed_events_random_offset[:-1, 0],
            axis=0,
        )
        defined_offset_monotonic = np.all(
            mixed_events_defined_offset[1:, 0] >= mixed_events_defined_offset[:-1, 0],
            axis=0,
        )
        conflict_offset_monotonic = np.all(
            mixed_events_conflict[1:, 0] >= mixed_events_conflict[:-1, 0], axis=0
        )
        all_colisions_detected = len(stream_1[0]) == num_conflicts

        self.assertTrue(
            all_colisions_detected,
            "Missed some event colisions, may cause processing problems.",
        )
        self.assertTrue(no_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(random_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(defined_offset_monotonic, "Result was not monotonic.")
        self.assertTrue(conflict_offset_monotonic, "Result was not monotonic.")

    def testRefractoryPeriodXytp(self):
        original_events = self.random_xytp[0].copy()

        augmented_events = F.refractory_period_numpy(
            original_events,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            refractory_period=0.1,
        )

        self.assertTrue(
            len(augmented_events) <= len(original_events),
            "Result can not be longer than original event stream",
        )
        self.assertTrue(
            np.isin(augmented_events, original_events).all(),
            "Added additional events that were not present in original event stream",
        )

    def testRefractoryPeriodTxyp(self):
        original_events = self.random_txyp[0].copy()

        augmented_events = F.refractory_period_numpy(
            original_events,
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
            refractory_period=0.1,
        )

        self.assertTrue(
            len(augmented_events) <= len(original_events),
            "Result can not be longer than original event stream",
        )
        self.assertTrue(
            np.isin(augmented_events, original_events).all(),
            "Added additional events that were not present in original event stream",
        )

    def testUniformNoise(self):
        original_events = self.random_xytp[0].copy()

        noisy_events = F.uniform_noise_numpy(
            original_events,
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            scaling_factor_to_micro_sec=1000000,
            noise_density=1e-8,
        )

        self.assertTrue(len(noisy_events) > len(original_events))
        self.assertTrue(np.isin(original_events, noisy_events).all())

    def testTimeSkew(self):
        original_events = self.random_xytp[0].copy()

        augmented_events = F.time_skew_numpy(
            original_events, ordering=self.random_xytp[3], coefficient=3.1, offset=100
        )

        self.assertTrue(len(augmented_events) == len(original_events))
        self.assertTrue((augmented_events[:, 2] >= original_events[:, 2]).all())
        self.assertTrue(np.min(augmented_events[:, 2]) >= 0)

    def testTemporalFlipXytp(self):
        original_t = self.random_xytp[0][0, 2].copy()
        original_p = self.random_xytp[0][0, 3].copy()

        max_t = np.max(self.random_xytp[0][:, 2])

        events, images = F.time_reversal_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            flip_probability=1.0,
        )

        same_time = np.isclose(max_t - original_t, events[0, 2])

        same_polarity = np.isclose(events[0, 3], -1.0 * original_p)

        self.assertTrue(same_time, "When flipping time must map t_i' = max(t) - t_i")
        self.assertTrue(same_polarity, "When flipping time polarity should be flipped")

    def testTemporalFlipTxyp(self):
        original_t = self.random_txyp[0][0, 0].copy()
        original_p = self.random_txyp[0][0, 3].copy()

        max_t = np.max(self.random_txyp[0][:, 0])

        events, images = F.time_reversal_numpy(
            self.random_txyp[0],
            images=self.random_txyp[1],
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
            multi_image=self.random_txyp[4],
            flip_probability=1.0,
        )

        same_time = np.isclose(max_t - original_t, events[0, 0])

        same_polarity = np.isclose(events[0, 3], -1.0 * original_p)

        self.assertTrue(same_time, "When flipping time must map t_i' = max(t) - t_i")
        self.assertTrue(same_polarity, "When flipping time polarity should be flipped")

    def testCropXytp(self):
        events, images = F.crop_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            target_size=(50, 50),
        )

        self.assertTrue(
            np.all(events[:, 0]) < 50 and np.all(events[:, 1] < 50),
            "Cropping needs to map the events into the new space",
        )

        self.assertTrue(
            images.shape[1] == 50 and images.shape[2] == 50,
            "Cropping needs to map the images into the new space",
        )

    def testCropTxyp(self):
        events, images = F.crop_numpy(
            self.random_txyp[0],
            images=self.random_txyp[1],
            sensor_size=self.random_txyp[2],
            ordering=self.random_txyp[3],
            multi_image=self.random_txyp[4],
            target_size=(50, 50),
        )

        self.assertTrue(
            np.all(events[:, 0]) < 50 and np.all(events[:, 1] < 50),
            "Cropping needs to map the events into the new space",
        )

        self.assertTrue(
            images.shape[1] == 50 and images.shape[2] == 50,
            "Cropping needs to map the images into the new space",

    def testStTransform(self):
        spatial_transform = np.array(((1, 0, 10), (0, 1, 10), (0, 0, 1)))
        temporal_transform = np.array((2, 0))
        events = F.st_transform(
            self.random_xytp[0],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            roll=False,
        )

        self.assertTrue(
            np.all(events[:, 0]) < self.random_xytp[2][0]
            and np.all(events[:, 1] < self.random_xytp[2][1]),
            "Transformation does not map beyond sensor size",
        )
