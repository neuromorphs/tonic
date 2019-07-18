import unittest
import numpy as np

import spike_data_augmentation.functional as F

import utils


class TestFunctionalAPI(unittest.TestCase):
    def setUp(self):
        self.random_xytp = utils.create_random_input_xytp()

    def testFlipLR(self):
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

    def testFlipUD(self):
        original_y = self.random_xytp[0][0, 1].copy()

        events, images = F.flip_ud_numpy(
            self.random_xytp[0],
            images=self.random_xytp[1],
            sensor_size=self.random_xytp[2],
            ordering=self.random_xytp[3],
            multi_image=self.random_xytp[4],
            flip_probability=1.0,
        )

        print(self.random_xytp[2], self.random_xytp[0][0, 1])
        print(events[0, 0])

        same_pixel = np.isclose(
            self.random_xytp[2][1] - self.random_xytp[0][0, 1], original_y
        )

        self.assertTrue(
            same_pixel,
            "When flipping up and down y must map to the opposite pixel, i.e. y' = sensor width - y",
        )

    def testMixEv(self):

        stream_1 = utils.create_random_input_xytp()
        stream_2 = utils.create_random_input_xytp()
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
