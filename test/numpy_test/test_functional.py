import unittest

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

        print(self.random_xytp[2], self.random_xytp[0][0, 0])
        print(events[0, 0])

        self.assertEqual(
            self.random_xytp[2][0] - self.random_xytp[0][0, 0],
            original_x,
            "When flipping left and right x must map to the opposite pixel, i.e. x' = sensor width - x",
        )
