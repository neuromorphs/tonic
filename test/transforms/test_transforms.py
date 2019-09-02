import unittest
import numpy as np
import spike_data_augmentation.transforms as transforms
import utils


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.random_xytp = utils.create_random_input_xytp()

    def testTimeJitter(self):
        original_events = self.random_xytp[0].copy()
        events = self.random_xytp[0].copy()
        variance = 0.1
        transform = transforms.Compose([transforms.TimeJitter(variance=variance)])
        transform(events, self.random_xytp[2], self.random_xytp[3])

        self.assertTrue(len(events) == len(original_events))
        self.assertTrue((events[:, 0] == original_events[:, 0]).all())
        self.assertTrue((events[:, 1] == original_events[:, 1]).all())
        self.assertFalse((events[:, 2] == original_events[:, 2]).all())
        self.assertTrue((events[:, 3] == original_events[:, 3]).all())
        self.assertTrue(
            np.isclose(events[:, 2].all(), original_events[:, 2].all(), atol=variance)
        )
