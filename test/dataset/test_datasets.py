import unittest
import numpy as np
from parameterized import parameterized
import tonic.datasets as datasets


# @unittest.skip("Super slow!")
class TestDatasets(unittest.TestCase):
    download = False

    @parameterized.expand(
        [(True, 440860, 7, 1077), (False, 288123, 7, 264),]
    )
    def testIBMGesture(self, train, n_events, true_label, n_samples):
        dataset = datasets.IBMGesture(
            save_to="./data", train=train, download=self.download
        )
        dataloader = datasets.DataLoader(dataset, batch_size=None, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[0], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand(
        [(True, 4733, 0, 60000), (False, 5100, 0, 10000),]
    )
    def testNMNIST(self, train, n_events, true_label, n_samples):
        dataset = datasets.NMNIST(save_to="./data", train=train, download=self.download)
        dataloader = datasets.DataLoader(dataset, batch_size=None, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[0], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)
