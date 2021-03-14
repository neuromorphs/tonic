import unittest
import numpy as np
from parameterized import parameterized
import tonic.datasets as datasets


# @unittest.skip("Super slow!")
class TestDatasets(unittest.TestCase):
    download = False

    @parameterized.expand([(True, 440860, 7, 1077), (False, 288123, 7, 264)])
    def testDVSGesture(self, train, n_events, true_label, n_samples):
        dataset = datasets.DVSGesture(
            save_to="./data", train=train, download=self.download
        )
        dataloader = datasets.DataLoader(dataset, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[1], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand(
        [
            (True, False, 4733, 0, 60000),
            (False, False, 5100, 0, 10000),
            (False, True, 1706, 0, 10000),
        ]
    )
    def testNMNIST(self, train, first_saccade_only, n_events, true_label, n_samples):
        dataset = datasets.NMNIST(
            save_to="./data",
            train=train,
            download=self.download,
            first_saccade_only=first_saccade_only,
        )
        dataloader = datasets.DataLoader(dataset, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[1], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand([(True, 10066, 0, 15422), (False, 4938, 0, 8607)])
    def testNCARS(self, train, n_events, true_label, n_samples):
        dataset = datasets.NCARS(save_to="./data", train=train, download=self.download)
        dataloader = datasets.DataLoader(dataset, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[1], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand([(True, 3773, 0, 48), (False, 2515, 0, 20)])
    def testPOKERDVS(self, train, n_events, true_label, n_samples):
        dataset = datasets.POKERDVS(
            save_to="./data", train=train, download=self.download
        )
        dataloader = datasets.DataLoader(dataset, batch_size=None, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[0], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand([(163302, "BACKGROUND_Google", 8709)])
    def testNCALTECH101(self, n_events, true_label, n_samples):
        dataset = datasets.NCALTECH101(save_to="./data", download=self.download)
        dataloader = datasets.DataLoader(dataset, batch_size=None, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[0], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand([(915556, 0, 304, True), (227321, 0, 1342, False)])
    def testNavGestures(self, n_events, true_label, n_samples, walk_subset):
        dataset = datasets.NavGesture(
            save_to="./data", walk_subset=walk_subset, download=self.download
        )
        dataloader = datasets.DataLoader(dataset, batch_size=None, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[0], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)

    @parameterized.expand([(15951, 0, 100800)])
    def testASLDVS(self, n_events, true_label, n_samples):
        dataset = datasets.ASLDVS(save_to="./data", download=self.download)
        dataloader = datasets.DataLoader(dataset, batch_size=None, shuffle=False)
        events, label = next(iter(dataloader))

        self.assertEqual(events.shape[0], n_events)
        self.assertEqual(label, true_label)
        self.assertEqual(len(dataset), n_samples)
