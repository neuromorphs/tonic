import unittest
import numpy as np
import tonic.datasets as datasets


@unittest.skip("Super slow!")
class TestDatasets(unittest.TestCase):
    def testIBMGesturetraining(self):
        trainingset = datasets.IBMGesture(save_to="./data", train=True)
        trainloader = datasets.Dataloader(trainingset, shuffle=False)
        events, label = next(iter(trainloader))

        self.assertEqual(events.shape[0], 520335)
        self.assertEqual(label, 8)
        self.assertEqual(len(trainingset), 1077)

    def testIBMGesturetesting(self):
        testset = datasets.IBMGesture(save_to="./data", train=False)
        testloader = datasets.Dataloader(testset, shuffle=False)
        events, label = next(iter(testloader))

        self.assertEqual(events.shape[0], 546808)
        self.assertEqual(label, 8)
        self.assertEqual(len(testset), 264)

    def testNMNISTtraining(self):
        trainingset = datasets.NMNIST(save_to="./data", train=True)
        trainloader = datasets.Dataloader(trainingset, shuffle=False)
        events, label = next(iter(trainloader))

        self.assertEqual(events.shape[0], 4893)
        self.assertEqual(label, 0)
        self.assertEqual(len(trainingset), 60000)

    def testNMNISTtesting(self):
        testset = datasets.NMNIST(save_to="./data", train=False)
        testloader = datasets.Dataloader(testset, shuffle=False)
        events, label = next(iter(testloader))

        self.assertEqual(events.shape[0], 5412)
        self.assertEqual(label, 0)
        self.assertEqual(len(testset), 10000)
