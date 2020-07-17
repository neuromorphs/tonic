import unittest
import numpy as np
import tonic.datasets as datasets


#@unittest.skip("Super slow!")
class TestDatasets(unittest.TestCase):
    def testIBMGesturetraining(self):
        trainingset = datasets.IBMGesture(save_to="./data", train=True)
        trainloader = datasets.Dataloader(trainingset, shuffle=False)
        events, label = next(iter(trainloader))

        self.assertEqual(events.shape[0], 440860)
        self.assertEqual(label, 7)
        self.assertEqual(len(trainingset), 1077)

    def testIBMGesturetesting(self):
        testset = datasets.IBMGesture(save_to="./data", train=False)
        testloader = datasets.Dataloader(testset, shuffle=False)
        events, label = next(iter(testloader))

        self.assertEqual(events.shape[0], 288123)
        self.assertEqual(label, 7)
        self.assertEqual(len(testset), 264)

    def testNMNISTtraining(self):
        trainingset = datasets.NMNIST(save_to="./data", train=True)
        trainloader = datasets.Dataloader(trainingset, shuffle=False)
        events, label = next(iter(trainloader))

        self.assertEqual(events.shape[0], 4733)
        self.assertEqual(label, 0)
        self.assertEqual(len(trainingset), 60000)

    def testNMNISTtesting(self):
        testset = datasets.NMNIST(save_to="./data", train=False)
        testloader = datasets.Dataloader(testset, shuffle=False)
        events, label = next(iter(testloader))

        self.assertEqual(events.shape[0], 5100)
        self.assertEqual(label, 0)
        self.assertEqual(len(testset), 10000)
