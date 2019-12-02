import unittest
import numpy as np
import spike_data_augmentation.datasets as datasets


class TestDatasets(unittest.TestCase):
    def testIBMGesturetraining(self):
        trainingset = datasets.IBMGesture(save_to="./data", train=True)
        trainloader = datasets.Dataloader(trainingset, shuffle=False)
        events, label = next(iter(trainloader))

        self.assertEqual(events.shape[0], 3000)
        self.assertEqual(label, 0)
        self.assertEqual(len(trainingset), 60000)

    def testIBMGesturetesting(self):
        testset = datasets.IBMGesture(save_to="./data", train=False)
        testloader = datasets.Dataloader(testset, shuffle=False)
        events, label = next(iter(testloader))

        self.assertEqual(events.shape[0], 3000)
        self.assertEqual(label, 0)
        self.assertEqual(len(testset), 10000)

    def testNMNISTtraining(self):
        trainingset = datasets.NMNIST(save_to="./data", train=True)
        trainloader = datasets.Dataloader(trainingset, shuffle=False)
        events, label = next(iter(trainloader))

        self.assertEqual(events.shape[0], 3000)
        self.assertEqual(label, 0)
        self.assertEqual(len(trainingset), 60000)

    def testNMNISTtesting(self):
        testset = datasets.NMNIST(save_to="./data", train=False)
        testloader = datasets.Dataloader(testset, shuffle=False)
        events, label = next(iter(testloader))

        self.assertEqual(events.shape[0], 3000)
        self.assertEqual(label, 0)
        self.assertEqual(len(testset), 10000)
