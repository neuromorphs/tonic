import unittest
import numpy as np
import pytest
import tonic.datasets as datasets
import dataset_utils
import os
from utils import create_random_input
import itertools


class DVSGestureTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.DVSGesture
    FEATURE_TYPES = (datasets.DVSGesture.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "ibmGestureTrain/user24_led")
        os.makedirs(testfolder, exist_ok=True)
        events, sensor_size = create_random_input(dtype=datasets.DVSGesture.dtype)
        events = np.lib.recfunctions.structured_to_unstructured(events)
        np.save(testfolder+"/0.npy", events)
        np.save(testfolder+"/1.npy", events)
        return {"n_samples": 2}

class DVSGestureTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.DVSGesture
    FEATURE_TYPES = (datasets.DVSGesture.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "ibmGestureTest/user24_led")
        os.makedirs(testfolder, exist_ok=True)
        events, sensor_size = create_random_input(dtype=datasets.DVSGesture.dtype)
        events = np.lib.recfunctions.structured_to_unstructured(events)
        np.save(testfolder+"/0.npy", events)
        return {"n_samples": 1}